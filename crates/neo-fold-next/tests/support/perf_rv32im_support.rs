// Performance/debugging reports for the current RV32IM proof path.

use std::collections::BTreeSet;
use std::env;
use std::io::{self, Write};
use std::time::Instant;

use neo_fold_next::nightstream::rv32im::{
    build_rv32im_nightstream_from_published_proof_seam_with_perf, verify_rv32im_nightstream_with_perf,
};
use neo_fold_next::proof::{FoldSchedule, PackagedProof};
use neo_fold_next::rv32im::audit::audit_rv32im_public_proof_with_perf;
use neo_fold_next::rv32im::audit::build_rv32im_published_proof_seam_with_perf;
use neo_fold_next::rv32im::ccs::{rv32im_root_main_lane_ccs, RV32IM_ROOT_PUBLIC_INPUTS, RV32IM_ROOT_ROW_WIDTH};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::ivc::Rv32imIvcState;
use neo_fold_next::rv32im::layout::RV32_REGISTER_COUNT;
use neo_fold_next::rv32im::stage1::build_stage1_summary;
use neo_fold_next::rv32im::stage2::{build_stage2_summary, RamAccessKind, RegisterReadRole};
use neo_fold_next::rv32im::stage3::build_stage3_summary;
use neo_fold_next::rv32im::tables::Rv32FamilyTag;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_parity_case_from_source, build_program,
    build_rv32im_accepted_proof_artifact, build_rv32im_chunk_step_ivc_relations, build_simple_kernel_witness_with_perf,
    mixed_opcode_perf_expected_x1, prove_rv32im_public_proof_with_options_and_perf, rv32im_simple_root_params,
    setup_rv32im_ivc_snark_cached, setup_rv32im_ivc_snark_cached_with_trace,
    setup_rv32im_ivc_snark_from_final_cached, OpeningAccumulator, OpeningAccumulatorStats, OpeningPointLabel, Rv32Program,
    Rv32State, Rv32imProofInput, Rv32imPublicProofOptions, SimpleKernelBuildPerf,
    RV32IM_MIXED_OPCODE_PERF_BLOCK_LEN, RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
use serde::Serialize;

const FAMILY_ORDER: [Rv32FamilyTag; 7] = [
    Rv32FamilyTag::NativeAlu,
    Rv32FamilyTag::AlignedMemory,
    Rv32FamilyTag::NarrowMemory,
    Rv32FamilyTag::Multiply,
    Rv32FamilyTag::UnsignedDivRem,
    Rv32FamilyTag::SignedDivRem,
    Rv32FamilyTag::ControlFlow,
];

#[derive(Clone, Copy, Default)]
struct FamilyRowStats {
    rows: usize,
    real_rows: usize,
    effect_rows: usize,
    commit_rows: usize,
}

#[derive(Clone, Copy, Default)]
struct LookupSummary {
    register_reads: usize,
    register_reads_rs1: usize,
    register_reads_rs2: usize,
    unique_read_regs: usize,
    register_writes: usize,
    unique_write_regs: usize,
    ram_events: usize,
    ram_reads: usize,
    ram_writes: usize,
    unique_ram_addrs: usize,
    twist_links: usize,
    twist_write_routes: usize,
    twist_memory_before_routes: usize,
    twist_memory_after_routes: usize,
}

#[derive(Clone, Copy, Default)]
struct ExactOpeningClaimStats {
    claims: usize,
    logical_width: usize,
    packed_rows: usize,
    packed_cols: usize,
}

#[derive(Clone, Copy, Default)]
struct PackagedProofStats {
    public_steps: usize,
    public_chunks: usize,
    proof_chunks: usize,
    final_main_claims: usize,
    ccs_outputs: usize,
    dec_children: usize,
}

#[derive(Clone, Copy, Default)]
struct OpeningSurfaceTotals {
    exact_claims: usize,
    flatten_u64_words: usize,
    logical_width: usize,
    packed_rows: usize,
    packed_cols: usize,
    selected_labels: usize,
    selected_claim_words: usize,
    packaged_public_steps: usize,
    packaged_public_chunks: usize,
    packaged_proof_chunks: usize,
    packaged_final_main_claims: usize,
    packaged_ccs_outputs: usize,
    packaged_dec_children: usize,
}

#[derive(Clone, Copy, Default)]
struct OpeningLabelBuckets {
    stage1: usize,
    stage2: usize,
    stage3: usize,
    kernel_binding: usize,
    kernel_prepared_steps: usize,
}

#[derive(Clone, Copy)]
struct ExactStagePerfRow<'a> {
    label: &'a str,
    records: usize,
    selected_labels: usize,
    selected_claim_words: usize,
    flatten_u64_words: usize,
    field_limb_width: usize,
    packed_rows: usize,
    packed_cols: usize,
    flatten_ms: f64,
    limb_encode_ms: f64,
    context_setup_ms: f64,
    ccs_encode_ms: f64,
    ajtai_commit_ms: f64,
    opening_manifest_ms: f64,
    opening_prove_ms: f64,
}

#[derive(Clone, Copy)]
struct SerializedSizeRow<'a> {
    label: &'a str,
    bytes: usize,
}

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

fn family_label(family: Rv32FamilyTag) -> &'static str {
    match family {
        Rv32FamilyTag::NativeAlu => "native_alu",
        Rv32FamilyTag::AlignedMemory => "aligned_memory",
        Rv32FamilyTag::NarrowMemory => "narrow_memory",
        Rv32FamilyTag::Multiply => "multiply",
        Rv32FamilyTag::UnsignedDivRem => "unsigned_divrem",
        Rv32FamilyTag::SignedDivRem => "signed_divrem",
        Rv32FamilyTag::ControlFlow => "control_flow",
    }
}

fn family_index(family: Rv32FamilyTag) -> usize {
    match family {
        Rv32FamilyTag::NativeAlu => 0,
        Rv32FamilyTag::AlignedMemory => 1,
        Rv32FamilyTag::NarrowMemory => 2,
        Rv32FamilyTag::Multiply => 3,
        Rv32FamilyTag::UnsignedDivRem => 4,
        Rv32FamilyTag::SignedDivRem => 5,
        Rv32FamilyTag::ControlFlow => 6,
    }
}

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn per_unit(ms: f64, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        ms / units as f64
    }
}

fn format_ms_per_opcode(ms: f64, opcode_count: usize) -> String {
    format!("{ms:.3} ms ({:.4} ms/op)", per_unit(ms, opcode_count))
}

fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:30} {}", label, value);
}

fn format_count(value: usize) -> String {
    let raw = value.to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3);
    for (idx, ch) in raw.chars().rev().enumerate() {
        if idx != 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

// ── Tree-formatted timing helpers ──────────────────────────────────────────

const BAR_WIDTH: usize = 20;

fn bar_str(ms: f64, max_ms: f64) -> String {
    if max_ms <= 0.0 {
        return " ".repeat(BAR_WIDTH);
    }
    let filled = ((ms / max_ms) * BAR_WIDTH as f64).round() as usize;
    let filled = filled.min(BAR_WIDTH);
    format!("{}{}", "█".repeat(filled), " ".repeat(BAR_WIDTH - filled))
}

fn pct_str(ms: f64, total_ms: f64) -> String {
    if total_ms <= 0.0 {
        return String::new();
    }
    format!("{:5.1}%", ms / total_ms * 100.0)
}

/// Print a tree row with optional bar+percent.  `show_bar` = false for deep children.
fn tree_row(prefix: &str, label: &str, ms: f64, max_ms: f64, total_ms: f64, show_bar: bool) {
    if show_bar {
        println!(
            "  {}{:.<30} {:>8.1} ms  {}  {}",
            prefix,
            format!("{} ", label),
            ms,
            bar_str(ms, max_ms),
            pct_str(ms, total_ms),
        );
    } else {
        println!("  {}{:.<30} {:>8.1} ms", prefix, format!("{} ", label), ms);
    }
}

fn tree_row_annotated(prefix: &str, label: &str, ms: f64, note: &str) {
    println!("  {}{:.<30} {:>8.1} ms  {}", prefix, format!("{} ", label), ms, note,);
}

fn tree_header(title: &str, total_ms: f64, ms_per_op: f64) {
    println!();
    println!("  {} {:>8.1} ms  ({:.2} ms/op)", title, total_ms, ms_per_op);
    println!("  {}", "─".repeat(72));
}

fn format_fold_schedule(schedule: FoldSchedule) -> String {
    match schedule {
        FoldSchedule::WholeTrace => "WholeTrace".to_string(),
        FoldSchedule::RowsPerChunk(rows) => format!("RowsPerChunk({rows})"),
    }
}

fn collect_unique_opcode_labels(build: &neo_fold_next::rv32im::builder::Rv32ProgramBuild) -> String {
    let mut labels = BTreeSet::new();
    for step in &build.executed_steps {
        labels.insert(format!("{:?}", step.decoded.opcode));
    }
    labels.into_iter().collect::<Vec<_>>().join(", ")
}

fn print_timing_table(title: &str, rows: &[(&str, f64)], opcode_count: usize, execution_rows: usize) {
    print_section(title);
    println!("  {:26} {:>12} {:>14} {:>14}", "phase", "wall ms", "ms/op", "ms/row");
    for (label, ms) in rows {
        println!(
            "  {:26} {:>12.3} {:>14.4} {:>14.4}",
            label,
            ms,
            per_unit(*ms, opcode_count),
            per_unit(*ms, execution_rows),
        );
    }
}

fn serialized_size_bytes<T: Serialize>(value: &T) -> usize {
    bincode::serialize(value)
        .expect("serialize perf snapshot component")
        .len()
}

fn bytes_to_kib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

fn is_direct_child_of_total(label: &str, total_label: &str) -> bool {
    let Some((root, _)) = total_label.split_once('.') else {
        return false;
    };
    let Some((label_root, rest)) = label.split_once('.') else {
        return false;
    };
    label_root == root && !rest.contains('.') && rest != "total"
}

fn serialized_size_share(label: &str, total_label: &str, bytes: usize, total_bytes: usize) -> String {
    if total_bytes == 0 {
        return "--".to_string();
    }
    if label == total_label || is_direct_child_of_total(label, total_label) {
        return format!("{:.2}%", bytes as f64 * 100.0 / total_bytes as f64);
    }
    "--".to_string()
}

fn print_serialized_size_table(title: &str, rows: &[SerializedSizeRow<'_>], total_bytes: usize) {
    print_section(title);
    let total_label = rows
        .first()
        .map(|row| row.label)
        .expect("serialized size table must have a total row");
    println!("  {:48} {:>14} {:>11} {:>9}", "component", "bytes", "KiB", "share");
    for row in rows {
        println!(
            "  {:48} {:>14} {:>11.3} {:>9}",
            row.label,
            format_count(row.bytes),
            bytes_to_kib(row.bytes),
            serialized_size_share(row.label, total_label, row.bytes, total_bytes),
        );
    }
    println!();
    println!("  note: share is shown only for the total row and its direct children.");
    println!("  note: nested rows are standalone bincode sizes for inspection and overlap heavily.");
}

fn print_hotspot_table(title: &str, total_ms: f64, opcode_count: usize, rows: &[(&str, f64)], limit: usize) {
    let mut rows = rows.to_vec();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    print_section(title);
    println!("  {:32} {:>10} {:>10} {:>10}", "phase", "wall ms", "ms/op", "% total");
    for (label, ms) in rows.into_iter().take(limit) {
        println!(
            "  {:32} {:>10.3} {:>10.4} {:>10.2}",
            label,
            ms,
            per_unit(ms, opcode_count),
            if total_ms <= 0.0 { 0.0 } else { ms * 100.0 / total_ms }
        );
    }
}

fn exact_stage_path_is_live(rows: &[ExactStagePerfRow<'_>]) -> bool {
    rows.iter().any(|row| {
        row.records != 0
            && (row.packed_rows != 0
                || row.packed_cols != 0
                || row.flatten_u64_words != 0
                || row.field_limb_width != 0
                || row.flatten_ms != 0.0
                || row.limb_encode_ms != 0.0
                || row.context_setup_ms != 0.0
                || row.ccs_encode_ms != 0.0
                || row.ajtai_commit_ms != 0.0
                || row.opening_manifest_ms != 0.0
                || row.opening_prove_ms != 0.0)
    })
}

fn exact_opening_claims_are_live(rows: &[(&str, ExactOpeningClaimStats)]) -> bool {
    rows.iter().any(|(_, stats)| {
        stats.claims != 0 || stats.logical_width != 0 || stats.packed_rows != 0 || stats.packed_cols != 0
    })
}

fn print_family_rows(title: &str, stats: &[FamilyRowStats], opcode_count: usize) {
    print_section(title);
    println!(
        "  {:18} {:>8} {:>8} {:>8} {:>8} {:>12}",
        "family", "rows", "real", "effect", "commit", "rows/op"
    );
    for family in FAMILY_ORDER {
        let stats = stats[family_index(family)];
        if stats.rows == 0 {
            continue;
        }
        println!(
            "  {:18} {:>8} {:>8} {:>8} {:>8} {:>12.4}",
            family_label(family),
            stats.rows,
            stats.real_rows,
            stats.effect_rows,
            stats.commit_rows,
            per_unit(stats.rows as f64, opcode_count),
        );
    }
}

fn print_lookup_summary(summary: LookupSummary, opcode_count: usize, twist_family_counts: &[usize]) {
    print_section("Lookup Summary");
    println!("  {:20} {:>10} {:>10} {:>12}", "kind", "count", "per op", "extra");
    println!(
        "  {:20} {:>10} {:>10.4} {:>12}",
        "register_reads",
        summary.register_reads,
        per_unit(summary.register_reads as f64, opcode_count),
        summary.unique_read_regs
    );
    println!(
        "  {:20} {:>10} {:>10.4} {:>12}",
        "register_writes",
        summary.register_writes,
        per_unit(summary.register_writes as f64, opcode_count),
        summary.unique_write_regs
    );
    println!(
        "  {:20} {:>10} {:>10.4} {:>12}",
        "ram_events",
        summary.ram_events,
        per_unit(summary.ram_events as f64, opcode_count),
        summary.unique_ram_addrs
    );
    println!(
        "  {:20} {:>10} {:>10.4} {:>12}",
        "twist_links",
        summary.twist_links,
        per_unit(summary.twist_links as f64, opcode_count),
        FAMILY_ORDER.len()
    );
    print_kv(
        "register_read_roles",
        format!("rs1={} rs2={}", summary.register_reads_rs1, summary.register_reads_rs2),
    );
    print_kv(
        "ram_access_split",
        format!("read={} write={}", summary.ram_reads, summary.ram_writes),
    );
    print_kv(
        "twist_routed_payloads",
        format!(
            "write={} mem_before={} mem_after={}",
            summary.twist_write_routes, summary.twist_memory_before_routes, summary.twist_memory_after_routes
        ),
    );

    println!();
    println!("  {:18} {:>8} {:>12}", "twist_family", "count", "per op");
    for family in FAMILY_ORDER {
        let count = twist_family_counts[family_index(family)];
        if count == 0 {
            continue;
        }
        println!(
            "  {:18} {:>8} {:>12.4}",
            family_label(family),
            count,
            per_unit(count as f64, opcode_count),
        );
    }
}

fn print_lookup_group_density(
    summary: LookupSummary,
    opcode_count: usize,
    twist_family_counts: &[usize],
    active_twist_family_count: usize,
) {
    print_section("Lookup Group Density");
    println!(
        "  {:20} {:>12} {:>10} {:>14} {:>16}",
        "group_kind", "active_groups", "events", "events/group", "inactive_slots"
    );
    println!(
        "  {:20} {:>12} {:>10} {:>14.4} {:>16}",
        "read_regs",
        summary.unique_read_regs,
        summary.register_reads,
        per_unit(summary.register_reads as f64, summary.unique_read_regs),
        RV32_REGISTER_COUNT.saturating_sub(summary.unique_read_regs)
    );
    println!(
        "  {:20} {:>12} {:>10} {:>14.4} {:>16}",
        "write_regs",
        summary.unique_write_regs,
        summary.register_writes,
        per_unit(summary.register_writes as f64, summary.unique_write_regs),
        RV32_REGISTER_COUNT.saturating_sub(summary.unique_write_regs)
    );
    println!(
        "  {:20} {:>12} {:>10} {:>14.4} {:>16}",
        "ram_addrs",
        summary.unique_ram_addrs,
        summary.ram_events,
        per_unit(summary.ram_events as f64, summary.unique_ram_addrs),
        "n/a"
    );
    println!(
        "  {:20} {:>12} {:>10} {:>14.4} {:>16}",
        "twist_families",
        active_twist_family_count,
        summary.twist_links,
        per_unit(summary.twist_links as f64, active_twist_family_count),
        FAMILY_ORDER.len().saturating_sub(active_twist_family_count)
    );
    print_kv(
        "used_lookup_groups (current proxy)",
        format!(
            "read_regs={} write_regs={} ram_addrs={} twist_families={}",
            summary.unique_read_regs, summary.unique_write_regs, summary.unique_ram_addrs, active_twist_family_count
        ),
    );
    print_kv(
        "avg_lookup_events_per_non-halt_opcode",
        format!(
            "reads={:.4} writes={:.4} ram={:.4} twist={:.4}",
            per_unit(summary.register_reads as f64, opcode_count),
            per_unit(summary.register_writes as f64, opcode_count),
            per_unit(summary.ram_events as f64, opcode_count),
            per_unit(summary.twist_links as f64, opcode_count),
        ),
    );
    print_kv(
        "active_twist_families",
        twist_family_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(idx, _)| family_label(FAMILY_ORDER[idx]))
            .collect::<Vec<_>>()
            .join(", "),
    );
}

fn exact_stage_perf_rows(
    output: &neo_fold_next::rv32im::SimpleKernelOutput,
    perf: &SimpleKernelBuildPerf,
) -> [ExactStagePerfRow<'static>; 3] {
    [
        ExactStagePerfRow {
            label: "stage1",
            records: output.stages.stage1.rows.len(),
            selected_labels: perf.stage_package_bundle.stage1.selected_labels,
            selected_claim_words: perf.stage_package_bundle.stage1.claim_words,
            flatten_u64_words: perf.stage_claim_bundle.stage1.flatten_u64_words,
            field_limb_width: perf.stage_claim_bundle.stage1.field_limb_width,
            packed_rows: perf.stage_claim_bundle.stage1.packed_rows,
            packed_cols: perf.stage_claim_bundle.stage1.packed_cols,
            flatten_ms: perf.stage_claim_bundle.stage1.flatten_ms,
            limb_encode_ms: perf.stage_claim_bundle.stage1.limb_encode_ms,
            context_setup_ms: perf.stage_claim_bundle.stage1.context_setup_ms,
            ccs_encode_ms: perf.stage_claim_bundle.stage1.ccs_encode_ms,
            ajtai_commit_ms: perf.stage_claim_bundle.stage1.ajtai_commit_ms,
            opening_manifest_ms: perf.stage_claim_bundle.stage1.opening_manifest_ms,
            opening_prove_ms: perf.stage_claim_bundle.stage1.opening_prove_ms,
        },
        ExactStagePerfRow {
            label: "stage2",
            records: output.stages.stage2.register_reads.len()
                + output.stages.stage2.register_writes.len()
                + output.stages.stage2.ram_events.len()
                + output.stages.stage2.twist_links.len()
                + 4,
            selected_labels: perf.stage_package_bundle.stage2.selected_labels,
            selected_claim_words: perf.stage_package_bundle.stage2.claim_words,
            flatten_u64_words: perf.stage_claim_bundle.stage2.flatten_u64_words,
            field_limb_width: perf.stage_claim_bundle.stage2.field_limb_width,
            packed_rows: perf.stage_claim_bundle.stage2.packed_rows,
            packed_cols: perf.stage_claim_bundle.stage2.packed_cols,
            flatten_ms: perf.stage_claim_bundle.stage2.flatten_ms,
            limb_encode_ms: perf.stage_claim_bundle.stage2.limb_encode_ms,
            context_setup_ms: perf.stage_claim_bundle.stage2.context_setup_ms,
            ccs_encode_ms: perf.stage_claim_bundle.stage2.ccs_encode_ms,
            ajtai_commit_ms: perf.stage_claim_bundle.stage2.ajtai_commit_ms,
            opening_manifest_ms: perf.stage_claim_bundle.stage2.opening_manifest_ms,
            opening_prove_ms: perf.stage_claim_bundle.stage2.opening_prove_ms,
        },
        ExactStagePerfRow {
            label: "stage3",
            records: output.stages.stage3.continuity.len() + 2,
            selected_labels: perf.stage_package_bundle.stage3.selected_labels,
            selected_claim_words: perf.stage_package_bundle.stage3.claim_words,
            flatten_u64_words: perf.stage_claim_bundle.stage3.flatten_u64_words,
            field_limb_width: perf.stage_claim_bundle.stage3.field_limb_width,
            packed_rows: perf.stage_claim_bundle.stage3.packed_rows,
            packed_cols: perf.stage_claim_bundle.stage3.packed_cols,
            flatten_ms: perf.stage_claim_bundle.stage3.flatten_ms,
            limb_encode_ms: perf.stage_claim_bundle.stage3.limb_encode_ms,
            context_setup_ms: perf.stage_claim_bundle.stage3.context_setup_ms,
            ccs_encode_ms: perf.stage_claim_bundle.stage3.ccs_encode_ms,
            ajtai_commit_ms: perf.stage_claim_bundle.stage3.ajtai_commit_ms,
            opening_manifest_ms: perf.stage_claim_bundle.stage3.opening_manifest_ms,
            opening_prove_ms: perf.stage_claim_bundle.stage3.opening_prove_ms,
        },
    ]
}

fn opening_reuse_stats(output: &neo_fold_next::rv32im::SimpleKernelOutput) -> (OpeningAccumulatorStats, Vec<[u8; 32]>) {
    let mut accumulator = OpeningAccumulator::default();
    for reference in output.root_lane_columns.opening_refs() {
        accumulator
            .observe(reference)
            .expect("root-lane canonical opening alias");
    }
    for reference in output.stage_packages.stage1.claim.opening_refs() {
        accumulator
            .observe(reference)
            .expect("stage1 canonical opening alias");
    }
    for reference in output.stage_packages.stage2.claim.opening_refs() {
        accumulator
            .observe(reference)
            .expect("stage2 canonical opening alias");
    }
    for reference in output.stage_packages.stage3.claim.opening_refs() {
        accumulator
            .observe(reference)
            .expect("stage3 canonical opening alias");
    }
    for reference in output.kernel_opening.claim.opening_refs() {
        accumulator
            .observe(reference)
            .expect("kernel canonical opening alias");
    }
    let opening_ids = accumulator.opening_id_digests();
    (accumulator.stats(), opening_ids)
}

fn print_root_main_lane_family(
    output: &neo_fold_next::rv32im::SimpleKernelOutput,
    proof: &neo_fold_next::rv32im::Rv32imProof,
) {
    print_section("Root Main Lane Columns");
    print_kv("canonical_lane_objects", 1);
    print_kv("row_width", output.root_lane_columns.row_width);
    print_kv("time_len", output.root_lane_columns.time_len);
    print_kv("padded_time_len", output.root_lane_commitment.padded_time_len);
    print_kv("column_count", output.root_lane_columns.column_digests.len());
    print_kv(
        "column_commitments",
        output.root_lane_commitment.commitments.commitments.len(),
    );
    print_kv("selected_openings", output.root_lane_columns.opening_refs().len());
    print_kv(
        "opening_proofs",
        usize::from(output.root_lane_commitment.first_opening.is_some())
            + usize::from(output.root_lane_commitment.last_opening.is_some()),
    );
    print_kv(
        "first_logical_index",
        output
            .root_lane_columns
            .first_row
            .as_ref()
            .map(|reference| reference.id.logical_index)
            .unwrap_or(0),
    );
    print_kv(
        "last_logical_index",
        output
            .root_lane_columns
            .last_row
            .as_ref()
            .map(|reference| reference.id.logical_index)
            .unwrap_or(0),
    );
    print_kv(
        "fold_schedule",
        format_fold_schedule(proof.kernel.main_lane.fold_schedule()),
    );
    print_kv("proof_chunks", proof.kernel.main_lane.chunk_count());
    print_kv(
        "bridge_status",
        "column family has Ajtai commitments and selected row openings; root reductions now prove schedule-bound contiguous chunks",
    );
}

fn print_exact_stage_witness_shape(rows: &[ExactStagePerfRow<'_>]) {
    if !exact_stage_path_is_live(rows) {
        return;
    }
    print_section("Exact Stage Witness Shape");
    println!(
        "  {:10} {:>8} {:>10} {:>10} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10}",
        "surface",
        "records",
        "pack_rows",
        "pack_cols",
        "u64_words",
        "field_limbs",
        "blowup",
        "u64/record",
        "limbs/record",
        "selected"
    );
    for row in rows {
        println!(
            "  {:10} {:>8} {:>10} {:>10} {:>12} {:>12} {:>10.4} {:>12.4} {:>12.4} {:>10}",
            row.label,
            row.records,
            row.packed_rows,
            row.packed_cols,
            row.flatten_u64_words,
            row.field_limb_width,
            per_unit(row.field_limb_width as f64, row.flatten_u64_words),
            per_unit(row.flatten_u64_words as f64, row.records),
            per_unit(row.field_limb_width as f64, row.records),
            row.selected_labels,
        );
    }
}

fn print_selected_vs_exact_amplification(rows: &[ExactStagePerfRow<'_>]) {
    if !exact_stage_path_is_live(rows) {
        return;
    }
    print_section("Selected vs Exact Amplification");
    println!(
        "  {:10} {:>12} {:>12} {:>12} {:>14} {:>12} {:>12}",
        "surface", "field_limbs", "claim_words", "labels", "exact/claim", "claim/label", "ms/label"
    );
    for row in rows {
        println!(
            "  {:10} {:>12} {:>12} {:>12} {:>14.4} {:>12.4} {:>12.4}",
            row.label,
            row.field_limb_width,
            row.selected_claim_words,
            row.selected_labels,
            per_unit(row.field_limb_width as f64, row.selected_claim_words),
            per_unit(row.selected_claim_words as f64, row.selected_labels),
            per_unit(
                row.flatten_ms
                    + row.limb_encode_ms
                    + row.context_setup_ms
                    + row.ccs_encode_ms
                    + row.ajtai_commit_ms
                    + row.opening_manifest_ms
                    + row.opening_prove_ms,
                row.selected_labels,
            ),
        );
    }
}

fn print_exact_stage_build_breakdown(rows: &[ExactStagePerfRow<'_>]) {
    if !exact_stage_path_is_live(rows) {
        return;
    }
    print_section("Exact Stage Build Breakdown");
    println!(
        "  {:10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "surface", "flatten", "limb", "context", "ccs", "commit", "manifest", "proof"
    );
    for row in rows {
        println!(
            "  {:10} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            row.label,
            row.flatten_ms,
            row.limb_encode_ms,
            row.context_setup_ms,
            row.ccs_encode_ms,
            row.ajtai_commit_ms,
            row.opening_manifest_ms,
            row.opening_prove_ms,
        );
    }
}

fn print_opening_reuse_proxy(output: &neo_fold_next::rv32im::SimpleKernelOutput) {
    let (stats, unique_opening_ids) = opening_reuse_stats(output);
    print_section("Opening Reuse");
    print_kv("opening_requests_total", stats.total_requests);
    print_kv("opening_requests_unique", stats.unique_requests);
    print_kv("opening_requests_aliased", stats.aliased_requests);
    print_kv(
        "opening_request_reuse_ratio",
        format!("{:.4}", per_unit(stats.aliased_requests as f64, stats.total_requests)),
    );
    print_kv("opening_id_digests_recorded", unique_opening_ids.len());
}

fn print_compact_opening_build_breakdown(perf: &SimpleKernelBuildPerf) {
    print_section("Compact Opening Build Breakdown");
    println!(
        "  {:18} {:>8} {:>12} {:>12} {:>12}",
        "surface", "labels", "claim_words", "package_ms", "ms/label"
    );
    for (label, stats) in [
        ("stage1", perf.stage_package_bundle.stage1),
        ("stage2", perf.stage_package_bundle.stage2),
        ("stage3", perf.stage_package_bundle.stage3),
        ("kernel_bindings", perf.kernel_opening_bundle.bindings),
        ("kernel_prepared", perf.kernel_opening_bundle.prepared_steps),
    ] {
        println!(
            "  {:18} {:>8} {:>12} {:>12.3} {:>12.4}",
            label,
            stats.selected_labels,
            stats.claim_words,
            stats.package_ms,
            per_unit(stats.package_ms, stats.selected_labels),
        );
    }
}

fn print_verify_breakdown(
    title: &str,
    perf: &neo_fold_next::rv32im::Rv32imPublicProofVerifyPerf,
    opcode_count: usize,
    execution_rows: usize,
) {
    print_section(title);
    println!("  {:26} {:>12} {:>14} {:>14}", "phase", "wall ms", "ms/op", "ms/row");
    for (label, ms) in [
        ("public_claim_digests", perf.public_claim_digests_ms),
        ("public_bundle_digests", perf.public_bundle_digests_ms),
        ("public_bundle_bindings", perf.public_bundle_bindings_ms),
        ("native_stage_bundle_verify", perf.native_stage_bundle_verify_ms),
        ("stage_package_verify", perf.stage_package_verify_ms),
        ("root_execution_verify", perf.root_execution_verify_ms),
        ("root_main_lane_proof", perf.root_main_lane_proof_ms),
        ("kernel_opening_verify", perf.kernel_opening_verify_ms),
        ("summary_consistency", perf.summary_consistency_ms),
    ] {
        println!(
            "  {:26} {:>12.3} {:>14.4} {:>14.4}",
            label,
            ms,
            per_unit(ms, opcode_count),
            per_unit(ms, execution_rows),
        );
    }

    if perf.public_kernel_build.total_ms > 0.0 {
        println!(
            "  {:26} {:>12.3} {:>14.4} {:>14.4}",
            "build_public_kernel",
            perf.public_kernel_build.total_ms,
            per_unit(perf.public_kernel_build.total_ms, opcode_count),
            per_unit(perf.public_kernel_build.total_ms, execution_rows),
        );
        println!();
        println!("  {:26} {:>12}", "build_public_kernel subphase", "wall ms");
        println!(
            "  {:26} {:>12.3}",
            "root_lane_witness", perf.public_kernel_build.root_lane_witness_ms
        );
        println!(
            "  {:26} {:>12.3}",
            "root_lane_columns", perf.public_kernel_build.root_lane_columns_ms
        );
        println!(
            "  {:26} {:>12.3}",
            "root_lane_commitment", perf.public_kernel_build.root_lane_commitment_ms
        );
        println!(
            "  {:26} {:>12.3}",
            "prepared_step_bindings", perf.public_kernel_build.prepared_step_bindings_ms
        );
        println!(
            "  {:26} {:>12.3}",
            "stage_claim_build", perf.public_kernel_build.stage_claim_bundle.total_ms
        );
        println!(
            "  {:26} {:>12.3}",
            "stage_package_build", perf.public_kernel_build.stage_package_bundle.total_ms
        );
        println!(
            "  {:26} {:>12.3}",
            "kernel_opening_build", perf.public_kernel_build.kernel_opening_bundle.total_ms
        );
    } else {
        println!();
        println!("  theorem verify uses the carried proof witness; no public-kernel replay runs in this path");
        if perf.accepted_stage_package.total_ms > 0.0 {
            println!();
            println!("  {:26} {:>12}", "accepted stage subphase", "wall ms");
            println!(
                "  {:26} {:>12.3}",
                "stage1_verify", perf.accepted_stage_package.stage1_ms
            );
            if perf.accepted_stage_package.stage1_breakdown.total_ms > 0.0 {
                println!();
                println!("  {:26} {:>12}", "accepted stage1 subphase", "wall ms");
                println!(
                    "  {:26} {:>12.3}",
                    "stage1_sem_inputs_surface",
                    perf.accepted_stage_package
                        .stage1_breakdown
                        .sem_inputs_surface_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage1_semantics_verify",
                    perf.accepted_stage_package
                        .stage1_breakdown
                        .semantics_verify_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage1_row_bindings_surface",
                    perf.accepted_stage_package
                        .stage1_breakdown
                        .row_bindings_surface_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage1_surface_digest_checks",
                    perf.accepted_stage_package
                        .stage1_breakdown
                        .surface_digest_checks_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage1_selected_opening",
                    perf.accepted_stage_package
                        .stage1_breakdown
                        .selected_opening_ms
                );
            }
            println!(
                "  {:26} {:>12.3}",
                "stage2_verify", perf.accepted_stage_package.stage2_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "stage3_verify", perf.accepted_stage_package.stage3_ms
            );
            if perf.accepted_stage_package.stage2_breakdown.total_ms > 0.0 {
                println!();
                println!("  {:26} {:>12}", "accepted stage2 subphase", "wall ms");
                println!(
                    "  {:26} {:>12.3}",
                    "stage2_semantics", perf.accepted_stage_package.stage2_breakdown.semantics_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage2_temporal", perf.accepted_stage_package.stage2_breakdown.temporal_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage2_family_digests",
                    perf.accepted_stage_package
                        .stage2_breakdown
                        .family_digests_ms
                );
                println!(
                    "  {:26} {:>12.3}",
                    "stage2_selected_opening",
                    perf.accepted_stage_package
                        .stage2_breakdown
                        .selected_opening_ms
                );
            }
        }
        if perf.accepted_root_execution.total_ms > 0.0 {
            println!();
            println!("  {:26} {:>12}", "accepted root-execution subphase", "wall ms");
            println!(
                "  {:26} {:>12.3}",
                "preflight", perf.accepted_root_execution.preflight_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "semantic_rows", perf.accepted_root_execution.semantic_rows_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "statement_chunk_layout", perf.accepted_root_execution.statement_chunk_layout_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "prepared_step_bindings", perf.accepted_root_execution.prepared_step_bindings_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "kernel_claim_bindings", perf.accepted_root_execution.kernel_claim_bindings_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "row_chunk_routes", perf.accepted_root_execution.row_chunk_routes_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "row_local_ccs_acceptance", perf.accepted_root_execution.row_local_ccs_acceptance_ms
            );
            println!(
                "  {:26} {:>12.3}",
                "semantics_refinement", perf.accepted_root_execution.semantics_refinement_ms
            );
        }
    }
}

fn packaged_proof_stats(packaged: &PackagedProof) -> PackagedProofStats {
    let mut stats = PackagedProofStats {
        public_steps: packaged.statement.public_step_count(),
        public_chunks: packaged.statement.chunks.len(),
        proof_chunks: packaged.proof.session.chunks.len(),
        final_main_claims: packaged.proof.session.final_main_claims.len(),
        ..PackagedProofStats::default()
    };
    for chunk in &packaged.proof.session.chunks {
        stats.ccs_outputs += chunk.ccs_outputs.len();
        stats.dec_children += chunk.dec.children.len();
    }
    stats
}

fn opening_surface_totals(
    build_perf: &SimpleKernelBuildPerf,
    exact_claims: &[ExactOpeningClaimStats],
    packaged_proofs: &[PackagedProofStats],
    selected_labels: usize,
) -> OpeningSurfaceTotals {
    let mut totals = OpeningSurfaceTotals {
        selected_labels,
        flatten_u64_words: build_perf.stage_claim_bundle.stage1.flatten_u64_words
            + build_perf.stage_claim_bundle.stage2.flatten_u64_words
            + build_perf.stage_claim_bundle.stage3.flatten_u64_words,
        selected_claim_words: build_perf.stage_package_bundle.stage1.claim_words
            + build_perf.stage_package_bundle.stage2.claim_words
            + build_perf.stage_package_bundle.stage3.claim_words
            + build_perf.kernel_opening_bundle.bindings.claim_words
            + build_perf.kernel_opening_bundle.prepared_steps.claim_words,
        ..OpeningSurfaceTotals::default()
    };
    for stats in exact_claims {
        totals.exact_claims += stats.claims;
        totals.logical_width += stats.logical_width;
        totals.packed_rows += stats.packed_rows;
        totals.packed_cols += stats.packed_cols;
    }
    for stats in packaged_proofs {
        totals.packaged_public_steps += stats.public_steps;
        totals.packaged_public_chunks += stats.public_chunks;
        totals.packaged_proof_chunks += stats.proof_chunks;
        totals.packaged_final_main_claims += stats.final_main_claims;
        totals.packaged_ccs_outputs += stats.ccs_outputs;
        totals.packaged_dec_children += stats.dec_children;
    }
    totals
}

fn opening_label_buckets(labels: &[OpeningPointLabel]) -> OpeningLabelBuckets {
    let mut buckets = OpeningLabelBuckets::default();
    for label in labels {
        match label {
            OpeningPointLabel::Stage1First
            | OpeningPointLabel::Stage1Effect
            | OpeningPointLabel::Stage1Commit
            | OpeningPointLabel::Stage1Last => buckets.stage1 += 1,
            OpeningPointLabel::Stage2FirstRead
            | OpeningPointLabel::Stage2LastRead
            | OpeningPointLabel::Stage2FirstWrite
            | OpeningPointLabel::Stage2LastWrite
            | OpeningPointLabel::Stage2FirstRam
            | OpeningPointLabel::Stage2LastRam
            | OpeningPointLabel::Stage2FirstTwist
            | OpeningPointLabel::Stage2LastTwist => buckets.stage2 += 1,
            OpeningPointLabel::Stage3FirstContinuity | OpeningPointLabel::Stage3LastContinuity => buckets.stage3 += 1,
            OpeningPointLabel::KernelFirstBinding | OpeningPointLabel::KernelLastBinding => buckets.kernel_binding += 1,
            OpeningPointLabel::KernelFirstPreparedStep | OpeningPointLabel::KernelLastPreparedStep => {
                buckets.kernel_prepared_steps += 1
            }
        }
    }
    buckets
}

fn print_exact_opening_table(rows: &[(&str, ExactOpeningClaimStats)], opcode_count: usize, execution_rows: usize) {
    if !exact_opening_claims_are_live(rows) {
        return;
    }
    print_section("Exact Opening Claims");
    println!(
        "  {:18} {:>8} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "surface", "claims", "field_limbs", "packed_rows", "packed_cols", "claims/op", "claims/row"
    );
    for (label, stats) in rows {
        println!(
            "  {:18} {:>8} {:>12} {:>12} {:>12} {:>10.4} {:>10.4}",
            label,
            stats.claims,
            stats.logical_width,
            stats.packed_rows,
            stats.packed_cols,
            per_unit(stats.claims as f64, opcode_count),
            per_unit(stats.claims as f64, execution_rows),
        );
    }
}

fn print_packaged_proof_table(rows: &[(&str, PackagedProofStats)]) {
    print_section("Packaged Opening Proofs");
    println!(
        "  {:18} {:>12} {:>13} {:>12} {:>12} {:>12} {:>12}",
        "surface", "public_steps", "public_chunks", "proof_chunks", "final_main", "ccs_outputs", "dec_children"
    );
    for (label, stats) in rows {
        println!(
            "  {:18} {:>12} {:>13} {:>12} {:>12} {:>12} {:>12}",
            label,
            stats.public_steps,
            stats.public_chunks,
            stats.proof_chunks,
            stats.final_main_claims,
            stats.ccs_outputs,
            stats.dec_children,
        );
    }
}

fn print_opening_surface_totals(totals: OpeningSurfaceTotals, opcode_count: usize, execution_rows: usize) {
    print_section("Opening Surface Totals");
    print_kv("selected_labels_total", totals.selected_labels);
    print_kv("selected_claim_words_total", totals.selected_claim_words);
    print_kv("packaged_public_steps_total", totals.packaged_public_steps);
    print_kv("packaged_public_chunks_total", totals.packaged_public_chunks);
    print_kv("packaged_proof_chunks_total", totals.packaged_proof_chunks);
    print_kv("packaged_final_main_claims_total", totals.packaged_final_main_claims);
    print_kv("packaged_ccs_outputs_total", totals.packaged_ccs_outputs);
    print_kv("packaged_dec_children_total", totals.packaged_dec_children);
    if totals.exact_claims != 0 || totals.flatten_u64_words != 0 || totals.logical_width != 0 {
        print_kv("exact_claims_total", totals.exact_claims);
        print_kv("exact_stage_flatten_u64_words_total", totals.flatten_u64_words);
        print_kv("exact_field_limb_width_total", totals.logical_width);
        print_kv("packed_rows_total", totals.packed_rows);
        print_kv("packed_cols_total", totals.packed_cols);
        print_kv(
            "exact_claims_per_non-halt_opcode",
            format!("{:.4}", per_unit(totals.exact_claims as f64, opcode_count)),
        );
        print_kv(
            "selected_labels_per_exact_claim",
            format!("{:.4}", per_unit(totals.selected_labels as f64, totals.exact_claims)),
        );
        print_kv(
            "exact_to_selected_amplification",
            format!(
                "{:.4}",
                per_unit(totals.logical_width as f64, totals.selected_claim_words)
            ),
        );
    }
    print_kv(
        "packaged_dec_children_per_execution_row",
        format!("{:.4}", per_unit(totals.packaged_dec_children as f64, execution_rows)),
    );
}

fn print_opening_label_summary(labels: &[OpeningPointLabel]) {
    let buckets = opening_label_buckets(labels);
    let rendered = labels
        .iter()
        .map(|label| format!("{label:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    print_section("Selected Opening Labels");
    print_kv("total_labels", labels.len());
    print_kv(
        "bucket_counts",
        format!(
            "stage1={} stage2={} stage3={} kernel_binding={} kernel_prepared={}",
            buckets.stage1, buckets.stage2, buckets.stage3, buckets.kernel_binding, buckets.kernel_prepared_steps
        ),
    );
    print_kv("labels", rendered);
}

fn aggregate_family_rows(output: &neo_fold_next::rv32im::SimpleKernelOutput) -> [FamilyRowStats; FAMILY_ORDER.len()] {
    let mut stats = [FamilyRowStats::default(); FAMILY_ORDER.len()];
    for row in &output.trace.execution_rows {
        let family = &mut stats[family_index(row.family)];
        family.rows += 1;
        family.real_rows += usize::from(row.is_real);
        family.effect_rows += usize::from(row.is_effect_row);
        family.commit_rows += usize::from(row.is_commit_row);
    }
    stats
}

fn aggregate_lookups(
    output: &neo_fold_next::rv32im::SimpleKernelOutput,
) -> (LookupSummary, [usize; FAMILY_ORDER.len()]) {
    let mut read_regs = [false; RV32_REGISTER_COUNT];
    let mut write_regs = [false; RV32_REGISTER_COUNT];
    let mut ram_addrs = BTreeSet::new();
    let mut twist_family_counts = [0usize; FAMILY_ORDER.len()];
    let mut summary = LookupSummary::default();

    for event in &output.stages.stage2.register_reads {
        summary.register_reads += 1;
        match event.role {
            RegisterReadRole::Rs1 => summary.register_reads_rs1 += 1,
            RegisterReadRole::Rs2 => summary.register_reads_rs2 += 1,
        }
        if let Some(seen) = read_regs.get_mut(event.reg as usize) {
            *seen = true;
        }
    }

    for event in &output.stages.stage2.register_writes {
        summary.register_writes += 1;
        if let Some(seen) = write_regs.get_mut(event.reg as usize) {
            *seen = true;
        }
    }

    for event in &output.stages.stage2.ram_events {
        summary.ram_events += 1;
        match event.kind {
            RamAccessKind::Read => summary.ram_reads += 1,
            RamAccessKind::Write => summary.ram_writes += 1,
        }
        ram_addrs.insert(event.addr);
    }

    for event in &output.stages.stage2.twist_links {
        summary.twist_links += 1;
        twist_family_counts[family_index(event.family)] += 1;
        summary.twist_write_routes += usize::from(event.routed_write_value.is_some());
        summary.twist_memory_before_routes += usize::from(event.routed_memory_before.is_some());
        summary.twist_memory_after_routes += usize::from(event.routed_memory_after.is_some());
    }

    summary.unique_read_regs = read_regs.iter().filter(|seen| **seen).count();
    summary.unique_write_regs = write_regs.iter().filter(|seen| **seen).count();
    summary.unique_ram_addrs = ram_addrs.len();
    (summary, twist_family_counts)
}
