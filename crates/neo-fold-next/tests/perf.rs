//! Performance/debugging reports for the current RV64IM proof path.

#[path = "support/chip8.rs"]
mod chip8_support;

use std::collections::BTreeSet;
use std::env;
use std::time::Instant;

use serde::Serialize;

use neo_fold_next::chip8::decider::{prove_chip8_spartan2_decider, setup_chip8_spartan2_decider};
use neo_fold_next::chip8::proof::prove_recursive as prove_chip8_recursive;
use neo_fold_next::decider::spartan2::Spartan2DeciderProvePerf;
use neo_fold_next::nightstream::chip8::{
    build_chip8_nightstream_from_recursive_proof, verify_chip8_nightstream_from_recursive_proof,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_nightstream_from_published_proof_seam_with_perf, verify_rv64im_nightstream_with_perf,
};
use neo_fold_next::proof::{FoldSchedule, PackagedProof};
use neo_fold_next::rv64im::audit::{
    build_rv64im_spartan2_decider_setup_shape, prove_rv64im_public_proof_and_published_seam_with_perf,
    prove_rv64im_spartan2_decider_cached, setup_rv64im_spartan2_decider_cached_from_shape,
};
use neo_fold_next::rv64im::ccs::{rv64im_root_main_lane_ccs, RV64IM_ROOT_PUBLIC_INPUTS, RV64IM_ROOT_ROW_WIDTH};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::layout::RV64_REGISTER_COUNT;
use neo_fold_next::rv64im::stage1::build_stage1_summary;
use neo_fold_next::rv64im::stage2::{build_stage2_summary, RamAccessKind, RegisterReadRole};
use neo_fold_next::rv64im::stage3::build_stage3_summary;
use neo_fold_next::rv64im::tables::Rv64FamilyTag;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_parity_case_from_source, build_program,
    build_rv64im_audit_witness_bundle as build_rv64im_proof_witness,
    build_rv64im_opening_bundle_from_accepted_artifact, build_simple_kernel_witness_with_perf,
    mixed_opcode_perf_expected_x1, rv64im_simple_root_params, validate_rv64im_public_proof_against_input_with_perf,
    verify_rv64im_audit_proof as verify_rv64im_proof, verify_rv64im_public_proof_with_perf, OpeningAccumulator,
    OpeningAccumulatorStats, OpeningPointLabel, Rv64Program, Rv64State, Rv64imProofInput, SimpleKernelBuildPerf,
    RV64IM_MIXED_OPCODE_PERF_BLOCK_LEN, RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
use neo_fold_next::time_opening::prove_time_opening;

const FAMILY_ORDER: [Rv64FamilyTag; 7] = [
    Rv64FamilyTag::NativeAlu,
    Rv64FamilyTag::AlignedMemory,
    Rv64FamilyTag::NarrowMemory,
    Rv64FamilyTag::Multiply,
    Rv64FamilyTag::UnsignedDivRem,
    Rv64FamilyTag::SignedDivRem,
    Rv64FamilyTag::ControlFlow,
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
        Err(_) => RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

fn family_label(family: Rv64FamilyTag) -> &'static str {
    match family {
        Rv64FamilyTag::NativeAlu => "native_alu",
        Rv64FamilyTag::AlignedMemory => "aligned_memory",
        Rv64FamilyTag::NarrowMemory => "narrow_memory",
        Rv64FamilyTag::Multiply => "multiply",
        Rv64FamilyTag::UnsignedDivRem => "unsigned_divrem",
        Rv64FamilyTag::SignedDivRem => "signed_divrem",
        Rv64FamilyTag::ControlFlow => "control_flow",
    }
}

fn family_index(family: Rv64FamilyTag) -> usize {
    match family {
        Rv64FamilyTag::NativeAlu => 0,
        Rv64FamilyTag::AlignedMemory => 1,
        Rv64FamilyTag::NarrowMemory => 2,
        Rv64FamilyTag::Multiply => 3,
        Rv64FamilyTag::UnsignedDivRem => 4,
        Rv64FamilyTag::SignedDivRem => 5,
        Rv64FamilyTag::ControlFlow => 6,
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

fn collect_unique_opcode_labels(build: &neo_fold_next::rv64im::builder::Rv64ProgramBuild) -> String {
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
        RV64_REGISTER_COUNT.saturating_sub(summary.unique_read_regs)
    );
    println!(
        "  {:20} {:>12} {:>10} {:>14.4} {:>16}",
        "write_regs",
        summary.unique_write_regs,
        summary.register_writes,
        per_unit(summary.register_writes as f64, summary.unique_write_regs),
        RV64_REGISTER_COUNT.saturating_sub(summary.unique_write_regs)
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
    output: &neo_fold_next::rv64im::SimpleKernelOutput,
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

fn opening_reuse_stats(output: &neo_fold_next::rv64im::SimpleKernelOutput) -> (OpeningAccumulatorStats, Vec<[u8; 32]>) {
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
    output: &neo_fold_next::rv64im::SimpleKernelOutput,
    proof: &neo_fold_next::rv64im::Rv64imProof,
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

fn print_opening_reuse_proxy(output: &neo_fold_next::rv64im::SimpleKernelOutput) {
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
    perf: &neo_fold_next::rv64im::Rv64imPublicProofVerifyPerf,
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

fn aggregate_family_rows(output: &neo_fold_next::rv64im::SimpleKernelOutput) -> [FamilyRowStats; FAMILY_ORDER.len()] {
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
    output: &neo_fold_next::rv64im::SimpleKernelOutput,
) -> (LookupSummary, [usize; FAMILY_ORDER.len()]) {
    let mut read_regs = [false; RV64_REGISTER_COUNT];
    let mut write_regs = [false; RV64_REGISTER_COUNT];
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

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv64im_mixed_opcode_perf_snapshot() {
    let end_to_end_started = Instant::now();
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let x1_increment_count = mixed_opcode_perf_expected_x1(opcode_count);
    let total_opcodes = source.program_words.len();
    let input = Rv64imProofInput {
        source: source.clone(),
        max_steps: total_opcodes,
    };

    let program = Rv64Program::new(source.start_pc, source.program_words.clone());
    let initial_state = Rv64State::new(source.start_pc, source.initial_registers, &source.initial_memory);

    let build_program_started = Instant::now();
    let build = build_program(&program, &initial_state, total_opcodes).expect("build program");
    let build_program_ms = millis_since(build_program_started);

    let stage1_started = Instant::now();
    let stage1 = build_stage1_summary(&build.rows);
    let stage1_ms = millis_since(stage1_started);

    let stage2_started = Instant::now();
    let stage2 = build_stage2_summary(&build.rows);
    let stage2_ms = millis_since(stage2_started);

    let stage3_started = Instant::now();
    let stage3 = build_stage3_summary(&build.rows);
    let stage3_ms = millis_since(stage3_started);

    let parity_started = Instant::now();
    let (_, derived) = build_parity_case_from_source(source.clone(), total_opcodes).expect("build derived parity case");
    let parity_ms = millis_since(parity_started);

    let build_started = Instant::now();
    let (output, build_perf) = build_simple_kernel_witness_with_perf(&input).expect("build simple kernel witness");
    let build_ms = millis_since(build_started);

    let ((proof, published_seam), prove_and_seam_perf) =
        prove_rv64im_public_proof_and_published_seam_with_perf(&input).expect("prove rv64im public proof and seam");
    let prove_perf = prove_and_seam_perf.proof;
    let prove_ms = prove_perf.total_ms;

    let verify_started = Instant::now();
    let verify_perf = verify_rv64im_public_proof_with_perf(&proof).expect("verify rv64im public proof");
    let verify_ms = millis_since(verify_started);

    let _ = validate_rv64im_public_proof_against_input_with_perf(&input, &proof)
        .expect("validate rv64im public proof against input");

    let published_seam_perf = prove_and_seam_perf.seam;
    let accepted_artifact = &published_seam.accepted_artifact;
    let kernel_export_source = published_seam.kernel_export_source();
    let (decider_final_statement, decider_final_proof) =
        prove_rv64im_final_statement_from_accepted(accepted_artifact).expect("build rv64im final seam");

    let decider_setup_started = Instant::now();
    let decider_shape = build_rv64im_spartan2_decider_setup_shape(&decider_final_statement, &decider_final_proof)
        .expect("build rv64im spartan2 decider shape");
    let decider_keys =
        setup_rv64im_spartan2_decider_cached_from_shape(&decider_shape).expect("setup rv64im spartan2 decider");
    let decider_setup_ms = millis_since(decider_setup_started);
    let decider_shape_sizes = decider_keys.as_ref().0.sizes();
    let decider_shape_debug_stats = decider_keys.as_ref().0.shape_debug_stats();

    let decider_prove_started = Instant::now();
    let decider_proof = prove_rv64im_spartan2_decider_cached(&decider_final_statement, &decider_final_proof)
        .expect("prove rv64im spartan2 decider");
    let decider_prove_perf = Spartan2DeciderProvePerf::default();
    let decider_prove_ms = millis_since(decider_prove_started);

    let ((nightstream_statement, nightstream_proof), nightstream_build_perf) =
        build_rv64im_nightstream_from_published_proof_seam_with_perf(&published_seam, &published_seam_perf)
            .expect("build rv64im nightstream proof");
    let public_statement = proof.statement.clone();
    let nightstream_build_ms = nightstream_build_perf.total_ms;
    let nightstream_opening_bundle =
        build_rv64im_opening_bundle_from_accepted_artifact(accepted_artifact).expect("build rv64im opening bundle");
    let side_statement = nightstream_proof
        .side_proof()
        .binding_statement(&nightstream_statement)
        .expect("build rv64im side binding statement");
    let side_keys = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_binding_cached(
        &side_statement,
        nightstream_proof.side_proof().opening_public(),
    )
    .expect("setup rv64im side binding");
    let (opening_statement, opening_witness) =
        neo_fold_next::nightstream::rv64im::audit::build_rv64im_side_opening_relation_from_accepted_artifact(
            accepted_artifact,
        )
        .expect("build rv64im side opening relation");
    let side_opening_keys = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_opening_spartan_cached(
        &opening_statement,
        &opening_witness,
    )
    .expect("setup rv64im side opening");

    let nightstream_verify_perf = verify_rv64im_nightstream_with_perf(
        &nightstream_statement,
        &nightstream_proof,
        proof.statement.root_params_id,
        &side_opening_keys.as_ref().1,
        &side_keys.as_ref().1,
        &public_statement,
    )
    .expect("verify rv64im nightstream proof");
    let nightstream_verify_ms = nightstream_verify_perf.total_ms;

    let prove_witness_started = Instant::now();
    let proved_witness = build_rv64im_proof_witness(&input).expect("build rv64im proof witness");
    let prove_witness_ms = millis_since(prove_witness_started);

    let verify_witness_started = Instant::now();
    let verified_witness = verify_rv64im_proof(&proof).expect("verify rv64im proof witness");
    let verify_witness_ms = millis_since(verify_witness_started);

    let execution_row_count = output.trace.execution_rows.len();
    let real_row_count = output
        .trace
        .execution_rows
        .iter()
        .filter(|row| row.is_real)
        .count();
    let effect_row_count = output
        .trace
        .execution_rows
        .iter()
        .filter(|row| row.is_effect_row)
        .count();
    let commit_row_count = output
        .trace
        .execution_rows
        .iter()
        .filter(|row| row.is_commit_row)
        .count();

    let root_ccs = rv64im_root_main_lane_ccs().expect("build RV64IM root CCS");
    let root_params = rv64im_simple_root_params();
    let root_ccs_n_p2 = root_ccs.n.next_power_of_two();
    let root_ccs_m_p2 = root_ccs.m.next_power_of_two();
    let ccs_total_nnz: usize = root_ccs
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .as_csc()
                .map(|csc| csc.vals.len())
                .unwrap_or(matrix.rows())
        })
        .sum();
    let ccs_identity_matrices = root_ccs
        .matrices
        .iter()
        .filter(|matrix| matrix.as_csc().is_none())
        .count();
    let approx_trace_constraints = root_ccs.n.saturating_mul(output.prepared_steps.len());
    let approx_trace_nnz = ccs_total_nnz.saturating_mul(output.prepared_steps.len());
    let family_rows = aggregate_family_rows(&output);
    let (lookup_summary, twist_family_counts) = aggregate_lookups(&output);
    let active_twist_family_count = twist_family_counts
        .iter()
        .filter(|count| **count > 0)
        .count();
    let stage1_exact_openings = ExactOpeningClaimStats::default();
    let stage2_exact_openings = ExactOpeningClaimStats::default();
    let stage3_exact_openings = ExactOpeningClaimStats::default();
    let stage1_packaged = packaged_proof_stats(&output.stage_packages.stage1.packaged);
    let stage2_packaged = packaged_proof_stats(&output.stage_packages.stage2.packaged);
    let stage3_packaged = packaged_proof_stats(&output.stage_packages.stage3.packaged);
    let kernel_binding_packaged = packaged_proof_stats(&output.kernel_opening.bindings.packaged);
    let kernel_prepared_packaged = packaged_proof_stats(&output.kernel_opening.prepared_steps.packaged);
    let mut selected_opening_labels = output.stage_packages.stage1.claim.labels();
    selected_opening_labels.extend(output.stage_packages.stage2.claim.labels());
    selected_opening_labels.extend(output.stage_packages.stage3.claim.labels());
    selected_opening_labels.extend(output.kernel_opening.claim.labels());
    let opening_totals = opening_surface_totals(
        &build_perf,
        &[stage1_exact_openings, stage2_exact_openings, stage3_exact_openings],
        &[
            stage1_packaged,
            stage2_packaged,
            stage3_packaged,
            kernel_binding_packaged,
            kernel_prepared_packaged,
        ],
        selected_opening_labels.len(),
    );
    let exact_stage_rows = exact_stage_perf_rows(&output, &build_perf);
    let serialized_sizes = [
        SerializedSizeRow {
            label: "proof.total",
            bytes: serialized_size_bytes(&proof),
        },
        SerializedSizeRow {
            label: "proof.statement",
            bytes: serialized_size_bytes(&proof.statement),
        },
        SerializedSizeRow {
            label: "proof.claim",
            bytes: serialized_size_bytes(&proof.claim),
        },
        SerializedSizeRow {
            label: "claim.accepted.terminal",
            bytes: serialized_size_bytes(&proof.claim.accepted.terminal),
        },
        SerializedSizeRow {
            label: "claim.opening.terminal",
            bytes: serialized_size_bytes(&proof.claim.opening.terminal),
        },
        SerializedSizeRow {
            label: "claim.root0.terminal",
            bytes: serialized_size_bytes(&proof.claim.root0.terminal),
        },
        SerializedSizeRow {
            label: "proof.kernel",
            bytes: serialized_size_bytes(&proof.kernel),
        },
        SerializedSizeRow {
            label: "proof.witness",
            bytes: serialized_size_bytes(&proof.witness),
        },
        SerializedSizeRow {
            label: "kernel.trace",
            bytes: serialized_size_bytes(&proof.kernel.trace),
        },
        SerializedSizeRow {
            label: "kernel.stages",
            bytes: serialized_size_bytes(&proof.kernel.stages),
        },
        SerializedSizeRow {
            label: "kernel.stage_claims",
            bytes: serialized_size_bytes(&proof.kernel.stage_claims),
        },
        SerializedSizeRow {
            label: "kernel.stage_claims.summary",
            bytes: serialized_size_bytes(&proof.kernel.stage_claims.summary),
        },
        SerializedSizeRow {
            label: "kernel.stage_claims.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_claims.packaged),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.summary",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.summary),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.stage1.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.packages.stage1.packaged),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.stage2.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.packages.stage2.packaged),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.stage3.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.packages.stage3.packaged),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening.bindings",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening.bindings),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening.bindings.packaged",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening.opening.bindings.packaged),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening.prepared_steps.packaged",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening.opening.prepared_steps.packaged),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims.summary",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims.summary),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims.summary.terminal",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims.summary.terminal),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims.packaged",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims.packaged),
        },
        SerializedSizeRow {
            label: "kernel.main_lane",
            bytes: serialized_size_bytes(&proof.kernel.main_lane),
        },
        SerializedSizeRow {
            label: "kernel.root_lane_columns",
            bytes: serialized_size_bytes(&proof.kernel.root_lane_columns),
        },
        SerializedSizeRow {
            label: "kernel.root_lane_commitment",
            bytes: serialized_size_bytes(&proof.kernel.root_lane_commitment),
        },
        SerializedSizeRow {
            label: "kernel_export.source",
            bytes: serialized_size_bytes(&kernel_export_source),
        },
        SerializedSizeRow {
            label: "witness.trace",
            bytes: serialized_size_bytes(&proof.witness.trace),
        },
        SerializedSizeRow {
            label: "witness.stages",
            bytes: serialized_size_bytes(&proof.witness.stages),
        },
        SerializedSizeRow {
            label: "witness.stage_claims",
            bytes: serialized_size_bytes(&proof.witness.stage_claims),
        },
        SerializedSizeRow {
            label: "witness.stage_packages",
            bytes: serialized_size_bytes(&proof.witness.stage_packages),
        },
        SerializedSizeRow {
            label: "witness.kernel_opening",
            bytes: serialized_size_bytes(&proof.witness.kernel_opening),
        },
        SerializedSizeRow {
            label: "witness.kernel_claims",
            bytes: serialized_size_bytes(&proof.witness.kernel_claims),
        },
    ];
    let accepted_artifact_total_bytes = serialized_size_bytes(&accepted_artifact);
    let final_statement_bytes = serialized_size_bytes(
        published_seam
            .main_proof
            .final_statement_cache()
            .expect("locally built published seam should retain the final-statement cache"),
    );
    let decider_proof_bytes = serialized_size_bytes(&decider_proof);
    let nightstream_serialized_sizes = [
        SerializedSizeRow {
            label: "nightstream.total",
            bytes: serialized_size_bytes(&(nightstream_statement.clone(), nightstream_proof.clone())),
        },
        SerializedSizeRow {
            label: "nightstream.statement",
            bytes: serialized_size_bytes(&nightstream_statement),
        },
        SerializedSizeRow {
            label: "nightstream.proof",
            bytes: serialized_size_bytes(&nightstream_proof),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof",
            bytes: serialized_size_bytes(nightstream_proof.main_proof()),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.final_statement",
            bytes: serialized_size_bytes(
                nightstream_proof
                    .main_proof()
                    .final_statement_cache()
                    .expect("locally built Nightstream main proof should retain the final-statement cache"),
            ),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.final_surface",
            bytes: serialized_size_bytes(
                nightstream_proof
                    .main_proof()
                    .final_surface_cache()
                    .expect("locally built Nightstream main proof should retain the final-surface cache"),
            ),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.kernel_export_cache",
            bytes: serialized_size_bytes(
                nightstream_proof
                    .main_proof()
                    .kernel_export_cache()
                    .expect("locally built Nightstream main proof must carry a kernel-export cache"),
            ),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.recursion_proof",
            bytes: serialized_size_bytes(nightstream_proof.main_proof().recursion_proof()),
        },
        SerializedSizeRow {
            label: "nightstream.side_proof",
            bytes: serialized_size_bytes(nightstream_proof.side_proof()),
        },
    ];
    let proof_total_bytes = serialized_sizes[0].bytes;
    let proof_total_kib = bytes_to_kib(proof_total_bytes);
    let nightstream_total_bytes = nightstream_serialized_sizes[0].bytes;
    let nightstream_total_kib = bytes_to_kib(nightstream_total_bytes);

    assert_eq!(build.rows, output.trace.execution_rows);
    assert_eq!(build.final_state.pc, output.kernel_claims.kernel.final_pc);
    assert_eq!(stage1, output.stages.stage1);
    assert_eq!(stage2, output.stages.stage2);
    assert_eq!(stage3, output.stages.stage3);
    assert_eq!(derived.execution_rows, output.trace.execution_rows);
    assert_eq!(derived.stage1, output.stages.stage1);
    assert_eq!(derived.stage2, output.stages.stage2);
    assert_eq!(derived.stage3, output.stages.stage3);
    assert_eq!(derived.transcript, output.stages.transcript);
    assert_eq!(derived.kernel, output.kernel_claims.kernel);

    assert_eq!(proved_witness.digest, verified_witness.digest);
    assert_eq!(proved_witness.trace.digest, verified_witness.trace.digest);
    assert_eq!(
        proved_witness.kernel_claims.digest,
        verified_witness.kernel_claims.digest
    );
    assert_eq!(
        proved_witness.root_lane_columns.time_len as usize,
        output.prepared_steps.len()
    );
    assert_eq!(proof.statement.public_step_count as usize, output.prepared_steps.len());
    assert_eq!(
        proof.kernel.root_lane_columns.time_len as usize,
        output.prepared_steps.len()
    );
    assert_eq!(execution_row_count, output.prepared_steps.len());
    assert_eq!(execution_row_count, output.root_lane_columns.time_len as usize);
    assert_eq!(
        proved_witness.trace.shape.execution_row_count as usize,
        execution_row_count
    );
    assert_eq!(proved_witness.trace.shape.real_row_count as usize, real_row_count);
    assert_eq!(proved_witness.trace.shape.effect_row_count as usize, effect_row_count);
    assert_eq!(proved_witness.trace.shape.commit_row_count as usize, commit_row_count);
    assert_eq!(
        proof.kernel.stages.summary.stage1_row_count as usize,
        output.stages.stage1.rows.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_register_read_count as usize,
        output.stages.stage2.register_reads.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_register_write_count as usize,
        output.stages.stage2.register_writes.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_ram_event_count as usize,
        output.stages.stage2.ram_events.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_twist_link_count as usize,
        output.stages.stage2.twist_links.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage3_continuity_count as usize,
        output.stages.stage3.continuity.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.transcript_event_count as usize,
        output.stages.transcript.events.len()
    );
    assert_eq!(proof.statement.final_pc, source.start_pc + (total_opcodes as u64) * 4);
    assert!(proof.statement.halted);
    assert_eq!(
        output.kernel_claims.kernel.final_registers[1],
        x1_increment_count as u64
    );
    assert_eq!(output.kernel_claims.kernel.final_pc, proof.statement.final_pc);
    assert!(output.kernel_claims.kernel.halted);
    assert_eq!(output.kernel_claims.kernel.final_memory.len(), 1);
    assert_eq!(
        output.kernel_claims.kernel.final_memory[0].addr,
        source.initial_memory[0].addr
    );
    assert_eq!(
        nightstream_statement.public_io_digest,
        published_seam
            .main_proof
            .published_statement()
            .expected_digest()
    );

    // ── Precompute published pipeline totals for executive summary ───────
    let total_executed_opcodes = build.executed_steps.len();
    let unique_opcode_labels = collect_unique_opcode_labels(&build);
    let published_prove_before_spartan_ms = prove_ms + published_seam_perf.total_ms + nightstream_build_ms;
    let spartan_setup_ms = decider_setup_ms;
    let spartan_prove_ms = decider_prove_ms;
    let published_verify_before_main_proof_ms = nightstream_verify_perf.before_main_proof_ms();
    let main_proof_verify_ms = nightstream_verify_perf.main_proof_ms;
    let published_pipeline_total_ms = spartan_setup_ms
        + published_prove_before_spartan_ms
        + spartan_prove_ms
        + published_verify_before_main_proof_ms
        + main_proof_verify_ms;
    let full_benchmark_wall_ms = millis_since(end_to_end_started);
    let benchmark_extras_ms = (full_benchmark_wall_ms - published_pipeline_total_ms).max(0.0);

    let recursive_relation_core_ms = nightstream_build_perf.final_statement_recursive_prepare_inputs_ms
        + nightstream_build_perf.final_statement_recursive_ccs_ms
        + nightstream_build_perf.final_statement_recursive_dims_ms
        + nightstream_build_perf.final_statement_recursive_rlc_prepare_ms
        + nightstream_build_perf.final_statement_recursive_rlc_ms
        + nightstream_build_perf.final_statement_recursive_dec_split_ms
        + nightstream_build_perf.final_statement_recursive_dec_commit_ms
        + nightstream_build_perf.final_statement_recursive_dec_ms;
    let recursive_wrapper_ms =
        (nightstream_build_perf.final_statement_recursive_proof_ms - recursive_relation_core_ms).max(0.0);

    // ── Input Shape ────────────────────────────────────────────────────────
    print_section("RV64IM Mixed Opcode Perf Snapshot");
    print_kv("ns_debug_n (non-halt ops)", opcode_count);
    print_kv("program_opcodes_total", total_opcodes);
    print_kv("mixed_block_len", RV64IM_MIXED_OPCODE_PERF_BLOCK_LEN);
    print_kv("family_tags", source.manifest.family_tags.len());
    print_kv("final_pc", proof.statement.final_pc);
    print_kv("final_x1", output.kernel_claims.kernel.final_registers[1]);
    print_kv("final_x7", output.kernel_claims.kernel.final_registers[7]);
    print_kv("final_mem_0x100", output.kernel_claims.kernel.final_memory[0].value);
    print_kv(
        "row_expansion",
        format!(
            "{execution_row_count}/{opcode_count} = {:.4} rows/op",
            per_unit(execution_row_count as f64, opcode_count)
        ),
    );
    print_kv(
        "prepared_step_expansion",
        format!(
            "{}/{} = {:.4} steps/op",
            output.prepared_steps.len(),
            opcode_count,
            per_unit(output.prepared_steps.len() as f64, opcode_count)
        ),
    );

    print_timing_table(
        "Raw Proving Timing",
        &[
            ("build_program", build_program_ms),
            ("stage1_summary", stage1_ms),
            ("stage2_summary", stage2_ms),
            ("stage3_summary", stage3_ms),
            ("build_parity_case", parity_ms),
            ("root_lane_witness", build_perf.root_lane_witness_ms),
            ("root_lane_columns", build_perf.root_lane_columns_ms),
            ("root_lane_commitment", build_perf.root_lane_commitment_ms),
            ("build_simple_kernel", build_ms),
            ("public.shared_trace", prove_perf.shared_trace_ms),
            ("public.kernel_projection", prove_perf.simple_kernel.total_ms),
            ("public.parallel_overlap", -prove_perf.parallel_overlap_ms),
            ("prove_rv64im_public_proof", prove_ms),
            (
                "build_rv64im_published_seam.accepted_artifact",
                published_seam_perf.accepted_artifact_ms,
            ),
            (
                "build_rv64im_published_seam.kernel_export_source",
                published_seam_perf.kernel_export_source_ms,
            ),
            (
                "build_rv64im_published_seam.final_statement",
                published_seam_perf.final_statement_ms,
            ),
            (
                "build_rv64im_published_seam.main_proof",
                published_seam_perf.main_proof_ms,
            ),
            ("setup_rv64im_spartan2_decider.direct", decider_setup_ms),
            ("prove_rv64im_spartan2_decider.direct", decider_prove_ms),
            ("build_rv64im_nightstream", nightstream_build_ms),
        ],
        opcode_count,
        execution_row_count,
    );

    print_timing_table(
        "Raw Verify Timing",
        &[
            ("verify_rv64im_public_proof", verify_ms),
            ("verify_rv64im_nightstream", nightstream_verify_ms),
        ],
        opcode_count,
        execution_row_count,
    );

    print_timing_table(
        "Raw Diagnostic Timing",
        &[
            ("build_rv64im_proof_witness", prove_witness_ms),
            ("verify_rv64im_proof_witness", verify_witness_ms),
        ],
        opcode_count,
        execution_row_count,
    );

    print_section("Benchmark Extras");
    print_kv(
        "diagnostics and extra benchmark work",
        format_ms_per_opcode(benchmark_extras_ms, total_executed_opcodes),
    );
    print_kv("includes", "public verify/replay and audit witness path".to_string());
    print_kv(
        "full benchmark wall time",
        format_ms_per_opcode(full_benchmark_wall_ms, total_executed_opcodes),
    );

    let prove_total_ms = published_prove_before_spartan_ms + spartan_setup_ms + spartan_prove_ms;
    let verify_total_ms = published_verify_before_main_proof_ms + main_proof_verify_ms;
    let amortized_prove_ms =
        prove_total_ms - spartan_setup_ms - nightstream_build_perf.verified_seams.side_binding_setup_ms;

    print_section("Nightstream Opening Diagnostics");
    println!("  total: {:.3} ms", nightstream_build_perf.total_ms);
    println!();
    println!("  note: recursive/final-statement timers are summarized in the proving tree.");
    println!("  note: phase0 values below are nested accumulators and overlap by design.");
    println!();
    println!("  phase0 opening (nested accumulators — do not sum as flat partition):");
    {
        let vs = &nightstream_build_perf.verified_seams;
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "claim_witnesses",
            vs.opening_phase0_claim_witnesses_ms,
            "relation_artifact",
            vs.opening_phase0_relation_artifact_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "pack_columns",
            vs.opening_phase0_packed_columns_ms,
            "commit_vector",
            vs.opening_phase0_commitment_vector_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "commit_many",
            vs.opening_phase0_commitment_commit_many_ms,
            "commit_root",
            vs.opening_phase0_commitment_root_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "object_total",
            vs.opening_phase0_opened_object_total_ms,
            "object_id",
            vs.opening_phase0_opened_object_id_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "bind_digest", vs.opening_phase0_binding_digest_ms, "point", vs.opening_phase0_point_derivation_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "payload_eval", vs.opening_phase0_payload_eval_ms, "claim_build", vs.opening_phase0_claim_build_ms
        );
        println!(
            "    {:28} {:>9.3}",
            "slot_total", vs.opening_phase0_slot_claims_total_ms
        );
    }
    println!();
    println!("  opening convergence:");
    {
        let vs = &nightstream_build_perf.verified_seams;
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}  {:18} {:>7.3}",
            "phase1",
            vs.opening_convergence_phase1_ms,
            "phase2",
            vs.opening_convergence_phase2_ms,
            "final_targets",
            vs.opening_convergence_final_openings_ms
        );
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}  {:18} {:>7.3}",
            "targets.map",
            vs.opening_convergence_final_openings_witness_map_ms,
            "targets.rep",
            vs.opening_convergence_final_openings_representative_ms,
            "targets.commit",
            vs.opening_convergence_final_openings_commitment_validate_ms
        );
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}  {:18} {:>7.3}",
            "targets.obj_digest",
            vs.opening_convergence_final_openings_opened_commitment_digest_ms,
            "targets.proof_dig",
            vs.opening_convergence_final_openings_opening_proof_digest_ms,
            "targets.target",
            vs.opening_convergence_final_openings_target_build_ms
        );
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}",
            "digest", vs.opening_convergence_digest_ms, "support_wrap", vs.opening_support_wrap_ms
        );
    }
    println!();
    println!("  verified seams (other components):");
    {
        let vs = &nightstream_build_perf.verified_seams;
        println!(
            "    {:24} {:>9.3}  {:24} {:>9.3}",
            "final_surface_guard", vs.final_surface_guard_ms, "decider_relation", vs.decider_relation_ms
        );
        println!("    {:24} {:>9.3}", "main_proof", vs.main_proof_ms);
        println!(
            "    {:24} {:>9.3}  {:24} {:>9.3}",
            "linkage_claims", vs.linkage_claims_ms, "linkage_root", vs.linkage_root_ms
        );
        println!(
            "    {:24} {:>9.3}  {:24} {:>9.3}",
            "statement", vs.statement_ms, "bind_side_core", vs.bind_side_statement_core_ms
        );
        println!("    {:24} {:>9.3}", "proof_binding_root", vs.proof_binding_root_ms);
    }

    print_section("CCS / Constraint Shape");
    print_kv("root_row_width", RV64IM_ROOT_ROW_WIDTH);
    print_kv("root_public_inputs", RV64IM_ROOT_PUBLIC_INPUTS);
    print_kv("constraints_per_step (n)", root_ccs.n);
    print_kv("columns_per_step (m)", root_ccs.m);
    print_kv("constraints_per_step_p2", root_ccs_n_p2);
    print_kv("columns_per_step_p2", root_ccs_m_p2);
    print_kv("matrix_count (t)", root_ccs.t());
    print_kv("max_degree", root_ccs.max_degree());
    print_kv("identity_matrices", ccs_identity_matrices);
    print_kv("total_nnz_per_step (non-zero matrix entries)", ccs_total_nnz);
    print_kv(
        "avg_nnz_per_constraint (non-zero matrix entries)",
        format!("{:.4}", per_unit(ccs_total_nnz as f64, root_ccs.n)),
    );
    print_kv("approx_constraints_for_trace", approx_trace_constraints);
    print_kv(
        "approx_constraints_per_non-halt_opcode",
        format!("{:.4}", per_unit(approx_trace_constraints as f64, opcode_count)),
    );
    print_kv("approx_nnz_for_trace (non-zero matrix entries)", approx_trace_nnz);
    print_kv(
        "approx_nnz_per_non-halt_opcode (non-zero matrix entries)",
        format!("{:.4}", per_unit(approx_trace_nnz as f64, opcode_count)),
    );
    print_kv(
        "root_params",
        format!(
            "d={} kappa={} m={} b={} k_rho={} B={} T={} s={} lambda={}",
            root_params.d,
            root_params.kappa,
            root_params.m,
            root_params.b,
            root_params.k_rho,
            root_params.B,
            root_params.T,
            root_params.s,
            root_params.lambda
        ),
    );

    print_section("Row / Step Shape");
    print_kv("execution_rows", execution_row_count);
    print_kv("real_rows", real_row_count);
    print_kv("effect_rows", effect_row_count);
    print_kv("commit_rows", commit_row_count);
    print_kv("prepared_steps", output.prepared_steps.len());
    print_kv("public_steps", output.root_lane_columns.time_len);
    print_kv("stage1_rows", output.stages.stage1.rows.len());
    print_kv("stage3_continuity", output.stages.stage3.continuity.len());
    print_kv("transcript_events", output.stages.transcript.events.len());
    print_root_main_lane_family(&output, &proof);

    print_section("Spartan Decider Shape");
    print_kv("num_cons_unpadded", decider_shape_sizes[0]);
    print_kv("num_shared_unpadded", decider_shape_sizes[1]);
    print_kv("num_precommitted_unpadded", decider_shape_sizes[2]);
    print_kv("num_rest_unpadded", decider_shape_sizes[3]);
    print_kv("num_cons_padded", decider_shape_sizes[4]);
    print_kv("num_shared_padded", decider_shape_sizes[5]);
    print_kv("num_precommitted_padded", decider_shape_sizes[6]);
    print_kv("num_rest_padded", decider_shape_sizes[7]);
    print_kv("num_public", decider_shape_sizes[8]);
    print_kv("num_challenges", decider_shape_sizes[9]);
    print_kv("a_nnz", decider_shape_debug_stats.a_nnz);
    print_kv("b_nnz", decider_shape_debug_stats.b_nnz);
    print_kv("c_nnz", decider_shape_debug_stats.c_nnz);
    print_kv("abc_total_nnz", decider_shape_debug_stats.total_nnz);
    print_kv("a_max_row_nnz", decider_shape_debug_stats.max_row_nnz_a);
    print_kv("b_max_row_nnz", decider_shape_debug_stats.max_row_nnz_b);
    print_kv("c_max_row_nnz", decider_shape_debug_stats.max_row_nnz_c);
    print_kv("abc_max_row_nnz", decider_shape_debug_stats.max_row_nnz_total);
    print_kv(
        "abc_avg_row_nnz",
        format!(
            "{:.2}",
            decider_shape_debug_stats.total_nnz as f64 / decider_shape_sizes[4].max(1) as f64
        ),
    );

    print_section("Spartan Direct Prove Diagnostics");
    print_kv(
        "published_seam_total_ms",
        format!("{:.3}", published_seam_perf.total_ms),
    );
    print_kv(
        "spartan_direct_relation_surface_ms",
        format!("{:.3}", decider_prove_perf.relation_surface_ms),
    );
    print_kv(
        "spartan_direct_prep_ms",
        format!("{:.3}", decider_prove_perf.shell.prep_ms),
    );
    print_kv(
        "spartan_direct_snark_total_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.total_ms),
    );
    print_kv(
        "spartan_direct_encode_ms",
        format!("{:.3}", decider_prove_perf.shell.encode_ms),
    );
    print_kv("spartan_direct_total_ms", format!("{:.3}", decider_prove_perf.total_ms));
    print_kv(
        "spartan.prepare_poly_tau_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.prepare_poly_tau_ms),
    );
    print_kv(
        "spartan.matrix_vector_multiply_ms",
        format!(
            "{:.3}",
            decider_prove_perf
                .shell
                .snark_perf
                .matrix_vector_multiply_ms
        ),
    );
    print_kv(
        "spartan.prepare_multilinear_polys_ms",
        format!(
            "{:.3}",
            decider_prove_perf
                .shell
                .snark_perf
                .prepare_multilinear_polys_ms
        ),
    );
    print_kv(
        "spartan.outer_sumcheck_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.outer_sumcheck_ms),
    );
    print_kv(
        "spartan.prepare_inner_claims_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.prepare_inner_claims_ms),
    );
    print_kv(
        "spartan.compute_eval_rx_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.compute_eval_rx_ms),
    );
    print_kv(
        "spartan.compute_eval_table_sparse_ms",
        format!(
            "{:.3}",
            decider_prove_perf
                .shell
                .snark_perf
                .compute_eval_table_sparse_ms
        ),
    );
    print_kv(
        "spartan.prepare_poly_abc_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.prepare_poly_abc_ms),
    );
    print_kv(
        "spartan.prepare_poly_z_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.prepare_poly_z_ms),
    );
    print_kv(
        "spartan.inner_sumcheck_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.inner_sumcheck_ms),
    );
    print_kv(
        "spartan.pcs_prove_ms",
        format!("{:.3}", decider_prove_perf.shell.snark_perf.pcs_prove_ms),
    );

    print_family_rows("Row Expansion by Family", &family_rows, opcode_count);
    print_lookup_summary(lookup_summary, opcode_count, &twist_family_counts);
    print_lookup_group_density(
        lookup_summary,
        opcode_count,
        &twist_family_counts,
        active_twist_family_count,
    );
    print_exact_stage_witness_shape(&exact_stage_rows);
    print_exact_opening_table(
        &[
            ("stage1", stage1_exact_openings),
            ("stage2", stage2_exact_openings),
            ("stage3", stage3_exact_openings),
        ],
        opcode_count,
        execution_row_count,
    );
    print_selected_vs_exact_amplification(&exact_stage_rows);
    print_exact_stage_build_breakdown(&exact_stage_rows);
    print_packaged_proof_table(&[
        ("stage1", stage1_packaged),
        ("stage2", stage2_packaged),
        ("stage3", stage3_packaged),
        ("kernel_bindings", kernel_binding_packaged),
        ("kernel_prepared", kernel_prepared_packaged),
    ]);
    print_compact_opening_build_breakdown(&build_perf);
    print_opening_surface_totals(opening_totals, opcode_count, execution_row_count);
    print_opening_reuse_proxy(&output);
    print_opening_label_summary(&selected_opening_labels);
    print_serialized_size_table("Serialized Sizes (Public Proof)", &serialized_sizes, proof_total_bytes);
    print_section("Nightstream Published Boundary");
    print_kv(
        "accepted_artifact_size",
        format!(
            "{accepted_artifact_total_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(accepted_artifact_total_bytes)
        ),
    );
    print_kv(
        "final_statement_size",
        format!(
            "{final_statement_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(final_statement_bytes)
        ),
    );
    print_kv(
        "kernel_export_source_size",
        format!(
            "{} bytes ({:.3} KiB)",
            serialized_size_bytes(&kernel_export_source),
            bytes_to_kib(serialized_size_bytes(&kernel_export_source))
        ),
    );
    print_kv(
        "spartan_decider_proof_size",
        format!(
            "{decider_proof_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(decider_proof_bytes)
        ),
    );
    print_kv(
        "nightstream_main_recursion_proof_size",
        format!(
            "{} bytes ({:.3} KiB)",
            serialized_size_bytes(nightstream_proof.main_proof().recursion_proof()),
            bytes_to_kib(serialized_size_bytes(nightstream_proof.main_proof().recursion_proof()))
        ),
    );
    let opening_group_count = prove_time_opening(&[], &nightstream_opening_bundle.claims)
        .expect("rebuild opening summary for perf output")
        .groups
        .len();
    print_kv("opening_group_count", opening_group_count);
    print_kv("opening_claim_count", nightstream_opening_bundle.claims.len());
    print_serialized_size_table(
        "Serialized Sizes (Nightstream)",
        &nightstream_serialized_sizes,
        nightstream_total_bytes,
    );
    print_section("Side Decider");
    let artifact_bytes = serialized_size_bytes(nightstream_proof.side_proof());
    println!("  {:32} {:>10} {:>10}", "component", "bytes", "KiB");
    println!(
        "  {:32} {:>10} {:>10.3}",
        "artifact (published)",
        artifact_bytes,
        bytes_to_kib(artifact_bytes)
    );
    print_verify_breakdown(
        "Theorem Verify Breakdown",
        &verify_perf,
        opcode_count,
        execution_row_count,
    );

    print_hotspot_table(
        "Critical Hotspots",
        published_pipeline_total_ms,
        total_executed_opcodes,
        &[
            ("spartan.setup", spartan_setup_ms),
            ("spartan.prove", spartan_prove_ms),
            (
                "published_seam.final.kernel_export",
                nightstream_build_perf.final_statement_kernel_export_ms,
            ),
            (
                "published_seam.recursive.rlc",
                nightstream_build_perf.final_statement_recursive_rlc_ms,
            ),
            ("public.root_main_lane.rlc", prove_perf.root_main_lane.session.rlc_ms()),
            ("public.root_main_lane.package", {
                let root_prove = &prove_perf.root_main_lane;
                (root_prove.total_ms - root_prove.prepare_steps_ms - root_prove.session.total_ms).max(0.0)
            }),
            (
                "public.root_main_lane.prepare_inputs",
                prove_perf.root_main_lane.session.prepare_inputs_ms(),
            ),
            (
                "published_seam.accepted_artifact",
                published_seam_perf.accepted_artifact_ms,
            ),
            (
                "nightstream.side_proof",
                nightstream_build_perf.verified_seams.side_binding_ms,
            ),
            ("public.kernel_projection", prove_perf.simple_kernel.total_ms),
        ],
        8,
    );

    {
        let total = prove_total_ms;
        let max_bar = total;
        tree_header("PROVING BREAKDOWN", total, per_unit(total, total_executed_opcodes));
        tree_row("├─ ", "public proof", prove_ms, max_bar, total, true);
        tree_row(
            "│  ├─ ",
            "shared trace",
            prove_perf.shared_trace_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  ├─ ",
            "kernel projection",
            prove_perf.simple_kernel.total_ms,
            max_bar,
            total,
            false,
        );

        let root_prove = &prove_perf.root_main_lane;
        let package_overhead_ms =
            (root_prove.total_ms - root_prove.prepare_steps_ms - root_prove.session.total_ms).max(0.0);
        tree_row("│  ├─ ", "root main lane", root_prove.total_ms, max_bar, total, false);
        tree_row("│     ├─ ", "package", package_overhead_ms, max_bar, total, false);
        tree_row("│     ├─ ", "Π_RLC", root_prove.session.rlc_ms(), max_bar, total, false);
        tree_row(
            "│     ├─ ",
            "prepare_inputs",
            root_prove.session.prepare_inputs_ms(),
            max_bar,
            total,
            false,
        );
        tree_row_annotated(
            "│     ├─ ",
            "Π_CCS",
            root_prove.session.ccs_ms(),
            &format!(
                "(FE {:.1}, NC {:.1})",
                root_prove.session.ccs_fe_sumcheck_ms(),
                root_prove.session.ccs_nc_sumcheck_ms()
            ),
        );
        tree_row("│     ├─ ", "Π_DEC", root_prove.session.dec_ms(), max_bar, total, false);
        tree_row(
            "│     └─ ",
            "prepare_steps",
            root_prove.prepare_steps_ms,
            max_bar,
            total,
            false,
        );
        if prove_perf.parallel_overlap_ms > 0.0 {
            tree_row_annotated(
                "│  ├─ ",
                "parallel overlap",
                -prove_perf.parallel_overlap_ms,
                "(kernel projection overlapped with root main lane)",
            );
        }
        tree_row_annotated(
            "│  └─ ",
            "other",
            (prove_ms - prove_perf.shared_trace_ms - prove_perf.simple_kernel.total_ms - root_prove.total_ms
                + prove_perf.parallel_overlap_ms)
                .max(0.0),
            "(main-lane binding + proof export)",
        );
        println!("  │");

        tree_row(
            "├─ ",
            "published seam",
            published_seam_perf.total_ms,
            max_bar,
            total,
            true,
        );
        tree_row(
            "│  ├─ ",
            "accepted_artifact",
            published_seam_perf.accepted_artifact_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  ├─ ",
            "kernel_export_source",
            published_seam_perf.kernel_export_source_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  ├─ ",
            "final_statement",
            published_seam_perf.final_statement_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "recursive_proof",
            nightstream_build_perf.final_statement_recursive_proof_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  │  ├─ ",
            "Π_RLC",
            nightstream_build_perf.final_statement_recursive_rlc_ms,
            max_bar,
            total,
            false,
        );
        tree_row("│  │  │  ├─ ", "wrapper", recursive_wrapper_ms, max_bar, total, false);
        tree_row_annotated(
            "│  │  │  ├─ ",
            "Π_CCS",
            nightstream_build_perf.final_statement_recursive_ccs_ms,
            &format!(
                "(FE {:.1}, NC {:.1})",
                nightstream_build_perf.final_statement_recursive_ccs_fe_sumcheck_ms,
                nightstream_build_perf.final_statement_recursive_ccs_nc_sumcheck_ms
            ),
        );
        tree_row(
            "│  │  │  └─ ",
            "Π_DEC",
            nightstream_build_perf.final_statement_recursive_dec_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "kernel_export",
            nightstream_build_perf.final_statement_kernel_export_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "folded_digest",
            nightstream_build_perf.final_statement_folded_digest_ms,
            max_bar,
            total,
            false,
        );
        let final_other = (published_seam_perf.final_statement_ms
            - nightstream_build_perf.final_statement_recursive_proof_ms
            - nightstream_build_perf.final_statement_kernel_export_ms
            - nightstream_build_perf.final_statement_folded_digest_ms)
            .max(0.0);
        tree_row("│  │  └─ ", "other", final_other, max_bar, total, false);
        println!("  │");

        tree_row(
            "├─ ",
            "nightstream residual build",
            nightstream_build_ms,
            max_bar,
            total,
            true,
        );

        let vs = &nightstream_build_perf.verified_seams;
        tree_row("│  ├─ ", "verified_seams", vs.total_ms, max_bar, total, false);
        tree_row_annotated("│  │  ├─ ", "side_binding *", vs.side_binding_ms, "★ biggest item");
        tree_row_annotated("│  │  │  ├─ ", "setup", vs.side_binding_setup_ms, "← amortizable");
        tree_row("│  │  │  ├─ ", "prove", vs.side_binding_prove_ms, max_bar, total, false);
        tree_row(
            "│  │  │  └─ ",
            "prepare",
            vs.side_binding_prepare_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "phase0 opening",
            vs.opening_phase0_artifact_ms,
            max_bar,
            total,
            false,
        );
        tree_row_annotated(
            "│  │  ├─ ",
            "opening_support_bundle",
            vs.opening_support_bundle_ms,
            &format!(
                "(p1 {:.1}, p2 {:.1}, tgt {:.1})",
                vs.opening_convergence_phase1_ms,
                vs.opening_convergence_phase2_ms,
                vs.opening_convergence_final_openings_ms
            ),
        );
        let seam_other =
            (vs.total_ms - vs.side_binding_ms - vs.opening_phase0_artifact_ms - vs.opening_support_bundle_ms).max(0.0);
        tree_row("│  │  └─ ", "other seams", seam_other, max_bar, total, false);

        tree_row(
            "│  └─ ",
            "side_support_bundle",
            nightstream_build_perf.side_support_bundle_ms,
            max_bar,
            total,
            false,
        );
        println!("  │");
        tree_row("├─ ", "Spartan setup/keygen", spartan_setup_ms, max_bar, total, true);
        tree_row("└─ ", "Spartan proving", spartan_prove_ms, max_bar, total, true);
        println!("  ─────────────────────────────────────────────────────────────────────");
        println!(
            "  prove total {:>7.1} ms  ({:.2} ms/op)    amortized: {:.1} ms ({:.2} ms/op)",
            total,
            per_unit(total, total_executed_opcodes),
            amortized_prove_ms,
            per_unit(amortized_prove_ms, total_executed_opcodes),
        );
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  RV64IM Perf Snapshot   N={:<6}  rows={:<6}  {:.2} rows/op              ║",
        opcode_count,
        execution_row_count,
        per_unit(execution_row_count as f64, opcode_count)
    );
    println!("╠═══════════════════════════ PROVING ══════════════════════════════════════╣");
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "pre-Spartan (public + nightstream)",
        published_prove_before_spartan_ms,
        per_unit(published_prove_before_spartan_ms, total_executed_opcodes)
    );
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op  (one-time) ║",
        "Spartan setup/keygen",
        spartan_setup_ms,
        per_unit(spartan_setup_ms, total_executed_opcodes)
    );
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "Spartan proving",
        spartan_prove_ms,
        per_unit(spartan_prove_ms, total_executed_opcodes)
    );
    println!("║                                          ─────────────────────────────  ║");
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "prove total",
        prove_total_ms,
        per_unit(prove_total_ms, total_executed_opcodes)
    );
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "  amortized (−keygen/setup)",
        amortized_prove_ms,
        per_unit(amortized_prove_ms, total_executed_opcodes)
    );
    println!("╠═══════════════════════════ VERIFYING ════════════════════════════════════╣");
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "verify total",
        verify_total_ms,
        per_unit(verify_total_ms, total_executed_opcodes)
    );
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "PIPELINE TOTAL",
        published_pipeline_total_ms,
        per_unit(published_pipeline_total_ms, total_executed_opcodes)
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!(
        "  proof: {proof_total_bytes} bytes ({proof_total_kib:.1} KiB)  |  nightstream: {nightstream_total_bytes} bytes ({nightstream_total_kib:.1} KiB)"
    );
    println!("  opcodes: {total_executed_opcodes} ({unique_opcode_labels})");
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn chip8_nightstream_perf_snapshot() {
    let input = chip8_support::build_jump_kernel_input(4);

    let recursive_started = Instant::now();
    let (recursive_statement, final_proof) = prove_chip8_recursive(&input).expect("prove chip8 recursive");
    let recursive_ms = millis_since(recursive_started);

    let decider_setup_started = Instant::now();
    let (decider_pk, _decider_vk) =
        setup_chip8_spartan2_decider(&recursive_statement, &final_proof).expect("setup chip8 spartan2 decider");
    let decider_setup_ms = millis_since(decider_setup_started);

    let decider_prove_started = Instant::now();
    let decider_proof = prove_chip8_spartan2_decider(&decider_pk, &recursive_statement, &final_proof)
        .expect("prove chip8 spartan2 decider");
    let decider_prove_ms = millis_since(decider_prove_started);

    let nightstream_build_started = Instant::now();
    let (nightstream_statement, nightstream_proof) =
        build_chip8_nightstream_from_recursive_proof(&recursive_statement, &final_proof)
            .expect("build chip8 nightstream proof");
    let nightstream_build_ms = millis_since(nightstream_build_started);

    let nightstream_verify_started = Instant::now();
    verify_chip8_nightstream_from_recursive_proof(
        &recursive_statement,
        &final_proof,
        &nightstream_statement,
        &nightstream_proof,
    )
    .expect("verify chip8 nightstream proof");
    let nightstream_verify_ms = millis_since(nightstream_verify_started);

    let decider_proof_bytes = serialized_size_bytes(&decider_proof);
    let nightstream_serialized_sizes = [
        SerializedSizeRow {
            label: "nightstream.total",
            bytes: serialized_size_bytes(&(nightstream_statement.clone(), nightstream_proof.clone())),
        },
        SerializedSizeRow {
            label: "nightstream.statement",
            bytes: serialized_size_bytes(&nightstream_statement),
        },
        SerializedSizeRow {
            label: "nightstream.proof",
            bytes: serialized_size_bytes(&nightstream_proof),
        },
        SerializedSizeRow {
            label: "nightstream.main_decider_proof",
            bytes: serialized_size_bytes(&nightstream_proof.main_decider_proof),
        },
        SerializedSizeRow {
            label: "nightstream.main_residual_proof",
            bytes: serialized_size_bytes(&nightstream_proof.main_residual_proof),
        },
    ];
    let nightstream_total_bytes = nightstream_serialized_sizes[0].bytes;

    print_section("CHIP-8 Nightstream Perf Snapshot");
    print_kv("semantic_rows", input.witness.semantic_trace_rows.len());
    print_kv("chunk_count", recursive_statement.folded.chunk_count);
    print_kv("semantic_step_count", recursive_statement.folded.semantic_step_count);
    print_kv(
        "fold_schedule",
        format_fold_schedule(recursive_statement.folded.fold_schedule),
    );
    print_kv("final_pc_word", recursive_statement.final_state.pc_word);

    print_timing_table(
        "Raw Timing",
        &[
            ("prove_chip8_recursive", recursive_ms),
            ("setup_chip8_spartan2_decider", decider_setup_ms),
            ("prove_chip8_spartan2_decider", decider_prove_ms),
            ("build_chip8_nightstream", nightstream_build_ms),
            ("verify_chip8_nightstream", nightstream_verify_ms),
        ],
        recursive_statement.folded.semantic_step_count as usize,
        recursive_statement.folded.semantic_step_count as usize,
    );

    print_section("Nightstream Published Boundary");
    print_kv(
        "spartan_decider_proof_size",
        format!(
            "{decider_proof_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(decider_proof_bytes)
        ),
    );
    print_serialized_size_table(
        "Serialized Sizes (Nightstream)",
        &nightstream_serialized_sizes,
        nightstream_total_bytes,
    );

    print_section("Final Summary");
    print_kv(
        "nightstream published size",
        format!(
            "{nightstream_total_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(nightstream_total_bytes)
        ),
    );
}
