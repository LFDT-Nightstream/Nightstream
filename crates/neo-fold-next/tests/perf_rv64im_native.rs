use std::env;
use std::time::Instant;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::build_rv64im_chunk_step_ivc_relations;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc::{
    derive_rv64im_ivc_step_cap, Rv64imIvcAppendPerf, Rv64imIvcState, Rv64imIvcVerifyPerf,
};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_recursion_shape_for_step_cap,
    prove_rv64im_accepted_proof_with_options, rv64im_simple_root_params_for_step_cap, Rv64imProofInput,
    Rv64imPublicProofOptions, RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
use serde::Serialize;

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
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

fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:30} {}", label, value);
}

fn format_ms_per_opcode(ms: f64, opcode_count: usize) -> String {
    format!("{ms:.3} ms ({:.4} ms/op)", per_unit(ms, opcode_count))
}

fn format_fold_schedule(schedule: FoldSchedule) -> String {
    match schedule {
        FoldSchedule::WholeTrace => "WholeTrace".to_string(),
        FoldSchedule::RowsPerChunk(rows) => format!("RowsPerChunk({rows})"),
    }
}

fn serialize_len<T: Serialize>(value: &T) -> usize {
    bincode::serialize(value)
        .expect("serialize perf artifact")
        .len()
}

fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes < KB {
        return format!("{bytes}b");
    }
    if bytes < MB {
        return format!("{}kb", (bytes + (KB / 2)) / KB);
    }
    if bytes < GB {
        return format!("{:.1}mb", bytes as f64 / MB as f64);
    }
    format!("{:.1}gb", bytes as f64 / GB as f64)
}

fn print_perf_rows(title: &str, rows: &[(&str, f64)], total_ms: f64, opcode_count: usize) {
    print_section(title);
    for (label, ms) in rows {
        let pct = if total_ms == 0.0 { 0.0 } else { (ms / total_ms) * 100.0 };
        print_kv(
            label,
            format!("{ms:.3} ms ({:.4} ms/op, {pct:.1}%)", per_unit(*ms, opcode_count)),
        );
    }
}

fn append_stage_rows(perfs: &[Rv64imIvcAppendPerf]) -> Vec<(&'static str, f64)> {
    let mut validate_state_surface_ms = 0.0;
    let mut validate_relation_surface_ms = 0.0;
    let mut validate_next_relation_surface_ms = 0.0;
    let mut verified_step_statement_ms = 0.0;
    let mut fixed_shape_chunk_summary_ms = 0.0;
    let mut main_circuit_trace_ms = 0.0;
    let mut construction2_pi_fold_ms = 0.0;
    let mut advice_build_ms = 0.0;
    let mut evaluate_f_prime_ms = 0.0;
    let mut finalize_state_ms = 0.0;

    for perf in perfs {
        validate_state_surface_ms += perf.validate_state_surface_ms;
        validate_relation_surface_ms += perf.validate_relation_surface_ms;
        validate_next_relation_surface_ms += perf.validate_next_relation_surface_ms;
        verified_step_statement_ms += perf.verified_step_statement_ms;
        fixed_shape_chunk_summary_ms += perf.fixed_shape_chunk_summary_ms;
        main_circuit_trace_ms += perf.main_circuit_trace_ms;
        construction2_pi_fold_ms += perf.construction2_pi_fold_ms;
        advice_build_ms += perf.advice_build_ms;
        evaluate_f_prime_ms += perf.evaluate_f_prime_ms;
        finalize_state_ms += perf.finalize_state_ms;
    }

    vec![
        ("validate_state_surface", validate_state_surface_ms),
        ("validate_relation_surface", validate_relation_surface_ms),
        ("validate_next_relation", validate_next_relation_surface_ms),
        ("verified_step_statement", verified_step_statement_ms),
        ("fixed_shape_chunk_summary", fixed_shape_chunk_summary_ms),
        ("main_circuit_trace", main_circuit_trace_ms),
        ("construction2_pi_fold", construction2_pi_fold_ms),
        ("advice_build", advice_build_ms),
        ("evaluate_f_prime", evaluate_f_prime_ms),
        ("finalize_state", finalize_state_ms),
    ]
}

fn verify_stage_rows(perf: Rv64imIvcVerifyPerf) -> Vec<(&'static str, f64)> {
    vec![
        ("validate_state_surface", perf.validate_state_surface_ms),
        ("build_terminal_relation", perf.build_terminal_relation_ms),
        ("verified_step_statement", perf.verified_step_statement_ms),
        ("context_lookup", perf.context_lookup_ms),
        ("replay_step", perf.replay_step_ms),
        ("compare_running_state", perf.compare_running_state_ms),
        ("transcript_snapshot", perf.transcript_snapshot_ms),
        ("compare_step_public", perf.compare_step_public_ms),
    ]
}

fn print_append_step_rows(perfs: &[Rv64imIvcAppendPerf], opcode_count: usize) {
    print_section("Native IVC Append Steps");
    let head = 8usize;
    let tail = 4usize;
    let len = perfs.len();
    for (idx, perf) in perfs.iter().enumerate() {
        let show = len <= head + tail || idx < head || idx >= len.saturating_sub(tail);
        if !show {
            if idx == head {
                println!("  ...");
            }
            continue;
        }
        println!(
            "  step {:>3}  total={:>8.3} ms ({:>7.4} ms/op)  validate={:>6.3}  statement={:>6.3}  trace={:>7.3}  pi_fold={:>6.3}  advice={:>6.3}  f_prime={:>7.3}",
            idx,
            perf.total_ms,
            per_unit(perf.total_ms, opcode_count),
            perf.validate_state_surface_ms
                + perf.validate_relation_surface_ms
                + perf.validate_next_relation_surface_ms,
            perf.verified_step_statement_ms + perf.fixed_shape_chunk_summary_ms,
            perf.main_circuit_trace_ms,
            perf.construction2_pi_fold_ms,
            perf.advice_build_ms,
            perf.evaluate_f_prime_ms,
        );
    }
}

fn print_append_step_summary(perfs: &[Rv64imIvcAppendPerf], opcode_count: usize) {
    let first_ms = perfs.first().map_or(0.0, |perf| perf.total_ms);
    let steady_state = if perfs.len() > 1 { &perfs[1..] } else { &[][..] };
    let steady_state_avg_ms = if steady_state.is_empty() {
        0.0
    } else {
        steady_state.iter().map(|perf| perf.total_ms).sum::<f64>() / steady_state.len() as f64
    };
    let min_ms = perfs
        .iter()
        .map(|perf| perf.total_ms)
        .fold(f64::INFINITY, f64::min);
    let max_ms = perfs.iter().map(|perf| perf.total_ms).fold(0.0, f64::max);

    print_section("Native IVC Append Step Summary");
    print_kv("fold_count", perfs.len());
    print_kv("first_step", format_ms_per_opcode(first_ms, opcode_count));
    print_kv(
        "steady_state_avg",
        format_ms_per_opcode(steady_state_avg_ms, opcode_count),
    );
    print_kv("min_step", format_ms_per_opcode(min_ms, opcode_count));
    print_kv("max_step", format_ms_per_opcode(max_ms, opcode_count));
}

fn run_rv64im_mixed_opcode_native_ivc_perf_snapshot(schedule: FoldSchedule, title: &str) {
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let total_opcodes = source.program_words.len();
    let input = Rv64imProofInput {
        source,
        max_steps: total_opcodes,
    };
    let public_proof_options = Rv64imPublicProofOptions {
        root_fold_schedule: schedule,
    };

    let relation_prep_started = Instant::now();
    let (accepted_artifact, _audit) =
        prove_rv64im_accepted_proof_with_options(&input, public_proof_options).expect("prove rv64im accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("prove rv64im final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");
    let relation_prep_ms = millis_since(relation_prep_started);

    assert!(
        !relations.is_empty(),
        "mixed-opcode native perf fixture must expose at least one chunk-step relation"
    );
    let semantic_step_count = relations
        .last()
        .map(|relation| relation.statement.step_public.step_hi as usize)
        .expect("non-empty native perf fixture must expose a terminal step count");
    let step_cap = derive_rv64im_ivc_step_cap(schedule, semantic_step_count)
        .expect("derive native step_cap from the configured fold schedule");

    let native_append_started = Instant::now();
    let mut ivc_state = Rv64imIvcState::init_with_step_cap(step_cap).expect("build initial rv64im ivc state");
    let mut append_perfs = Vec::with_capacity(relations.len());
    for relation in &relations {
        let (next_state, perf) = ivc_state
            .append_with_perf(relation)
            .expect("append rv64im ivc relation");
        append_perfs.push(perf);
        ivc_state = next_state;
    }
    let native_append_ms = millis_since(native_append_started);

    let native_verify_started = Instant::now();
    let native_verify_perf = ivc_state
        .verify_with_perf()
        .expect("verify native rv64im ivc state");
    let native_verify_ms = millis_since(native_verify_started);
    let native_total_ms = native_append_ms + native_verify_ms;

    let public_image = ivc_state.public_image();
    let kernel_params = rv64im_simple_root_params_for_step_cap(step_cap);
    let recursion_shape = build_rv64im_recursion_shape_for_step_cap(step_cap).expect("build rv64im recursion shape");
    let final_statement_bytes = serialize_len(&final_statement);
    let final_proof_bytes = serialize_len(&final_proof);
    let relation_statements_total_bytes: usize = relations
        .iter()
        .map(|relation| serialize_len(&relation.statement))
        .sum();
    let ivc_state_bytes = serialize_len(&ivc_state);
    let public_image_bytes = serialize_len(&public_image);
    let terminal_statement = ivc_state
        .latest_terminal_statement()
        .expect("non-empty native IVC state must carry the latest terminal statement");
    let terminal_statement_bytes = serialize_len(terminal_statement);
    assert_eq!(
        terminal_statement.step_public.step_hi, public_image.step_count,
        "native IVC terminal statement must match the carried semantic step count"
    );

    print_section(title);
    print_kv("mixed_opcode_non_halt_ops", opcode_count);
    print_kv("total_program_words", total_opcodes);
    print_kv("root_fold_schedule", format_fold_schedule(schedule));
    print_kv("native_step_cap", step_cap);
    print_kv("relation_count", relations.len());
    print_kv("fold_count", relations.len());
    print_kv("chunk_count", public_image.chunk_count);
    print_kv("step_count", public_image.step_count);

    print_section("Fixture Prep (not native IVC)");
    print_kv(
        "accepted+final+relations",
        format_ms_per_opcode(relation_prep_ms, total_opcodes),
    );

    print_section("Native IVC");
    print_kv("native append", format_ms_per_opcode(native_append_ms, total_opcodes));
    print_kv("native verify", format_ms_per_opcode(native_verify_ms, total_opcodes));
    print_kv(
        "native append+verify",
        format_ms_per_opcode(native_total_ms, total_opcodes),
    );

    let append_rows = append_stage_rows(&append_perfs);
    print_perf_rows(
        "Native IVC Append Breakdown",
        &append_rows,
        native_append_ms,
        total_opcodes,
    );
    print_append_step_rows(&append_perfs, total_opcodes);
    print_append_step_summary(&append_perfs, total_opcodes);

    let verify_rows = verify_stage_rows(native_verify_perf);
    print_perf_rows(
        "Native IVC Verify Breakdown",
        &verify_rows,
        native_verify_ms,
        total_opcodes,
    );

    print_section("Artifact Sizes");
    print_kv("final_statement_size", format_bytes(final_statement_bytes));
    print_kv("final_proof_size", format_bytes(final_proof_bytes));
    print_kv(
        "relation_statements_total_size",
        format_bytes(relation_statements_total_bytes),
    );
    print_kv("ivc_state_size", format_bytes(ivc_state_bytes));
    print_kv("public_image_size", format_bytes(public_image_bytes));
    print_kv("terminal_statement_size", format_bytes(terminal_statement_bytes));

    print_section("Live Kernel Params");
    print_kv("b", kernel_params.b);
    print_kv("k_rho", kernel_params.k_rho);
    print_kv("B", kernel_params.B);
    print_kv("T", kernel_params.T);

    print_section("Fixed Recursion Shape");
    print_kv("shape_step_cap", recursion_shape.step_cap);
    print_kv("shape_soundness_k", recursion_shape.soundness_k);
    print_kv("shape_soundness_big_k", recursion_shape.soundness_big_k);
    print_kv("t_matrices", recursion_shape.t_matrices);
    print_kv("log_m", recursion_shape.log_m);
    print_kv("d_sc", recursion_shape.d_sc);
    print_kv("n_R", recursion_shape.n_R);
    print_kv("n_R_in", recursion_shape.n_R_in);
    print_kv("shape_b", recursion_shape.b);
    print_kv("shape_decomposition_k", recursion_shape.decomposition_k);
    print_kv("side_families_active", recursion_shape.side_families_active.len());
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv64im_mixed_opcode_native_ivc_perf_snapshot() {
    run_rv64im_mixed_opcode_native_ivc_perf_snapshot(
        FoldSchedule::WholeTrace,
        "RV64IM Native IVC Perf Snapshot (no Spartan, whole trace)",
    );
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv64im_mixed_opcode_native_ivc_perf_snapshot_rows_per_chunk_1() {
    run_rv64im_mixed_opcode_native_ivc_perf_snapshot(
        FoldSchedule::RowsPerChunk(1),
        "RV64IM Native IVC Perf Snapshot (no Spartan, per-op folds)",
    );
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv64im_mixed_opcode_native_ivc_perf_snapshot_whole_trace() {
    run_rv64im_mixed_opcode_native_ivc_perf_snapshot(
        FoldSchedule::WholeTrace,
        "RV64IM Native IVC Perf Snapshot (no Spartan, whole trace)",
    );
}
