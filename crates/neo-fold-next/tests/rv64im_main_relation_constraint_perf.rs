//! Constraint-focused perf snapshot for the RV64IM main-relation Spartan circuit.

use std::collections::BTreeSet;
use std::env;
use std::time::Instant;

use neo_fold_next::rv64im::audit::{
    measure_rv64im_spartan2_decider_circuit, Rv64imMainRelationCircuitMetrics, Rv64imMainRelationCountBucket,
    Rv64imMainRelationHotspotDetail,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, prove_rv64im_public_proof,
    Rv64imProofInput, RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};

const RV64IM_MAIN_RELATION_TOTAL_BUDGET: usize = 50_000;
const RV64IM_MAIN_RELATION_MARGINAL_CE_BUDGET: usize = 500;

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
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

fn per_unit(total: usize, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        total as f64 / units as f64
    }
}

fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:34} {}", label, value);
}

fn print_hotspot_detail(metrics: &Rv64imMainRelationCircuitMetrics, detail: &Rv64imMainRelationHotspotDetail) {
    print_section(&format!("Hotspot Deep Dive: {}", detail.parent_namespace));
    print_kv("total_constraints", format_count(detail.total_constraint_count));
    print_kv(
        "coverage",
        format!(
            "{:.2}%",
            detail.leaf_coverage_constraint_count as f64 * 100.0 / detail.total_constraint_count.max(1) as f64
        ),
    );
    println!(
        "  {:56} {:>12} {:>8} {:>8} {:>12} {:>12}",
        "sub_namespace", "constraints", "share", "cum%", "terms", "max_row"
    );
    let mut cumulative = 0usize;
    for bucket in &detail.leaf_buckets {
        if bucket.constraint_count == 0 {
            continue;
        }
        cumulative += bucket.constraint_count;
        println!(
            "  {:56} {:>12} {:>7.2}% {:>7.2}% {:>12} {:>12}",
            bucket.namespace,
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / detail.total_constraint_count.max(1) as f64,
            cumulative as f64 * 100.0 / detail.total_constraint_count.max(1) as f64,
            format_count(bucket.total_term_count),
            format_count(bucket.max_constraint_term_count),
        );
    }
    print_kv(
        "global_share",
        format!(
            "{:.2}%",
            detail.total_constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64
        ),
    );
}

fn print_hotspot_table(
    title: &str,
    metrics: &Rv64imMainRelationCircuitMetrics,
    buckets: impl Iterator<Item = Rv64imMainRelationCountBucket>,
) {
    print_section(title);
    println!(
        "  {:42} {:>12} {:>8} {:>8} {:>12} {:>12}",
        "namespace", "constraints", "share", "cum%", "terms", "max_row"
    );
    let mut cumulative = 0usize;
    for bucket in buckets {
        if bucket.constraint_count == 0 {
            continue;
        }
        cumulative += bucket.constraint_count;
        println!(
            "  {:42} {:>12} {:>7.2}% {:>7.2}% {:>12} {:>12}",
            bucket.namespace,
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            cumulative as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.total_term_count),
            format_count(bucket.max_constraint_term_count),
        );
    }
    print_kv(
        "hotspot_coverage",
        format!(
            "{:.2}%",
            cumulative as f64 * 100.0 / metrics.constraint_count.max(1) as f64
        ),
    );
}

fn hotspot_family(namespace: &str) -> Option<&'static str> {
    if namespace.contains("_carrier_outputs") {
        Some("carrier_output")
    } else if namespace.contains("_carrier_children") {
        Some("carrier_child")
    } else if namespace.contains("_carrier_parent") {
        Some("carrier_parent")
    } else {
        None
    }
}

fn representative_namespace<'a>(metrics: &'a Rv64imMainRelationCircuitMetrics, prefix: &str) -> &'a str {
    metrics
        .representative_claim_details
        .iter()
        .find(|detail| detail.parent_namespace.contains(prefix))
        .map(|detail| detail.parent_namespace.as_str())
        .unwrap_or("<none>")
}

fn avg_leaf_constraints_with_substring(metrics: &Rv64imMainRelationCircuitMetrics, parent: &str, needle: &str) -> f64 {
    let Some(detail) = metrics
        .hotspot_details
        .iter()
        .find(|detail| detail.parent_namespace == parent)
    else {
        return 0.0;
    };
    let matching = detail
        .leaf_buckets
        .iter()
        .filter(|bucket| bucket.namespace.contains(needle))
        .collect::<Vec<_>>();
    if matching.is_empty() {
        0.0
    } else {
        matching
            .iter()
            .map(|bucket| bucket.constraint_count)
            .sum::<usize>() as f64
            / matching.len() as f64
    }
}

fn phase_constraint(metrics: &Rv64imMainRelationCircuitMetrics, phase: &str) -> usize {
    metrics
        .phase_rollup
        .iter()
        .find(|bucket| bucket.phase == phase)
        .map(|bucket| bucket.constraint_count)
        .unwrap_or(0)
}

fn phase_constraints(metrics: &Rv64imMainRelationCircuitMetrics, phases: &[&str]) -> usize {
    phases
        .iter()
        .map(|phase| phase_constraint(metrics, phase))
        .sum()
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv64im_main_relation_constraint_perf_snapshot() {
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };

    let prove_started = Instant::now();
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let prove_ms = millis_since(prove_started);

    let artifact_started = Instant::now();
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let artifact_ms = millis_since(artifact_started);

    let final_started = Instant::now();
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let final_ms = millis_since(final_started);

    let measure_started = Instant::now();
    let metrics = measure_rv64im_spartan2_decider_circuit(&statement, &final_proof)
        .expect("measure rv64im spartan2 decider circuit");
    let measure_ms = millis_since(measure_started);

    let carrier_output_constraints = phase_constraint(&metrics, "carrier_outputs");
    let carrier_parent_constraints = phase_constraint(&metrics, "carrier_parent");
    let carrier_child_constraints = phase_constraint(&metrics, "carrier_children");
    let carrier_relation_constraints =
        carrier_output_constraints + carrier_parent_constraints + carrier_child_constraints;
    let replay_transcript_constraints = phase_constraints(
        &metrics,
        &[
            "transcript_core",
            "transcript_bind",
            "challenge_sampling",
            "initial_sum",
            "sumcheck_fe",
            "sumcheck_nc",
            "fold_digest",
        ],
    );
    let replay_relation_constraints = metrics
        .constraint_count
        .saturating_sub(carrier_relation_constraints);
    let replay_non_transcript_constraints = replay_relation_constraints.saturating_sub(replay_transcript_constraints);

    print_section("Benchmark Context");
    print_kv(
        "command",
        "NS_DEBUG_N=<n> cargo test -p neo-fold-next --release --test rv64im_main_relation_constraint_perf -- --ignored --nocapture rv64im_main_relation_constraint_perf_snapshot",
    );
    print_kv("NS_DEBUG_N", opcode_count);
    print_kv("public_proof_ms", format!("{prove_ms:.3}"));
    print_kv("accepted_artifact_ms", format!("{artifact_ms:.3}"));
    print_kv("final_statement_ms", format!("{final_ms:.3}"));
    print_kv("counting_measure_ms", format!("{measure_ms:.3}"));

    print_section("Trace Shape");
    print_kv("chunks", metrics.trace.chunk_count);
    print_kv("final_claims", metrics.trace.final_claim_count);
    print_kv("fresh_claims", metrics.trace.fresh_claim_count);
    print_kv("ccs_outputs", metrics.trace.ccs_output_count);
    print_kv("child_claims", metrics.trace.child_claim_count);
    print_kv("fresh_witness_cells", format_count(metrics.trace.fresh_witness_cells));
    print_kv("child_witness_cells", format_count(metrics.trace.child_witness_cells));
    print_kv(
        "max_chunk_witness_cells",
        format_count(metrics.trace.max_chunk_witness_cells),
    );
    print_kv("fe_rounds", metrics.trace.fe_round_count);
    print_kv("nc_rounds", metrics.trace.nc_round_count);
    print_kv("fe_round_coeffs", metrics.trace.fe_round_coeff_count);
    print_kv("nc_round_coeffs", metrics.trace.nc_round_coeff_count);
    print_kv(
        "fe_coeffs_per_round",
        format!(
            "{:.1}",
            per_unit(metrics.trace.fe_round_coeff_count, metrics.trace.fe_round_count)
        ),
    );
    print_kv(
        "nc_coeffs_per_round",
        format!(
            "{:.1}",
            per_unit(metrics.trace.nc_round_coeff_count, metrics.trace.nc_round_count)
        ),
    );

    print_section("Circuit Totals");
    print_kv("public_inputs", format_count(metrics.public_input_count));
    print_kv("aux_vars", format_count(metrics.aux_count));
    print_kv("constraints", format_count(metrics.constraint_count));
    print_kv(
        "linear_constraints",
        format!(
            "{} ({:.2}%)",
            format_count(metrics.linear_constraint_count),
            metrics.linear_constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
        ),
    );
    print_kv(
        "quadratic_constraints",
        format!(
            "{} ({:.2}%)",
            format_count(metrics.quadratic_constraint_count),
            metrics.quadratic_constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
        ),
    );
    print_kv("a_terms", format_count(metrics.a_term_count));
    print_kv("b_terms", format_count(metrics.b_term_count));
    print_kv("c_terms", format_count(metrics.c_term_count));
    print_kv("total_terms", format_count(metrics.total_term_count));
    print_kv("max_constraint_terms", format_count(metrics.max_constraint_term_count));
    print_kv(
        "constraints_per_witness_cell",
        format!(
            "{:.1}",
            per_unit(
                metrics.constraint_count,
                metrics.trace.fresh_witness_cells + metrics.trace.child_witness_cells,
            )
        ),
    );
    print_kv(
        "amortized_per_final_claim",
        format!(
            "{:.1}",
            per_unit(metrics.constraint_count, metrics.trace.final_claim_count)
        ),
    );
    print_section("Amortization");
    print_kv(
        "replay_per_chunk",
        format!(
            "{:.1}",
            per_unit(replay_relation_constraints, metrics.trace.chunk_count)
        ),
    );
    print_kv(
        "carrier_output_per_claim",
        format!(
            "{:.1}",
            per_unit(carrier_output_constraints, metrics.trace.ccs_output_count)
        ),
    );
    print_kv(
        "carrier_child_per_claim",
        format!(
            "{:.1}",
            per_unit(carrier_child_constraints, metrics.trace.child_claim_count)
        ),
    );
    print_kv(
        "carrier_parent_per_chunk",
        format!("{:.1}", per_unit(carrier_parent_constraints, metrics.trace.chunk_count)),
    );

    let dense_output_digit_constraints =
        metrics.trace.output_logical_col_count + 2 * metrics.trace.output_digit_slot_count;
    let sparse_output_digit_constraints =
        metrics.trace.output_logical_col_count + 2 * metrics.trace.output_nonzero_digit_count;
    print_section("Digit Sparsity");
    print_kv(
        "output_logical_cols",
        format_count(metrics.trace.output_logical_col_count),
    );
    print_kv(
        "output_digit_slots",
        format_count(metrics.trace.output_digit_slot_count),
    );
    print_kv(
        "output_nonzero_digits",
        format_count(metrics.trace.output_nonzero_digit_count),
    );
    print_kv(
        "output_nonzero_share",
        format!(
            "{:.2}%",
            metrics.trace.output_nonzero_digit_count as f64 * 100.0
                / metrics.trace.output_digit_slot_count.max(1) as f64
        ),
    );
    print_kv("dense_output_digit_cost", format_count(dense_output_digit_constraints));
    print_kv(
        "sparse_output_digit_cost",
        format_count(sparse_output_digit_constraints),
    );
    print_kv(
        "digit_constraints_saved",
        format!(
            "{} ({:.2}%)",
            format_count(dense_output_digit_constraints.saturating_sub(sparse_output_digit_constraints)),
            dense_output_digit_constraints.saturating_sub(sparse_output_digit_constraints) as f64 * 100.0
                / dense_output_digit_constraints.max(1) as f64
        ),
    );
    print_section("Output NC Sparsity");
    print_kv(
        "output_y_zcol_slots",
        format_count(metrics.trace.output_y_zcol_slot_count),
    );
    print_kv(
        "output_nonzero_y_zcol",
        format_count(metrics.trace.output_nonzero_y_zcol_count),
    );
    print_kv(
        "output_y_zcol_nonzero_share",
        format!(
            "{:.2}%",
            metrics.trace.output_nonzero_y_zcol_count as f64 * 100.0
                / metrics.trace.output_y_zcol_slot_count.max(1) as f64
        ),
    );

    print_section("Proof-Complete Split");
    print_kv(
        "replay_relation",
        format!(
            "{} ({:.2}%)",
            format_count(replay_relation_constraints),
            replay_relation_constraints as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
        ),
    );
    print_kv(
        "carrier_relation",
        format!(
            "{} ({:.2}%)",
            format_count(carrier_relation_constraints),
            carrier_relation_constraints as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
        ),
    );
    print_kv(
        "replay_transcript",
        format!(
            "{} ({:.2}%)",
            format_count(replay_transcript_constraints),
            replay_transcript_constraints as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
        ),
    );
    print_kv(
        "replay_non_transcript",
        format!(
            "{} ({:.2}%)",
            format_count(replay_non_transcript_constraints),
            replay_non_transcript_constraints as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
        ),
    );
    print_kv(
        "fe_append_round_avg",
        format!(
            "{:.1}",
            avg_leaf_constraints_with_substring(&metrics, "chunk_0_fe_sumcheck", "_append_round_")
        ),
    );
    print_kv(
        "fe_challenge_round_avg",
        format!(
            "{:.1}",
            avg_leaf_constraints_with_substring(&metrics, "chunk_0_fe_sumcheck", "_challenge_")
        ),
    );
    print_kv(
        "fe_sumcheck_round_avg",
        format!(
            "{:.1}",
            per_unit(phase_constraint(&metrics, "sumcheck_fe"), metrics.trace.fe_round_count)
        ),
    );
    print_kv(
        "nc_append_round_avg",
        format!(
            "{:.1}",
            avg_leaf_constraints_with_substring(&metrics, "chunk_0_nc_sumcheck", "_append_round_")
        ),
    );
    print_kv(
        "nc_challenge_round_avg",
        format!(
            "{:.1}",
            avg_leaf_constraints_with_substring(&metrics, "chunk_0_nc_sumcheck", "_challenge_")
        ),
    );
    print_kv(
        "nc_sumcheck_round_avg",
        format!(
            "{:.1}",
            per_unit(phase_constraint(&metrics, "sumcheck_nc"), metrics.trace.nc_round_count)
        ),
    );

    print_section("Budgets");
    print_kv(
        "total",
        format!(
            "{} / {} ({:.1}x)",
            format_count(metrics.constraint_count),
            format_count(RV64IM_MAIN_RELATION_TOTAL_BUDGET),
            metrics.constraint_count as f64 / RV64IM_MAIN_RELATION_TOTAL_BUDGET as f64,
        ),
    );
    print_kv(
        "carrier_output_per_claim",
        format!(
            "{:.1} / {} ({:.1}x)",
            per_unit(carrier_output_constraints, metrics.trace.ccs_output_count),
            format_count(RV64IM_MAIN_RELATION_MARGINAL_CE_BUDGET),
            per_unit(carrier_output_constraints, metrics.trace.ccs_output_count)
                / RV64IM_MAIN_RELATION_MARGINAL_CE_BUDGET as f64,
        ),
    );
    print_kv(
        "carrier_child_per_claim",
        format!(
            "{:.1} / {} ({:.1}x)",
            per_unit(carrier_child_constraints, metrics.trace.child_claim_count),
            format_count(RV64IM_MAIN_RELATION_MARGINAL_CE_BUDGET),
            per_unit(carrier_child_constraints, metrics.trace.child_claim_count)
                / RV64IM_MAIN_RELATION_MARGINAL_CE_BUDGET as f64,
        ),
    );
    print_section("Phase Rollup");
    println!(
        "  {:18} {:>12} {:>8} {:>8} {:>12} {:>20}",
        "phase", "constraints", "share", "buckets", "avg/bucket", "hottest_bucket"
    );
    for bucket in &metrics.phase_rollup {
        if bucket.constraint_count == 0 {
            continue;
        }
        println!(
            "  {:18} {:>12} {:>7.2}% {:>8} {:>12.1} {:>20}",
            bucket.phase,
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.bucket_count),
            per_unit(bucket.constraint_count, bucket.bucket_count),
            bucket.max_bucket_namespace,
        );
    }

    print_section("Component Rollup");
    println!(
        "  {:18} {:>12} {:>8} {:>8} {:>12} {:>20}",
        "component", "constraints", "share", "buckets", "avg/bucket", "hottest_bucket"
    );
    for bucket in &metrics.component_rollup {
        if bucket.constraint_count == 0 {
            continue;
        }
        println!(
            "  {:18} {:>12} {:>7.2}% {:>8} {:>12.1} {:>20}",
            bucket.component,
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.bucket_count),
            per_unit(bucket.constraint_count, bucket.bucket_count),
            bucket.max_bucket_namespace,
        );
    }

    print_section("Repeated Claim Families");
    println!(
        "  {:18} {:>8} {:>12} {:>10} {:>24}",
        "family", "count", "total", "avg", "representative"
    );
    println!(
        "  {:18} {:>8} {:>12} {:>10.1} {:>24}",
        "carrier_output",
        format_count(metrics.trace.ccs_output_count),
        format_count(carrier_output_constraints),
        per_unit(carrier_output_constraints, metrics.trace.ccs_output_count),
        representative_namespace(&metrics, "_carrier_outputs"),
    );
    println!(
        "  {:18} {:>8} {:>12} {:>10.1} {:>24}",
        "carrier_child",
        format_count(metrics.trace.child_claim_count),
        format_count(carrier_child_constraints),
        per_unit(carrier_child_constraints, metrics.trace.child_claim_count),
        representative_namespace(&metrics, "_carrier_children"),
    );
    println!(
        "  {:18} {:>8} {:>12} {:>10.1} {:>24}",
        "carrier_parent",
        format_count(metrics.trace.chunk_count),
        format_count(carrier_parent_constraints),
        per_unit(carrier_parent_constraints, metrics.trace.chunk_count),
        representative_namespace(&metrics, "_carrier_parent"),
    );

    print_section("Carrier Component Split");
    println!(
        "  {:16} {:16} {:>12} {:>8} {:>8} {:>12} {:>20}",
        "family", "component", "constraints", "share", "buckets", "avg/bucket", "hottest_bucket"
    );
    for bucket in &metrics.family_component_rollup {
        if bucket.constraint_count == 0 {
            continue;
        }
        println!(
            "  {:16} {:16} {:>12} {:>7.2}% {:>8} {:>12.1} {:>20}",
            bucket.family,
            bucket.component,
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.bucket_count),
            per_unit(bucket.constraint_count, bucket.bucket_count),
            bucket.max_bucket_namespace,
        );
    }

    print_hotspot_table("Global Hotspots", &metrics, metrics.hotspots.iter().cloned());
    print_hotspot_table(
        "Shared Hotspots",
        &metrics,
        metrics
            .hotspots
            .iter()
            .filter(|bucket| hotspot_family(&bucket.namespace).is_none())
            .cloned(),
    );

    let mut printed_detail_namespaces = BTreeSet::new();
    for detail in metrics
        .hotspot_details
        .iter()
        .chain(metrics.representative_claim_details.iter())
    {
        if matches!(
            hotspot_family(&detail.parent_namespace),
            Some("carrier_output" | "carrier_child" | "carrier_parent")
        ) {
            continue;
        }
        if printed_detail_namespaces.insert(detail.parent_namespace.clone()) {
            print_hotspot_detail(&metrics, detail);
        }
    }

    print_section("Definitions");
    print_kv("constraints", "total R1CS rows (cs.enforce calls)");
    print_kv("linear_constraints", "rows where one multiplicative side is exactly 1");
    print_kv(
        "quadratic_constraints",
        "rows that keep a real multiplication on both sides",
    );
    print_kv("a/b/c_terms", "nonzero entries across A, B, C rows");
    print_kv(
        "replay_per_chunk",
        "chunk-scoped replay/verifier work above the explicit carrier layer",
    );
    print_kv(
        "carrier_*_per_claim",
        "carrier-layer total divided by output or child claim count",
    );
    print_kv(
        "replay_relation",
        "verifier-style chunk replay theorem proved inside Spartan",
    );
    print_kv(
        "carrier_relation",
        "explicit witness-opening layer tying carried claims to packed Z matrices",
    );

    assert!(metrics.constraint_count > 0);
}
