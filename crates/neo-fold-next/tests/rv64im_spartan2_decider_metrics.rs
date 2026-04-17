//! Bounded diagnostic tests for the RV64IM main-relation Spartan circuit shape.

use std::collections::BTreeMap;

use neo_fold_next::rv64im::audit::{
    inspect_rv64im_spartan2_decider_trace, measure_rv64im_spartan2_decider_circuit, Rv64imMainRelationCircuitMetrics,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
};

const RV64IM_MAIN_RELATION_CONSTRAINT_FLOOR: usize = 50_000;
const RV64IM_MAIN_RELATION_CONSTRAINT_BUDGET: usize = 400_000;
const RV64IM_MAIN_RELATION_PER_CLAIM_DIGEST_CONSTRAINT_BUDGET: usize = 0;

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn final_fixture(
    name: &str,
) -> (
    neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
) {
    let input = proof_input(name);
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement")
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

fn print_section(title: &str) {
    eprintln!();
    eprintln!("{title}");
    eprintln!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    eprintln!("  {:28} {}", label, value);
}

fn print_count_buckets(metrics: &Rv64imMainRelationCircuitMetrics, title: &str, max_rows: usize) {
    print_section(title);
    eprintln!(
        "  {:42} {:>12} {:>8} {:>12} {:>12}",
        "namespace", "constraints", "share", "aux", "terms"
    );
    for bucket in metrics.hotspots.iter().take(max_rows) {
        if bucket.constraint_count == 0 {
            continue;
        }
        eprintln!(
            "  {:42} {:>12} {:>7.2}% {:>12} {:>12}",
            bucket.namespace,
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.aux_count),
            format_count(bucket.total_term_count),
        );
    }
}

fn print_phase_rollup(metrics: &Rv64imMainRelationCircuitMetrics, max_rows: usize) {
    print_section("Phase Rollup");
    eprintln!(
        "  {:22} {:>8} {:>12} {:>8} {:>12} {:>12}",
        "phase", "buckets", "constraints", "share", "aux", "max_bucket"
    );
    for bucket in metrics.phase_rollup.iter().take(max_rows) {
        eprintln!(
            "  {:22} {:>8} {:>12} {:>7.2}% {:>12} {:>12}",
            bucket.phase,
            format_count(bucket.bucket_count),
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.aux_count),
            format_count(bucket.max_bucket_constraint_count),
        );
    }
}

fn print_component_rollup(metrics: &Rv64imMainRelationCircuitMetrics, max_rows: usize) {
    print_section("Component Rollup");
    eprintln!(
        "  {:22} {:>8} {:>12} {:>8} {:>12} {:>12}",
        "component", "buckets", "constraints", "share", "aux", "max_bucket"
    );
    for bucket in metrics.component_rollup.iter().take(max_rows) {
        eprintln!(
            "  {:22} {:>8} {:>12} {:>7.2}% {:>12} {:>12}",
            bucket.component,
            format_count(bucket.bucket_count),
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.aux_count),
            format_count(bucket.max_bucket_constraint_count),
        );
    }
}

fn print_sumcheck_rollup(metrics: &Rv64imMainRelationCircuitMetrics) {
    print_section("Sumcheck Breakdown");
    eprintln!(
        "  {:18} {:>8} {:>12} {:>8} {:>12} {:>12}",
        "bucket", "buckets", "constraints", "share", "aux", "max_bucket"
    );
    for bucket in &metrics.sumcheck_rollup {
        eprintln!(
            "  {:18} {:>8} {:>12} {:>7.2}% {:>12} {:>12}",
            bucket.bucket,
            format_count(bucket.bucket_count),
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.aux_count),
            format_count(bucket.max_bucket_constraint_count),
        );
    }
}

fn print_rho_rollup(metrics: &Rv64imMainRelationCircuitMetrics) {
    print_section("Rho Breakdown");
    eprintln!(
        "  {:18} {:>8} {:>12} {:>8} {:>12} {:>12}",
        "bucket", "buckets", "constraints", "share", "aux", "max_bucket"
    );
    for bucket in &metrics.rho_rollup {
        eprintln!(
            "  {:18} {:>8} {:>12} {:>7.2}% {:>12} {:>12}",
            bucket.bucket,
            format_count(bucket.bucket_count),
            format_count(bucket.constraint_count),
            bucket.constraint_count as f64 * 100.0 / metrics.constraint_count.max(1) as f64,
            format_count(bucket.aux_count),
            format_count(bucket.max_bucket_constraint_count),
        );
    }
}

fn print_claim_family_rollup(metrics: &Rv64imMainRelationCircuitMetrics) {
    let surface_by_family = metrics
        .surface
        .families
        .iter()
        .map(|family| (family.family.as_str(), family))
        .collect::<BTreeMap<_, _>>();
    print_section("Claim Family Rollup");
    eprintln!(
        "  {:18} {:>12} {:>8} {:>8} {:>12} {:>12} {:>8} {:>8} {:>12}",
        "family", "rlc_constraints", "rlc_F", "rlc_K", "final_constraints", "delta", "final_F", "final_K", "combined"
    );
    for bucket in &metrics.claim_family_rollup {
        let surface = surface_by_family.get(bucket.family.as_str());
        let rlc_f = surface.map_or(0, |surface| surface.rlc_public_field_coords_total);
        let rlc_k = surface.map_or(0, |surface| surface.rlc_public_k_coords_total);
        let final_f = surface.map_or(0, |surface| surface.final_claim_field_coords_total);
        let final_k = surface.map_or(0, |surface| surface.final_claim_k_coords_total);
        eprintln!(
            "  {:18} {:>12} {:>8} {:>8} {:>12} {:>12} {:>8} {:>8} {:>12}",
            bucket.family,
            format_count(bucket.rlc_constraint_count),
            format_count(rlc_f),
            format_count(rlc_k),
            format_count(bucket.final_constraint_count),
            format_count(
                bucket
                    .final_constraint_count
                    .saturating_sub(bucket.rlc_constraint_count)
            ),
            format_count(final_f),
            format_count(final_k),
            format_count(bucket.rlc_constraint_count + bucket.final_constraint_count),
        );
    }
}

fn print_metrics_report(metrics: &Rv64imMainRelationCircuitMetrics) {
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
    print_kv("total_terms", format_count(metrics.total_term_count));
    print_kv("max_row_terms", format_count(metrics.max_constraint_term_count));

    print_section("Trace Shape");
    print_kv("chunks", metrics.trace.chunk_count);
    print_kv("final_claims", metrics.trace.final_claim_count);
    print_kv("fresh_claims", metrics.trace.fresh_claim_count);
    print_kv("ccs_outputs", metrics.trace.ccs_output_count);
    print_kv("child_claims", metrics.trace.child_claim_count);
    print_kv("fresh_witness_cells", format_count(metrics.trace.fresh_witness_cells));
    print_kv("child_witness_cells", format_count(metrics.trace.child_witness_cells));
    print_kv("fe_rounds", metrics.trace.fe_round_count);
    print_kv("nc_rounds", metrics.trace.nc_round_count);

    print_section("CE Surface Width");
    print_kv("rlc_parent_claims", metrics.surface.rlc_parent_claim_count);
    print_kv(
        "rlc_field_coords_total",
        format_count(metrics.surface.rlc_public_field_coords_total),
    );
    print_kv(
        "rlc_k_coords_total",
        format_count(metrics.surface.rlc_public_k_coords_total),
    );
    print_kv("final_claims", metrics.surface.final_claim_count);
    print_kv(
        "final_field_coords_total",
        format_count(metrics.surface.final_claim_field_coords_total),
    );
    print_kv(
        "final_k_coords_total",
        format_count(metrics.surface.final_claim_k_coords_total),
    );
    print_kv(
        "final_field_coords_per_claim",
        format!(
            "{:.1}",
            if metrics.surface.final_claim_count == 0 {
                0.0
            } else {
                metrics.surface.final_claim_field_coords_total as f64 / metrics.surface.final_claim_count as f64
            }
        ),
    );
    print_kv(
        "final_k_coords_per_claim",
        format!(
            "{:.1}",
            if metrics.surface.final_claim_count == 0 {
                0.0
            } else {
                metrics.surface.final_claim_k_coords_total as f64 / metrics.surface.final_claim_count as f64
            }
        ),
    );
    eprintln!(
        "  {:22} {:>12} {:>12} {:>12} {:>12}",
        "family", "rlc_F", "rlc_K", "final_F", "final_K"
    );
    for family in &metrics.surface.families {
        eprintln!(
            "  {:22} {:>12} {:>12} {:>12} {:>12}",
            family.family,
            format_count(family.rlc_public_field_coords_total),
            format_count(family.rlc_public_k_coords_total),
            format_count(family.final_claim_field_coords_total),
            format_count(family.final_claim_k_coords_total),
        );
    }

    print_phase_rollup(metrics, 16);
    print_component_rollup(metrics, 16);
    print_sumcheck_rollup(metrics);
    print_rho_rollup(metrics);
    print_claim_family_rollup(metrics);
    print_count_buckets(metrics, "Top Hotspots", 24);
}

#[test]
#[ignore = "diagnostic: bounded trace-shape probe for the main-relation Spartan circuit"]
fn rv64im_spartan2_decider_trace_stats_only() {
    let (statement, final_proof) = final_fixture("control_flow_jal_skip_ecall");
    let stats =
        inspect_rv64im_spartan2_decider_trace(&statement, &final_proof).expect("inspect rv64im spartan2 decider trace");
    print_section("Trace Shape");
    print_kv("chunks", stats.chunk_count);
    print_kv("final_claims", stats.final_claim_count);
    print_kv("fresh_claims", stats.fresh_claim_count);
    print_kv("ccs_outputs", stats.ccs_output_count);
    print_kv("child_claims", stats.child_claim_count);
    print_kv("fresh_witness_cells", format_count(stats.fresh_witness_cells));
    print_kv("child_witness_cells", format_count(stats.child_witness_cells));
    print_kv("max_chunk_witness_cells", format_count(stats.max_chunk_witness_cells));
    print_kv("fe_rounds", stats.fe_round_count);
    print_kv("nc_rounds", stats.nc_round_count);
    assert!(stats.chunk_count > 0);
    assert!(stats.max_chunk_witness_cells > 0);
}

#[test]
#[ignore = "diagnostic: bounded no-storage synthesis count for the main-relation Spartan circuit"]
fn rv64im_spartan2_decider_counting_cs_metrics() {
    let (statement, final_proof) = final_fixture("control_flow_jal_skip_ecall");
    let metrics = measure_rv64im_spartan2_decider_circuit(&statement, &final_proof)
        .expect("measure rv64im spartan2 decider circuit");
    print_metrics_report(&metrics);
    assert!(metrics.constraint_count > 0);
    assert!(metrics.trace.chunk_count > 0);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_spartan2_decider_stays_theorem_bearing_within_current_budget() {
    let (statement, final_proof) = final_fixture("control_flow_jal_skip_ecall");
    let metrics = measure_rv64im_spartan2_decider_circuit(&statement, &final_proof)
        .expect("measure rv64im spartan2 decider circuit");
    assert!(
        metrics.constraint_count >= RV64IM_MAIN_RELATION_CONSTRAINT_FLOOR,
        "main-relation circuit collapsed back toward an empty wrapper: got {} constraints, expected at least {}, hotspots={:#?}",
        metrics.constraint_count,
        RV64IM_MAIN_RELATION_CONSTRAINT_FLOOR,
        metrics.hotspots,
    );
    assert!(
        metrics.constraint_count <= RV64IM_MAIN_RELATION_CONSTRAINT_BUDGET,
        "main-relation circuit current budget exceeded: got {} constraints, budget {}, hotspots={:#?}",
        metrics.constraint_count,
        RV64IM_MAIN_RELATION_CONSTRAINT_BUDGET,
        metrics.hotspots,
    );
    assert!(
        metrics.max_claim_digest_constraint_count <= RV64IM_MAIN_RELATION_PER_CLAIM_DIGEST_CONSTRAINT_BUDGET,
        "main-relation claim-digest constraints reappeared: namespace {}, got {} constraints, budget {}",
        metrics.max_claim_digest_namespace,
        metrics.max_claim_digest_constraint_count,
        RV64IM_MAIN_RELATION_PER_CLAIM_DIGEST_CONSTRAINT_BUDGET,
    );
}
