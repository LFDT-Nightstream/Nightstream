use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_fold_next::core::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::{
    audit_rv32im_main_recursion_step_spartan_fixed_shape_at_chunk_positions, build_rv32im_chunk_step_ivc_relations,
    build_rv32im_main_recursion_f_prime_advices_single_step,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_measure_rv32im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv32im_main_recursion_step_chunk_replay_stages,
    debug_trace_rv32im_main_recursion_construction2_default_pair_for_full_width,
    debug_trace_rv32im_main_recursion_step_spartan_shape_synthesis, Rv32imMainRecursionFPrimeBackendRelation,
};
use neo_fold_next::rv32im::f_prime::{build_rv32im_main_recursion_verifier_key_fs, Rv32imMainRecursionPhiSide};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_main_recursion_construction2_canonical_full_width,
    prove_rv32im_accepted_proof_with_options_and_perf, Rv32imProofInput, Rv32imPublicProofOptions,
};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::support::{fast_structural_backend_relations, fast_structural_spartan_shape};

fn print_stage_ms(label: &str, elapsed_ms: f64) {
    eprintln!("{label}={elapsed_ms:.2}ms");
    let _ = io::stderr().flush();
}

fn print_count(label: &str, value: usize) {
    eprintln!("{label}={value}");
    let _ = io::stderr().flush();
}

fn perturb_ce_claim_values(claim: &mut neo_ccs::CeClaim<Commitment, F, K>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if claim.X.rows() > 0 && claim.X.cols() > 0 {
        claim.X[(0, 0)] += F::ONE;
    }
    if let Some(first) = claim.r.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.s_col.first_mut() {
        *first += K::ONE;
    }
    if let Some(row) = claim.y_ring.first_mut() {
        if let Some(first) = row.first_mut() {
            *first += K::ONE;
        }
    }
    if let Some(first) = claim.ct.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.aux_openings.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.y_zcol.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.c_step_coords.first_mut() {
        *first += F::ONE;
    }
    claim.fold_digest[0] ^= 1;
}

fn perturb_ccs_claim_values(claim: &mut neo_ccs::CcsClaim<Commitment, F>) {
    if let Some(first) = claim.c.data.first_mut() {
        *first += F::ONE;
    }
    if let Some(first) = claim.x.first_mut() {
        *first += F::ONE;
    }
}

fn perturb_ccs_witness_values(witness: &mut neo_ccs::CcsWitness<F>) {
    if let Some(first) = witness.w.first_mut() {
        *first += F::ONE;
    }
    if witness.Z.rows() > 0 && witness.Z.cols() > 0 {
        witness.Z[(0, 0)] += F::ONE;
    }
}

fn perturb_backend_relation_values(relation: &mut Rv32imMainRecursionFPrimeBackendRelation) {
    for claim in &mut relation.payload.state_in_claims {
        perturb_ce_claim_values(claim);
    }
    for claim in &mut relation.payload.state_out_claims {
        perturb_ce_claim_values(claim);
    }
    for claim in &mut relation.payload.pi_ccs.ccs_outputs {
        perturb_ce_claim_values(claim);
    }
    perturb_ce_claim_values(&mut relation.payload.pi_rlc.parent);
    for child in &mut relation.payload.pi_dec.children {
        perturb_ce_claim_values(child);
    }
    for claim in &mut relation.payload.fresh_claims {
        perturb_ccs_claim_values(claim);
    }
    for witness in &mut relation.payload.fresh_witnesses {
        perturb_ccs_witness_values(witness);
    }
}

#[test]
#[ignore = "manual default-pair timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_default_pair_breakdown_probe() {
    let vk_fs = build_rv32im_main_recursion_verifier_key_fs().expect("build canonical vk_fs for default-pair probe");
    let full_width =
        build_rv32im_main_recursion_construction2_canonical_full_width(&vk_fs, &Rv32imMainRecursionPhiSide::zero())
            .expect("derive canonical full width for default-pair probe");
    let started = Instant::now();
    let _ = debug_trace_rv32im_main_recursion_construction2_default_pair_for_full_width(
        &vk_fs,
        full_width,
        "goal2_probe.default_pair",
    )
    .expect("build traced default pair for default-pair probe");
    print_stage_ms(
        "goal2_probe.default_pair_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "manual Goal 2 value-invariance timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_value_invariant_breakdown_probe() {
    let source = build_mixed_opcode_perf_source_case(0);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };

    let started = Instant::now();
    let ((accepted, _), _) = prove_rv32im_accepted_proof_with_options_and_perf(&input, options)
        .expect("prove accepted artifact for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.accepted_artifact_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let (final_statement, final_proof) = prove_rv32im_final_statement_from_accepted(&accepted)
        .expect("build final statement for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.final_statement_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let relations = build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step relations for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.chunk_step_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let advices = build_rv32im_main_recursion_f_prime_advices_single_step(&relations)
        .expect("build recursive-step advices for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.f_prime_advices_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let (spartan_shape, backend_relations) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build backend relations for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.backend_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let baseline_relation = backend_relations
        .first()
        .expect("value-invariance probe requires one backend relation");
    let started = Instant::now();
    let baseline_synthesis =
        debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis(&spartan_shape, baseline_relation)
            .expect("measure baseline shape synthesis for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.baseline_shape_synthesis_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms(
        "goal2_probe.value_invariant.baseline_shape_synthesis_shared",
        baseline_synthesis.shared_ms,
    );
    print_stage_ms(
        "goal2_probe.value_invariant.baseline_shape_synthesis_precommitted",
        baseline_synthesis.precommitted_ms,
    );
    print_stage_ms(
        "goal2_probe.value_invariant.baseline_shape_synthesis_synthesize",
        baseline_synthesis.synthesize_ms,
    );

    let started = Instant::now();
    let baseline = debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, baseline_relation)
        .expect("measure baseline shape for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.baseline_shape_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms(
        "goal2_probe.value_invariant.baseline_num_constraints",
        baseline.num_constraints as f64,
    );

    let mut perturbed_relation = baseline_relation.clone();
    perturb_backend_relation_values(&mut perturbed_relation);
    let started = Instant::now();
    let perturbed = debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, &perturbed_relation)
        .expect("measure perturbed shape for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.perturbed_shape_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms(
        "goal2_probe.value_invariant.perturbed_num_constraints",
        perturbed.num_constraints as f64,
    );
}

#[test]
#[ignore = "manual Goal 2 n-invariance timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_n_invariant_breakdown_probe() {
    let source = build_mixed_opcode_perf_source_case(0);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };

    let started = Instant::now();
    let ((accepted, _), _) = prove_rv32im_accepted_proof_with_options_and_perf(&input, options)
        .expect("prove accepted artifact for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.accepted_artifact_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("build final statement for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.final_statement_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let relations = build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step relations for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.chunk_step_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let measured = audit_rv32im_main_recursion_step_spartan_fixed_shape_at_chunk_positions(&relations, &[0, 1])
        .expect("measure fixed shape across chunk positions for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.audit_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_count("goal2_probe.n_invariant.measurement_count", measured.len());
}

#[test]
#[ignore = "manual Goal 2 shape-synthesis timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_shape_synthesis_breakdown_probe() {
    let started = Instant::now();
    let backend_relations = fast_structural_backend_relations();
    let spartan_shape = fast_structural_spartan_shape();
    print_stage_ms(
        "goal2_probe.shape_synthesis.fixture_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let first = backend_relations
        .first()
        .expect("shape-synthesis probe requires one backend relation");
    let started = Instant::now();
    let metrics = debug_trace_rv32im_main_recursion_step_spartan_shape_synthesis(
        spartan_shape,
        first,
        "goal2_probe.shape_synthesis",
    )
    .expect("measure recursive-step shape synthesis");
    print_stage_ms(
        "goal2_probe.shape_synthesis.total_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms("goal2_probe.shape_synthesis.shared", metrics.shared_ms);
    print_stage_ms("goal2_probe.shape_synthesis.precommitted", metrics.precommitted_ms);
    print_stage_ms("goal2_probe.shape_synthesis.synthesize", metrics.synthesize_ms);
    print_count("goal2_probe.shape_synthesis.num_inputs", metrics.num_inputs);
    print_count("goal2_probe.shape_synthesis.num_aux", metrics.num_aux);
    print_count("goal2_probe.shape_synthesis.num_constraints", metrics.num_constraints);
}

#[test]
#[ignore = "manual Goal 2 chunk-replay stage profile probe; run exact with --ignored --nocapture"]
fn goal2_manual_chunk_replay_stage_profile_probe() {
    let started = Instant::now();
    let backend_relations = fast_structural_backend_relations();
    print_stage_ms(
        "goal2_probe.chunk_replay_profile.fixture_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let first = backend_relations
        .first()
        .expect("chunk-replay profile probe requires one backend relation");
    let started = Instant::now();
    debug_profile_rv32im_main_recursion_step_chunk_replay_stages(first)
        .expect("profile recursive-step chunk replay stages");
    print_stage_ms(
        "goal2_probe.chunk_replay_profile.total_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}
