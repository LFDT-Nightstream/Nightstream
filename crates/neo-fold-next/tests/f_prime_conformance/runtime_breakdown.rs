use std::io::{self, Write};
use std::time::Instant;

use neo_ajtai::Commitment;
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions, build_rv64im_chunk_step_ivc_relations,
    build_rv64im_main_recursion_f_prime_advices_single_step,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages,
    debug_trace_rv64im_main_recursion_construction2_default_pair_for_full_width,
    debug_trace_rv64im_main_recursion_f_prime_advices_single_step_build,
    debug_trace_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_trace_rv64im_main_recursion_step_spartan_shape_synthesis, prove_rv64im_main_recursion_step_spartan,
    setup_rv64im_main_recursion_step_spartan_cached, setup_rv64im_main_recursion_step_spartan_shape_cached,
    verify_rv64im_main_recursion_step_spartan_and_extract_published_target, Rv64imMainRecursionFPrimeBackendRelation,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::main_recursion::{build_rv64im_main_recursion_verifier_key_fs, Rv64imMainRecursionPhiSide};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_main_recursion_construction2_canonical_full_width,
    prove_rv64im_accepted_proof_with_options_and_perf, Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::support::{fast_structural_backend_relations, fast_structural_spartan_shape, single_step_spartan_shape};

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

fn perturb_backend_relation_values(relation: &mut Rv64imMainRecursionFPrimeBackendRelation) {
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
#[ignore = "manual Goal 2 timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_runtime_breakdown_probe() {
    let source = build_mixed_opcode_perf_source_case(1);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };

    let started = Instant::now();
    let ((accepted, _audit), proof_perf) = prove_rv64im_accepted_proof_with_options_and_perf(&input, options)
        .expect("prove accepted artifact for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.accepted_artifact_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms("goal2_probe.accepted_artifact_perf_total", proof_perf.total_ms);
    print_stage_ms("goal2_probe.accepted_artifact_perf_main_lane", proof_perf.main_lane_ms);
    print_stage_ms(
        "goal2_probe.accepted_artifact_perf_public_export",
        proof_perf.public_export_ms,
    );

    let started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("build final statement for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.final_statement_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step relations for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.chunk_step_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_count("goal2_probe.relations_len", relations.len());
    if let Some(first_relation) = relations.first() {
        print_count(
            "goal2_probe.first_relation_public_steps_len",
            first_relation.witness.handoff.public_chunk.steps.len(),
        );
        print_count(
            "goal2_probe.first_relation_state_in_claims_len",
            first_relation.witness.state_in.carry.main.claims.len(),
        );
        print_count(
            "goal2_probe.first_relation_state_out_claims_len",
            first_relation.witness.state_out.carry.main.claims.len(),
        );
    }

    let started = Instant::now();
    let (advices, advice_perf) =
        debug_trace_rv64im_main_recursion_f_prime_advices_single_step_build(&relations, "goal2_probe.f_prime_advices")
            .expect("build native F' advices for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.f_prime_advices_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms("goal2_probe.f_prime_advices_perf_total", advice_perf.total_ms);
    print_stage_ms(
        "goal2_probe.f_prime_advices_perf_relation_validation",
        advice_perf.relation_validation_ms,
    );
    print_stage_ms(
        "goal2_probe.f_prime_advices_perf_verifier_key",
        advice_perf.verifier_key_ms,
    );
    print_stage_ms(
        "goal2_probe.f_prime_advices_perf_canonical_full_width",
        advice_perf.canonical_full_width_ms,
    );
    print_stage_ms(
        "goal2_probe.f_prime_advices_perf_canonical_u_perp",
        advice_perf.canonical_u_perp_ms,
    );
    print_count("goal2_probe.f_prime_advices_step_count", advice_perf.step_count);
    for (step_index, step_perf) in advice_perf.per_step.iter().enumerate() {
        print_stage_ms(
            &format!("goal2_probe.f_prime_advices_step_{step_index}_build_advice"),
            step_perf.build_advice_ms,
        );
        print_stage_ms(
            &format!("goal2_probe.f_prime_advices_step_{step_index}_evaluate_step"),
            step_perf.evaluate_step_ms,
        );
        print_stage_ms(
            &format!("goal2_probe.f_prime_advices_step_{step_index}_apply_step_image"),
            step_perf.apply_step_image_ms,
        );
    }
    if let Some(first_advice) = advices.first() {
        print_count(
            "goal2_probe.first_advice_state_in_claims_len",
            first_advice.running_state().carry.main.claims.len(),
        );
    }

    let started = Instant::now();
    let ((spartan_shape, backend_relations), backend_perf) =
        debug_trace_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
            &relations,
            &advices,
            "goal2_probe.backend_relations",
        )
        .expect("build recursive-step backend relations for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.backend_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_stage_ms("goal2_probe.backend_relations_perf_total", backend_perf.total_ms);
    print_stage_ms(
        "goal2_probe.backend_relations_perf_spartan_shape",
        backend_perf.spartan_shape_ms,
    );
    print_stage_ms("goal2_probe.backend_relations_perf_payloads", backend_perf.payloads_ms);
    print_stage_ms(
        "goal2_probe.backend_relations_perf_statement_build",
        backend_perf.statement_build_ms,
    );
    print_stage_ms(
        "goal2_probe.backend_relations_perf_semantics_check",
        backend_perf.semantics_check_ms,
    );
    print_count(
        "goal2_probe.backend_relations_relation_count",
        backend_perf.relation_count,
    );
    print_count("goal2_probe.backend_relations_len", backend_relations.len());

    let first = backend_relations
        .first()
        .expect("Goal 2 timing probe requires at least one recursive-step backend relation");
    print_count(
        "goal2_probe.first_backend_state_in_claims_len",
        first.payload.state_in_claims.len(),
    );
    print_count(
        "goal2_probe.first_backend_state_out_claims_len",
        first.payload.state_out_claims.len(),
    );
    print_count(
        "goal2_probe.first_backend_fresh_claims_len",
        first.payload.fresh_claims.len(),
    );
    print_count(
        "goal2_probe.first_backend_ccs_outputs_len",
        first.payload.pi_ccs.ccs_outputs.len(),
    );
    print_count(
        "goal2_probe.first_backend_dec_children_len",
        first.payload.pi_dec.children.len(),
    );

    let started = Instant::now();
    let keys = setup_rv64im_main_recursion_step_spartan_cached(&spartan_shape, first)
        .expect("setup recursive-step Spartan keys for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.recursive_step_setup_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let (pk, vk) = &*keys;

    let started = Instant::now();
    let proof = prove_rv64im_main_recursion_step_spartan(pk, &spartan_shape, first)
        .expect("prove recursive-step Spartan proof for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.recursive_step_prove_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    verify_rv64im_main_recursion_step_spartan_and_extract_published_target(vk, &proof)
        .expect("verify recursive-step Spartan proof for Goal 2 timing probe");
    print_stage_ms(
        "goal2_probe.recursive_step_verify_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "manual default-pair timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_default_pair_breakdown_probe() {
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs().expect("build canonical vk_fs for default-pair probe");
    let full_width =
        build_rv64im_main_recursion_construction2_canonical_full_width(&vk_fs, &Rv64imMainRecursionPhiSide::zero())
            .expect("derive canonical full width for default-pair probe");
    let started = Instant::now();
    let _ = debug_trace_rv64im_main_recursion_construction2_default_pair_for_full_width(
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
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };

    let started = Instant::now();
    let ((accepted, _), _) = prove_rv64im_accepted_proof_with_options_and_perf(&input, options)
        .expect("prove accepted artifact for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.accepted_artifact_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&accepted)
        .expect("build final statement for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.final_statement_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step relations for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.chunk_step_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&relations)
        .expect("build recursive-step advices for value-invariance probe");
    print_stage_ms(
        "goal2_probe.value_invariant.f_prime_advices_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
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
        debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis(&spartan_shape, baseline_relation)
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
    let baseline = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, baseline_relation)
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
    let perturbed = debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, &perturbed_relation)
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
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };

    let started = Instant::now();
    let ((accepted, _), _) = prove_rv64im_accepted_proof_with_options_and_perf(&input, options)
        .expect("prove accepted artifact for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.accepted_artifact_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("build final statement for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.final_statement_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step relations for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.chunk_step_relations_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let measured = audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions(&relations, &[0, 1])
        .expect("measure fixed shape across chunk positions for n-invariance probe");
    print_stage_ms(
        "goal2_probe.n_invariant.audit_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_count("goal2_probe.n_invariant.measurement_count", measured.len());
}

#[test]
#[ignore = "manual Goal 2 parity timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_parity_breakdown_probe() {
    let started = Instant::now();
    let backend_relations = fast_structural_backend_relations();
    let spartan_shape = fast_structural_spartan_shape();
    print_stage_ms(
        "goal2_probe.parity.fixture_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    print_count("goal2_probe.parity.backend_relations_len", backend_relations.len());

    let started = Instant::now();
    let keys = setup_rv64im_main_recursion_step_spartan_cached(spartan_shape, &backend_relations[0])
        .expect("setup recursive-step Spartan keys for parity probe");
    print_stage_ms(
        "goal2_probe.parity.setup_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    let (pk, vk) = &*keys;

    for (step, backend_relation) in backend_relations.iter().enumerate() {
        let started = Instant::now();
        let proof = prove_rv64im_main_recursion_step_spartan(pk, spartan_shape, backend_relation)
            .unwrap_or_else(|err| panic!("step {step}: prove parity probe: {err}"));
        print_stage_ms(
            &format!("goal2_probe.parity.step_{step}.prove_wall"),
            started.elapsed().as_secs_f64() * 1_000.0,
        );

        let started = Instant::now();
        let _ = verify_rv64im_main_recursion_step_spartan_and_extract_published_target(vk, &proof)
            .unwrap_or_else(|err| panic!("step {step}: verify parity probe: {err}"));
        print_stage_ms(
            &format!("goal2_probe.parity.step_{step}.verify_wall"),
            started.elapsed().as_secs_f64() * 1_000.0,
        );
    }
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
    let metrics = debug_trace_rv64im_main_recursion_step_spartan_shape_synthesis(
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
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages(first)
        .expect("profile recursive-step chunk replay stages");
    print_stage_ms(
        "goal2_probe.chunk_replay_profile.total_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "manual Goal 2 shape-only setup timing probe; run exact with --ignored --nocapture"]
fn goal2_manual_shape_only_setup_breakdown_probe() {
    let started = Instant::now();
    let spartan_shape = single_step_spartan_shape();
    print_stage_ms(
        "goal2_probe.shape_only_setup.fixture_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let _keys = setup_rv64im_main_recursion_step_spartan_shape_cached(spartan_shape)
        .expect("setup recursive-step Spartan shape-only keys");
    print_stage_ms(
        "goal2_probe.shape_only_setup.setup_wall",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}
