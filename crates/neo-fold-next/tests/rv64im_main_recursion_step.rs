#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    audit_build_rv64im_main_circuit_chunk_trace_authoritative_summary,
    audit_build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs,
    audit_rv64im_main_recursion_backend_statement_matches_native_f_prime,
    audit_rv64im_main_recursion_construction2_bridge_next_running,
    audit_rv64im_main_recursion_construction2_verified_step_statement_digest,
    audit_rv64im_main_recursion_step_spartan_fixed_shape_across_chain,
    audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions,
    audit_rv64im_nifs_round_trip_from_chunk_step_relation, build_rv64im_chunk_step_ivc_relations,
    build_rv64im_main_recursion_f_prime_advices, build_rv64im_main_recursion_f_prime_advices_single_step,
    build_rv64im_main_recursion_f_prime_claim_cover, build_rv64im_main_recursion_f_prime_public_output,
    evaluate_rv64im_main_recursion_f_prime_advice, rv64im_main_recursion_advice_retarget_x_hash_to_current_accumulator,
    rv64im_main_recursion_advice_tamper_bridge_handoff_chain_digest_first_byte,
    rv64im_main_recursion_advice_tamper_ccs_replay_first_round_coeff, rv64im_main_recursion_advice_tamper_chunk_index,
    rv64im_main_recursion_advice_tamper_dec_child_commitment_first_word,
    rv64im_main_recursion_advice_tamper_fresh_state_out_terminal_handle_first_byte,
    rv64im_main_recursion_advice_tamper_fresh_state_out_transcript_absorbed,
    rv64im_main_recursion_advice_tamper_legacy_bridge_binding_digest_first_byte,
    rv64im_main_recursion_advice_tamper_legacy_bridge_handoff_digest_first_byte,
    rv64im_main_recursion_advice_tamper_legacy_prepared_step_digest_first_byte,
    rv64im_main_recursion_advice_tamper_pc_i,
    rv64im_main_recursion_advice_tamper_running_state_terminal_handle_first_byte,
    rv64im_main_recursion_advice_tamper_running_state_terminal_handle_only_first_byte,
    rv64im_main_recursion_advice_tamper_side_witness_nonzero,
    rv64im_main_recursion_advice_tamper_step_statement_chain_digest_first_byte,
    rv64im_main_recursion_advice_tamper_terminal_step,
    rv64im_main_recursion_advice_tamper_vk_fs_main_lane_shape_digest_first_byte,
    rv64im_main_recursion_advice_tamper_x_hash_first_byte, rv64im_main_recursion_advice_tamper_z_i_first_byte,
    rv64im_recursion_step_statement_chain_digest, verify_rv64im_main_recursion_f_prime_public_output,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::Rv64imEncodedPublicInput;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact,
    build_rv64im_main_recursion_construction2_default_fresh_instance,
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape,
    build_rv64im_main_recursion_construction2_f_prime_ccs_shape,
    build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image,
    build_rv64im_main_recursion_construction2_f_prime_witness_image,
    build_rv64im_main_recursion_construction2_fresh_instance,
    build_rv64im_main_recursion_construction2_fresh_instance_with_input,
    build_rv64im_main_recursion_construction2_input_state_image,
    build_rv64im_main_recursion_construction2_output_state_image, build_rv64im_main_recursion_construction2_x_i,
    build_rv64im_main_recursion_f_prime_advices_with_side_opening_public, prove_rv64im_public_proof_with_options,
    Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn set_digest_bit_le(digest: &mut [u8; 32], bit_index: usize) {
    digest[bit_index / 8] |= 1 << (bit_index % 8);
}

fn default_full_width_from_advice(advice: &neo_fold_next::rv64im::Rv64imMainRecursionFPrimeAdvice) -> usize {
    let shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(core::slice::from_ref(advice))
        .expect("derive explicit native F' shape");
    build_rv64im_main_recursion_construction2_default_full_width_from_ccs_shape(&shape)
        .expect("derive explicit default width from native shape")
}

#[test]
fn rv64im_main_recursion_step_chain_round_trip() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");

    let mut last_output = None;
    for advice in &f_prime_advices {
        let public_output =
            build_rv64im_main_recursion_f_prime_public_output(advice).expect("build native F' public output");
        let output = verify_rv64im_main_recursion_f_prime_public_output(&public_output, advice)
            .expect("verify F' public output");
        assert_eq!(public_output.x_out(), output.x_out());
        assert_eq!(output.chunk_count(), advice.chunk_count_in() + 1);
        last_output = Some(output);
    }

    let final_output = last_output.expect("non-empty step relation chain");
    assert_eq!(final_output.chunk_count() as usize, f_prime_advices.len());
}

#[test]
fn rv64im_main_recursion_first_advice_exposes_explicit_base_case_state() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let first = &f_prime_advices[0];

    assert_eq!(first.chunk_count_in(), 0);
    assert_eq!(first.z_0(), first.z_i(), "base-case advice must expose z_0 == z_i");
    assert_eq!(
        first.pc_i(),
        1,
        "RV64IM recursion currently specializes to trivial pc = 1"
    );
    assert!(
        first.side_witness().is_zero(),
        "current unwired RV64IM recursion path must expose a zero side lane"
    );
}

#[test]
fn rv64im_main_recursion_single_step_builder_rejects_multi_step_chunk_relations() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");

    let err = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect_err("whole-trace chunk relations must not pass the single-step native F' builder");
    assert!(err.to_string().contains("single-step"));
}

#[test]
fn rv64im_main_recursion_single_step_builder_accepts_rows_per_chunk_one() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");

    assert_eq!(f_prime_advices.len(), input.max_steps);

    for advice in &f_prime_advices {
        let public_output =
            build_rv64im_main_recursion_f_prime_public_output(advice).expect("build native F' public output");
        verify_rv64im_main_recursion_f_prime_public_output(&public_output, advice)
            .expect("verify single-step native F' public output");
    }
}

#[test]
fn rv64im_main_recursion_construction2_x_i_matches_native_public_output() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let advice = &f_prime_advices[0];
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(advice).expect("build native F' public output");

    assert_eq!(
        build_rv64im_main_recursion_construction2_x_i(advice).expect("build construction2 x_i"),
        *public_output.x_out(),
        "Construction-2 x_i must reuse the native F' public output image"
    );
    assert_eq!(
        build_rv64im_main_recursion_construction2_output_state_image(advice)
            .expect("build construction2 output state image")
            .encoded_public_input(),
        *public_output.x_out(),
        "Construction-2 output state image must be the canonical owner of x_i"
    );
}

#[test]
fn rv64im_main_recursion_construction2_input_state_image_matches_carried_x_i() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let advice = &f_prime_advices[0];

    assert_eq!(
        build_rv64im_main_recursion_construction2_input_state_image(advice).encoded_public_input(),
        *advice.x_i(),
        "Construction-2 input state image must own the carried x_i semantics"
    );
}

#[test]
fn rv64im_main_recursion_construction2_shape_builder_rejects_multi_step_advices() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");

    let err = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&advices)
        .expect_err("Construction-2 shape builder must reject multi-step native F' advice chains");
    assert!(err.to_string().contains("one public step"));
}

#[test]
fn rv64im_main_recursion_construction2_shape_builder_accepts_single_step_advices() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");

    let shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&advices)
        .expect("build Construction-2 native F' CCS shape");

    assert_eq!(shape.x_i_bit_len, 256);
    assert_eq!(shape.claim_cover.fresh_claim_shapes.len(), 1);
    assert_eq!(shape.claim_cover.fresh_witness_shapes.len(), 1);
    assert_eq!(shape.step_cover_shape.fresh_claim_count, 1);
    assert_eq!(shape.step_cover_shape.fresh_witness_count, 1);
}

#[test]
fn rv64im_main_recursion_construction2_witness_image_builds_for_base_case_single_step_advice() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");
    let _shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&f_prime_advices)
        .expect("build Construction-2 native F' CCS shape");
    let u_perp = build_rv64im_main_recursion_construction2_default_fresh_instance(
        f_prime_advices[0].verifier_key_fs(),
        default_full_width_from_advice(&f_prime_advices[0]),
    )
    .expect("build canonical u_perp");

    let witness_image = build_rv64im_main_recursion_construction2_f_prime_witness_image(&f_prime_advices[0], &u_perp)
        .expect("build base-case Construction-2 native F' witness image");

    assert!(witness_image.logical_field_count() > 256);
    assert_ne!(witness_image.expected_digest(), [0; 32]);
}

#[test]
fn rv64im_main_recursion_construction2_witness_image_binds_pi_dec_children() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");
    let u_perp = build_rv64im_main_recursion_construction2_default_fresh_instance(
        f_prime_advices[0].verifier_key_fs(),
        default_full_width_from_advice(&f_prime_advices[0]),
    )
    .expect("build canonical u_perp");
    let baseline = build_rv64im_main_recursion_construction2_f_prime_witness_image(&f_prime_advices[0], &u_perp)
        .expect("build baseline Construction-2 native F' witness image");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_dec_child_commitment_first_word(&mut tampered_advice, 0);
    let tampered = build_rv64im_main_recursion_construction2_f_prime_witness_image(&tampered_advice, &u_perp)
        .expect("build tampered Construction-2 native F' witness image");

    assert_ne!(
        baseline.expected_digest(),
        tampered.expected_digest(),
        "Construction-2 native F' witness image must bind the paper-owned Pi_DEC child commitment cargo carried in pi_fold"
    );
}

#[test]
fn rv64im_main_recursion_construction2_low_norm_witness_image_is_binary_for_base_case() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");
    let _shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&f_prime_advices)
        .expect("build Construction-2 native F' CCS shape");
    let u_perp = build_rv64im_main_recursion_construction2_default_fresh_instance(
        f_prime_advices[0].verifier_key_fs(),
        default_full_width_from_advice(&f_prime_advices[0]),
    )
    .expect("build canonical u_perp");

    let low_norm_witness =
        build_rv64im_main_recursion_construction2_f_prime_low_norm_witness_image(&f_prime_advices[0], &u_perp)
            .expect("build base-case Construction-2 low-norm witness image");

    assert!(low_norm_witness.low_norm_field_count() > 256);
    assert_ne!(low_norm_witness.expected_digest(), [0; 32]);
    assert!(
        low_norm_witness
            .binary_values()
            .iter()
            .all(|&value| value == F::ZERO || value == F::ONE),
        "Construction-2 low-norm witness image must be binary under b=2"
    );
}

#[test]
fn rv64im_main_recursion_construction2_shape_builder_ignores_tampered_native_terminal_step() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");
    let baseline = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&f_prime_advices)
        .expect("build baseline Construction-2 native F' shape");
    let mut tampered_advices = f_prime_advices.clone();
    rv64im_main_recursion_advice_tamper_terminal_step(&mut tampered_advices[0]);

    let rebuilt = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&tampered_advices)
        .expect("Construction-2 native F' shape builder must derive terminality-independent cover shape");

    assert_eq!(
        rebuilt.expected_digest(),
        baseline.expected_digest(),
        "Construction-2 native F' shape digest must be invariant under tampered native terminal-step selector"
    );
}

#[test]
fn rv64im_main_recursion_payload_builder_ignores_tampered_legacy_relation_halted_out() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let baseline_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build baseline native F' advices");
    let spartan_shape =
        neo_fold_next::rv64im::audit::build_rv64im_main_recursion_step_spartan_shape(&chunk_step_relations)
            .expect("build recursive-step spartan shape");
    let baseline_payloads =
        neo_fold_next::rv64im::audit::build_rv64im_main_recursion_f_prime_payloads(&baseline_advices, &spartan_shape)
            .expect("build baseline recursive-step payloads");
    let mut tampered_relations = chunk_step_relations.clone();
    tampered_relations[0].statement.step_public.halted_out = !tampered_relations[0].statement.step_public.halted_out;
    let rebuilt_advices = build_rv64im_main_recursion_f_prime_advices(&tampered_relations)
        .expect("rebuild native F' advices from tampered legacy halted_out relation");
    let rebuilt_payloads =
        neo_fold_next::rv64im::audit::build_rv64im_main_recursion_f_prime_payloads(&rebuilt_advices, &spartan_shape)
            .expect("build recursive-step payloads from rebuilt native advices");

    assert_eq!(
        rebuilt_payloads[0].step_shape.expected_digest(),
        baseline_payloads[0].step_shape.expected_digest(),
        "native recursive-step step-shape payload must be invariant under tampered legacy relation step_public.halted_out"
    );
}

#[test]
fn rv64im_main_circuit_chunk_trace_builder_ignores_legacy_relation_step_public_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain")
        .into_iter()
        .next()
        .expect("first relation");
    let baseline = audit_build_rv64im_main_circuit_chunk_trace_authoritative_summary(&relation)
        .expect("build authoritative chunk trace summary");
    let mut tampered_relation = relation.clone();
    tampered_relation.statement.step_public.program_digest[0] ^= 1;
    tampered_relation.statement.step_public.chunk_index ^= 1;
    tampered_relation.statement.step_public.step_lo ^= 1;
    tampered_relation.statement.step_public.step_hi ^= 1;
    tampered_relation.statement.step_public.halted_out = !tampered_relation.statement.step_public.halted_out;
    let rebuilt = audit_build_rv64im_main_circuit_chunk_trace_authoritative_summary(&tampered_relation)
        .expect("rebuild authoritative chunk trace summary");

    assert_eq!(
        rebuilt, baseline,
        "authoritative main-circuit chunk trace builder must ignore the legacy relation.statement.step_public shell"
    );
}

#[test]
fn rv64im_main_recursion_advice_builder_ignores_legacy_relation_chunk_summary_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let baseline_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build baseline native F' advices");
    let baseline_step = evaluate_rv64im_main_recursion_f_prime_advice(&baseline_advices[0])
        .expect("evaluate baseline native F' advice");
    let mut tampered_relations = chunk_step_relations.clone();
    tampered_relations[0].statement.chunk_summary.start_index ^= 1;
    tampered_relations[0]
        .statement
        .chunk_summary
        .public_step_count ^= 1;
    tampered_relations[0]
        .statement
        .chunk_summary
        .public_chunk_digest[0] ^= 1;
    tampered_relations[0]
        .statement
        .chunk_summary
        .chunk_relation_digest[0] ^= 1;
    let rebuilt_advices = build_rv64im_main_recursion_f_prime_advices(&tampered_relations)
        .expect("rebuild native F' advices from tampered legacy chunk_summary relation");
    let rebuilt_step =
        evaluate_rv64im_main_recursion_f_prime_advice(&rebuilt_advices[0]).expect("evaluate rebuilt native F' advice");

    assert_eq!(
        rebuilt_step.x_out(),
        baseline_step.x_out(),
        "native F' advice builder must ignore the legacy relation.statement.chunk_summary shell"
    );
    assert_eq!(
        rebuilt_step.step_statement_chain_digest(),
        baseline_step.step_statement_chain_digest(),
        "native F' step chain must ignore the legacy relation.statement.chunk_summary shell"
    );
}

#[test]
fn rv64im_main_recursion_construction2_verified_step_statement_ignores_legacy_relation_chunk_index() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain")
        .into_iter()
        .next()
        .expect("first relation");
    let baseline = audit_rv64im_main_recursion_construction2_verified_step_statement_digest(&relation)
        .expect("build baseline Construction-2 verified step statement");
    let mut tampered_relation = relation.clone();
    tampered_relation.statement.step_public.chunk_index ^= 1;
    let rebuilt = audit_rv64im_main_recursion_construction2_verified_step_statement_digest(&tampered_relation)
        .expect("rebuild Construction-2 verified step statement from tampered legacy chunk_index relation");

    assert_eq!(
        rebuilt, baseline,
        "Construction-2 verified step statement must ignore legacy relation.statement.step_public.chunk_index"
    );
}

#[test]
fn rv64im_main_recursion_construction2_verified_step_statement_ignores_legacy_relation_statement_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain")
        .into_iter()
        .next()
        .expect("first relation");
    let baseline = audit_rv64im_main_recursion_construction2_verified_step_statement_digest(&relation)
        .expect("build baseline Construction-2 verified step statement");
    let mut tampered_relation = relation.clone();
    tampered_relation.statement.step_public.program_digest[0] ^= 1;
    tampered_relation.statement.step_public.step_lo ^= 1;
    tampered_relation.statement.step_public.step_hi ^= 1;
    tampered_relation.statement.step_public.state_in[0] ^= 1;
    tampered_relation.statement.step_public.state_out[0] ^= 1;
    tampered_relation.statement.step_public.halted_out = !tampered_relation.statement.step_public.halted_out;
    tampered_relation.statement.chunk_summary.start_index ^= 1;
    tampered_relation.statement.chunk_summary.public_step_count ^= 1;
    tampered_relation
        .statement
        .chunk_summary
        .public_chunk_digest[0] ^= 1;
    tampered_relation
        .statement
        .chunk_summary
        .chunk_relation_digest[0] ^= 1;
    let rebuilt = audit_rv64im_main_recursion_construction2_verified_step_statement_digest(&tampered_relation)
        .expect("rebuild Construction-2 verified step statement from tampered legacy relation statement shell");

    assert_eq!(
        rebuilt, baseline,
        "Construction-2 verified step statement must ignore the legacy relation.statement shell"
    );
}

#[test]
fn rv64im_main_recursion_claim_cover_ignores_legacy_relation_statement_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain")
        .into_iter()
        .next()
        .expect("first relation");
    let advices = build_rv64im_main_recursion_f_prime_advices(std::slice::from_ref(&relation))
        .expect("build native F' advice for recursive claim cover");
    let claim_cover =
        build_rv64im_main_recursion_f_prime_claim_cover(&advices).expect("build recursive-step claim cover");
    let mut tampered_relation = relation.clone();
    tampered_relation.statement.step_public.program_digest[0] ^= 1;
    tampered_relation.statement.step_public.chunk_index ^= 1;
    tampered_relation.statement.step_public.step_lo ^= 1;
    tampered_relation.statement.step_public.step_hi ^= 1;
    tampered_relation.statement.step_public.state_in[0] ^= 1;
    tampered_relation.statement.step_public.state_out[0] ^= 1;
    tampered_relation.statement.step_public.halted_out = !tampered_relation.statement.step_public.halted_out;
    tampered_relation.statement.chunk_summary.start_index ^= 1;
    tampered_relation.statement.chunk_summary.public_step_count ^= 1;
    tampered_relation
        .statement
        .chunk_summary
        .public_chunk_digest[0] ^= 1;
    tampered_relation
        .statement
        .chunk_summary
        .chunk_relation_digest[0] ^= 1;

    assert!(
        claim_cover.covers_relation(&tampered_relation),
        "recursive-step claim cover must ignore the legacy relation.statement shell"
    );
}

#[test]
fn rv64im_main_recursion_construction2_shape_builder_ignores_tampered_legacy_fresh_state_out_transcript_absorbed() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");
    let baseline = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&f_prime_advices)
        .expect("build baseline Construction-2 native F' shape");
    let mut tampered_advices = f_prime_advices.clone();
    rv64im_main_recursion_advice_tamper_fresh_state_out_transcript_absorbed(&mut tampered_advices[0]);

    let rebuilt = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&tampered_advices)
        .expect("Construction-2 native F' shape builder must derive transcript_out from the verified step, not legacy fresh_state_out");

    assert_eq!(
        rebuilt.expected_digest(),
        baseline.expected_digest(),
        "Construction-2 native F' shape digest must be invariant under tampered legacy fresh_state_out transcript absorbed count"
    );
}

#[test]
fn rv64im_main_recursion_construction2_fresh_instance_builder_builds_base_case_u1() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");

    let u_1 = build_rv64im_main_recursion_construction2_fresh_instance(&f_prime_advices[0])
        .expect("build base-case HyperNova Construction-2 fresh instance");

    assert_eq!(
        u_1.x_i(),
        &build_rv64im_main_recursion_construction2_x_i(&f_prime_advices[0]).expect("build construction2 x_i"),
    );
    assert!(
        u_1.commitment()
            .commitment()
            .data
            .iter()
            .any(|&value| value != F::ZERO),
        "base-case Construction-2 fresh commitment should not remain the zero placeholder once enc(F') is wired"
    );
}

#[test]
fn rv64im_main_recursion_construction2_fresh_instance_builder_builds_inductive_step_with_threaded_prior_u_i() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");

    assert!(
        f_prime_advices.len() >= 2,
        "expected at least two single-step advices for inductive Construction-2 coverage"
    );

    let u_1 = build_rv64im_main_recursion_construction2_fresh_instance(&f_prime_advices[0])
        .expect("build base-case HyperNova Construction-2 fresh instance");
    let u_2 = build_rv64im_main_recursion_construction2_fresh_instance_with_input(&f_prime_advices[1], &u_1)
        .expect("build inductive HyperNova Construction-2 fresh instance with threaded prior u_i");

    assert_eq!(
        u_2.x_i(),
        &build_rv64im_main_recursion_construction2_x_i(&f_prime_advices[1]).expect("build construction2 x_i"),
    );
    assert!(
        u_2.commitment()
            .commitment()
            .data
            .iter()
            .any(|&value| value != F::ZERO),
        "inductive Construction-2 fresh commitment should be non-zero once prior u_i is threaded"
    );
}

#[test]
fn rv64im_main_recursion_construction2_fresh_instance_builder_rejects_inductive_step_without_prior_output_threading() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let f_prime_advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");

    assert!(
        f_prime_advices.len() >= 2,
        "expected at least two single-step advices for inductive Construction-2 coverage"
    );

    let err = build_rv64im_main_recursion_construction2_fresh_instance(&f_prime_advices[1])
        .expect_err("inductive Construction-2 fresh-instance construction must require the prior-step output u_i");
    assert!(err.to_string().contains("prior-step output"));
}

#[test]
fn rv64im_main_recursion_construction2_shape_builder_rejects_empty_advice_chain() {
    let err = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&[])
        .expect_err("Construction-2 shape builder must reject an empty native F' advice chain");
    assert!(err.to_string().contains("at least one"));
}

#[test]
fn rv64im_main_recursion_base_case_rejects_non_default_accumulator_even_if_x_i_is_retargeted() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline_advice = &f_prime_advices[0];
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(baseline_advice).expect("build native F' public output");
    let mut tampered_advice = baseline_advice.clone();
    rv64im_main_recursion_advice_tamper_running_state_terminal_handle_first_byte(&mut tampered_advice);
    rv64im_main_recursion_advice_retarget_x_hash_to_current_accumulator(&mut tampered_advice);

    assert_ne!(
        baseline_advice.x_i(),
        tampered_advice.x_i(),
        "retargeting x_i should make the tampered base-case accumulator self-consistent"
    );

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("base-case accumulator drift must fail even after x_i retargeting");
    assert!(err.to_string().contains("U_perp"));
}

#[test]
fn rv64im_main_recursion_step_x_i_ignores_tampered_input_step_statement_chain_digest() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline = &f_prime_advices[0];
    let mut tampered = baseline.clone();
    rv64im_main_recursion_advice_tamper_step_statement_chain_digest_first_byte(&mut tampered);

    assert_eq!(
        build_rv64im_main_recursion_construction2_input_state_image(&tampered).encoded_public_input(),
        *baseline.x_i(),
        "authoritative input x_i must ignore the legacy input step-statement chain digest shell"
    );
}

#[test]
fn rv64im_main_recursion_step_x_i_ignores_tampered_input_running_state_terminal_handle_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline = &f_prime_advices[0];
    let mut tampered = baseline.clone();
    rv64im_main_recursion_advice_tamper_running_state_terminal_handle_only_first_byte(&mut tampered);

    assert_eq!(
        build_rv64im_main_recursion_construction2_input_state_image(&tampered).encoded_public_input(),
        *baseline.x_i(),
        "authoritative input x_i must depend only on the paper-facing running U_i digest, not the legacy running terminal-handle shell"
    );
}

#[test]
fn rv64im_main_recursion_step_x_out_ignores_tampered_input_step_statement_chain_digest() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline =
        evaluate_rv64im_main_recursion_f_prime_advice(&f_prime_advices[0]).expect("evaluate baseline native F' step");
    let mut tampered = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_step_statement_chain_digest_first_byte(&mut tampered);

    let rebuilt = evaluate_rv64im_main_recursion_f_prime_advice(&tampered)
        .expect("native F' must ignore the legacy input step-statement chain digest shell");

    assert_eq!(
        rebuilt.x_out(),
        baseline.x_out(),
        "authoritative x_out must ignore the legacy input step-statement chain digest shell"
    );
}

#[test]
fn rv64im_main_recursion_step_x_i_ignores_tampered_input_bridge_handoff_chain_digest() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline = &f_prime_advices[0];
    let mut tampered = baseline.clone();
    rv64im_main_recursion_advice_tamper_bridge_handoff_chain_digest_first_byte(&mut tampered);

    assert_eq!(
        build_rv64im_main_recursion_construction2_input_state_image(&tampered).encoded_public_input(),
        *baseline.x_i(),
        "authoritative input x_i must ignore the legacy input bridge-handoff chain digest shell"
    );
}

#[test]
fn rv64im_main_recursion_step_x_out_ignores_tampered_input_bridge_handoff_chain_digest() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline =
        evaluate_rv64im_main_recursion_f_prime_advice(&f_prime_advices[0]).expect("evaluate baseline native F' step");
    let mut tampered = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_bridge_handoff_chain_digest_first_byte(&mut tampered);

    let rebuilt = evaluate_rv64im_main_recursion_f_prime_advice(&tampered)
        .expect("native F' must ignore the legacy input bridge-handoff chain digest shell");

    assert_eq!(
        rebuilt.x_out(),
        baseline.x_out(),
        "authoritative x_out must ignore the legacy input bridge-handoff chain digest shell"
    );
}

#[test]
fn rv64im_main_recursion_step_image_exposes_explicit_next_state() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let output = verify_rv64im_main_recursion_f_prime_public_output(
        &build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output"),
        &f_prime_advices[0],
    )
    .expect("verify F' public output");

    assert_eq!(
        output.z_next(),
        &output.running_out_state().carry.terminal_handle.0,
        "step image z_next must expose the carried next terminal handle"
    );
    assert_eq!(
        output.pc_next(),
        1,
        "RV64IM recursion currently specializes to trivial pc = 1"
    );
    assert!(
        output.phi_side().is_zero(),
        "current unwired RV64IM recursion path must emit a zero phi_side projection"
    );
}

#[test]
fn rv64im_main_recursion_step_emits_authoritative_phi_side_without_changing_x_out() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let zero_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build zero-side recursion advices");
    let side_advices = build_rv64im_main_recursion_f_prime_advices_with_side_opening_public(
        &chunk_step_relations,
        fixture.side_proof.opening_public(),
    )
    .expect("build side-aware recursion advices");

    assert_eq!(zero_advices.len(), side_advices.len());
    assert!(
        side_advices
            .iter()
            .all(|advice| advice.side_witness().is_zero() && !advice.phi_side().is_zero()),
        "stable authoritative phi_side should be wired without pretending step-local side witnesses are active"
    );

    for (zero_advice, side_advice) in zero_advices.iter().zip(&side_advices) {
        let zero_output = verify_rv64im_main_recursion_f_prime_public_output(
            &build_rv64im_main_recursion_f_prime_public_output(zero_advice)
                .expect("build zero-side native F' public output"),
            zero_advice,
        )
        .expect("verify zero-side F' public output");
        let side_output = verify_rv64im_main_recursion_f_prime_public_output(
            &build_rv64im_main_recursion_f_prime_public_output(side_advice)
                .expect("build side-aware native F' public output"),
            side_advice,
        )
        .expect("verify side-aware F' public output");

        assert_eq!(side_output.x_out(), zero_output.x_out());
        assert_eq!(
            side_output.folded_accumulator_digest(),
            zero_output.folded_accumulator_digest()
        );
        assert_eq!(
            side_output.terminal_handle_digest(),
            zero_output.terminal_handle_digest()
        );
        assert_eq!(
            side_output.step_statement_chain_digest(),
            zero_output.step_statement_chain_digest()
        );
        assert_eq!(
            side_output.bridge_handoff_chain_digest(),
            zero_output.bridge_handoff_chain_digest()
        );
        assert_eq!(
            side_output.phi_side(),
            side_advice.phi_side(),
            "side-aware native F' output must carry the authoritative stable phi_side projection"
        );
        assert!(
            !side_output.phi_side().is_zero(),
            "side-aware native F' output should expose a non-zero phi_side projection"
        );
    }
}

#[test]
fn rv64im_main_recursion_native_chain_closes_to_final_accumulator_and_x_last() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");

    let last_output = f_prime_advices
        .iter()
        .map(|advice| {
            verify_rv64im_main_recursion_f_prime_public_output(
                &build_rv64im_main_recursion_f_prime_public_output(advice).expect("build native F' public output"),
                advice,
            )
            .expect("verify native F' public output")
        })
        .last()
        .expect("non-empty step relation chain");
    let rebuilt_x_last = audit_build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
        f_prime_advices
            .last()
            .expect("non-empty step relation chain")
            .verifier_key_fs(),
        last_output.chunk_count(),
        &fixture.final_statement.folded.final_accumulator,
        last_output.step_statement_chain_digest(),
        last_output.bridge_handoff_chain_digest(),
    )
    .expect("rebuild x_last from final accumulator");

    assert_eq!(
        last_output.running_out_state().carry.main.claims,
        fixture
            .final_statement
            .folded
            .final_accumulator
            .final_main_claims,
        "native F' chain must land on the final folded main-claim surface"
    );
    assert_eq!(
        last_output.running_out_state().carry.terminal_handle.0,
        fixture
            .final_statement
            .folded
            .final_accumulator
            .terminal_handle
            .0,
        "native F' chain must land on the final folded terminal handle"
    );
    assert_eq!(
        last_output.x_out(),
        &rebuilt_x_last,
        "native F' chain must rebuild the theorem-facing final x_last from the final accumulator"
    );
}

#[test]
fn rv64im_nifs_proof_round_trip_from_structured_surfaces() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let relation = &chunk_step_relations[0];

    audit_rv64im_nifs_round_trip_from_chunk_step_relation(relation)
        .expect("round-trip NIFS proof from structured surfaces");
}

#[test]
fn rv64im_nifs_proof_round_trip_holds_for_each_step_in_structured_chain() {
    let source = build_mixed_opcode_perf_source_case(1);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove multi-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");

    assert!(
        chunk_step_relations.len() > 1,
        "expected a multi-step structured chain for native NIFS replay coverage"
    );

    for (step_index, relation) in chunk_step_relations.iter().enumerate() {
        audit_rv64im_nifs_round_trip_from_chunk_step_relation(relation)
            .unwrap_or_else(|err| panic!("native NIFS replay round-trip failed at step {step_index}: {err}"));
    }
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_chunk_index() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_chunk_index(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered chunk_index must fail");
    assert!(err.to_string().contains("chunk_index"));
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_x_i() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_x_hash_first_byte(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered x_i must fail");
    assert!(err.to_string().contains("x_i"));
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_z_i() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_z_i_first_byte(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered z_i must fail");
    assert!(err.to_string().contains("z_i"));
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_pc_i() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_pc_i(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered pc_i must fail");
    assert!(err.to_string().contains("pc_i"));
}

#[test]
fn rv64im_main_recursion_step_rejects_nonzero_side_witness() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_side_witness_nonzero(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("non-zero side_witness must fail before phi_side is wired");
    assert!(err.to_string().contains("side_witness"));
}

#[test]
fn rv64im_main_recursion_fresh_instance_digest_binds_x_i() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline_advice = &f_prime_advices[0];
    let mut tampered_advice = baseline_advice.clone();
    rv64im_main_recursion_advice_tamper_x_hash_first_byte(&mut tampered_advice);

    assert_eq!(
        baseline_advice.step_statement_digest(),
        tampered_advice.step_statement_digest(),
        "step-statement digest must remain the native NIFS digest"
    );
    assert_ne!(
        baseline_advice.fresh_instance_digest(),
        tampered_advice.fresh_instance_digest(),
        "fresh-instance digest must bind the carried Construction-2 fresh input image"
    );
}

#[test]
fn rv64im_main_recursion_external_step_statement_chain_matches_native_f_prime() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");

    let last_output = f_prime_advices
        .iter()
        .map(|advice| evaluate_rv64im_main_recursion_f_prime_advice(advice).expect("evaluate main recursion F'"))
        .last()
        .expect("non-empty step relation chain");

    assert_eq!(
        rv64im_recursion_step_statement_chain_digest(&chunk_step_relations),
        last_output.step_statement_chain_digest(),
        "external recursion step-statement chain must agree with the native Construction-2 F' chain"
    );
}

#[test]
fn rv64im_main_recursion_advice_chain_threads_construction2_input_fresh_instances() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");
    let advices = build_rv64im_main_recursion_f_prime_advices_single_step(&chunk_step_relations)
        .expect("build single-step native F' advices");
    let _shape = build_rv64im_main_recursion_construction2_f_prime_ccs_shape(&advices)
        .expect("build Construction-2 native F' CCS shape");
    let mut current_u_i = build_rv64im_main_recursion_construction2_default_fresh_instance(
        advices[0].verifier_key_fs(),
        default_full_width_from_advice(&advices[0]),
    )
    .expect("build canonical u_perp");

    for advice in &advices {
        let carried_u_i = advice
            .construction2_input_fresh_instance()
            .expect("native F' advice must carry the threaded Construction-2 input u_i");
        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice).expect("evaluate native F' step image");
        assert_eq!(
            carried_u_i, &current_u_i,
            "native F' advice must expose the threaded Construction-2 input u_i"
        );
        assert_eq!(
            carried_u_i.x_i(),
            advice.x_i(),
            "native F' advice Construction-2 input u_i must agree with the carried x_i image"
        );
        assert_eq!(
            step_image.construction2_u_next(),
            &build_rv64im_main_recursion_construction2_fresh_instance_with_input(advice, &current_u_i)
                .expect("rebuild threaded Construction-2 output fresh instance"),
            "native F' step image must own the canonical Construction-2 output fresh instance u_{{i+1}}"
        );
        assert_eq!(
            step_image.construction2_u_next().x_i(),
            step_image.x_out(),
            "native F' Construction-2 output fresh instance must carry x_{{i+1}}"
        );
        current_u_i = step_image.construction2_u_next().clone();
    }
}

#[test]
fn rv64im_main_recursion_step_ignores_tampered_legacy_fresh_state_out_terminal_handle() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline = evaluate_rv64im_main_recursion_f_prime_advice(&f_prime_advices[0])
        .expect("evaluate baseline native F' step image");
    let mut tampered = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_fresh_state_out_terminal_handle_first_byte(&mut tampered);

    let rebuilt = evaluate_rv64im_main_recursion_f_prime_advice(&tampered)
        .expect("native F' must derive next state from verified replay, not trusted legacy state_out cargo");

    assert_eq!(
        rebuilt.terminal_handle_digest(),
        baseline.terminal_handle_digest(),
        "native F' terminal handle must come from verified replay, not legacy state_out cargo"
    );
    assert_eq!(
        rebuilt.folded_accumulator_digest(),
        baseline.folded_accumulator_digest(),
        "native F' folded accumulator digest must come from verified replay, not legacy state_out cargo"
    );
    assert_eq!(
        rebuilt.x_out(),
        baseline.x_out(),
        "native F' x_out must be invariant under tampered legacy state_out cargo"
    );
}

#[test]
fn rv64im_main_recursion_construction2_bridge_ignores_tampered_legacy_prepared_step_digest_vector() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline = audit_rv64im_main_recursion_construction2_bridge_next_running(&f_prime_advices[0])
        .expect("derive baseline next running state through the Construction-2 bridge");
    let mut tampered = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_legacy_prepared_step_digest_first_byte(&mut tampered);

    let rebuilt = audit_rv64im_main_recursion_construction2_bridge_next_running(&tampered)
        .expect("Construction-2 bridge must derive next running state without legacy prepared-step digest cargo");

    assert_eq!(
        rebuilt.carry.terminal_handle, baseline.carry.terminal_handle,
        "Construction-2 bridge terminal handle must be invariant under tampered legacy prepared-step digest cargo"
    );
    assert_eq!(
        rebuilt.transcript, baseline.transcript,
        "Construction-2 bridge transcript_out must be invariant under tampered legacy prepared-step digest cargo"
    );
    assert_eq!(
        rebuilt.carry.main.claims, baseline.carry.main.claims,
        "Construction-2 bridge carried claims must be invariant under tampered legacy prepared-step digest cargo"
    );
    assert_eq!(
        rebuilt.carry.main.witnesses, baseline.carry.main.witnesses,
        "Construction-2 bridge carried witnesses must be invariant under tampered legacy prepared-step digest cargo"
    );
}

#[test]
fn rv64im_main_recursion_backend_statement_matches_native_f_prime_step_image() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");

    for advice in &f_prime_advices {
        audit_rv64im_main_recursion_backend_statement_matches_native_f_prime(advice)
            .expect("backend statement must match native F' step image");
    }
}

#[test]
fn rv64im_main_recursion_step_spartan_fixed_shape_contract_holds_across_chain() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");

    let (first_shape, last_shape) =
        audit_rv64im_main_recursion_step_spartan_fixed_shape_across_chain(&chunk_step_relations)
            .expect("recursive-step circuit should keep a fixed shape across the native F' chain");

    assert_eq!(
        first_shape, last_shape,
        "recursive-step circuit shape fingerprint drifted across the native F' chain"
    );
}

#[test]
fn rv64im_main_recursion_step_spartan_fixed_shape_contract_holds_at_positions_0_10_100() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof =
        prove_rv64im_public_proof_with_options(&input, options).expect("prove per-step rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let chunk_step_relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step chain");

    let position_shapes = audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions(
        &chunk_step_relations[..1],
        &[0, 10, 100],
    )
    .expect("recursive-step circuit should keep a fixed shape at chunk positions 0, 10, and 100");

    assert_eq!(position_shapes.len(), 3);
    assert_eq!(position_shapes[0].0, 0);
    assert_eq!(position_shapes[1].0, 10);
    assert_eq!(position_shapes[2].0, 100);
    assert_eq!(
        position_shapes[0].1, position_shapes[1].1,
        "shared recursive-step Spartan shape digest drifted between chunk positions 0 and 10"
    );
    assert_eq!(
        position_shapes[0].1, position_shapes[2].1,
        "shared recursive-step Spartan shape digest drifted between chunk positions 0 and 100"
    );
    assert_eq!(
        position_shapes[0].2, position_shapes[1].2,
        "recursive-step circuit shape drifted between chunk positions 0 and 10"
    );
    assert_eq!(
        position_shapes[0].2, position_shapes[2].2,
        "recursive-step circuit shape drifted between chunk positions 0 and 100"
    );
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_vk_fs() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_vk_fs_main_lane_shape_digest_first_byte(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered vk_fs must fail");
    assert!(err.to_string().contains("vk_fs"));
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_public_output_x_out() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let mut public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    public_output.x_out_mut().bytes_mut()[0] ^= 1;

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &f_prime_advices[0])
        .expect_err("tampered x_out must fail");
    assert!(err.to_string().contains("x_out"));
}

#[test]
fn rv64im_main_recursion_step_ignores_tampered_legacy_bridge_handoff_digest_field() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline =
        evaluate_rv64im_main_recursion_f_prime_advice(&f_prime_advices[0]).expect("evaluate baseline native F' step");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_legacy_bridge_handoff_digest_first_byte(&mut tampered_advice);

    let rebuilt = evaluate_rv64im_main_recursion_f_prime_advice(&tampered_advice)
        .expect("native F' must recompute bridge_handoff.digest instead of trusting stored digest bytes");

    assert_eq!(
        rebuilt.bridge_handoff_chain_digest(),
        baseline.bridge_handoff_chain_digest(),
        "native F' bridge_handoff_chain_digest must be invariant under tampered stored bridge_handoff.digest bytes"
    );
    assert_eq!(
        rebuilt.x_out(),
        baseline.x_out(),
        "native F' x_out must be invariant under tampered stored bridge_handoff.digest bytes"
    );
}

#[test]
fn rv64im_main_recursion_step_ignores_tampered_legacy_bridge_binding_digest_field() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let baseline =
        evaluate_rv64im_main_recursion_f_prime_advice(&f_prime_advices[0]).expect("evaluate baseline native F' step");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_legacy_bridge_binding_digest_first_byte(&mut tampered_advice);

    let rebuilt = evaluate_rv64im_main_recursion_f_prime_advice(&tampered_advice)
        .expect("native F' must recompute bridge binding digests instead of trusting stored digest bytes");

    assert_eq!(
        rebuilt.bridge_handoff_chain_digest(),
        baseline.bridge_handoff_chain_digest(),
        "native F' bridge_handoff_chain_digest must be invariant under tampered stored bridge binding digest bytes"
    );
    assert_eq!(
        rebuilt.x_out(),
        baseline.x_out(),
        "native F' x_out must be invariant under tampered stored bridge binding digest bytes"
    );
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_pi_ccs_replay() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_ccs_replay_first_round_coeff(&mut tampered_advice);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered Pi_CCS replay must fail");
    assert!(err.to_string().contains("Pi_CCS replay"));
}

#[test]
fn rv64im_main_recursion_step_rejects_tampered_pi_dec_children() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let chunk_step_relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step chain");
    let f_prime_advices =
        build_rv64im_main_recursion_f_prime_advices(&chunk_step_relations).expect("build main recursion F' advices");
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(&f_prime_advices[0]).expect("build native F' public output");
    let mut tampered_advice = f_prime_advices[0].clone();
    rv64im_main_recursion_advice_tamper_dec_child_commitment_first_word(&mut tampered_advice, 0);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered_advice)
        .expect_err("tampered Pi_DEC children must fail");
    assert!(err.to_string().contains("DEC child"));
}

#[test]
fn rv64im_encoded_public_input_bit_image_uses_little_endian_bits_per_byte() {
    let mut digest = [0u8; 32];
    digest[0] = 0xA5;
    digest[1] = 0x03;
    let encoded = Rv64imEncodedPublicInput::from_digest_bytes(digest);
    let bits = encoded.bit_image();

    assert_eq!(bits[..10], [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]);
}

#[test]
fn rv64im_encoded_public_input_field_image_is_binary_low_norm() {
    let mut digest = [0u8; 32];
    digest[0] = 0xA5;
    digest[31] = 0x80;
    let encoded = Rv64imEncodedPublicInput::from_digest_bytes(digest);
    let bits = encoded.bit_image();
    let fields = encoded.field_image();

    assert!(encoded.is_binary_low_norm());
    for (field, bit) in fields.iter().zip(bits.into_iter()) {
        assert_eq!(field.as_canonical_u64(), bit as u64);
        assert!(field.as_canonical_u64() <= 1);
    }
}

#[test]
fn rv64im_encoded_public_input_ring_image_packs_by_q_d_plus_r() {
    let mut digest = [0u8; 32];
    for bit_index in [0usize, 53, 54, 55, 255] {
        set_digest_bit_le(&mut digest, bit_index);
    }

    let encoded = Rv64imEncodedPublicInput::from_digest_bytes(digest);
    let fields = encoded.field_image();
    let ring = encoded.ring_image();

    assert_eq!(ring.len(), 5);
    assert_eq!(ring[0].len(), 54);
    assert_eq!(ring[0][0].as_canonical_u64(), 1);
    assert_eq!(ring[0][53].as_canonical_u64(), 1);
    assert_eq!(ring[1][0].as_canonical_u64(), 1);
    assert_eq!(ring[1][1].as_canonical_u64(), 1);
    assert_eq!(ring[4][39].as_canonical_u64(), 1);
    assert_eq!(ring[4][40].as_canonical_u64(), 0);

    for (field_index, field) in fields.into_iter().enumerate() {
        assert_eq!(ring[field_index / 54][field_index % 54], field);
    }
}

#[test]
fn rv64im_encoded_public_input_ring_image_zero_pads_tail_after_bit_255() {
    let encoded = Rv64imEncodedPublicInput::from_digest_bytes([0xFF; 32]);
    let ring = encoded.ring_image();

    for coeff in ring[4].iter().take(40) {
        assert_eq!(coeff.as_canonical_u64(), 1);
    }
    for coeff in ring[4].iter().skip(40) {
        assert_eq!(coeff.as_canonical_u64(), 0);
    }
}
