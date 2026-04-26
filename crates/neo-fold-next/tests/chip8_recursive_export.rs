#[path = "support/chip8.rs"]
mod chip8_support;

use neo_fold_next::chip8::kernel::{
    build_chip8_bridge_chunk_proof_bundle, prepared_step_digest, verify_kernel_execution_relation,
};
use neo_fold_next::chip8::proof::{
    audit_check_final_statement_replay, audit_check_folded_statement_replay, audit_check_recursive_artifact_replay,
    prove_final_statement, prove_folded_statement, prove_kernel_export, prove_recursive,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn chip8_recursive_step_verifies_single_chunk_transition() {
    let input = chip8_support::build_jump_kernel_input(1);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");

    assert_eq!(proof.steps.len(), 1);
    assert_eq!(statement.folded.final_accumulator.final_main_claims.len(), 16);
    assert_ne!(statement.folded.kernel_relation_digest, [0; 32]);
    assert!(proof.steps[0].main_transition.step_witness_slots[0].is_some());
    assert_ne!(statement.folded.final_accumulator.bridge_state, [0; 32]);

    audit_check_recursive_artifact_replay(&statement, &proof).expect("audit-check recursive artifact replay");
}

#[test]
fn chip8_recursive_chain_verifies_multi_chunk_execution() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");

    assert!(proof.steps.len() >= 2);
    assert_eq!(statement.folded.final_accumulator.final_main_claims.len(), 16);

    audit_check_recursive_artifact_replay(&statement, &proof).expect("audit-check recursive artifact replay");
}

#[test]
fn chip8_recursive_bridge_transitions_are_derived_from_relation_witness() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (_statement, proof) = prove_recursive(&input).expect("prove recursive");
    let (_relation_digest, native_export) = prove_kernel_export(&input).expect("prove native kernel export");

    let verified = verify_kernel_execution_relation(
        &chip8_support::verifier_input_from_public(&input.public),
        &native_export,
    )
    .expect("verify kernel execution relation");
    let rebuilt_bundle = build_chip8_bridge_chunk_proof_bundle(
        &verified.kernel_opening_manifest,
        &verified.opening_refinement_summary,
        &native_export
            .bridge_chunk_transitions()
            .iter()
            .flat_map(|transition| transition.row_slots.iter().flatten())
            .map(|row| row.row_binding.clone())
            .collect::<Vec<_>>(),
    )
    .expect("rebuild bridge bundle from native export witness");

    assert_eq!(proof.kernel_export.chunk_handoffs.len(), proof.steps.len());
    assert_eq!(
        rebuilt_bundle.chunk_transitions.len(),
        proof.kernel_export.chunk_handoffs.len()
    );
    for (chunk_index, compact_handoff) in proof.kernel_export.chunk_handoffs.iter().enumerate() {
        let verified_handoff = &verified.chunk_handoffs[chunk_index];
        assert_eq!(
            compact_handoff.public_chunk.start_index,
            verified_handoff.public_chunk.start_index
        );
        assert_eq!(
            compact_handoff.public_chunk.steps.len(),
            verified_handoff.public_chunk.steps.len()
        );
        assert_eq!(compact_handoff.bridge_handoff, verified_handoff.bridge_handoff);
        let active_row_count = compact_handoff
            .bridge_handoff
            .step_bindings
            .iter()
            .take_while(|slot| slot.is_some())
            .count();
        let first_row_index = compact_handoff
            .bridge_handoff
            .step_bindings
            .iter()
            .flatten()
            .next()
            .expect("active compact bridge binding")
            .row_index;
        assert_eq!(first_row_index, chunk_index * 2);
        assert_eq!(active_row_count, 2);
    }
}

#[test]
fn chip8_recursive_steps_carry_prepared_step_bridge_bindings() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (output, _native_proof) = chip8_support::run_native_kernel(&input).expect("prove simple kernel");
    let (_statement, proof) = prove_recursive(&input).expect("prove recursive");

    assert_eq!(proof.steps.len(), 2);
    for (chunk_index, step) in proof.steps.iter().enumerate() {
        for slot_index in 0..2 {
            let binding = step.bridge_bindings[slot_index]
                .as_ref()
                .expect("two-row chunk must carry two bridge bindings");
            let expected_row_index = chunk_index * 2 + slot_index;
            assert_eq!(binding.row_index, expected_row_index);
            assert_eq!(
                binding.prepared_step_digest,
                prepared_step_digest(&output.prepared_steps[expected_row_index])
            );
        }
    }
}

#[test]
fn chip8_recursive_proof_envelope_round_trip() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    audit_check_recursive_artifact_replay(&statement, &proof).expect("audit-check recursive artifact replay");
}

#[test]
fn chip8_folded_statement_round_trip() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (folded, proof) = prove_folded_statement(&input).expect("prove folded statement");
    audit_check_folded_statement_replay(&input.public, &folded, &proof).expect("audit-check folded statement replay");
}

#[test]
fn chip8_final_statement_round_trip() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (folded, proof) = prove_final_statement(&input).expect("prove final statement");
    audit_check_final_statement_replay(&input.public, &folded, &proof).expect("audit-check final statement replay");
}

#[test]
fn chip8_final_statement_rejects_tampered_folded_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut folded, proof) = prove_final_statement(&input).expect("prove final statement");
    folded.digest[0] ^= 1;

    let err = audit_check_final_statement_replay(&input.public, &folded, &proof)
        .expect_err("tampered folded digest must fail");
    assert!(format!("{err}").contains("folded") || format!("{err}").contains("digest"));
}

#[test]
fn chip8_final_statement_rejects_tampered_final_proof_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (folded, mut proof) = prove_final_statement(&input).expect("prove final statement");
    proof.proof_digest[0] ^= 1;

    let err = audit_check_final_statement_replay(&input.public, &folded, &proof)
        .expect_err("tampered final proof digest must fail");
    assert!(format!("{err}").contains("final proof") || format!("{err}").contains("digest"));
}

#[test]
fn chip8_final_statement_rejects_swapped_self_consistent_public() {
    let input_a = chip8_support::build_jump_kernel_input(4);
    let (folded_a, proof_a) = prove_final_statement(&input_a).expect("prove final statement A");

    let mut input_b = chip8_support::build_jump_kernel_input(4);
    input_b.public.transcript_seed = vec![1, 2, 3, 4];
    let (_folded_b, _proof_b) = prove_final_statement(&input_b).expect("prove final statement B");
    assert_ne!(input_a.public.transcript_seed, input_b.public.transcript_seed);

    let err = audit_check_final_statement_replay(&input_b.public, &folded_a, &proof_a)
        .expect_err("swapped self-consistent public input must fail");
    assert!(!format!("{err}").is_empty());

    audit_check_final_statement_replay(&input_a.public, &folded_a, &proof_a)
        .expect("audit-check original final statement replay");
}

#[test]
fn chip8_audit_replay_uses_only_statement_and_proof() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");

    assert_eq!(statement.folded.semantic_step_count, 4);
    audit_check_recursive_artifact_replay(&statement, &proof).expect("audit-check recursive artifact replay");
}

#[test]
fn chip8_recursive_proof_envelope_rejects_swapped_self_consistent_statement() {
    let input_a = chip8_support::build_jump_kernel_input(4);
    let (proof_statement_a, proof_a) = prove_recursive(&input_a).expect("prove recursive A");

    let input_b = chip8_support::build_jump_kernel_input(1);
    let (statement_b, _proof_b) = prove_recursive(&input_b).expect("prove recursive B");

    let err = audit_check_recursive_artifact_replay(&statement_b, &proof_a)
        .expect_err("swapped self-consistent statement must fail");
    assert!(
        format!("{err}").contains("final proof")
            || format!("{err}").contains("kernel")
            || format!("{err}").contains("digest")
    );

    // Keep the original pair exercised so the test is not vacuously passing on broken setup.
    audit_check_recursive_artifact_replay(&proof_statement_a, &proof_a)
        .expect("audit-check original recursive artifact replay");
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_terminal_handle() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut statement, proof) = prove_recursive(&input).expect("prove recursive");
    statement.folded.final_accumulator.terminal_handle.0[0] ^= 1;

    let err =
        audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered terminal handle must fail");
    assert!(
        format!("{err}").contains("digest")
            || format!("{err}").contains("terminal")
            || format!("{err}").contains("handle")
    );
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_final_bridge_accumulator() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut statement, proof) = prove_recursive(&input).expect("prove recursive");
    statement.folded.final_accumulator.bridge_state[0] ^= 1;

    let err = audit_check_recursive_artifact_replay(&statement, &proof)
        .expect_err("tampered final bridge accumulator must fail");
    assert!(format!("{err}").contains("bridge"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_statement_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut statement, proof) = prove_recursive(&input).expect("prove recursive");
    statement.digest[0] ^= 1;

    let err =
        audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered statement digest must fail");
    assert!(format!("{err}").contains("digest"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_chunk_relation_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut statement, proof) = prove_recursive(&input).expect("prove recursive");
    statement.folded.final_accumulator.final_main_claims[0]
        .c
        .data[0] += F::ONE;

    let err = audit_check_recursive_artifact_replay(&statement, &proof)
        .expect_err("tampered chunk relation digest must fail");
    assert!(format!("{err}").contains("digest") || format!("{err}").contains("main claims"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_main_replay_witness() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    proof.steps[0].main_transition.replay_witness.header_digest[0] ^= 1;

    let err =
        audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered main replay witness must fail");
    assert!(format!("{err}").contains("chunk") || format!("{err}").contains("digest"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_bridge_binding() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    proof.steps[0].bridge_bindings[0]
        .as_mut()
        .expect("active bridge binding")
        .prepared_step_digest[0] ^= 1;

    let err = audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered bridge binding must fail");
    assert!(format!("{err}").contains("binding") || format!("{err}").contains("bridge"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_nonempty_inactive_main_replay_output_slot() {
    let input = chip8_support::build_jump_kernel_input(1);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    let cloned = proof.steps[0]
        .main_transition
        .replay_witness
        .ccs_output_slots[0]
        .clone()
        .expect("active replay output slot");
    proof.steps[0]
        .main_transition
        .replay_witness
        .ccs_output_slots[1] = Some(cloned);

    let err = audit_check_recursive_artifact_replay(&statement, &proof)
        .expect_err("inactive replay output slot pollution must fail");
    assert!(
        format!("{err}").contains("output slot")
            || format!("{err}").contains("inactive")
            || format!("{err}").contains("final proof")
            || format!("{err}").contains("digest")
    );
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_chunk_main_witness() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    proof.steps[0].main_transition.step_witness_slots[0]
        .as_mut()
        .expect("active slot witness")
        .Z[(0, 0)] += F::ONE;

    let err =
        audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered chunk main witness must fail");
    assert!(
        format!("{err}").contains("prepared step")
            || format!("{err}").contains("final proof")
            || format!("{err}").contains("digest")
    );
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_kernel_relation_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut statement, proof) = prove_recursive(&input).expect("prove recursive");
    statement.folded.kernel_relation_digest[0] ^= 1;

    let err = audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered kernel digest must fail");
    assert!(format!("{err}").contains("kernel") || format!("{err}").contains("digest"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_kernel_bridge_row_bits() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    let row_binding_digest = &mut proof.kernel_export.chunk_handoffs[0]
        .bridge_handoff
        .step_bindings[0]
        .as_mut()
        .expect("active compact bridge binding")
        .row_binding_claim_digest;
    row_binding_digest[0] ^= 1;

    let err = audit_check_recursive_artifact_replay(&statement, &proof)
        .expect_err("tampered compact bridge binding must fail");
    assert!(format!("{err}").contains("bridge") || format!("{err}").contains("row"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_kernel_bridge_source() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    proof.kernel_export.chunk_handoffs[0]
        .bridge_handoff
        .step_bindings[0]
        .as_mut()
        .expect("active compact bridge binding")
        .row_index ^= 1;

    let err =
        audit_check_recursive_artifact_replay(&statement, &proof).expect_err("tampered kernel bridge source must fail");
    assert!(format!("{err}").contains("bridge") || format!("{err}").contains("kernel"));
}

#[test]
fn chip8_recursive_proof_envelope_rejects_tampered_later_chunk_chain_handle() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    proof.steps[1].main_transition.replay_witness.header_digest[0] ^= 1;

    let err = audit_check_recursive_artifact_replay(&statement, &proof)
        .expect_err("tampered later chunk chain handle must fail");
    assert!(format!("{err}").contains("chunk") || format!("{err}").contains("digest"));
}
