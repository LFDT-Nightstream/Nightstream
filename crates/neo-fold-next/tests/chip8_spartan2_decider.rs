#[path = "support/chip8.rs"]
mod chip8_support;

use neo_fold_next::chip8::decider::{
    build_chip8_decider_relation, build_chip8_spartan2_decider_target, prove_chip8_spartan2_decider,
    setup_chip8_spartan2_decider, verify_chip8_decider_relation, verify_chip8_spartan2_decider,
};
use neo_fold_next::chip8::proof::prove_recursive;

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_decider_relation_round_trip() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");

    let relation = build_chip8_decider_relation(&statement, &proof).expect("build decider relation");

    assert_eq!(relation.public_statement_digest, statement.digest);
    assert_eq!(relation.relation_digest, statement.folded.digest);
    assert_eq!(relation.final_proof_digest, proof.proof_digest);
    assert_eq!(relation.fold_schedule, statement.folded.fold_schedule);
    assert_eq!(relation.semantic_step_count, statement.folded.semantic_step_count);
    assert_eq!(relation.chunk_summaries, proof.chunk_summaries);
    assert_eq!(
        relation.base_component_digests[0],
        build_chip8_spartan2_decider_target(&statement, &proof)
            .expect("build chip8 decider target")
            .witness
            .base_component_digests[0]
    );
    assert_eq!(relation.chunk_transition_bindings.len(), proof.steps.len());
    assert_ne!(relation.digest, [0; 32]);

    verify_chip8_decider_relation(&relation, &statement, &proof).expect("verify decider relation");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_spartan2_decider_target_projects_final_seam() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let relation = build_chip8_decider_relation(&statement, &proof).expect("build decider relation");

    let target = build_chip8_spartan2_decider_target(&statement, &proof).expect("build decider target");

    assert_eq!(
        target.statement.public_statement_digest,
        relation.public_statement_digest
    );
    assert_eq!(target.statement.relation_digest, relation.relation_digest);
    assert_eq!(target.statement.final_proof_digest, relation.final_proof_digest);
    assert_eq!(target.statement.fold_schedule, statement.folded.fold_schedule);
    assert_eq!(target.statement.chunk_count, statement.folded.chunk_count);
    assert_eq!(
        target.statement.semantic_step_count,
        statement.folded.semantic_step_count
    );
    assert_eq!(target.statement.chunk_summaries, relation.chunk_summaries);
    assert_eq!(target.witness.base_component_digests.len(), 1);
    assert_eq!(target.witness.chunk_transition_bindings.len(), proof.steps.len());
    assert_eq!(
        target.witness.base_component_digests[0],
        relation.base_component_digests[0]
    );
    assert_eq!(
        target
            .witness
            .chunk_transition_bindings
            .iter()
            .map(|binding| binding.transition_witness_digest)
            .collect::<Vec<_>>(),
        relation
            .chunk_transition_bindings
            .iter()
            .map(|binding| binding.transition_witness_digest)
            .collect::<Vec<_>>()
    );
    assert!(!target.statement.public_io().is_empty());
    assert_ne!(target.statement.digest(), [0; 32]);
    assert_ne!(target.witness.digest(), [0; 32]);
    assert_ne!(target.digest(), [0; 32]);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_spartan2_decider_target_rejects_tampered_statement_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (mut statement, proof) = prove_recursive(&input).expect("prove recursive");
    statement.digest[0] ^= 1;

    let err = build_chip8_spartan2_decider_target(&statement, &proof).expect_err("tampered statement digest must fail");
    assert!(format!("{err}").contains("statement") || format!("{err}").contains("digest"));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_spartan2_decider_target_rejects_tampered_final_proof_digest() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, mut proof) = prove_recursive(&input).expect("prove recursive");
    proof.proof_digest[0] ^= 1;

    let err =
        build_chip8_spartan2_decider_target(&statement, &proof).expect_err("tampered final proof digest must fail");
    assert!(format!("{err}").contains("final proof") || format!("{err}").contains("digest"));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_spartan2_decider_round_trip() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");

    let (pk, vk) = setup_chip8_spartan2_decider(&statement, &proof).expect("setup chip8 spartan2 decider");
    let decider_proof = prove_chip8_spartan2_decider(&pk, &statement, &proof).expect("prove chip8 spartan2 decider");

    verify_chip8_spartan2_decider(&vk, &statement, &proof, &decider_proof).expect("verify chip8 spartan2 decider");
    assert!(decider_proof.snark_bytes_len() > 0);
}
