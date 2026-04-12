//! Focused tests for the owned RV64IM decider relation seam.

use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_decider_relation,
    build_rv64im_decider_relation_from_final_surface, parity_source_cases, prove_rv64im_public_proof,
    verify_rv64im_decider_relation, Rv64imProofInput,
};

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

#[test]
fn rv64im_decider_relation_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let relation = build_rv64im_decider_relation(&proof).expect("build decider relation");

    assert_eq!(relation.public_statement_digest, statement.digest);
    assert_eq!(relation.relation_digest, statement.folded.digest);
    assert_eq!(relation.final_proof_digest, final_proof.proof_digest);
    assert_eq!(relation.fold_schedule, statement.folded.fold_schedule);
    assert_eq!(relation.semantic_step_count, statement.folded.semantic_step_count);
    assert_eq!(relation.chunk_summaries, final_proof.chunk_summaries);
    assert_eq!(relation.base_component_digests, vec![final_proof.kernel_export.digest]);
    assert_eq!(
        relation.chunk_transition_bindings.len(),
        statement.folded.chunk_count as usize
    );
    assert_ne!(relation.digest, [0; 32]);

    verify_rv64im_decider_relation(&relation, &proof).expect("verify decider relation");
}

#[test]
fn rv64im_decider_relation_from_final_surface_matches_verified_relation() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let from_verified = build_rv64im_decider_relation(&proof).expect("build verified decider relation");
    let from_surface =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build surface relation");

    assert_eq!(from_surface, from_verified);
}

#[test]
fn rv64im_decider_relation_rejects_tampered_statement_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let mut relation = build_rv64im_decider_relation(&proof).expect("build decider relation");
    relation.public_statement_digest[0] ^= 1;

    let err = verify_rv64im_decider_relation(&relation, &proof).expect_err("tampered statement digest must fail");
    assert!(format!("{err}").contains("decider relation") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_decider_relation_rejects_tampered_component_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let mut relation = build_rv64im_decider_relation(&proof).expect("build decider relation");
    relation.base_component_digests[0][0] ^= 1;

    let err = verify_rv64im_decider_relation(&relation, &proof).expect_err("tampered component digest must fail");
    assert!(format!("{err}").contains("decider relation") || format!("{err}").contains("digest"));
}
