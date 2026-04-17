//! Focused tests for the surviving RV64IM decider relation surface.

use neo_fold_next::rv64im::audit::{
    build_rv64im_decider_relation_from_final_surface, validate_rv64im_decider_relation_surface,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
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
fn rv64im_decider_relation_surface_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let relation =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build decider relation");

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

    validate_rv64im_decider_relation_surface(&relation).expect("validate decider relation surface");
}

#[test]
fn rv64im_decider_relation_surface_rejects_tampered_chunk_summary() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut relation =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build decider relation");
    relation.chunk_summaries[0].public_step_count += 1;

    let err = validate_rv64im_decider_relation_surface(&relation)
        .expect_err("tampered chunk summary must fail decider relation surface validation");
    assert!(format!("{err}").contains("chunk") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_decider_relation_surface_rejects_tampered_transition_binding() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut relation =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build decider relation");
    relation.chunk_transition_bindings[0].claimed_chunk_relation_digest[0] ^= 1;

    let err = validate_rv64im_decider_relation_surface(&relation)
        .expect_err("tampered transition binding must fail decider relation surface validation");
    assert!(format!("{err}").contains("chunk") || format!("{err}").contains("digest"));
}
