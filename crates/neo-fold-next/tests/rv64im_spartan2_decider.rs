//! Focused tests for the RV64IM adapter into the generic Spartan2 decider target.

use neo_fold_next::decider::spartan2::{
    prove_spartan2_decider, setup_spartan2_decider, verify_spartan2_decider, Spartan2BackendBindingShellError,
    Spartan2DeciderError,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_decider_relation, build_rv64im_spartan2_decider_target,
    parity_source_cases, prove_rv64im_public_proof, prove_rv64im_spartan2_decider,
    prove_rv64im_spartan2_decider_from_public_proof, setup_rv64im_spartan2_decider,
    setup_rv64im_spartan2_decider_from_public_proof, verify_rv64im_spartan2_decider,
    verify_rv64im_spartan2_decider_from_public_proof, Rv64imProofInput,
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
fn rv64im_spartan2_decider_target_projects_decider_relation_seam() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let relation = build_rv64im_decider_relation(&proof).expect("build decider relation");

    let target = build_rv64im_spartan2_decider_target(&statement, &final_proof).expect("build decider target");

    assert_eq!(
        target.statement.public_statement_digest,
        relation.public_statement_digest
    );
    assert_eq!(target.statement.relation_digest, relation.relation_digest);
    assert_eq!(target.statement.fold_schedule, statement.folded.fold_schedule);
    assert_eq!(
        target.statement.semantic_step_count,
        statement.folded.semantic_step_count
    );
    assert_eq!(target.statement.chunk_summaries, relation.chunk_summaries);
    assert_eq!(
        target.statement.chunk_summaries.len(),
        statement.folded.chunk_count as usize
    );
    assert_eq!(target.witness.base_component_digests.len(), 1);
    assert_eq!(
        target.witness.chunk_transition_bindings.len(),
        statement.folded.chunk_count as usize
    );
    assert_eq!(target.witness.base_component_digests, relation.base_component_digests);
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
    assert!(!target.witness.chunk_transition_bindings.is_empty());
    assert!(!target.statement.public_io().is_empty());
    assert_eq!(target.statement.final_proof_digest, relation.final_proof_digest);
    assert_ne!(target.statement.digest(), [0; 32]);
    assert_ne!(target.witness.digest(), [0; 32]);
    assert_ne!(target.digest(), [0; 32]);
}

#[test]
fn rv64im_spartan2_decider_target_rejects_tampered_statement_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    statement.digest[0] ^= 1;

    let err = build_rv64im_spartan2_decider_target(&statement, &final_proof)
        .expect_err("tampered statement digest must fail");
    assert!(format!("{err}").contains("statement") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_spartan2_decider_target_rejects_tampered_relation_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (mut statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    statement.folded.digest[0] ^= 1;

    let err =
        build_rv64im_spartan2_decider_target(&statement, &final_proof).expect_err("tampered relation digest must fail");
    assert!(format!("{err}").contains("relation") || format!("{err}").contains("digest"));
}

#[test]
fn rv64im_spartan2_decider_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let (pk, vk) = setup_rv64im_spartan2_decider(&statement, &final_proof).expect("setup rv64im spartan2 decider");
    let decider_proof =
        prove_rv64im_spartan2_decider(&pk, &statement, &final_proof).expect("prove rv64im spartan2 decider");

    verify_rv64im_spartan2_decider(&vk, &statement, &final_proof, &decider_proof)
        .expect("verify rv64im spartan2 decider");
    assert!(decider_proof.snark_bytes_len() > 0);
}

#[test]
fn rv64im_spartan2_decider_from_public_proof_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");

    let (pk, vk) =
        setup_rv64im_spartan2_decider_from_public_proof(&proof).expect("setup rv64im spartan2 decider from proof");
    let decider_proof =
        prove_rv64im_spartan2_decider_from_public_proof(&pk, &proof).expect("prove rv64im spartan2 decider from proof");

    verify_rv64im_spartan2_decider_from_public_proof(&vk, &proof, &decider_proof)
        .expect("verify rv64im spartan2 decider from proof");
    assert!(decider_proof.snark_bytes_len() > 0);
}

#[test]
fn rv64im_spartan2_decider_rejects_tampered_base_component_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let target = build_rv64im_spartan2_decider_target(&statement, &final_proof).expect("build decider target");

    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup generic spartan2 decider");
    let decider_proof = prove_spartan2_decider(&pk, &target).expect("prove generic spartan2 decider");

    let mut tampered_target = target.clone();
    tampered_target.witness.base_component_digests[0][0] ^= 1;
    let err = verify_spartan2_decider(&vk, &tampered_target, &decider_proof)
        .expect_err("tampered base component digest must fail");
    assert!(matches!(
        err,
        Spartan2DeciderError::RelationDigestMismatch
            | Spartan2DeciderError::FinalProofDigestMismatch
            | Spartan2DeciderError::RelationSurface(_)
            | Spartan2DeciderError::Backend(Spartan2BackendBindingShellError::RelationSurface(_))
            | Spartan2DeciderError::Backend(Spartan2BackendBindingShellError::PublicIoMismatch)
            | Spartan2DeciderError::Backend(Spartan2BackendBindingShellError::Verify(_))
    ));
}

#[test]
fn rv64im_spartan2_decider_rejects_tampered_chunk_transition_binding() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let target = build_rv64im_spartan2_decider_target(&statement, &final_proof).expect("build decider target");

    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup generic spartan2 decider");
    let decider_proof = prove_spartan2_decider(&pk, &target).expect("prove generic spartan2 decider");

    let mut tampered_target = target.clone();
    tampered_target.witness.chunk_transition_bindings[0].transition_witness_digest[0] ^= 1;
    let err = verify_spartan2_decider(&vk, &tampered_target, &decider_proof)
        .expect_err("tampered chunk transition binding must fail");
    assert!(matches!(
        err,
        Spartan2DeciderError::RelationDigestMismatch
            | Spartan2DeciderError::FinalProofDigestMismatch
            | Spartan2DeciderError::RelationSurface(_)
            | Spartan2DeciderError::Backend(Spartan2BackendBindingShellError::RelationSurface(_))
            | Spartan2DeciderError::Backend(Spartan2BackendBindingShellError::PublicIoMismatch)
            | Spartan2DeciderError::Backend(Spartan2BackendBindingShellError::Verify(_))
    ));
}
