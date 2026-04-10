//! Focused RV64IM round-trip over the shared Spartan2 public-target shell.

use neo_fold_next::decider::spartan2::{
    prove_spartan2_public_target_shell, setup_spartan2_public_target_shell, verify_spartan2_public_target_shell,
};
use neo_fold_next::nightstream::rv64im::{
    audit::{
        build_rv64im_hybrid_side_bridge_public_target, build_rv64im_opening_artifact_from_accepted_artifact,
        build_rv64im_side_proof_bundle_from_accepted_artifact, Rv64imWitnessBackedSideBridgeStatement,
    },
    build_rv64im_nightstream_from_public_proof,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_spartan2_decider_target, parity_source_cases,
    prove_rv64im_public_proof, Rv64imProofInput,
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
fn rv64im_spartan2_public_target_shell_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let target = build_rv64im_spartan2_decider_target(&statement, &final_proof).expect("build rv64im decider target");

    let (pk, vk) = setup_spartan2_public_target_shell(&target.shape()).expect("setup public-target shell");
    let shell = prove_spartan2_public_target_shell(&pk, &target).expect("prove public-target shell");

    verify_spartan2_public_target_shell(&vk, &target, &shell).expect("verify public-target shell");
    assert!(shell.snark_bytes_len() > 0);
}

#[test]
fn rv64im_hybrid_side_bridge_spartan2_public_target_shell_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");
    let (pk, vk) = setup_spartan2_public_target_shell(&target.shape()).expect("setup hybrid-side-bridge shell");
    let shell = prove_spartan2_public_target_shell(&pk, &target).expect("prove hybrid-side-bridge shell");

    verify_spartan2_public_target_shell(&vk, &target, &shell).expect("verify hybrid-side-bridge shell");
    assert!(shell.snark_bytes_len() > 0);
}

#[test]
fn rv64im_hybrid_side_bridge_spartan2_public_target_shell_rejects_unbound_side_bundle() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let unbound_side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&artifact).expect("build unbound side proof bundle");
    let opening_artifact =
        build_rv64im_opening_artifact_from_accepted_artifact(&proof.statement, &unbound_side_bundle, &artifact)
            .expect("build opening artifact for unbound side proof bundle");
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: proof.statement.clone(),
        side_bundle_digest: unbound_side_bundle.digest,
        opening_artifact_digest: opening_artifact.digest,
        bridge_handoff_digests: nightstream_proof
            .main_residual_proof
            .bridge_handoff_digests
            .clone(),
    };
    let mut tampered_proof = nightstream_proof.clone();
    tampered_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle = unbound_side_bundle;
    tampered_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .opening_artifact = opening_artifact;
    tampered_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .digest = tampered_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .expected_digest(bridge_statement.digest());
    tampered_proof.hybrid_side_bridge_artifact.digest = tampered_proof.hybrid_side_bridge_artifact.expected_digest();

    let err = build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &tampered_proof, &proof.statement)
        .expect_err("accepted-artifact side bundle must not satisfy the Nightstream-bound hybrid-side-bridge target");

    assert!(err.to_string().contains("statement core"), "unexpected error: {err}");
}

#[test]
fn rv64im_hybrid_side_bridge_public_target_uses_bridge_artifact_not_duplicate_public_copies() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let expected =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build baseline hybrid-side-bridge public target");

    let mut unrelated_artifacts_tampered = nightstream_proof.clone();
    unrelated_artifacts_tampered.linkage_artifact.digest[0] ^= 1;

    let rebuilt = build_rv64im_hybrid_side_bridge_public_target(
        &nightstream_statement,
        &unrelated_artifacts_tampered,
        &proof.statement,
    )
    .expect("unrelated public artifacts must not affect the bridge-bound public target");

    assert_eq!(rebuilt.digest(), expected.digest());
}

#[test]
fn rv64im_hybrid_side_bridge_public_target_freezes_base_component_layout() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");

    assert_eq!(target.shape().base_component_count, 4);
    assert_eq!(target.shape().chunk_transition_count, 64);
    assert_eq!(target.statement.chunk_summaries.len(), 64);
    for summary in &target.statement.chunk_summaries[proof.statement.chunk_count as usize..] {
        assert_eq!(summary.start_index, nightstream_statement.semantic_step_count);
        assert_eq!(summary.public_step_count, 0);
        assert_eq!(summary.public_chunk_digest, [0; 32]);
        assert_eq!(summary.chunk_relation_digest, [0; 32]);
    }
}

#[test]
fn rv64im_hybrid_side_bridge_public_target_rejects_noncanonical_padded_tail() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let mut target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");
    let padded_index = proof.statement.chunk_count as usize;
    target.statement.chunk_summaries[padded_index].public_chunk_digest[0] ^= 1;

    let (pk, _) = setup_spartan2_public_target_shell(&target.shape()).expect("setup hybrid-side-bridge shell");
    let err = prove_spartan2_public_target_shell(&pk, &target)
        .expect_err("noncanonical padded tails must not satisfy the fixed-shape hybrid-side-bridge target");

    assert!(
        err.to_string().contains("padded chunk summary"),
        "unexpected error: {err}"
    );
}
