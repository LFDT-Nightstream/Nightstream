//! Focused RV64IM round-trip over the stricter Spartan2 public-relation shell.

use neo_fold_next::decider::spartan2::{
    prove_spartan2_public_relation_shell, setup_spartan2_public_relation_shell, verify_spartan2_public_relation_shell,
};
use neo_fold_next::nightstream::rv64im::{
    audit::{build_rv64im_hybrid_side_bridge_public_target, Rv64imWitnessBackedSideBridgeStatement},
    build_rv64im_nightstream_from_public_proof,
};
use neo_fold_next::rv64im::{parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput};

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
fn rv64im_hybrid_side_bridge_spartan2_public_relation_shell_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");
    let (pk, vk) =
        setup_spartan2_public_relation_shell(&target.shape()).expect("setup hybrid-side-bridge public-relation shell");
    let shell =
        prove_spartan2_public_relation_shell(&pk, &target).expect("prove hybrid-side-bridge public-relation shell");

    verify_spartan2_public_relation_shell(&vk, &target, &shell)
        .expect("verify hybrid-side-bridge public-relation shell");
    assert!(shell.snark_bytes_len() > 0);
}

#[test]
fn rv64im_hybrid_side_bridge_spartan2_public_relation_shell_rejects_nightstream_side_tamper() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let public_statement = proof.statement.clone();
    let target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &public_statement)
            .expect("build hybrid-side-bridge public target");
    let (pk, vk) =
        setup_spartan2_public_relation_shell(&target.shape()).expect("setup hybrid-side-bridge public-relation shell");
    let shell =
        prove_spartan2_public_relation_shell(&pk, &target).expect("prove hybrid-side-bridge public-relation shell");

    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .statement_core_digest[0] ^= 1;
    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .digest = nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .expected_digest();
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: public_statement.clone(),
        side_bundle_digest: nightstream_proof
            .hybrid_side_bridge_artifact
            .bridge_artifact
            .witness
            .side_bundle
            .digest,
        opening_artifact_digest: nightstream_proof
            .hybrid_side_bridge_artifact
            .bridge_artifact
            .witness
            .opening_artifact
            .digest,
        bridge_handoff_digests: nightstream_proof
            .main_residual_proof
            .bridge_handoff_digests
            .clone(),
    };
    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .digest = nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .expected_digest(bridge_statement.digest());
    nightstream_proof.hybrid_side_bridge_artifact.digest = nightstream_proof
        .hybrid_side_bridge_artifact
        .expected_digest();

    let err =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &public_statement)
            .and_then(|tampered_target| {
                verify_spartan2_public_relation_shell(&vk, &tampered_target, &shell)
                    .map_err(|verify_err| neo_fold_next::rv64im::SimpleKernelError::Bridge(verify_err.to_string()))
            })
            .expect_err("tampered Nightstream side bundle must fail");

    assert!(
        err.to_string()
            .contains("side bundle does not match the carried Nightstream statement core")
            || err
                .to_string()
                .contains("opening artifact does not match the verified compact Phase 0 opening surface"),
        "unexpected error: {err}"
    );
}
