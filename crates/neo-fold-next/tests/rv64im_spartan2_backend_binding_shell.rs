//! Focused RV64IM round-trip over the stricter Spartan2 backend-binding shell.

use neo_fold_next::decider::spartan2::{
    prove_spartan2_backend_binding_shell, setup_spartan2_backend_binding_shell, verify_spartan2_backend_binding_shell,
    Spartan2BackendBindingShellProof,
};
use neo_fold_next::nightstream::rv64im::{
    audit::{build_rv64im_hybrid_side_bridge_public_target, Rv64imWitnessBackedSideBridgeStatement},
    build_rv64im_nightstream_from_public_proof, Rv64imHybridSideBridgeBackendProof,
};
use neo_fold_next::rv64im::{parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput};

fn rebind_tampered_public_statement(
    nightstream_statement: &mut neo_fold_next::nightstream::NightstreamStatement,
    nightstream_proof: &mut neo_fold_next::nightstream::rv64im::Rv64imNightstreamProof,
    public_statement: &neo_fold_next::rv64im::Rv64imProofStatement,
) {
    nightstream_statement.public_io_digest = public_statement.digest;
    let bridge_artifact = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact;
    bridge_artifact.witness.side_bundle.statement_core_digest = nightstream_statement.core_digest();
    bridge_artifact.witness.side_bundle.digest = bridge_artifact.witness.side_bundle.expected_digest();
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: public_statement.clone(),
    };
    bridge_artifact.digest = bridge_artifact.expected_digest(bridge_statement.digest());
    nightstream_proof.hybrid_side_bridge_artifact.digest = nightstream_proof
        .hybrid_side_bridge_artifact
        .expected_digest();
}

fn rebind_tampered_side_bundle(
    nightstream_statement: &neo_fold_next::nightstream::NightstreamStatement,
    nightstream_proof: &mut neo_fold_next::nightstream::rv64im::Rv64imNightstreamProof,
    public_statement: &neo_fold_next::rv64im::Rv64imProofStatement,
) {
    let bridge_artifact = &mut nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact;
    bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_bridge
        .digest = bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_bridge
        .expected_digest();
    bridge_artifact.witness.side_bundle.digest = bridge_artifact.witness.side_bundle.expected_digest();
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        public_statement: public_statement.clone(),
    };
    bridge_artifact.digest = bridge_artifact.expected_digest(bridge_statement.digest());
    nightstream_proof.hybrid_side_bridge_artifact.digest = nightstream_proof
        .hybrid_side_bridge_artifact
        .expected_digest();
}

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
fn rv64im_hybrid_side_bridge_spartan2_backend_binding_shell_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");
    let relation = target.backend_relation();
    let (pk, vk) = setup_spartan2_backend_binding_shell(&relation.shape())
        .expect("setup hybrid-side-bridge backend-binding shell");
    let shell =
        prove_spartan2_backend_binding_shell(&pk, &relation).expect("prove hybrid-side-bridge backend-binding shell");

    verify_spartan2_backend_binding_shell(&vk, &relation, &shell)
        .expect("verify hybrid-side-bridge backend-binding shell");
    assert!(shell.snark_bytes_len() > 0);
}

#[test]
fn rv64im_hybrid_side_bridge_spartan2_backend_binding_shell_rejects_private_base_digest_tamper() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let relation =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target")
            .backend_relation();
    let (pk, vk) = setup_spartan2_backend_binding_shell(&relation.shape())
        .expect("setup hybrid-side-bridge backend-binding shell");
    let shell =
        prove_spartan2_backend_binding_shell(&pk, &relation).expect("prove hybrid-side-bridge backend-binding shell");

    let mut tampered = relation;
    tampered.witness.base_component_digests[0][0] ^= 1;

    let err = verify_spartan2_backend_binding_shell(&vk, &tampered, &shell)
        .expect_err("tampered backend witness digest must fail");

    assert!(
        err.to_string()
            .contains("public final proof digest does not match the carried fixed-shape backend relation"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_backend_proof_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");
    let relation = target.backend_relation();
    let (pk, vk) =
        setup_spartan2_backend_binding_shell(&relation.shape()).expect("setup hybrid-side-bridge backend proof");
    let shell = prove_spartan2_backend_binding_shell(&pk, &relation).expect("prove hybrid-side-bridge backend proof");
    let backend_proof = Rv64imHybridSideBridgeBackendProof {
        snark_data: shell.snark_data.clone(),
    };
    let backend_shell = Spartan2BackendBindingShellProof {
        snark_data: backend_proof.snark_data.clone(),
    };

    verify_spartan2_backend_binding_shell(&vk, &relation, &backend_shell)
        .expect("verify hybrid-side-bridge backend proof");
    assert!(backend_proof.snark_bytes_len() > 0);
    assert_ne!(backend_proof.digest(), [0; 32]);
}

#[test]
fn rv64im_hybrid_side_bridge_backend_relation_rejects_recomputed_public_statement_with_wrong_stage_package_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (mut nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let mut public_statement = proof.statement.clone();

    public_statement.stage_packages_digest[0] ^= 1;
    public_statement.digest = public_statement.recompute_digest();
    rebind_tampered_public_statement(&mut nightstream_statement, &mut nightstream_proof, &public_statement);

    let err =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &public_statement)
            .expect_err("recomputed public statement with wrong compact stage-package digest must fail");

    assert!(
        err.to_string().contains(
            "RV64IM hybrid-side-bridge decider relation compact stage-package proof surface does not match the carried RV64IM public statement"
        ) || err
            .to_string()
            .contains("opening artifact does not match the verified compact Phase 0 opening surface"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_backend_relation_rejects_recomputed_public_statement_with_wrong_stage_claim_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (mut nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let mut public_statement = proof.statement.clone();

    public_statement.stage_claims_digest[0] ^= 1;
    public_statement.digest = public_statement.recompute_digest();
    rebind_tampered_public_statement(&mut nightstream_statement, &mut nightstream_proof, &public_statement);

    let err =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &public_statement)
            .expect_err("recomputed public statement with wrong compact stage-claim digest must fail");

    assert!(
        err.to_string().contains(
            "RV64IM hybrid-side-bridge decider relation compact stage-claim proof surface does not match the carried RV64IM public statement"
        ) || err
            .to_string()
            .contains("opening artifact does not match the verified compact Phase 0 opening surface"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv64im_hybrid_side_bridge_backend_relation_rejects_rebound_side_bundle_with_wrong_root0_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let (nightstream_statement, mut nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let public_statement = proof.statement.clone();

    nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_bridge
        .root0_digest[0] ^= 1;
    rebind_tampered_side_bundle(&nightstream_statement, &mut nightstream_proof, &public_statement);

    let err =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &public_statement)
            .expect_err("rebound side bundle with wrong root0 digest must fail");

    assert!(
        err.to_string().contains(
            "RV64IM hybrid-side-bridge decider relation compact kernel-claim proof surface does not match the carried RV64IM public statement"
        ) || err
            .to_string()
            .contains("opening artifact does not match the verified compact Phase 0 opening surface"),
        "unexpected error: {err}"
    );
}
