use std::fmt::Debug;

use neo_fold_next::decider::spartan2::{
    setup_spartan2_decider, Spartan2DeciderProof, Spartan2DeciderShape, Spartan2DeciderVerifierKey,
};
use neo_fold_next::nightstream::rv64im::{
    audit::Rv64imWitnessBackedSideBridgeStatement, build_rv64im_main_decider_proof, build_rv64im_main_residual_proof,
    build_rv64im_nightstream_decider_target, build_rv64im_nightstream_from_public_proof,
    build_rv64im_nightstream_linkage_claims, build_rv64im_nightstream_linkage_claims_from_relation,
    build_rv64im_nightstream_statement_from_final, rv64im_nightstream_decider_target_digest,
    rv64im_nightstream_linkage_root, rv64im_verifier_context_digest, verify_rv64im_main_decider_proof,
    verify_rv64im_main_residual_proof, verify_rv64im_nightstream, Rv64imNightstreamProof,
};
use neo_fold_next::nightstream::{
    nightstream_proof_binding_root, nightstream_statement_digest, NightstreamProofBindingInputs, NightstreamStatement,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_spartan2_decider_target, parity_source_cases,
    prove_rv64im_public_proof, prove_rv64im_spartan2_decider_from_public_proof,
    setup_rv64im_spartan2_decider_from_public_proof, Rv64imProofInput, Rv64imProofStatement, SimpleKernelError,
};
use serde::{de::DeserializeOwned, Serialize};

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn alternate_case_name(exclude: &str) -> String {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name != exclude)
        .unwrap_or_else(|| panic!("missing alternate parity source case for {exclude}"))
        .manifest
        .name
        .to_string()
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

struct ExternalNightstreamFixture {
    trusted_root_params_id: [u8; 32],
    public_statement: Rv64imProofStatement,
    decider_vk: Spartan2DeciderVerifierKey,
    decider_proof: Spartan2DeciderProof,
    statement: NightstreamStatement,
    nightstream_proof: Rv64imNightstreamProof,
}

fn external_fixture(name: &str) -> ExternalNightstreamFixture {
    let public_proof = prove_rv64im_public_proof(&proof_input(name)).expect("prove rv64im public proof");
    let trusted_root_params_id = public_proof.statement.root_params_id;
    let public_statement = public_proof.statement.clone();
    let (decider_pk, decider_vk) =
        setup_rv64im_spartan2_decider_from_public_proof(&public_proof).expect("setup rv64im spartan2 decider");
    let decider_proof = prove_rv64im_spartan2_decider_from_public_proof(&decider_pk, &public_proof)
        .expect("prove rv64im spartan2 decider");
    let (statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    ExternalNightstreamFixture {
        trusted_root_params_id,
        public_statement,
        decider_vk,
        decider_proof,
        statement,
        nightstream_proof,
    }
}

fn verify_fixture(fixture: &ExternalNightstreamFixture) -> Result<(), SimpleKernelError> {
    verify_rv64im_nightstream(
        &fixture.statement,
        &fixture.nightstream_proof,
        fixture.trusted_root_params_id,
        &fixture.decider_vk,
        &fixture.decider_proof,
        &fixture.public_statement,
    )
}

fn rebind_statement(statement: &mut NightstreamStatement, proof: &Rv64imNightstreamProof) {
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: proof.main_decider_proof.expected_digest(),
        main_residual_proof_digest: proof.main_residual_proof.expected_digest(),
        side_bridge_artifact_digest: proof.hybrid_side_bridge_artifact.digest,
        linkage_artifact_digest: proof.linkage_artifact.digest,
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &proof_binding_inputs);
}

fn rebind_bridge_artifact_digest(fixture: &mut ExternalNightstreamFixture) {
    let bridge_artifact = &mut fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact;
    bridge_artifact.witness.side_bundle.digest = bridge_artifact.witness.side_bundle.expected_digest();
    bridge_artifact.witness.opening_artifact.digest = bridge_artifact.witness.opening_artifact.expected_digest();
    let bridge_statement = Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: fixture.statement.clone(),
        public_statement: fixture.public_statement.clone(),
        side_bundle_digest: bridge_artifact.witness.side_bundle.digest,
        opening_artifact_digest: bridge_artifact.witness.opening_artifact.digest,
        bridge_handoff_digests: fixture
            .nightstream_proof
            .main_residual_proof
            .bridge_handoff_digests
            .clone(),
    };
    bridge_artifact.digest = bridge_artifact.expected_digest(bridge_statement.digest());
    fixture.nightstream_proof.hybrid_side_bridge_artifact.digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .expected_digest();
}

fn tamper_snark_bytes(snark_data: &mut Vec<u8>) {
    if let Some(first) = snark_data.first_mut() {
        *first ^= 1;
    } else {
        snark_data.push(1);
    }
}

#[test]
fn rv64im_nightstream_rejects_stale_stage1_claim_digest_inside_side_bundle() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .stage1
        .claim
        .points
        .first
        .value_digest[0] ^= 1;
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("tampered stage1 selected-opening claim must fail");
    assert!(
        format!("{err}").contains("stage1 selected-opening claim digest mismatch")
            || format!("{err}").contains("witness-backed side bridge")
    );
}

fn assert_bincode_roundtrip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + Eq + Debug,
{
    let bytes = bincode::serialize(value).expect("serialize roundtrip value");
    let decoded: T = bincode::deserialize(&bytes).expect("deserialize roundtrip value");
    assert_eq!(decoded, *value);
}

#[test]
fn nightstream_statement_digest_tracks_binding_root() {
    let mut statement = NightstreamStatement {
        public_io_digest: [1; 32],
        verifier_context_digest: [2; 32],
        fold_schedule: neo_fold_next::proof::FoldSchedule::WholeTrace,
        semantic_step_count: 7,
        chunk_summaries: Vec::new(),
        linkage_root: [3; 32],
        proof_binding_root: [0; 32],
    };
    let inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: [4; 32],
        main_residual_proof_digest: [5; 32],
        side_bridge_artifact_digest: [6; 32],
        linkage_artifact_digest: [9; 32],
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &inputs);
    let digest_before = nightstream_statement_digest(&statement);
    statement.proof_binding_root[0] ^= 1;
    let digest_after = nightstream_statement_digest(&statement);
    assert_ne!(digest_before, digest_after);
}

#[test]
fn rv64im_nightstream_decider_target_bridge_matches_current_target() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let existing = build_rv64im_spartan2_decider_target(&statement, &final_proof).expect("existing target");
    let bridged = build_rv64im_nightstream_decider_target(&statement, &final_proof).expect("nightstream target");

    assert_eq!(existing, bridged);
    assert_eq!(
        rv64im_nightstream_decider_target_digest(&statement, &final_proof).expect("target digest"),
        existing.digest(),
    );
}

#[test]
fn rv64im_nightstream_linkage_and_residual_follow_verified_final_seam() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let linkage_claims =
        build_rv64im_nightstream_linkage_claims(&statement, &final_proof).expect("build linkage claims");
    assert_eq!(
        linkage_claims.public_chunk_digests.len(),
        statement.folded.chunk_count as usize
    );
    assert_eq!(
        linkage_claims.bridge_handoff_digests.len(),
        statement.folded.chunk_count as usize
    );
    assert_ne!(linkage_claims.digest, [0; 32]);
    assert_ne!(
        rv64im_nightstream_linkage_root(final_proof.kernel_export.digest, &linkage_claims),
        [0; 32]
    );

    let residual = build_rv64im_main_residual_proof(&statement, &final_proof).expect("build residual proof");
    verify_rv64im_main_residual_proof(&statement, &final_proof, &residual).expect("verify residual proof");

    let main_decider_proof =
        build_rv64im_main_decider_proof(&statement, &final_proof).expect("build main decider proof");
    verify_rv64im_main_decider_proof(&statement, &final_proof, &main_decider_proof).expect("verify main decider proof");

    let verifier_context = rv64im_verifier_context_digest(proof.statement.root_params_id);
    assert_ne!(verifier_context, [0; 32]);
}

#[test]
fn rv64im_nightstream_linkage_claims_from_relation_match_verified_final_seam() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let from_final = build_rv64im_nightstream_linkage_claims(&statement, &final_proof).expect("build linkage claims");
    let residual = build_rv64im_main_residual_proof(&statement, &final_proof).expect("build residual proof");
    let from_relation = build_rv64im_nightstream_linkage_claims_from_relation(
        &residual.decider_relation,
        &residual.bridge_handoff_digests,
    )
    .expect("build linkage claims from carried relation");

    assert_eq!(from_relation, from_final);
}

#[test]
fn rv64im_main_residual_and_decider_relation_public_statement_digests_are_distinct_surfaces() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let residual = build_rv64im_main_residual_proof(&final_statement, &final_proof).expect("build main residual proof");

    assert_eq!(
        residual.public_statement_digest,
        final_statement.public_statement_digest
    );
    assert_eq!(
        residual.decider_relation.public_statement_digest,
        final_statement.digest
    );
    assert_ne!(
        residual.public_statement_digest,
        residual.decider_relation.public_statement_digest
    );
}

#[test]
fn rv64im_main_decider_proof_rejects_tampered_target_digest() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut main_decider_proof =
        build_rv64im_main_decider_proof(&statement, &final_proof).expect("build main decider proof");
    main_decider_proof.decider_target_digest[0] ^= 1;
    let err = verify_rv64im_main_decider_proof(&statement, &final_proof, &main_decider_proof)
        .expect_err("tampered main decider proof must fail");
    assert!(format!("{err}").contains("main decider proof"));
}

#[test]
fn rv64im_nightstream_statement_projects_verified_final_seam() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let verifier_context = rv64im_verifier_context_digest(proof.statement.root_params_id);
    let linkage_claims =
        build_rv64im_nightstream_linkage_claims(&statement, &final_proof).expect("build linkage claims");
    let linkage_root = rv64im_nightstream_linkage_root(final_proof.kernel_export.digest, &linkage_claims);

    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: [9; 32],
        main_residual_proof_digest: [10; 32],
        side_bridge_artifact_digest: [11; 32],
        linkage_artifact_digest: linkage_claims.digest,
    };
    let mut nightstream = build_rv64im_nightstream_statement_from_final(
        statement.public_statement_digest,
        verifier_context,
        &statement,
        &final_proof,
        linkage_root,
        [0; 32],
    )
    .expect("build nightstream statement");
    nightstream.proof_binding_root = nightstream_proof_binding_root(nightstream.core_digest(), &proof_binding_inputs);

    assert_eq!(nightstream.fold_schedule, statement.folded.fold_schedule);
    assert_eq!(nightstream.semantic_step_count, statement.folded.semantic_step_count);
    assert_eq!(nightstream.chunk_summaries, final_proof.chunk_summaries);
    assert_ne!(nightstream.digest(), [0; 32]);
}

#[test]
fn rv64im_nightstream_round_trips_against_current_public_proof_seam() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    verify_fixture(&fixture).expect("verify nightstream proof");
}

#[test]
fn rv64im_nightstream_rejects_tampered_statement_binding_root() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.statement.proof_binding_root[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered statement binding must fail");
    assert!(format!("{err}").contains("Nightstream statement"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_statement_verifier_context_with_rebound_binding() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.statement.verifier_context_digest[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .statement_core_digest = fixture.statement.core_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);
    let err = verify_fixture(&fixture).expect_err("tampered verifier context must fail");
    assert!(format!("{err}").contains("verifier context"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_statement_public_io_digest_with_rebound_binding() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.statement.public_io_digest[0] ^= 1;
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);
    let err = verify_fixture(&fixture).expect_err("tampered public io digest must fail");
    assert!(format!("{err}").contains("Nightstream statement") || format!("{err}").contains("public IO"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_hybrid_side_bridge_artifact() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.nightstream_proof.hybrid_side_bridge_artifact.digest[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered side-terminal artifact must fail");
    assert!(
        format!("{err}").contains("side-terminal proof artifact")
            || format!("{err}").contains("verified final seam")
            || format!("{err}").contains("Nightstream statement")
    );
}

#[test]
fn rv64im_nightstream_rejects_tampered_opening_artifact() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .opening_artifact
        .digest[0] ^= 1;
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);
    let err = verify_fixture(&fixture).expect_err("tampered opening artifact must fail");
    assert!(
        format!("{err}").contains("opening artifact")
            || format!("{err}").contains("witness-backed side bridge artifact statement digest")
    );
}

#[test]
fn rv64im_nightstream_rejects_opening_artifact_with_tampered_bundle() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .opening_artifact
        .convergence_artifact
        .digest[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("opening artifact with a tampered compact artifact must fail");
    assert!(format!("{err}").contains("opening"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_side_bundle_stage2_digest() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .stage2
        .digest[0] ^= 1;
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("nightstream proof with a tampered carried stage2 digest must fail");
    assert!(
        format!("{err}").contains("stage2 verified-claims digest mismatch")
            || format!("{err}").contains("witness-backed side bridge")
    );
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_root_execution_surface_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .semantic_rows_digest[0] ^= 1;
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("side root-execution summary tamper with rebound binding must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_kernel_opening_surface_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .bindings_opening_digest[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .expected_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side kernel-opening surface tamper must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_kernel_opening_binding_summary_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .last_binding_digest
        .as_mut()
        .expect("last binding digest")[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .expected_digest();
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .expected_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side kernel-opening binding summary tamper must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_kernel_opening_root_lane_summary_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .root_lane_commitment
        .last_selected_row
        .as_mut()
        .expect("last selected row")
        .value_digest[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .expected_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side kernel-opening root-lane summary tamper must fail");
    assert!(
        format!("{err}").contains("kernel-opening proof surface")
            || format!("{err}").contains("root-lane commitment summary")
            || format!("{err}").contains("witness-backed side bridge")
    );
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_stage_packages_surface_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .stage1
        .packaged_digest[0] ^= 1;
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side stage-package surface tamper must fail");
    assert!(
        format!("{err}").contains("stage-package proof surface")
            || format!("{err}").contains("stage1 verified-claims digest mismatch")
    );
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_stage_claim_proof_surface_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .stage_claim_proof_bridge
        .packaged_proof_digest[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .stage_claim_proof_bridge
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .stage_claim_proof_bridge
        .expected_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side stage-claim proof surface tamper must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_kernel_claim_proof_surface_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_proof_bridge
        .packaged_proof_digest[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_proof_bridge
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_proof_bridge
        .expected_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side kernel-claim proof surface tamper must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_side_kernel_claim_bridge_cross_instance_swap() {
    let mut primary = external_fixture("control_flow_jal_skip_ecall");
    let alternate = external_fixture(&alternate_case_name("control_flow_jal_skip_ecall"));

    primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_bridge = alternate
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_claim_bridge
        .clone();
    rebind_bridge_artifact_digest(&mut primary);
    rebind_statement(&mut primary.statement, &primary.nightstream_proof);

    let err = verify_fixture(&primary).expect_err("cross-instance kernel-claim bridge swap must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_side_kernel_export_bridge_cross_instance_swap() {
    let mut primary = external_fixture("control_flow_jal_skip_ecall");
    let alternate_name = alternate_case_name("control_flow_jal_skip_ecall");
    let alternate = external_fixture(&alternate_name);

    primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_export_bridge = alternate
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_export_bridge
        .clone();
    rebind_bridge_artifact_digest(&mut primary);
    rebind_statement(&mut primary.statement, &primary.nightstream_proof);

    let err = verify_fixture(&primary).expect_err("cross-instance kernel-export bridge swap must fail");
    assert!(!format!("{err}").is_empty());
}

#[test]
fn rv64im_nightstream_rejects_rebound_side_main_lane_surface_tamper() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .root_lane_commitment
        .first_selected_row
        .as_mut()
        .expect("fixture must carry the first selected row")
        .digest[0] ^= 1;
    fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .digest = fixture
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .kernel_opening_bridge
        .expected_digest();
    rebind_bridge_artifact_digest(&mut fixture);
    rebind_statement(&mut fixture.statement, &fixture.nightstream_proof);

    let err = verify_fixture(&fixture).expect_err("rebound side main-lane surface tamper must fail");
    assert!(
        format!("{err}").contains("main-lane proof surface")
            || format!("{err}").contains("root-lane commitment summary")
            || format!("{err}").contains("witness-backed side bridge")
    );
}

#[test]
fn rv64im_nightstream_rejects_tampered_main_decider_proof() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .main_decider_proof
        .decider_target_digest[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered main decider proof must fail");
    assert!(format!("{err}").contains("main decider proof"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_main_residual_proof() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture
        .nightstream_proof
        .main_residual_proof
        .decider_relation
        .digest[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered main residual proof must fail");
    assert!(format!("{err}").contains("main residual proof"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_linkage_artifact() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.nightstream_proof.linkage_artifact.digest[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered linkage artifact must fail");
    assert!(format!("{err}").contains("linkage artifact"));
}

/// Consolidated anchor test for the four artifact digests that feed
/// `NightstreamProofBindingInputs` inside `verify_rv64im_nightstream_carried_boundary`.
///
/// Walks the surviving `NightstreamProofBindingInputs` fields in one place and proves
/// each one is enforced by the full nightstream verifier. Empirically verified
/// by gutting `verify_rv64im_nightstream_carried_boundary` to `Ok(())`. The
/// carried-boundary rebuild still owns the published proof-binding root, even
/// when downstream verifiers also independently constrain some of these fields.
#[test]
fn rv64im_nightstream_carried_boundary_rejects_each_tampered_proof_binding_input() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    verify_fixture(&fixture).expect("baseline fixture must verify before running the anchor sweep");

    let verify_mutated = |proof: Rv64imNightstreamProof, label: &str| {
        verify_rv64im_nightstream(
            &fixture.statement,
            &proof,
            fixture.trusted_root_params_id,
            &fixture.decider_vk,
            &fixture.decider_proof,
            &fixture.public_statement,
        )
        .err()
        .unwrap_or_else(|| panic!("tampered {label} must be rejected by carried-boundary anchor"));
    };

    // 1. main_decider_proof — *exclusively* anchored by the Step 2
    //    rebuild-and-compare inside `verify_rv64im_nightstream_carried_boundary`.
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.main_decider_proof.decider_target_digest[0] ^= 1;
        verify_mutated(proof, "main_decider_proof");
    }

    // 2. main_residual_proof — anchored by the carried-boundary rebuild and by
    //    the downstream `public_statement_digest` equality check. Tampering the
    //    public-statement digest forces the carried-boundary verifier to rebuild
    //    an expected statement that diverges from the caller-supplied one.
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.main_residual_proof.public_statement_digest[0] ^= 1;
        verify_mutated(proof, "main_residual_proof");
    }

    // 3. hybrid_side_bridge_artifact — anchored by the final
    //    `expected_statement != statement` rebuild-and-compare over
    //    `proof_binding_root` inside the carried-boundary verifier (and by the
    //    downstream `verify_rv64im_hybrid_side_bridge_artifact` surface check).
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.hybrid_side_bridge_artifact.digest[0] ^= 1;
        verify_mutated(proof, "hybrid_side_bridge_artifact");
    }

    // 4. linkage_artifact — *exclusively* anchored by
    //    `verify_rv64im_linkage_artifact_from_claims` inside the carried-boundary
    //    verifier (self-consistency against the linkage claims rebuilt from the
    //    carried decider relation and bridge handoffs).
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.linkage_artifact.digest[0] ^= 1;
        verify_mutated(proof, "linkage_artifact");
    }
}

#[test]
fn rv64im_nightstream_rejects_tampered_spartan_snark_data() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    tamper_snark_bytes(&mut fixture.decider_proof.snark_data);
    let err = verify_fixture(&fixture).expect_err("tampered Spartan proof bytes must fail");
    assert!(format!("{err}").contains("Spartan proof"));
}

#[test]
fn rv64im_nightstream_rejects_tampered_spartan_shape_digest() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.decider_proof.shape_digest[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered Spartan proof shape digest must fail");
    assert!(format!("{err}").contains("Spartan proof"));
}

#[test]
fn rv64im_nightstream_rejects_wrong_spartan_verifier_key_shape() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    let target_shape = fixture
        .nightstream_proof
        .main_residual_proof
        .decider_relation
        .target()
        .shape();
    let wrong_shape = Spartan2DeciderShape {
        base_component_count: target_shape.base_component_count + 1,
        chunk_transition_count: target_shape.chunk_transition_count,
    };
    let (_, wrong_vk) = setup_spartan2_decider(&wrong_shape).expect("setup wrong-shape spartan2 decider");
    let err = verify_rv64im_nightstream(
        &fixture.statement,
        &fixture.nightstream_proof,
        fixture.trusted_root_params_id,
        &wrong_vk,
        &fixture.decider_proof,
        &fixture.public_statement,
    )
    .expect_err("wrong Spartan verifier key must fail");
    assert!(format!("{err}").contains("Spartan proof"));
}

#[test]
fn rv64im_nightstream_rejects_public_statement_with_stale_digest() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.public_statement.final_pc += 1;
    let err = verify_fixture(&fixture).expect_err("stale public statement digest must fail");
    assert!(format!("{err}").contains("public statement"));
}

#[test]
fn rv64im_nightstream_rejects_public_statement_with_recomputed_digest_but_wrong_fields() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.public_statement.final_pc += 1;
    fixture.public_statement.digest = fixture.public_statement.recompute_digest();
    let err = verify_fixture(&fixture).expect_err("wrong public statement fields with a recomputed digest must fail");
    assert!(format!("{err}").contains("public statement"));
}

#[test]
fn rv64im_nightstream_rejects_side_opening_cross_instance_swap() {
    let mut primary = external_fixture("control_flow_jal_skip_ecall");
    let alternate_name = alternate_case_name("control_flow_jal_skip_ecall");
    let alternate = external_fixture(&alternate_name);

    primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle = alternate
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .side_bundle
        .clone();
    primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .opening_artifact = alternate
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .witness
        .opening_artifact
        .clone();
    primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .digest = primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .bridge_artifact
        .expected_digest(
            Rv64imWitnessBackedSideBridgeStatement {
                nightstream_statement: primary.statement.clone(),
                public_statement: primary.public_statement.clone(),
                side_bundle_digest: primary
                    .nightstream_proof
                    .hybrid_side_bridge_artifact
                    .bridge_artifact
                    .witness
                    .side_bundle
                    .digest,
                opening_artifact_digest: primary
                    .nightstream_proof
                    .hybrid_side_bridge_artifact
                    .bridge_artifact
                    .witness
                    .opening_artifact
                    .digest,
                bridge_handoff_digests: primary
                    .nightstream_proof
                    .main_residual_proof
                    .bridge_handoff_digests
                    .clone(),
            }
            .digest(),
        );
    primary.nightstream_proof.hybrid_side_bridge_artifact.digest = primary
        .nightstream_proof
        .hybrid_side_bridge_artifact
        .expected_digest();
    rebind_statement(&mut primary.statement, &primary.nightstream_proof);

    let err = verify_fixture(&primary).expect_err("cross-instance side/opening swap must fail");
    assert!(
        format!("{err}").contains("opening artifact")
            || format!("{err}").contains("side-opening relation")
            || format!("{err}").contains("witness-backed side bridge")
    );
}

#[test]
fn rv64im_nightstream_serde_roundtrips_statement_proof_and_spartan_proof() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    assert_eq!(fixture.decider_vk.shape_digest(), fixture.decider_proof.shape_digest);
    assert_bincode_roundtrip(&fixture.statement);
    assert_bincode_roundtrip(&fixture.nightstream_proof);
    assert_bincode_roundtrip(&fixture.decider_proof);
}
