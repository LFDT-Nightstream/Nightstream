use std::fmt::Debug;

use neo_fold_next::nightstream::rv64im::{
    build_rv64im_bound_side_proof_bundle_from_accepted_artifact, build_rv64im_main_decider_proof,
    build_rv64im_main_residual_proof, build_rv64im_nightstream_from_public_proof,
    build_rv64im_nightstream_linkage_claims, build_rv64im_nightstream_linkage_claims_from_relation,
    build_rv64im_nightstream_statement_from_final, rv64im_nightstream_linkage_root, rv64im_verifier_context_digest,
    setup_rv64im_side_spartan_from_accepted_artifact, verify_rv64im_main_decider_proof,
    verify_rv64im_main_residual_proof, verify_rv64im_nightstream, Rv64imNightstreamProof, Rv64imSideSpartanVerifierKey,
};
use neo_fold_next::nightstream::{
    nightstream_proof_binding_root, nightstream_statement_digest, NightstreamProofBindingInputs, NightstreamStatement,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_published_proof_seam, parity_source_cases,
    prove_rv64im_public_proof, setup_rv64im_spartan2_decider_from_public_proof, Rv64imProofInput, Rv64imProofStatement,
    Rv64imSpartan2DeciderVerifierKey, SimpleKernelError,
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
    decider_vk: Rv64imSpartan2DeciderVerifierKey,
    side_decider_vk: Rv64imSideSpartanVerifierKey,
    statement: NightstreamStatement,
    nightstream_proof: Rv64imNightstreamProof,
}

fn external_fixture(name: &str) -> ExternalNightstreamFixture {
    let public_proof = prove_rv64im_public_proof(&proof_input(name)).expect("prove rv64im public proof");
    let published_seam = build_rv64im_published_proof_seam(&public_proof).expect("build published seam");
    let trusted_root_params_id = public_proof.statement.root_params_id;
    let public_statement = public_proof.statement.clone();
    let (_, decider_vk) =
        setup_rv64im_spartan2_decider_from_public_proof(&public_proof).expect("setup rv64im spartan2 decider");
    let (statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    let side_bundle =
        build_rv64im_bound_side_proof_bundle_from_accepted_artifact(&statement, &published_seam.accepted_artifact)
            .expect("build bound side bundle");
    let (_, side_decider_vk) =
        setup_rv64im_side_spartan_from_accepted_artifact(&statement, &side_bundle, &published_seam.accepted_artifact)
            .expect("setup rv64im side spartan");
    ExternalNightstreamFixture {
        trusted_root_params_id,
        public_statement,
        decider_vk,
        side_decider_vk,
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
        &fixture.side_decider_vk,
        &fixture.public_statement,
    )
}

fn tamper_snark_bytes(snark_data: &mut Vec<u8>) {
    if let Some(first) = snark_data.first_mut() {
        *first ^= 1;
    } else {
        snark_data.push(1);
    }
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
        linkage_artifact_digest: [7; 32],
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &inputs);
    let digest_before = nightstream_statement_digest(&statement);
    statement.proof_binding_root[0] ^= 1;
    let digest_after = nightstream_statement_digest(&statement);
    assert_ne!(digest_before, digest_after);
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
    let residual = build_rv64im_main_residual_proof(&statement, &final_proof).expect("build residual proof");
    verify_rv64im_main_residual_proof(&statement, &final_proof, &residual).expect("verify residual proof");

    let from_relation = build_rv64im_nightstream_linkage_claims_from_relation(
        &residual.decider_relation,
        &residual.bridge_handoff_digests,
    )
    .expect("build linkage claims from carried relation");
    assert_eq!(linkage_claims, from_relation);

    let main_decider_proof =
        build_rv64im_main_decider_proof(&statement, &final_proof).expect("build main decider proof");
    let (_, decider_vk) =
        setup_rv64im_spartan2_decider_from_public_proof(&proof).expect("setup rv64im spartan2 decider");

    let verifier_context = rv64im_verifier_context_digest(proof.statement.root_params_id);
    let linkage_root = rv64im_nightstream_linkage_root(final_proof.kernel_export.digest, &linkage_claims);
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: main_decider_proof.expected_digest(),
        main_residual_proof_digest: residual.expected_digest(),
        side_bridge_artifact_digest: [9; 32],
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
    verify_rv64im_main_decider_proof(&decider_vk, &residual, &main_decider_proof).expect("verify main decider proof");
    nightstream.proof_binding_root = nightstream_proof_binding_root(nightstream.core_digest(), &proof_binding_inputs);
    assert_eq!(nightstream.fold_schedule, statement.folded.fold_schedule);
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_round_trips_against_current_public_proof_seam() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    verify_fixture(&fixture).expect("verify nightstream proof");
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_tampered_statement_binding_root() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.statement.proof_binding_root[0] ^= 1;
    let err = verify_fixture(&fixture).expect_err("tampered statement binding must fail");
    assert!(format!("{err}").contains("Nightstream statement"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_tampered_main_decider_proof() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    tamper_snark_bytes(
        &mut fixture
            .nightstream_proof
            .main_decider_proof
            .spartan_proof
            .snark_data,
    );
    let err = verify_fixture(&fixture).expect_err("tampered main decider proof must fail");
    assert!(format!("{err}").contains("main decider proof"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_tampered_side_decider_proof_bytes() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    tamper_snark_bytes(
        &mut fixture
            .nightstream_proof
            .side_decider_proof
            .spartan_proof
            .snark_data,
    );
    let err = verify_fixture(&fixture).expect_err("tampered side decider proof must fail");
    assert!(format!("{err}").contains("side relation") || format!("{err}").contains("side decider proof"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_carried_boundary_rejects_each_tampered_proof_binding_input() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    verify_fixture(&fixture).expect("baseline fixture must verify");

    let verify_mutated = |proof: Rv64imNightstreamProof, label: &str| {
        verify_rv64im_nightstream(
            &fixture.statement,
            &proof,
            fixture.trusted_root_params_id,
            &fixture.decider_vk,
            &fixture.side_decider_vk,
            &fixture.public_statement,
        )
        .err()
        .unwrap_or_else(|| panic!("tampered {label} must be rejected"));
    };

    {
        let mut proof = fixture.nightstream_proof.clone();
        tamper_snark_bytes(&mut proof.main_decider_proof.spartan_proof.snark_data);
        verify_mutated(proof, "main_decider_proof");
    }
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.main_residual_proof.public_statement_digest[0] ^= 1;
        verify_mutated(proof, "main_residual_proof");
    }
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.linkage_artifact.digest[0] ^= 1;
        verify_mutated(proof, "linkage_artifact");
    }
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_wrong_main_spartan_verifier_key_shape() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    let alternate_public_proof =
        prove_rv64im_public_proof(&proof_input(&alternate_case_name("control_flow_jal_skip_ecall")))
            .expect("prove alternate public proof");
    let (_, wrong_vk) = setup_rv64im_spartan2_decider_from_public_proof(&alternate_public_proof)
        .expect("setup alternate-case Spartan decider");
    let err = verify_rv64im_nightstream(
        &fixture.statement,
        &fixture.nightstream_proof,
        fixture.trusted_root_params_id,
        &wrong_vk,
        &fixture.side_decider_vk,
        &fixture.public_statement,
    )
    .expect_err("wrong main verifier key must fail");
    assert!(format!("{err}").contains("Spartan proof"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_wrong_side_spartan_verifier_key_shape() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    let alternate = external_fixture(&alternate_case_name("control_flow_jal_skip_ecall"));
    let err = verify_rv64im_nightstream(
        &fixture.statement,
        &fixture.nightstream_proof,
        fixture.trusted_root_params_id,
        &fixture.decider_vk,
        &alternate.side_decider_vk,
        &fixture.public_statement,
    )
    .expect_err("wrong side verifier key must fail");
    assert!(format!("{err}").contains("side relation") || format!("{err}").contains("sumcheck"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_public_statement_with_stale_digest() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.public_statement.final_pc += 1;
    let err = verify_fixture(&fixture).expect_err("stale public statement digest must fail");
    assert!(format!("{err}").contains("public statement"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_serde_roundtrips_statement_proof_and_spartan_proofs() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    assert_bincode_roundtrip(&fixture.statement);
    assert_bincode_roundtrip(&fixture.nightstream_proof);
    assert_bincode_roundtrip(&fixture.nightstream_proof.main_decider_proof.spartan_proof);
    assert_bincode_roundtrip(&fixture.nightstream_proof.side_decider_proof.spartan_proof);
}
