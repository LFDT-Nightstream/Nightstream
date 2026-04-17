use std::fmt::Debug;

use neo_fold_next::nightstream::rv64im::audit::build_rv64im_nightstream_linkage_claims;
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_main_proof, build_rv64im_nightstream_from_public_proof,
    build_rv64im_nightstream_statement_from_main_proof, rv64im_nightstream_linkage_root,
    rv64im_verifier_context_digest, verify_rv64im_main_proof, verify_rv64im_nightstream, Rv64imNightstreamProof,
    Rv64imSideBindingVerifierKey, Rv64imSideOpeningSpartanVerifierKey,
};
use neo_fold_next::nightstream::{
    nightstream_proof_binding_root, nightstream_statement_digest, NightstreamProofBindingInputs, NightstreamStatement,
};
use neo_fold_next::rv64im::audit::{
    rv64im_main_recursion_proof_first_step_snark_bytes_mut, rv64im_main_recursion_proof_x_last_mut,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
    Rv64imProofStatement, SimpleKernelError,
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
    side_opening_vk: Rv64imSideOpeningSpartanVerifierKey,
    side_binding_vk: Rv64imSideBindingVerifierKey,
    statement: NightstreamStatement,
    nightstream_proof: Rv64imNightstreamProof,
}

fn external_fixture(name: &str) -> ExternalNightstreamFixture {
    let public_proof = prove_rv64im_public_proof(&proof_input(name)).expect("prove rv64im public proof");
    let trusted_root_params_id = public_proof.statement.root_params_id;
    let public_statement = public_proof.statement.clone();
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    let (opening_statement, opening_witness) =
        neo_fold_next::nightstream::rv64im::audit::build_rv64im_side_opening_relation_from_accepted_artifact(
            &accepted_artifact,
        )
        .expect("build rv64im side opening relation");
    let (_, side_opening_vk) = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_opening_spartan(
        &opening_statement,
        &opening_witness,
    )
    .expect("setup rv64im side opening");
    let side_statement = nightstream_proof
        .side_proof()
        .binding_statement(&statement)
        .expect("build rv64im side binding statement");
    let (_, side_binding_vk) = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_binding(
        &side_statement,
        nightstream_proof.side_proof().opening_public(),
    )
    .expect("setup rv64im side binding");
    ExternalNightstreamFixture {
        trusted_root_params_id,
        public_statement,
        side_opening_vk,
        side_binding_vk,
        statement,
        nightstream_proof,
    }
}

fn verify_fixture(fixture: &ExternalNightstreamFixture) -> Result<(), SimpleKernelError> {
    verify_rv64im_nightstream(
        &fixture.statement,
        &fixture.nightstream_proof,
        fixture.trusted_root_params_id,
        &fixture.side_opening_vk,
        &fixture.side_binding_vk,
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
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let bytes = bincode::serialize(value).expect("serialize roundtrip value");
    let decoded: T = bincode::deserialize(&bytes).expect("deserialize roundtrip value");
    assert_eq!(decoded, *value);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
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
        main_proof_digest: [4; 32],
        side_proof_digest: [6; 32],
        linkage_binding_digest: [7; 32],
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &inputs);
    let digest_before = nightstream_statement_digest(&statement);
    statement.proof_binding_root[0] ^= 1;
    let digest_after = nightstream_statement_digest(&statement);
    assert_ne!(digest_before, digest_after);
}

#[test]
#[ignore = "temporary during main-proof linkage ownership split; re-enable after the owner/runtime seam and Nightstream happy-path proving surface stabilize"]
fn rv64im_nightstream_linkage_and_main_proof_follow_verified_final_seam() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let linkage_claims =
        build_rv64im_nightstream_linkage_claims(&statement, &final_proof).expect("build linkage claims");
    let mut main_proof = build_rv64im_main_proof(&statement, &final_proof).expect("build main proof");
    let (_, nightstream_proof) = build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    assert_eq!(
        main_proof
            .final_statement_cache()
            .expect("locally built main proof should retain the final-statement cache"),
        &statement
    );
    assert_ne!(main_proof.published_statement().expected_digest(), [0; 32]);
    assert_eq!(main_proof.linkage_anchor_digest(), statement.public_statement_digest);
    assert_eq!(main_proof.linkage_anchor_digest(), statement.public_statement_digest);
    assert_eq!(main_proof.chunk_summaries().len(), final_proof.steps.len());
    assert!(!rv64im_main_recursion_proof_first_step_snark_bytes_mut(main_proof.recursion_proof_mut()).is_empty());
    assert_eq!(linkage_claims, *nightstream_proof.linkage_claims());

    let verifier_context = rv64im_verifier_context_digest(proof.statement.root_params_id);
    let linkage_root = rv64im_nightstream_linkage_root(main_proof.linkage_anchor_digest(), &linkage_claims);
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: main_proof.binding_digest(),
        side_proof_digest: [9; 32],
        linkage_binding_digest: linkage_claims.digest(),
    };
    let mut nightstream =
        build_rv64im_nightstream_statement_from_main_proof(verifier_context, &main_proof, linkage_root, [0; 32])
            .expect("build nightstream statement");
    verify_rv64im_main_proof(&main_proof).expect("verify main proof");
    nightstream.proof_binding_root = nightstream_proof_binding_root(nightstream.core_digest(), &proof_binding_inputs);
    assert_eq!(nightstream.fold_schedule, statement.folded.fold_schedule);
}

#[test]
#[ignore = "expensive manual tamper probe: Nightstream main-proof construction exceeds normal regression budget"]
fn rv64im_main_proof_binding_digest_tracks_backend_statement_not_private_bytes() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut main_proof = build_rv64im_main_proof(&statement, &final_proof).expect("build main proof");
    let baseline = main_proof.binding_digest();
    tamper_snark_bytes(rv64im_main_recursion_proof_first_step_snark_bytes_mut(
        main_proof.recursion_proof_mut(),
    ));
    assert_eq!(
        baseline,
        main_proof.binding_digest(),
        "Nightstream main-proof binding digest must not depend on private recursion step-proof bytes"
    );

    rv64im_main_recursion_proof_x_last_mut(main_proof.recursion_proof_mut())[0] ^= 1;
    assert_ne!(
        baseline,
        main_proof.binding_digest(),
        "Nightstream main-proof binding digest must change when recursion final public image changes"
    );
}

#[test]
#[ignore = "expensive manual tamper probe: Nightstream side-proof construction exceeds normal regression budget"]
fn rv64im_side_proof_digest_tracks_opening_statement_digest_bytes() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    let baseline = fixture.nightstream_proof.side_proof().expected_digest();
    fixture
        .nightstream_proof
        .side_proof_mut()
        .opening_statement_mut()
        .stage1
        .digest[0] ^= 1;
    assert_ne!(
        baseline,
        fixture.nightstream_proof.side_proof().expected_digest(),
        "Nightstream side-proof digest must bind the carried opening-statement digest bytes"
    );
}

#[test]
#[ignore = "expensive manual tamper probe: Nightstream side-proof construction exceeds normal regression budget"]
fn rv64im_side_proof_digest_tracks_linkage_bytes() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    let baseline = fixture.nightstream_proof.side_proof().expected_digest();
    fixture
        .nightstream_proof
        .side_proof_mut()
        .linkage_mut()
        .transcript_surface_digest_mut()[0] ^= 1;
    assert_ne!(
        baseline,
        fixture.nightstream_proof.side_proof().expected_digest(),
        "Nightstream side-proof digest must bind the carried linkage bytes"
    );
}

#[test]
#[ignore = "expensive manual tamper probe: Nightstream main-proof construction exceeds normal regression budget"]
fn rv64im_main_proof_tracks_tampered_linkage_anchor_but_ignores_local_kernel_export_cache() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut wrong_surface = build_rv64im_main_proof(&statement, &final_proof).expect("build main proof");
    let baseline_binding = wrong_surface.binding_digest();
    let baseline_expected = wrong_surface.expected_digest();
    wrong_surface.linkage_anchor_digest_mut()[0] ^= 1;
    verify_rv64im_main_proof(&wrong_surface)
        .expect("published main-proof verification must still ignore the Nightstream linkage anchor");
    assert_eq!(
        baseline_binding,
        wrong_surface.binding_digest(),
        "main-proof binding digest must be driven by the theorem-facing published boundary, not the Nightstream linkage anchor"
    );
    assert_eq!(
        baseline_expected,
        wrong_surface.expected_digest(),
        "main-proof digest must be driven by the theorem-facing published boundary, not the Nightstream linkage anchor"
    );

    let mut wrong_replay = build_rv64im_main_proof(&statement, &final_proof).expect("build main proof");
    let baseline_binding = wrong_replay.binding_digest();
    let baseline_expected = wrong_replay.expected_digest();
    wrong_replay
        .kernel_export_cache_mut()
        .expect("locally built Nightstream main proof must carry a kernel-export cache")
        .digest[0] ^= 1;
    wrong_replay
        .validate_local_build_caches()
        .expect_err("tampered local kernel-export cache must fail the local-cache validator");
    verify_rv64im_main_proof(&wrong_replay)
        .expect("published main-proof verification must ignore the local kernel-export cache");
    assert_eq!(
        baseline_binding,
        wrong_replay.binding_digest(),
        "main-proof binding digest must not depend on the local kernel-export cache"
    );
    assert_eq!(
        baseline_expected,
        wrong_replay.expected_digest(),
        "main-proof digest must not depend on the local kernel-export cache"
    );
}

#[test]
#[ignore = "expensive manual tamper probe: Nightstream main-proof construction exceeds normal regression budget"]
fn rv64im_main_proof_rejects_tampered_recursion_public_image_split() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");

    let mut wrong_surface = build_rv64im_main_proof(&statement, &final_proof).expect("build main proof");
    rv64im_main_recursion_proof_x_last_mut(wrong_surface.recursion_proof_mut())[0] ^= 1;
    let err = verify_rv64im_main_proof(&wrong_surface).expect_err("tampered recursion public image must fail");
    assert!(format!("{err}").contains("x_last") || format!("{err}").contains("public image"));
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
    tamper_snark_bytes(rv64im_main_recursion_proof_first_step_snark_bytes_mut(
        fixture
            .nightstream_proof
            .main_proof_mut()
            .recursion_proof_mut(),
    ));
    let err = verify_fixture(&fixture).expect_err("tampered main recursion proof must fail");
    assert!(format!("{err}").contains("chunk-step") || format!("{err}").contains("main proof"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_tampered_side_proof_bytes() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    tamper_snark_bytes(
        &mut fixture
            .nightstream_proof
            .side_proof_mut()
            .binding_mut()
            .snark_data,
    );
    let err = verify_fixture(&fixture).expect_err("tampered side proof must fail");
    assert!(format!("{err}").contains("side binding") || format!("{err}").contains("side proof"));
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
            &fixture.side_opening_vk,
            &fixture.side_binding_vk,
            &fixture.public_statement,
        )
        .err()
        .unwrap_or_else(|| panic!("tampered {label} must be rejected"));
    };

    {
        let mut proof = fixture.nightstream_proof.clone();
        tamper_snark_bytes(rv64im_main_recursion_proof_first_step_snark_bytes_mut(
            proof.main_proof_mut().recursion_proof_mut(),
        ));
        verify_mutated(proof, "main_recursion_proof");
    }
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof.main_proof_mut().linkage_anchor_digest_mut()[0] ^= 1;
        verify_mutated(proof, "main_final_statement");
    }
    {
        let mut proof = fixture.nightstream_proof.clone();
        rv64im_main_recursion_proof_x_last_mut(proof.main_proof_mut().recursion_proof_mut())[0] ^= 1;
        verify_mutated(proof, "main_final_surface");
    }
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_rejects_wrong_side_binding_verifier_key_shape() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    let alternate = external_fixture(&alternate_case_name("control_flow_jal_skip_ecall"));
    let err = verify_rv64im_nightstream(
        &fixture.statement,
        &fixture.nightstream_proof,
        fixture.trusted_root_params_id,
        &fixture.side_opening_vk,
        &alternate.side_binding_vk,
        &fixture.public_statement,
    )
    .expect_err("wrong side verifier key must fail");
    assert!(format!("{err}").contains("side binding") || format!("{err}").contains("sumcheck"));
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
    assert_bincode_roundtrip(fixture.nightstream_proof.main_proof().recursion_proof());
    assert_bincode_roundtrip(fixture.nightstream_proof.side_proof().binding());
}
