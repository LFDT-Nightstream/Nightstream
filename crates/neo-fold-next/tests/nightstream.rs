use std::fmt::Debug;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_nightstream_statement_from_published_statement, rv64im_main_nightstream_proof_digest,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_nightstream_from_public_proof, rv64im_verifier_context_digest, verify_rv64im_nightstream,
    Rv64imNightstreamProof, Rv64imSideBindingVerifierKey, Rv64imSideOpeningSpartanVerifierKey,
};
use neo_fold_next::nightstream::{
    nightstream_proof_binding_root, nightstream_statement_digest, NightstreamProofBindingInputs, NightstreamStatement,
};
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_ivc_recursion_snark_setup_shape_from_components,
    prove_rv64im_public_proof_and_published_seam_with_perf,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof_with_options,
    setup_rv64im_ivc_snark_from_final_cached, Rv64imIvcSnarkKeyPair, Rv64imProofInput, Rv64imProofStatement,
    Rv64imPublicProofOptions, SimpleKernelError,
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
    ivc_recursion_snark_keys: Rv64imIvcSnarkKeyPair,
    side_opening_vk: Rv64imSideOpeningSpartanVerifierKey,
    side_binding_vk: Rv64imSideBindingVerifierKey,
    statement: NightstreamStatement,
    nightstream_proof: Rv64imNightstreamProof,
}

fn external_fixture(name: &str) -> ExternalNightstreamFixture {
    let public_proof = prove_rv64im_public_proof_with_options(
        &proof_input(name),
        Rv64imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove rv64im public proof");
    let trusted_root_params_id = public_proof.statement.root_params_id;
    let public_statement = public_proof.statement.clone();
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("prove rv64im final statement");
    let (statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    let _ivc_recursion_snark_shape = build_rv64im_ivc_recursion_snark_setup_shape_from_components(
        &final_statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build rv64im IVC recursion SNARK setup shape");
    let ivc_recursion_snark_keys = setup_rv64im_ivc_snark_from_final_cached(&final_statement, &final_proof)
        .expect("setup rv64im IVC recursion SNARK");
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
        .binding_statement(&statement, &public_statement)
        .expect("build rv64im side binding statement");
    let (_, side_binding_vk) = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_binding(
        &side_statement,
        nightstream_proof.side_proof().opening_public(),
    )
    .expect("setup rv64im side binding");
    ExternalNightstreamFixture {
        trusted_root_params_id,
        public_statement,
        ivc_recursion_snark_keys,
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
        &fixture.ivc_recursion_snark_keys.1,
        &fixture.side_opening_vk,
        &fixture.side_binding_vk,
        &fixture.public_statement,
    )
}

fn published_seam_fixture(
    name: &str,
) -> (
    neo_fold_next::rv64im::Rv64imProof,
    neo_fold_next::rv64im::audit::Rv64imPublishedProofSeam,
) {
    let ((proof, seam), _) = prove_rv64im_public_proof_and_published_seam_with_perf(&proof_input(name))
        .expect("prove rv64im public proof and published seam");
    (proof, seam)
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
        proof_binding_root: [0; 32],
    };
    let inputs = NightstreamProofBindingInputs {
        main_proof_digest: [4; 32],
        side_proof_digest: [6; 32],
        public_statement_digest: [8; 32],
    };
    statement.proof_binding_root = nightstream_proof_binding_root(statement.core_digest(), &inputs);
    let digest_before = nightstream_statement_digest(&statement);
    statement.proof_binding_root[0] ^= 1;
    let digest_after = nightstream_statement_digest(&statement);
    assert_ne!(digest_before, digest_after);
}

#[test]
#[ignore = "published-seam rebuild remains too expensive for routine Nightstream boundary regression"]
fn rv64im_nightstream_statement_and_main_proof_follow_verified_final_seam() {
    let (proof, seam) = published_seam_fixture("control_flow_jal_skip_ecall");
    let statement = seam
        .rebuild_final_statement()
        .expect("rebuild final statement from one-step published seam");
    let final_proof = seam
        .final_proof()
        .expect("rebuild final proof from one-step published seam");

    let compressed_main_proof = seam.main_proof.clone();
    let (_, nightstream_proof) = build_rv64im_nightstream_from_public_proof(&proof).expect("build nightstream proof");

    assert_ne!(
        compressed_main_proof
            .published_statement()
            .expected_digest(),
        [0; 32]
    );
    assert_eq!(final_proof.chunk_summaries.len(), final_proof.steps.len());
    assert_eq!(compressed_main_proof, *nightstream_proof.main_proof());

    let ivc_recursion_snark_keys =
        setup_rv64im_ivc_snark_from_final_cached(&statement, &final_proof).expect("setup IVC recursion SNARK");
    let verifier_context = rv64im_verifier_context_digest(
        proof.statement.root_params_id,
        compressed_main_proof.published_statement(),
        &ivc_recursion_snark_keys.as_ref().1,
    )
    .expect("digest verifier context");
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: rv64im_main_nightstream_proof_digest(&compressed_main_proof),
        side_proof_digest: [9; 32],
        public_statement_digest: proof.statement.digest,
    };
    let mut nightstream = build_rv64im_nightstream_statement_from_published_statement(
        verifier_context,
        compressed_main_proof.published_statement(),
        [0; 32],
    )
    .expect("build nightstream statement");
    let expected_public_image = compressed_main_proof
        .expected_ivc_public_image()
        .expect("derive compact main proof public image");
    compressed_main_proof
        .ivc_snark()
        .verify(&ivc_recursion_snark_keys.1, &expected_public_image)
        .expect("verify compact main proof IVC SNARK");
    nightstream.proof_binding_root = nightstream_proof_binding_root(nightstream.core_digest(), &proof_binding_inputs);
    assert_eq!(nightstream.fold_schedule, statement.folded.fold_schedule);
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
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_round_trips_against_current_public_proof_seam() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    verify_fixture(&fixture).expect("verify Nightstream proof fixture");
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
fn rv64im_nightstream_rejects_tampered_ivc_recursion_snark_proof() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    tamper_snark_bytes(
        &mut fixture
            .nightstream_proof
            .main_proof_mut()
            .ivc_recursion_snark_proof_mut()
            .terminal_f_prime_committed_step_proof
            .snark_data,
    );
    let err = verify_fixture(&fixture).expect_err("tampered IVC recursion SNARK proof must fail");
    assert!(format!("{err}").contains("main relation") || format!("{err}").contains("recursion"));
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
    verify_fixture(&fixture).expect("verify baseline Nightstream proof fixture");

    let verify_mutated = |proof: Rv64imNightstreamProof, label: &str| {
        verify_rv64im_nightstream(
            &fixture.statement,
            &proof,
            fixture.trusted_root_params_id,
            &fixture.ivc_recursion_snark_keys.1,
            &fixture.side_opening_vk,
            &fixture.side_binding_vk,
            &fixture.public_statement,
        )
        .err()
        .unwrap_or_else(|| panic!("tampered {label} must be rejected"));
    };

    {
        let mut proof = fixture.nightstream_proof.clone();
        tamper_snark_bytes(
            &mut proof
                .main_proof_mut()
                .ivc_recursion_snark_proof_mut()
                .terminal_f_prime_committed_step_proof
                .snark_data,
        );
        verify_mutated(proof, "main_ivc_recursion_snark_proof");
    }
    {
        let mut proof = fixture.nightstream_proof.clone();
        proof
            .main_proof_mut()
            .published_statement_mut()
            .x_last_mut()
            .bytes_mut()[0] ^= 1;
        verify_mutated(proof, "main_published_statement");
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
        &fixture.ivc_recursion_snark_keys.1,
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
fn rv64im_nightstream_rejects_self_consistent_tampered_public_statement_root_params_id() {
    let mut fixture = external_fixture("control_flow_jal_skip_ecall");
    fixture.public_statement.root_params_id[0] ^= 1;
    fixture.public_statement.digest = fixture.public_statement.recompute_digest();
    let err = verify_fixture(&fixture).expect_err("tampered public statement root_params_id must fail");
    assert!(format!("{err}").contains("Nightstream statement"));
}

#[test]
#[ignore = "expensive: Nightstream end-to-end proof path exceeds developer-memory budget"]
fn rv64im_nightstream_serde_roundtrips_statement_proof_and_spartan_proofs() {
    let fixture = external_fixture("control_flow_jal_skip_ecall");
    assert_bincode_roundtrip(&fixture.statement);
    assert_bincode_roundtrip(&fixture.nightstream_proof);
    assert_bincode_roundtrip(fixture.nightstream_proof.main_proof());
    assert_bincode_roundtrip(fixture.nightstream_proof.main_proof().ivc_snark());
    assert_bincode_roundtrip(
        fixture
            .nightstream_proof
            .main_proof()
            .ivc_recursion_snark_proof(),
    );
    assert_bincode_roundtrip(fixture.nightstream_proof.side_proof().binding());
}
