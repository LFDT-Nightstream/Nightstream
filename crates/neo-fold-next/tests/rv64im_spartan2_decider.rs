//! Focused tests for the live RV64IM terminal Spartan decider.

use std::sync::Arc;

use neo_fold_next::nightstream::rv64im::audit::build_rv64im_nightstream_statement_from_final;
use neo_fold_next::nightstream::rv64im::rv64im_verifier_context_digest;
use neo_fold_next::rv64im::audit::{
    build_rv64im_terminal_decider_setup_shape_from_components, debug_check_rv64im_terminal_decider_circuit,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc_snark::{
    setup_rv64im_ivc_snark_from_final, setup_rv64im_ivc_snark_from_final_cached, verify_rv64im_ivc_snark_against_final,
};
use neo_fold_next::rv64im::main_proof::Rv64imCompressedMainProof;
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

fn proof_input_with_transcript_suffix(name: &str, suffix: &[u8]) -> Rv64imProofInput {
    let mut source = source_case(name);
    source.transcript_seed.extend_from_slice(suffix);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn final_fixture_from_input(
    input: Rv64imProofInput,
) -> (
    neo_fold_next::rv64im::Rv64imProof,
    neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
    neo_fold_next::nightstream::NightstreamStatement,
) {
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let nightstream_statement = build_rv64im_nightstream_statement_from_final(
        proof.statement.digest,
        rv64im_verifier_context_digest(proof.statement.root_params_id),
        &statement,
        &final_proof,
        [0; 32],
    )
    .expect("build nightstream statement");
    (proof, statement, final_proof, nightstream_statement)
}

fn final_fixture(
    name: &str,
) -> (
    neo_fold_next::rv64im::Rv64imProof,
    neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
    neo_fold_next::nightstream::NightstreamStatement,
) {
    final_fixture_from_input(proof_input(name))
}

fn compressed_main_proof_fixture(
    statement: &neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    final_proof: &neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
    final_pc: u64,
) -> Rv64imCompressedMainProof {
    Rv64imCompressedMainProof::from_verified_final_seam(statement, final_proof, final_pc)
        .expect("build compressed main proof")
}

#[test]
fn rv64im_spartan2_decider_setup_shape_is_transcript_seed_invariant() {
    let (_proof_a, statement_a, final_proof_a, _) = final_fixture("control_flow_jal_skip_ecall");
    let (_proof_b, statement_b, final_proof_b, _) = final_fixture_from_input(proof_input_with_transcript_suffix(
        "control_flow_jal_skip_ecall",
        b"/goal3-alt",
    ));

    let shape_a = build_rv64im_terminal_decider_setup_shape_from_components(
        &statement_a,
        final_proof_a.proof_digest,
        &final_proof_a.kernel_export,
        &final_proof_a.chunk_summaries,
        &final_proof_a.steps,
    )
    .expect("build setup shape A");
    let shape_b = build_rv64im_terminal_decider_setup_shape_from_components(
        &statement_b,
        final_proof_b.proof_digest,
        &final_proof_b.kernel_export,
        &final_proof_b.chunk_summaries,
        &final_proof_b.steps,
    )
    .expect("build setup shape B");

    assert_eq!(
        shape_a, shape_b,
        "same control-flow program with a different transcript seed must derive the same decider setup shape"
    );
}

#[test]
fn rv64im_spartan2_decider_setup_shape_rejects_tampered_public_statement_digest() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let mut tampered_statement = statement.clone();
    tampered_statement.public_statement_digest[0] ^= 1;

    let err = build_rv64im_terminal_decider_setup_shape_from_components(
        &tampered_statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect_err("tampered final statement digest must fail setup-shape derivation");
    assert!(
        format!("{err}").contains("final statement digest mismatch"),
        "expected final statement digest mismatch, got: {err}"
    );
}

#[test]
fn rv64im_spartan2_decider_setup_shape_rejects_tampered_replay_header_transport() {
    let (_proof, statement, mut final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    final_proof.steps[0]
        .replay_witness
        .ccs_replay_proof
        .header_digest[0] ^= 1;

    let err = build_rv64im_terminal_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect_err("tampered replay header transport must fail setup-shape derivation");
    assert!(
        format!("{err}").contains("header digest does not match transcript replay"),
        "expected replay header digest mismatch, got: {err}"
    );
}

#[test]
#[ignore = "expensive: terminal decider debug synthesis exceeds developer-memory budget"]
fn rv64im_spartan2_decider_debug_check_only() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    debug_check_rv64im_terminal_decider_circuit(&statement, &final_proof)
        .expect("rv64im terminal decider circuit must be satisfied");
}

#[test]
#[ignore = "expensive: terminal decider Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_setup_only() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let _ = setup_rv64im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv64im spartan2 decider");
}

#[test]
#[ignore = "expensive: terminal decider Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_cached_setup_reuses_same_final_seam() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let first = setup_rv64im_ivc_snark_from_final_cached(&statement, &final_proof)
        .expect("setup cached rv64im spartan2 decider");
    let second = setup_rv64im_ivc_snark_from_final_cached(&statement, &final_proof)
        .expect("setup cached rv64im spartan2 decider");
    assert!(
        Arc::ptr_eq(&first, &second),
        "exact same final seam should reuse cached setup"
    );
}

#[test]
#[ignore = "expensive: terminal decider Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_setup_from_shape_is_reproducible() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_first_pk, first_vk) = setup_rv64im_ivc_snark_from_final(&statement, &final_proof).expect("setup decider");
    let (_second_pk, second_vk) = setup_rv64im_ivc_snark_from_final(&statement, &final_proof).expect("setup decider");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    verify_rv64im_ivc_snark_against_final(&first_vk, &statement, &final_proof, compressed_main_proof.ivc_snark())
        .expect("first verifier key must accept the proof");
    verify_rv64im_ivc_snark_against_final(&second_vk, &statement, &final_proof, compressed_main_proof.ivc_snark())
        .expect("repeating setup from the same setup shape must reproduce a verifier key that accepts the same proof");
}

#[test]
#[ignore = "expensive: terminal decider Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_cached_shape_setup_is_deterministic() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let first = setup_rv64im_ivc_snark_from_final_cached(&statement, &final_proof).expect("cached decider setup");
    let second = setup_rv64im_ivc_snark_from_final_cached(&statement, &final_proof).expect("cached decider setup");
    assert!(
        Arc::ptr_eq(&first, &second),
        "exact same setup shape should reuse cached setup"
    );
}

#[test]
#[ignore = "expensive: terminal decider Spartan round-trip exceeds developer-memory budget"]
fn rv64im_spartan2_decider_round_trip_without_replay_verifier_input() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");

    let (_pk, vk) = setup_rv64im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv64im spartan2 decider");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    verify_rv64im_ivc_snark_against_final(&vk, &statement, &final_proof, compressed_main_proof.ivc_snark())
        .expect("verify rv64im spartan2 decider");
    assert!(
        compressed_main_proof
            .terminal_decider_proof()
            .snark_bytes_len()
            > 0
    );
}

#[test]
#[ignore = "expensive: terminal decider Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_rejects_tampered_chunk_relation_digest() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let (_pk, vk) = setup_rv64im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv64im spartan2 decider");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    let mut tampered_final_proof = final_proof.clone();
    tampered_final_proof.chunk_summaries[0].chunk_relation_digest[0] ^= 1;
    let err = verify_rv64im_ivc_snark_against_final(
        &vk,
        &statement,
        &tampered_final_proof,
        compressed_main_proof.ivc_snark(),
    )
    .expect_err("tampered chunk relation digest must fail");
    assert!(format!("{err}").contains("relation digest") || format!("{err}").contains("chunk"));
}

#[test]
#[ignore = "expensive: terminal decider Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_rejects_tampered_final_claim() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");

    let (_pk, vk) = setup_rv64im_ivc_snark_from_final(&statement, &final_proof).expect("setup rv64im spartan2 decider");
    let compressed_main_proof = compressed_main_proof_fixture(&statement, &final_proof, proof.statement.final_pc);

    verify_rv64im_ivc_snark_against_final(&vk, &statement, &final_proof, compressed_main_proof.ivc_snark())
        .expect("baseline theorem statement must verify");

    let mut tampered_statement = statement.clone();
    tampered_statement.public_statement_digest[0] ^= 1;
    let err = verify_rv64im_ivc_snark_against_final(
        &vk,
        &tampered_statement,
        &final_proof,
        compressed_main_proof.ivc_snark(),
    )
    .expect_err("tampered public-statement digest must fail");
    assert!(
        format!("{err}").contains("public statement") || format!("{err}").contains("digest"),
        "expected public-statement failure, got: {err}"
    );
}
