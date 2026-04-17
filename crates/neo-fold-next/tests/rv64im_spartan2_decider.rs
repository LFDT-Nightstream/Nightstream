//! Focused tests for the shell-free RV64IM main-relation Spartan decider.

use std::sync::Arc;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_nightstream_linkage_claims, build_rv64im_nightstream_statement_from_final,
};
use neo_fold_next::nightstream::rv64im::{rv64im_nightstream_linkage_root, rv64im_verifier_context_digest};
use neo_fold_next::rv64im::audit::{
    build_rv64im_decider_relation_from_final_surface, build_rv64im_spartan2_decider_setup_shape_from_components,
    debug_check_rv64im_spartan2_decider_circuit, prove_rv64im_spartan2_decider,
    setup_rv64im_spartan2_decider_cached_from_shape, setup_rv64im_spartan2_decider_from_shape,
    verify_rv64im_spartan2_decider,
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

fn final_fixture(
    name: &str,
) -> (
    neo_fold_next::rv64im::Rv64imProof,
    neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
    neo_fold_next::nightstream::NightstreamStatement,
) {
    let input = proof_input(name);
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let linkage_claims =
        build_rv64im_nightstream_linkage_claims(&statement, &final_proof).expect("build linkage claims");
    let linkage_root = rv64im_nightstream_linkage_root(final_proof.kernel_export.digest, &linkage_claims);
    let nightstream_statement = build_rv64im_nightstream_statement_from_final(
        proof.statement.digest,
        rv64im_verifier_context_digest(proof.statement.root_params_id),
        &statement,
        &final_proof,
        linkage_root,
        [0; 32],
    )
    .expect("build nightstream statement");
    (proof, statement, final_proof, nightstream_statement)
}

#[test]
fn rv64im_spartan2_decider_setup_shape_rejects_tampered_public_statement_digest() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let mut tampered_statement = statement.clone();
    tampered_statement.public_statement_digest[0] ^= 1;

    let err = build_rv64im_spartan2_decider_setup_shape_from_components(
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

    let err = build_rv64im_spartan2_decider_setup_shape_from_components(
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
#[ignore = "expensive: main-relation debug synthesis exceeds developer-memory budget"]
fn rv64im_spartan2_decider_debug_check_only() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    debug_check_rv64im_spartan2_decider_circuit(&statement, &final_proof)
        .expect("rv64im main relation circuit must be satisfied");
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_setup_only() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let _ = setup_rv64im_spartan2_decider_from_shape(&shape).expect("setup rv64im spartan2 decider from shape");
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_cached_setup_reuses_same_final_seam() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let first = setup_rv64im_spartan2_decider_cached_from_shape(&shape).expect("setup cached rv64im spartan2 decider");
    let second = setup_rv64im_spartan2_decider_cached_from_shape(&shape).expect("setup cached rv64im spartan2 decider");
    assert!(
        Arc::ptr_eq(&first, &second),
        "exact same final seam should reuse cached setup"
    );
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_setup_from_shape_is_reproducible() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let (_, first_vk) = setup_rv64im_spartan2_decider_from_shape(&shape).expect("setup from shape");
    let (_, second_vk) = setup_rv64im_spartan2_decider_from_shape(&shape).expect("setup from shape");
    let live_bytes = bincode::serialize(&first_vk).expect("encode live verifier key");
    let shape_bytes = bincode::serialize(&second_vk).expect("encode shape verifier key");
    assert_eq!(
        live_bytes, shape_bytes,
        "repeating setup from the same setup shape must reproduce the same verifier key"
    );
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_cached_shape_setup_is_deterministic() {
    let (_proof, statement, final_proof, _) = final_fixture("control_flow_jal_skip_ecall");
    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let first = setup_rv64im_spartan2_decider_cached_from_shape(&shape).expect("cached shape setup");
    let second = setup_rv64im_spartan2_decider_cached_from_shape(&shape).expect("cached shape setup");
    assert!(
        Arc::ptr_eq(&first, &second),
        "exact same setup shape should reuse cached setup"
    );
}

#[test]
#[ignore = "expensive: main-relation Spartan round-trip exceeds developer-memory budget"]
fn rv64im_spartan2_decider_round_trip_without_replay_verifier_input() {
    let (_proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let relation =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build decider relation");

    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let (pk, vk) = setup_rv64im_spartan2_decider_from_shape(&shape).expect("setup rv64im spartan2 decider");
    let decider_proof =
        prove_rv64im_spartan2_decider(&pk, &statement, &final_proof).expect("prove rv64im spartan2 decider");

    verify_rv64im_spartan2_decider(&vk, statement.public_statement_digest, &relation, &decider_proof)
        .expect("verify rv64im spartan2 decider");
    assert!(decider_proof.snark_bytes_len() > 0);
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_rejects_tampered_chunk_relation_digest() {
    let (_proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let mut relation =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build decider relation");

    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let (pk, vk) = setup_rv64im_spartan2_decider_from_shape(&shape).expect("setup rv64im spartan2 decider");
    let decider_proof =
        prove_rv64im_spartan2_decider(&pk, &statement, &final_proof).expect("prove rv64im spartan2 decider");

    relation.chunk_summaries[0].chunk_relation_digest[0] ^= 1;
    let err = verify_rv64im_spartan2_decider(&vk, statement.public_statement_digest, &relation, &decider_proof)
        .expect_err("tampered chunk relation digest must fail");
    assert!(format!("{err}").contains("relation digest") || format!("{err}").contains("chunk"));
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_rejects_tampered_final_claim() {
    let (_proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let relation =
        build_rv64im_decider_relation_from_final_surface(&statement, &final_proof).expect("build decider relation");

    let shape = build_rv64im_spartan2_decider_setup_shape_from_components(
        &statement,
        final_proof.proof_digest,
        &final_proof.kernel_export,
        &final_proof.chunk_summaries,
        &final_proof.steps,
    )
    .expect("build setup shape");
    let (pk, vk) = setup_rv64im_spartan2_decider_from_shape(&shape).expect("setup rv64im spartan2 decider");
    let decider_proof =
        prove_rv64im_spartan2_decider(&pk, &statement, &final_proof).expect("prove rv64im spartan2 decider");

    verify_rv64im_spartan2_decider(&vk, statement.public_statement_digest, &relation, &decider_proof)
        .expect("baseline theorem statement must verify");

    let mut tampered_public_statement_digest = statement.public_statement_digest;
    tampered_public_statement_digest[0] ^= 1;
    let err = verify_rv64im_spartan2_decider(&vk, tampered_public_statement_digest, &relation, &decider_proof)
        .expect_err("tampered public-statement digest must fail");
    assert!(format!("{err}").contains("public IO mismatch"));
}
