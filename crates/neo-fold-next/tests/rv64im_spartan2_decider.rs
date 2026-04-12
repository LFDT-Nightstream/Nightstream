//! Focused tests for the shell-free RV64IM main-relation Spartan decider.

use neo_fold_next::nightstream::rv64im::{
    build_rv64im_nightstream_linkage_claims, build_rv64im_nightstream_statement_from_final,
    rv64im_nightstream_linkage_root, rv64im_verifier_context_digest,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_decider_relation_from_final,
    main_relation_spartan::debug_check_rv64im_spartan2_decider_circuit, parity_source_cases, prove_rv64im_public_proof,
    prove_rv64im_spartan2_decider, prove_rv64im_spartan2_decider_from_public_proof, setup_rv64im_spartan2_decider,
    setup_rv64im_spartan2_decider_from_public_proof, verify_rv64im_spartan2_decider, Rv64imProofInput,
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
    neo_fold_next::rv64im::final_relation::Rv64imFinalProof,
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
    let _ = setup_rv64im_spartan2_decider(&statement, &final_proof).expect("setup rv64im spartan2 decider");
}

#[test]
#[ignore = "expensive: main-relation Spartan round-trip exceeds developer-memory budget"]
fn rv64im_spartan2_decider_round_trip_without_replay_verifier_input() {
    let (_proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let relation = build_rv64im_decider_relation_from_final(&statement, &final_proof).expect("build decider relation");

    let (pk, vk) = setup_rv64im_spartan2_decider(&statement, &final_proof).expect("setup rv64im spartan2 decider");
    let decider_proof =
        prove_rv64im_spartan2_decider(&pk, &statement, &final_proof).expect("prove rv64im spartan2 decider");

    verify_rv64im_spartan2_decider(&vk, statement.public_statement_digest, &relation, &decider_proof)
        .expect("verify rv64im spartan2 decider");
    assert!(decider_proof.snark_bytes_len() > 0);
}

#[test]
#[ignore = "expensive: main-relation Spartan round-trip exceeds developer-memory budget"]
fn rv64im_spartan2_decider_from_public_proof_round_trip() {
    let (proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let relation = build_rv64im_decider_relation_from_final(&statement, &final_proof).expect("build decider relation");

    let (pk, vk) =
        setup_rv64im_spartan2_decider_from_public_proof(&proof).expect("setup rv64im spartan2 decider from proof");
    let decider_proof =
        prove_rv64im_spartan2_decider_from_public_proof(&pk, &proof).expect("prove rv64im spartan2 decider from proof");

    verify_rv64im_spartan2_decider(&vk, statement.public_statement_digest, &relation, &decider_proof)
        .expect("verify rv64im spartan2 decider");
}

#[test]
#[ignore = "expensive: main-relation Spartan setup exceeds developer-memory budget"]
fn rv64im_spartan2_decider_rejects_tampered_chunk_relation_digest() {
    let (_proof, statement, final_proof, _nightstream_statement) = final_fixture("control_flow_jal_skip_ecall");
    let mut relation =
        build_rv64im_decider_relation_from_final(&statement, &final_proof).expect("build decider relation");

    let (pk, vk) = setup_rv64im_spartan2_decider(&statement, &final_proof).expect("setup rv64im spartan2 decider");
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
    let relation = build_rv64im_decider_relation_from_final(&statement, &final_proof).expect("build decider relation");

    let (pk, vk) = setup_rv64im_spartan2_decider(&statement, &final_proof).expect("setup rv64im spartan2 decider");
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
