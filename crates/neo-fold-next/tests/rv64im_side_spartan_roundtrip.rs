use std::sync::Arc;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_eval_claim_relation_from_accepted_artifact, debug_check_rv64im_side_binding_circuit,
    prove_rv64im_side_binding, setup_rv64im_side_binding, setup_rv64im_side_binding_cached, verify_rv64im_side_binding,
};
use neo_fold_next::nightstream::rv64im::{build_rv64im_nightstream_from_public_proof, build_rv64im_side_proof};
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

fn side_shape_signature(witness: &neo_fold_next::nightstream::rv64im::Rv64imSideOpeningPublic) -> (usize, usize) {
    (witness.opened_objects.len(), witness.evals.len())
}

#[test]
#[ignore = "expensive: side Spartan round-trip rebuilds the Nightstream side theorem path"]
fn rv64im_side_binding_roundtrip_with_same_and_rebuilt_vk() {
    let public_proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (nightstream_statement, _nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    let side_proof = build_rv64im_side_proof(&nightstream_statement, &accepted_artifact).expect("build side proof");
    let (_, phase0_witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&accepted_artifact)
        .expect("build side eval-claim relation");
    let statement = side_proof
        .binding_statement(&nightstream_statement)
        .expect("build side binding statement");
    let witness = side_proof.opening_public().clone();
    debug_check_rv64im_side_binding_circuit(&statement, &witness, &phase0_witness.claim_witnesses)
        .expect("debug check side binding");

    let (pk, vk) = setup_rv64im_side_binding(&statement, &witness).expect("setup side binding");
    let proof = prove_rv64im_side_binding(&pk, &statement, &witness, &phase0_witness.claim_witnesses)
        .expect("prove side binding");
    verify_rv64im_side_binding(&vk, &statement, &proof).expect("verify side binding with same vk");

    let (_, rebuilt_vk) = setup_rv64im_side_binding(&statement, &witness).expect("rebuild side binding vk");
    verify_rv64im_side_binding(&rebuilt_vk, &statement, &proof).expect("verify side binding with rebuilt vk");
}

#[test]
#[ignore = "expensive: validates that direct side Spartan setup cache keys on circuit shape"]
fn rv64im_side_binding_setup_reuses_same_shape_for_rebuilt_same_case() {
    let public_proof_a =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof A");
    let accepted_artifact_a = build_rv64im_accepted_proof_artifact(&public_proof_a).expect("build accepted artifact A");
    let (nightstream_statement_a, _nightstream_proof_a) =
        build_rv64im_nightstream_from_public_proof(&public_proof_a).expect("build nightstream proof A");
    let side_proof_a =
        build_rv64im_side_proof(&nightstream_statement_a, &accepted_artifact_a).expect("build side proof A");
    let statement_a = side_proof_a
        .binding_statement(&nightstream_statement_a)
        .expect("build side binding statement A");
    let witness_a = side_proof_a.opening_public().clone();
    let keys_a = setup_rv64im_side_binding_cached(&statement_a, &witness_a).expect("setup side binding A");

    let public_proof_b =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof B");
    let accepted_artifact_b = build_rv64im_accepted_proof_artifact(&public_proof_b).expect("build accepted artifact B");
    let (nightstream_statement_b, _nightstream_proof_b) =
        build_rv64im_nightstream_from_public_proof(&public_proof_b).expect("build nightstream proof B");
    let side_proof_b =
        build_rv64im_side_proof(&nightstream_statement_b, &accepted_artifact_b).expect("build side proof B");
    let (_, phase0_witness_b) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&accepted_artifact_b)
        .expect("build side eval-claim relation B");
    let statement_b = side_proof_b
        .binding_statement(&nightstream_statement_b)
        .expect("build side binding statement B");
    let witness_b = side_proof_b.opening_public().clone();
    debug_check_rv64im_side_binding_circuit(&statement_b, &witness_b, &phase0_witness_b.claim_witnesses)
        .expect("debug check side binding B");

    let shape_a = side_shape_signature(&witness_a);
    let shape_b = side_shape_signature(&witness_b);
    assert_eq!(
        shape_a, shape_b,
        "side Spartan setup reuse test requires same-shape witnesses"
    );
    assert_eq!(statement_a, statement_b, "rebuilt same-case side statements must match");
    let keys_b = setup_rv64im_side_binding_cached(&statement_b, &witness_b).expect("setup side binding B");
    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "same-shape side setup should reuse the cached keypair"
    );

    let proof_b = prove_rv64im_side_binding(
        &keys_a.as_ref().0,
        &statement_b,
        &witness_b,
        &phase0_witness_b.claim_witnesses,
    )
    .expect("prove side binding B with setup from A");
    verify_rv64im_side_binding(&keys_b.as_ref().1, &statement_b, &proof_b)
        .expect("verify side binding B with verifier key from A");
}
