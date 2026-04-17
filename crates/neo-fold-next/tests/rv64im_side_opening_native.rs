use std::sync::Arc;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_eval_claim_relation_from_accepted_artifact,
    build_rv64im_side_opening_relation_from_accepted_artifact, setup_rv64im_side_binding,
    setup_rv64im_side_opening_spartan_cached, verify_rv64im_side_eval_claim_relation,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_nightstream_from_public_proof, build_rv64im_side_proof, verify_rv64im_side_proof,
};
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imProofInput,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

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
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_opening_native_round_trips() {
    let public_proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (nightstream_statement, _nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    let side_proof = build_rv64im_side_proof(&nightstream_statement, &accepted_artifact).expect("build side proof");
    let statement = side_proof
        .binding_statement(&nightstream_statement)
        .expect("build side binding statement");
    let (opening_statement, opening_witness) =
        build_rv64im_side_opening_relation_from_accepted_artifact(&accepted_artifact)
            .expect("build side opening relation");
    let opening_keys =
        setup_rv64im_side_opening_spartan_cached(&opening_statement, &opening_witness).expect("setup side opening");
    let (_, vk) = setup_rv64im_side_binding(&statement, side_proof.opening_public()).expect("setup side binding");

    verify_rv64im_side_proof(
        &opening_keys.as_ref().1,
        &vk,
        &nightstream_statement,
        &accepted_artifact.statement,
        &side_proof,
    )
    .expect("verify side proof");
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_side_eval_relation_rejects_tampered_phase0_witness() {
    let public_proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (statement, mut witness) = build_rv64im_side_eval_claim_relation_from_accepted_artifact(&accepted_artifact)
        .expect("build side eval-claim relation");

    let first = witness
        .claim_witnesses
        .first_mut()
        .expect("phase0 claim witness");
    let opened_object_witness = Arc::make_mut(&mut first.witness);
    opened_object_witness.packed_columns[0].rows[0][0] += F::ONE;

    let err = verify_rv64im_side_eval_claim_relation(&statement, &witness)
        .expect_err("tampered phase0 witness must be rejected");
    assert!(
        err.to_string().contains("payload")
            || err.to_string().contains("claim witnesses")
            || err.to_string().contains("side-eval-claim relation"),
        "unexpected rejection error: {err}"
    );
}
