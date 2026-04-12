use std::sync::Arc;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_direct_side_relation_from_accepted_artifact, verify_rv64im_direct_side_relation,
};
use neo_fold_next::nightstream::rv64im::build_rv64im_side_proof_bundle_from_accepted_artifact;
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
#[ignore = "expensive: direct side relation currently depends on the full Nightstream proof build"]
fn rv64im_direct_side_relation_round_trips() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&accepted_artifact).expect("build side bundle");
    let (nightstream_statement, _nightstream_proof) =
        neo_fold_next::nightstream::rv64im::build_rv64im_nightstream_from_public_proof(&proof)
            .expect("build nightstream proof");

    let (statement, witness) = build_rv64im_direct_side_relation_from_accepted_artifact(
        &nightstream_statement,
        &side_bundle,
        &accepted_artifact,
    )
    .expect("build direct side relation");

    verify_rv64im_direct_side_relation(&statement, &witness).expect("verify direct side relation");
}

#[test]
#[ignore = "expensive: direct side relation currently depends on the full Nightstream proof build"]
fn rv64im_direct_side_relation_rejects_tampered_phase0_witness() {
    let proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let side_bundle =
        build_rv64im_side_proof_bundle_from_accepted_artifact(&accepted_artifact).expect("build side bundle");
    let (nightstream_statement, _nightstream_proof) =
        neo_fold_next::nightstream::rv64im::build_rv64im_nightstream_from_public_proof(&proof)
            .expect("build nightstream proof");

    let (statement, mut witness) = build_rv64im_direct_side_relation_from_accepted_artifact(
        &nightstream_statement,
        &side_bundle,
        &accepted_artifact,
    )
    .expect("build direct side relation");

    let opened_object_witness = Arc::make_mut(&mut witness.phase0_claim_witnesses[0].witness);
    opened_object_witness.packed_columns[0].rows[0][0] += F::ONE;

    let err =
        verify_rv64im_direct_side_relation(&statement, &witness).expect_err("tampered phase0 witness must be rejected");
    assert!(
        err.to_string().contains("side-eval-claim relation")
            || err.to_string().contains("payload")
            || err.to_string().contains("claim witnesses"),
        "unexpected error: {err}"
    );
}
