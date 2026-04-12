use std::sync::Arc;

use neo_fold_next::nightstream::rv64im::{
    build_rv64im_bound_side_proof_bundle_from_accepted_artifact, build_rv64im_nightstream_from_public_proof,
    build_rv64im_side_spartan_from_accepted_artifact, debug_check_rv64im_side_spartan_circuit,
    prove_rv64im_side_spartan, setup_rv64im_side_spartan, setup_rv64im_side_spartan_cached,
    setup_rv64im_side_spartan_from_accepted_artifact, verify_rv64im_side_spartan,
};
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_published_proof_seam, parity_source_cases,
    prove_rv64im_public_proof, Rv64imProofInput,
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

fn side_shape_signature(
    witness: &neo_fold_next::nightstream::rv64im::Rv64imAuthoritativeSidePublicInstance,
) -> (usize, usize, usize) {
    (
        witness.side_surface_public.targets.len(),
        witness.opened_objects.len(),
        witness.evals.len(),
    )
}

#[test]
#[ignore = "expensive: side Spartan round-trip rebuilds the Nightstream side theorem path"]
fn rv64im_side_spartan_roundtrip_with_same_and_rebuilt_vk() {
    let public_proof =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof");
    let published_seam = build_rv64im_published_proof_seam(&public_proof).expect("build published seam");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (nightstream_statement, _nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&public_proof).expect("build nightstream proof");
    let side_bundle =
        build_rv64im_bound_side_proof_bundle_from_accepted_artifact(&nightstream_statement, &accepted_artifact)
            .expect("build bound side bundle");

    let (statement, witness) =
        build_rv64im_side_spartan_from_accepted_artifact(&nightstream_statement, &side_bundle, &accepted_artifact)
            .expect("build side spartan statement/witness");
    debug_check_rv64im_side_spartan_circuit(&statement, &witness).expect("debug check side spartan");

    let (pk, vk) = setup_rv64im_side_spartan(&statement, &witness).expect("setup side spartan");
    let proof = prove_rv64im_side_spartan(&pk, &statement, &witness).expect("prove side spartan");
    verify_rv64im_side_spartan(&vk, &statement, &proof).expect("verify side spartan with same vk");

    let (_, rebuilt_vk) = setup_rv64im_side_spartan_from_accepted_artifact(
        &nightstream_statement,
        &side_bundle,
        &published_seam.accepted_artifact,
    )
    .expect("rebuild side spartan vk");
    verify_rv64im_side_spartan(&rebuilt_vk, &statement, &proof).expect("verify side spartan with rebuilt vk");
}

#[test]
#[ignore = "expensive: validates that direct side Spartan setup cache keys on circuit shape"]
fn rv64im_side_spartan_setup_reuses_same_shape_for_rebuilt_same_case() {
    let public_proof_a =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof A");
    let accepted_artifact_a = build_rv64im_accepted_proof_artifact(&public_proof_a).expect("build accepted artifact A");
    let (nightstream_statement_a, _nightstream_proof_a) =
        build_rv64im_nightstream_from_public_proof(&public_proof_a).expect("build nightstream proof A");
    let side_bundle_a =
        build_rv64im_bound_side_proof_bundle_from_accepted_artifact(&nightstream_statement_a, &accepted_artifact_a)
            .expect("build bound side bundle A");
    let (statement_a, witness_a) = build_rv64im_side_spartan_from_accepted_artifact(
        &nightstream_statement_a,
        &side_bundle_a,
        &accepted_artifact_a,
    )
    .expect("build side spartan statement/witness A");
    let keys_a = setup_rv64im_side_spartan_cached(&statement_a, &witness_a).expect("setup side spartan A");

    let public_proof_b =
        prove_rv64im_public_proof(&proof_input("control_flow_jal_skip_ecall")).expect("prove rv64im public proof B");
    let accepted_artifact_b = build_rv64im_accepted_proof_artifact(&public_proof_b).expect("build accepted artifact B");
    let (nightstream_statement_b, _nightstream_proof_b) =
        build_rv64im_nightstream_from_public_proof(&public_proof_b).expect("build nightstream proof B");
    let side_bundle_b =
        build_rv64im_bound_side_proof_bundle_from_accepted_artifact(&nightstream_statement_b, &accepted_artifact_b)
            .expect("build bound side bundle B");
    let (statement_b, witness_b) = build_rv64im_side_spartan_from_accepted_artifact(
        &nightstream_statement_b,
        &side_bundle_b,
        &accepted_artifact_b,
    )
    .expect("build side spartan statement/witness B");
    debug_check_rv64im_side_spartan_circuit(&statement_b, &witness_b).expect("debug check side spartan B");

    let shape_a = side_shape_signature(&witness_a);
    let shape_b = side_shape_signature(&witness_b);
    assert_eq!(
        shape_a, shape_b,
        "side Spartan setup reuse test requires same-shape witnesses"
    );
    assert_eq!(statement_a, statement_b, "rebuilt same-case side statements must match");
    let keys_b = setup_rv64im_side_spartan_cached(&statement_b, &witness_b).expect("setup side spartan B");
    assert!(
        Arc::ptr_eq(&keys_a, &keys_b),
        "same-shape side setup should reuse the cached keypair"
    );

    let proof_b = prove_rv64im_side_spartan(&keys_a.as_ref().0, &statement_b, &witness_b)
        .expect("prove side spartan B with setup from A");
    verify_rv64im_side_spartan(&keys_b.as_ref().1, &statement_b, &proof_b)
        .expect("verify side spartan B with verifier key from A");
}
