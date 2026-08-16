//! One real Nebula proof over an event-bound trace: host-event-anchored
//! preprocessing (bindings ROMs preloaded, host calls chain-bound), prove,
//! verify against the final-state claim, and check the claimed transcript
//! folds to the digest-bound final chain.

mod common;

use common::host_event_fixture::{expected_transcript, host_event_lifecycle_setup};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::paper::params::Params;
use p3_field::PrimeCharacteristicRing;

fn nebula_test_params() -> Params {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 14,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .expect("test SuperNeo parameters");
    Params::test_only_from_neo_params(raw)
}

#[test]
#[ignore = "full Nebula proof, ~30 minutes; run explicitly with --ignored"]
fn wasm_nebula_proves_a_host_event_template_trace() {
    let setup = host_event_lifecycle_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(setup.run_fref));
    // r = 12: the component fixture's pc ROMs plus the bindings ROM families
    // outgrow the default test geometry's 2^10 ROM cells.
    let geometry = NebulaParams::new(12, 12, 64, 1024, 16).expect("bindings test geometry");
    let prep = neo_wasm::nebula::preprocess_seeded_host_events_test_only(
        nebula_test_params(),
        neo_wasm::WasmNebulaProfile::test_profile_with_geometry(geometry),
        &artifacts,
        entry_pc,
        &setup.bindings,
        setup.run_fref,
        0x57a5_0002,
        Default::default(),
    )
    .expect("bindings Nebula preprocessing");

    let proof = neo_wasm::prove(&prep, &setup.trace).expect("bindings Nebula proof");
    let final_state = common::final_state(&setup.trace);
    neo_wasm::verify(&prep, &proof, final_state).expect("bindings Nebula verification");

    // Transcript binding on top of the digest-bound final state: the
    // claimed event blocks fold to the final chain; a different transcript
    // folds elsewhere.
    let expected = expected_transcript(&setup.bindings, setup.run_fref);
    let expected_chain = neo_wasm::comm_chain::fold_event_blocks(Default::default(), &expected).canonical_u64();
    assert_eq!(
        final_state.comm_chain, expected_chain,
        "the claimed transcript must fold to the proven final chain"
    );
    let mut wrong = expected;
    wrong[0][1] += p3_goldilocks::Goldilocks::ONE;
    let wrong_chain = neo_wasm::comm_chain::fold_event_blocks(Default::default(), &wrong).canonical_u64();
    assert_ne!(
        final_state.comm_chain, wrong_chain,
        "a different transcript must not fold to the final chain"
    );
}
