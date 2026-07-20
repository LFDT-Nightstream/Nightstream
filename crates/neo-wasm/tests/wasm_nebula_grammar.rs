//! One real Nebula proof over a grammar-mode trace: grammar-anchored
//! preprocessing (grammar ROMs preloaded, host calls chain-bound), prove,
//! verify against the final-state claim, and check the claimed transcript
//! folds to the digest-bound final chain.

mod common;

use common::grammar_fixture::{expected_transcript, grammar_lifecycle_setup, ENTRY_CLAIMS};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::paper::params::Params;
use p3_field::PrimeField64;

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
fn wasm_nebula_proves_a_grammar_template_trace() {
    let setup = grammar_lifecycle_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(setup.run_fref));
    // r = 12: the component fixture's pc ROMs plus the grammar ROM families
    // outgrow the default test geometry's 2^10 ROM cells.
    let geometry = NebulaParams::new(12, 12, 64, 1024, 16).expect("grammar test geometry");
    let prep = neo_wasm::nebula::preprocess_seeded_grammar_test_only(
        nebula_test_params(),
        neo_wasm::WasmNebulaProfile::test_profile_with_geometry(geometry),
        &artifacts,
        &setup.initial_locals,
        entry_pc,
        &setup.grammar,
        setup.run_fref,
        0x57a5_0002,
    )
    .expect("grammar Nebula preprocessing");

    let proof = neo_wasm::prove(&prep, &setup.trace).expect("grammar Nebula proof");
    let final_state = common::final_state(&setup.trace);
    neo_wasm::verify(&prep, &proof, final_state).expect("grammar Nebula verification");

    // Transcript binding on top of the digest-bound final state: the
    // claimed event blocks (with the true claim inputs) fold to the final
    // chain; a transcript claiming different inputs folds elsewhere.
    let fold = |inputs: &[u64]| {
        neo_wasm::comm_chain::fold_event_blocks(&expected_transcript(&setup.grammar, setup.run_fref, inputs))
            .map(|limb| limb.as_canonical_u64())
    };
    assert_eq!(
        final_state.comm_chain,
        fold(&ENTRY_CLAIMS),
        "the claimed transcript must fold to the proven final chain"
    );
    assert_ne!(
        final_state.comm_chain,
        fold(&[500, 999]),
        "a transcript claiming different inputs must not fold to the final chain"
    );
}
