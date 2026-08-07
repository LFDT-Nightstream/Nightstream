//! One real Nebula proof over a grammar-mode trace: grammar-anchored
//! preprocessing (grammar ROMs preloaded, host calls chain-bound), prove,
//! verify against the final-state claim, and check the claimed transcript
//! folds to the digest-bound final chain.

mod common;

use common::grammar_fixture::{expected_transcript, grammar_lifecycle_setup, ENTRY_CLAIMS};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::paper::params::Params;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use neo_prover_metal::MetalNifsProver;

fn nebula_test_params() -> Params {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 24,
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
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn wasm_nebula_proves_a_grammar_template_trace() {
    let setup = grammar_lifecycle_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(setup.run_fref));
    // r = 12: the component fixture's pc ROMs plus the grammar ROM families
    // outgrow the default test geometry's 2^10 ROM cells. Split the complete
    // 2^12-ROM + 2^12-RAM scan and the complete grammar trace over one
    // 16-step segment. Wider application batches exceed the fixed
    // SuperNeo row-domain bound; the old batch of three repeated this full
    // scan over several segments.
    let folded_steps = 16;
    let geometry = NebulaParams::new(12, 12, 64, 512, 16).expect("grammar test geometry");
    let batch_size = setup.trace.len().div_ceil(folded_steps);
    let profile = neo_wasm::WasmNebulaProfile::test_profile_with_schedule(geometry, batch_size);
    assert_eq!(profile.memory().steps_per_segment(), folded_steps);
    assert!(profile.memory().steps_per_segment() * profile.batch_size() >= setup.trace.len());
    let prep = neo_wasm::nebula::preprocess_seeded_grammar_test_only(
        nebula_test_params(),
        profile,
        &artifacts,
        &setup.initial_locals,
        entry_pc,
        &setup.grammar,
        setup.run_fref,
        0x57a5_0002,
        Default::default(),
    )
    .expect("grammar Nebula preprocessing");

    let mut prover = MetalNifsProver::new().expect("Metal prover");
    prover.session().reset_activity();
    let proof = neo_wasm::nebula::prove_with_nifs_adapter(&prep, &mut prover, &setup.trace)
        .expect("grammar Nebula proof on Metal");
    let proof_activity = prover.session().activity();
    assert!(
        proof_activity.dispatches > 0,
        "grammar proof must dispatch Metal kernels"
    );
    assert!(
        proof_activity.host_waits > 0,
        "grammar proof must wait for Metal results"
    );

    let final_state = common::final_state(&setup.trace);
    neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, final_state, &mut prover)
        .expect("grammar Nebula verification with Metal openings");
    assert!(
        prover.session().activity().dispatches > proof_activity.dispatches,
        "grammar verification must dispatch Metal opening kernels"
    );

    // Transcript binding on top of the digest-bound final state: the
    // claimed event blocks (with the true claim inputs) fold to the final
    // chain; a transcript claiming different inputs folds elsewhere.
    let fold = |inputs: &[u64]| {
        neo_wasm::comm_chain::fold_event_blocks(
            Default::default(),
            &expected_transcript(&setup.grammar, setup.run_fref, inputs),
        )
        .canonical_u64()
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
