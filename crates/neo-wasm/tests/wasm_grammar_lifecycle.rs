//! End-to-end folding proof of an event-bound trace: preprocess with the
//! grammar initial digest, fold a trace containing export entry events,
//! import events (multi-block, mid-args groups), and export exit events,
//! and verify against the final-state claim. This is the capstone showing
//! the grammar machinery composes with the lifecycle pipeline; per-row
//! CCS and native-checker coverage lives in the other grammar test files.

mod common;

use common::audit::{prove_batched, verify_with_transcript, AuditProveError};
use common::grammar_fixture::{expected_transcript, grammar_lifecycle_setup, GrammarLifecycleSetup, ENTRY_CLAIMS};
use neo_wasm::{grammar_top_level_initial_state_digest, preprocess_seeded_batched};
use p3_field::PrimeCharacteristicRing;

/// Every batch (including the padded tail) of the grammar trace satisfies
/// the batched R1CS; on failure the diagnostics name the step and tag.
#[test]
fn grammar_trace_satisfies_batched_ccs() {
    let setup = grammar_lifecycle_setup();
    let batch_size = 8;
    let batched = neo_wasm::batch::build_batched_wasm_ccs(batch_size).expect("batched CCS");
    let vm = neo_wasm::WasmVmSpec::default();
    let n_single = vm.core_ccs_spec().structure.n;
    let catalog = vm.constraint_catalog();
    let n_batches = neo_wasm::batch::batch_count(setup.trace.len(), batch_size);
    for batch_idx in 0..n_batches {
        let witness = neo_wasm::batch::build_batched_witness(&setup.trace, batch_size, batch_idx);
        batched
            .sparse_r1cs
            .is_satisfied_by(&witness)
            .unwrap_or_else(|err| {
                let detail = err.to_string();
                let row = detail
                    .split_once("row ")
                    .and_then(|(_, rest)| rest.split_once(|c: char| !c.is_ascii_digit()))
                    .and_then(|(digits, _)| digits.parse::<usize>().ok());
                let context = row.map(|row| {
                    if row < batch_size * n_single {
                        format!(
                            "step {} constraint {:?}",
                            batch_idx * batch_size + row / n_single,
                            catalog.row_tags.get(row % n_single)
                        )
                    } else {
                        format!("link row {}", row - batch_size * n_single)
                    }
                });
                panic!("batch {batch_idx} rejected: {err} ({context:?})");
            });
    }
}

#[test]
fn grammar_anchor_rejects_missing_or_mismatched_export() {
    let setup = grammar_lifecycle_setup();
    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&setup.component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(setup.run_fref));

    let missing = neo_wasm::grammar_top_level_initial_state(
        &artifacts.tables,
        entry_pc,
        &Default::default(),
        setup.run_fref,
        Default::default(),
    )
    .expect_err("selected export needs a template");
    assert!(missing.to_string().contains("no export template"));

    let mismatched = neo_wasm::grammar_top_level_initial_state(
        &artifacts.tables,
        entry_pc,
        &setup.grammar,
        setup.run_fref.wrapping_add(1),
        Default::default(),
    )
    .expect_err("selected export fref must own the entry pc");
    assert!(mismatched.to_string().contains("different pc"));
}

#[test]
fn grammar_folding_proof_covers_import_and_export_events() {
    let setup = grammar_lifecycle_setup();
    let GrammarLifecycleSetup {
        trace,
        grammar,
        run_fref,
        component_bytes,
        ..
    } = setup;

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(run_fref));
    // The anchor is per-program (mode, entry pc, entry schedule) — claim
    // inputs are NOT anchored; they are bound by the final-chain transcript
    // check below.
    let digest =
        grammar_top_level_initial_state_digest(&artifacts.tables, entry_pc, &grammar, run_fref, Default::default())
            .expect("grammar anchor");
    // The verifier's constructed initial state must be exactly the trace's
    // opening boundary.
    assert_eq!(
        digest,
        neo_wasm::semantic_state_digest(trace[0].state_before),
        "verifier initial state must match the trace's first before-state"
    );
    let f = p3_goldilocks::Goldilocks::from_u64;
    let initial_comm_chain = neo_wasm::CommChainState::new([f(11), f(22), f(33), f(44)]);
    let initial_state =
        neo_wasm::grammar_top_level_initial_state(&artifacts.tables, entry_pc, &grammar, run_fref, initial_comm_chain)
            .expect("grammar anchor");
    let initial_digest =
        grammar_top_level_initial_state_digest(&artifacts.tables, entry_pc, &grammar, run_fref, initial_comm_chain)
            .expect("grammar anchor");
    assert_eq!(initial_state.comm_chain, initial_comm_chain.canonical_u64());
    assert_eq!(initial_digest, neo_wasm::semantic_state_digest(initial_state));
    assert_ne!(initial_digest, digest);

    // batch_size 8 forces perm groups, gather runs, and the entry/exit
    // boundaries across batch edges, so the semantic digest must carry the
    // whole grammar state (chain, absorb, schedule, oracles) correctly.
    // Check the claim rather than trusting the trace shape: some batch
    // boundary must fall mid-permutation and some must carry live schedule
    // state, otherwise this test lost its cross-batch coverage.
    let batch_size = 8;
    let boundary_states = (batch_size..trace.len())
        .step_by(batch_size)
        .map(|row| trace[row].state_before)
        .collect::<Vec<_>>();
    assert!(
        boundary_states
            .iter()
            .any(|s| s.event_absorb.perm_round != 0 || s.event_absorb.perm_pending),
        "no batch boundary falls inside a permutation group"
    );
    assert!(
        boundary_states
            .iter()
            .any(|s| s.grammar.events_remaining != 0 || s.grammar.slot_cursor != 0),
        "no batch boundary carries live gather/schedule state"
    );

    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &trace, batch_size).expect("prove");
    let final_state = common::final_state(&trace);

    // Transcript binding: verification succeeds only with the claimed
    // transcript — export entry (with the claim inputs), the two import
    // calls, and the export exit — and rejects a transcript claiming
    // different inputs. This is the verifier's input check: per-invocation
    // data never touches preprocessing.
    verify_with_transcript(
        &prep,
        &proof,
        final_state,
        Default::default(),
        &expected_transcript(&grammar, run_fref, &ENTRY_CLAIMS),
    )
    .expect("verify with the claimed transcript");
    assert!(
        matches!(
            verify_with_transcript(
                &prep,
                &proof,
                final_state,
                Default::default(),
                &expected_transcript(&grammar, run_fref, &[500, 999])
            ),
            Err(AuditProveError::TranscriptMismatch)
        ),
        "a transcript claiming different inputs must be rejected"
    );
}
