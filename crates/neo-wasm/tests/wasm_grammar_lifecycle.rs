//! End-to-end folding proof of a grammar-mode trace: preprocess with the
//! grammar initial digest, fold a trace containing export entry events,
//! import events (multi-block, mid-args groups), and export exit events,
//! and verify against the final-state claim. This is the capstone showing
//! the grammar machinery composes with the lifecycle pipeline; per-row
//! CCS and native-checker coverage lives in the other grammar test files.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{ExportTemplate, GrammarEvent, HostEventGrammar, ImportTemplate, Limb, SlotSource};
use common::audit::{prove_batched, verify, verify_with_transcript, AuditProveError};
use neo_wasm::{grammar_top_level_initial_state_digest, preprocess_seeded_batched, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

fn mul_sink_component_wat() -> &'static str {
    r#"
    (component
      (type $host-mul (func (param "x" s32) (param "y" s32) (result s32)))
      (type $host-sink (func (param "x" s32)))
      (type $run-type (func (result s32)))
      (import "host-mul" (func $host-mul (type $host-mul)))
      (import "host-sink" (func $host-sink (type $host-sink)))
      (core module $m
        (import "" "0" (func $mul (param i32 i32) (result i32)))
        (import "" "1" (func $sink (param i32)))
        (func (export "run") (result i32)
          (local i32)
          i32.const 7
          i32.const 6
          call $mul
          local.tee 0
          call $sink
          local.get 0))
      (core func $lowered-mul (canon lower (func $host-mul)))
      (core func $lowered-sink (canon lower (func $host-sink)))
      (core instance $lowered-host
        (export "0" (func $lowered-mul))
        (export "1" (func $lowered-sink)))
      (core instance $i
        (instantiate $m
          (with "" (instance $lowered-host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

/// Import templates for mul/sink plus an export boundary for `run`:
/// Enter + Activation at entry, Return-with-output at exit.
fn test_grammar(mul_fref: u32, sink_fref: u32, run_fref: u32) -> HostEventGrammar {
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let oracle = |idx| SlotSource::Claim { idx };
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(
        mul_fref,
        ImportTemplate {
            pre_result: vec![GrammarEvent::op(
                10,
                slots(&[(0, oracle(0)), (1, arg(0, Limb::Lo)), (2, arg(1, Limb::Lo))]),
            )],
            post_result: vec![GrammarEvent::op(
                12,
                slots(&[(0, SlotSource::ResultElem { limb: Limb::Lo }), (1, oracle(0))]),
            )],
            claim_count: 1,
        },
    );
    grammar.imports.insert(
        sink_fref,
        ImportTemplate {
            pre_result: vec![GrammarEvent::op(7, slots(&[(0, arg(0, Limb::Lo))]))],
            post_result: vec![],
            claim_count: 0,
        },
    );
    grammar.exports.insert(
        run_fref,
        ExportTemplate {
            entry: vec![
                GrammarEvent::op(20, slots(&[(0, SlotSource::Const(55))])),
                GrammarEvent::op(
                    8,
                    slots(&[(1, SlotSource::Claim { idx: 0 }), (3, SlotSource::Claim { idx: 1 })]),
                ),
            ],
            exit: vec![GrammarEvent::op(
                17,
                slots(&[(1, SlotSource::OutputElem { limb: Limb::Lo })]),
            )],
            entry_claim_count: 2,
            exit_claim_count: 0,
        },
    );
    grammar
}

/// The mul import is the one with a post-result event; sink has none.
fn mul_fref(grammar: &HostEventGrammar) -> u32 {
    *grammar
        .imports
        .iter()
        .find(|(_, t)| !t.post_result.is_empty())
        .expect("mul template")
        .0
}

fn sink_fref(grammar: &HostEventGrammar) -> u32 {
    *grammar
        .imports
        .iter()
        .find(|(_, t)| t.post_result.is_empty())
        .expect("sink template")
        .0
}

fn host_call_frefs(trace: &[WasmVmStep]) -> Vec<u32> {
    trace
        .iter()
        .filter(|row| {
            row.row_kind.is_program()
                && matches!(row.opcode, neo_wasm::WasmOpcode::Call)
                && !row.target_function_is_guest
        })
        .map(|row| row.state_after.host_callee_fref)
        .collect()
}

struct GrammarLifecycleSetup {
    trace: Vec<WasmVmStep>,
    grammar: HostEventGrammar,
    run_fref: u32,
    component_bytes: Vec<u8>,
}

fn grammar_lifecycle_setup() -> GrammarLifecycleSetup {
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-mul", |mut store, (x, y): (i32, i32)| {
                // The mul template consumes one oracle word; record it at
                // call time (the grammar hand-off path).
                store.data_mut().record_call_claims(&[100])?;
                Ok((x * y,))
            })
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-mul: {err}")))?;
        linker
            .root()
            .func_wrap("host-sink", |_store, (_x,): (i32,)| Ok(()))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-sink: {err}")))
    })
    .expect("component run");

    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    let run_fref = raw
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref;
    let grammar = test_grammar(frefs[0], frefs[1], run_fref);

    let turns = [neo_wasm::event_grammar::TurnClaims {
        entry: vec![500, 501],
        exit: vec![],
    }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &turns).expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    GrammarLifecycleSetup {
        trace,
        grammar,
        run_fref,
        component_bytes,
    }
}

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
fn grammar_folding_proof_covers_import_and_export_events() {
    let setup = grammar_lifecycle_setup();
    let GrammarLifecycleSetup {
        trace,
        grammar,
        run_fref,
        component_bytes,
    } = setup;
    let entry_claims = [500u64, 501];

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let entry_pc = common::entry_pc_for_function_ref(&artifacts, u64::from(run_fref));
    // The anchor is per-program (mode, entry pc, entry schedule) — claim
    // inputs are NOT anchored; they are bound by the final-chain transcript
    // check below.
    let digest = grammar_top_level_initial_state_digest(&artifacts.tables, entry_pc, &grammar, run_fref);
    // The verifier's constructed initial state must be exactly the trace's
    // opening boundary.
    assert_eq!(
        digest,
        neo_wasm::semantic_state_digest(trace[0].state_before),
        "verifier initial state must match the trace's first before-state"
    );

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
    let template = grammar.exports.get(&run_fref).expect("export template");
    let expected_transcript = |inputs: &[u64]| -> Vec<[p3_goldilocks::Goldilocks; 8]> {
        let mut blocks = neo_wasm::event_grammar::expand_export_entry(template, inputs).expect("entry");
        let mul = neo_wasm::event_grammar::expand_import_events(
            &grammar.imports[&mul_fref(&grammar)],
            &[(7, 0), (6, 0)],
            Some((42, 0)),
            &[100],
        )
        .expect("mul events");
        blocks.extend(mul.pre_result_blocks);
        blocks.extend(mul.post_result_blocks);
        let sink = neo_wasm::event_grammar::expand_import_events(
            &grammar.imports[&sink_fref(&grammar)],
            &[(42, 0)],
            None,
            &[],
        )
        .expect("sink events");
        blocks.extend(sink.pre_result_blocks);
        blocks.extend(neo_wasm::event_grammar::expand_export_exit(template, Some((42, 0)), &[]).expect("exit"));
        blocks
            .into_iter()
            .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
            .collect()
    };
    verify_with_transcript(&prep, &proof, final_state, &expected_transcript(&entry_claims))
        .expect("verify with the claimed transcript");
    assert!(
        matches!(
            verify_with_transcript(&prep, &proof, final_state, &expected_transcript(&[500, 999])),
            Err(AuditProveError::TranscriptMismatch)
        ),
        "a transcript claiming different inputs must be rejected"
    );

    // The verifier's mode pinning is real: an anchor that differs from the
    // grammar initial state in *only* the grammar_mode bit must not accept
    // the trace — every other digested field agreeing means the rejection
    // can only come from the mode bit being bound. The base-step anchor
    // mismatch surfaces as the encoder's structure-violation panic (see
    // wasm_batch.rs semantic_state_rejects_wrong_initial_state_digest);
    // match on its message so an unrelated panic can't masquerade as a
    // successful rejection.
    let mut mode_flipped = neo_wasm::grammar_top_level_initial_state(&artifacts.tables, entry_pc, &grammar, run_fref);
    mode_flipped.grammar_mode = false;
    let flipped_digest = neo_wasm::semantic_state_digest(mode_flipped);
    assert_ne!(flipped_digest, digest, "grammar_mode must contribute to the digest");
    let flipped_prep = preprocess_seeded_batched(batch_size, flipped_digest).expect("flipped prep");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove_batched(&flipped_prep, &trace, batch_size)
    }));
    match outcome {
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&'static str>().copied())
                .unwrap_or("<non-string panic>");
            assert!(
                msg.contains("encoded R1CS F' step must satisfy its structure"),
                "expected encoder structure-violation panic, got: {msg}"
            );
        }
        Ok(Err(_)) => {}
        Ok(Ok(wrong_proof)) => {
            verify(&flipped_prep, &wrong_proof, common::final_state(&trace))
                .expect_err("a grammar trace must not verify against a mode-flipped anchor");
        }
    }
}
