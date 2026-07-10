//! Grammar-mode traces: the chain absorbs embedder grammar events staged by
//! `HostEventGather` rows instead of raw host-call records. Every row is
//! CCS-checked; the gather contents themselves are stage-C territory (their
//! binding to the grammar ROM), so soundness tests here cover the mode
//! gating, not gather forgery.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{GrammarEvent, HostEventGrammar, ImportTemplate, Limb, SlotSource};
use neo_wasm::layout::{COL_GATHER_ACTIVE, COL_GRAMMAR_MODE_AFTER, COL_RAW_ARGS_ACTIVE};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmRowKind, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// Example embedder grammar for the mul/sink component: `mul(x, y) -> r`
/// expands to a two-event template (args event + result event referencing a
/// shared oracle), `sink(x)` to a single event.
fn test_grammar(mul_fref: u32, sink_fref: u32) -> HostEventGrammar {
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(
        mul_fref,
        ImportTemplate {
            pre_result: vec![GrammarEvent {
                discriminant: 10,
                slots: slots(&[
                    (0, SlotSource::Oracle { idx: 0 }),
                    (1, arg(0, Limb::Lo)),
                    (2, arg(1, Limb::Lo)),
                    (3, SlotSource::Const(5)),
                ]),
            }],
            post_result: vec![GrammarEvent {
                discriminant: 12,
                slots: slots(&[
                    (0, SlotSource::ResultElem { limb: Limb::Lo }),
                    (1, SlotSource::Oracle { idx: 0 }),
                ]),
            }],
            oracle_count: 1,
        },
    );
    grammar.imports.insert(
        sink_fref,
        ImportTemplate {
            pre_result: vec![GrammarEvent {
                discriminant: 7,
                slots: slots(&[(0, arg(0, Limb::Lo))]),
            }],
            post_result: vec![],
            oracle_count: 0,
        },
    );
    grammar
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

fn run_component() -> neo_wasm::WasmtimeTraceRun {
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-mul", |_store, (x, y): (i32, i32)| Ok((x * y,)))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-mul: {err}")))?;
        linker
            .root()
            .func_wrap("host-sink", |_store, (_x,): (i32,)| Ok(()))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-sink: {err}")))
    })
    .expect("component trace run")
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

/// Grammar trace for the two-call component, with oracles `[100]` for mul
/// and `[]` for sink.
fn grammar_trace() -> Vec<WasmVmStep> {
    let run = run_component();
    // Resolve frefs from a raw normalization of the same run.
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let frefs = host_call_frefs(&raw);
    assert_eq!(frefs.len(), 2);
    let grammar = test_grammar(frefs[0], frefs[1]);
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[vec![100], vec![]])
        .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);
    trace
}

#[test]
fn grammar_trace_folds_expanded_blocks() {
    let trace = grammar_trace();
    assert!(trace.iter().all(|row| row.state_before.grammar_mode));

    // Three grammar events → three gather rows staging the expected blocks.
    let staged: Vec<[u64; 8]> = trace
        .iter()
        .filter(|row| row.row_kind.is_host_event_gather())
        .map(|row| row.state_after.event_absorb.evbuf)
        .collect();
    assert_eq!(
        staged,
        vec![
            [10, 100, 7, 6, 5, 0, 0, 0],  // mul pre-result event
            [12, 42, 100, 0, 0, 0, 0, 0], // mul post-result event
            [7, 42, 0, 0, 0, 0, 0, 0],    // sink event
        ],
    );

    // The carried chain equals the fold of exactly those blocks.
    let f = p3_goldilocks::Goldilocks::from_u64;
    let mut chain = [p3_goldilocks::Goldilocks::ZERO; 4];
    for block in &staged {
        chain = neo_wasm::comm_chain::commit_event(chain, f(block[0]), core::array::from_fn(|i| f(block[1 + i])));
    }
    let final_chain = trace.last().expect("rows").state_after.comm_chain;
    assert_eq!(
        final_chain,
        chain.map(|limb| p3_field::PrimeField64::as_canonical_u64(&limb))
    );
}

#[test]
fn missing_template_is_rejected() {
    let run = run_component();
    let grammar = HostEventGrammar::default();
    assert!(neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &[]).is_err());
}

/// The raw absorb machinery must stay de-gated in grammar mode: forging a
/// raw arg-row mask back on is CCS-rejected.
#[test]
fn ccs_rejects_raw_machinery_in_grammar_mode() {
    let trace = grammar_trace();
    let arg_row = trace
        .iter()
        .find(|row| row.row_kind == WasmRowKind::Aux(neo_wasm::WasmAuxOpcode::HostCallArg))
        .expect("arg row");
    let mut witness = build_witness_vector(arg_row);
    common::assert_satisfied(&witness, "untampered grammar arg row");
    witness[COL_RAW_ARGS_ACTIVE] = neo_math::F::ONE;
    common::assert_rejected(&witness, "grammar arg row with the raw machinery forged on");
}

/// Gather rows only exist in grammar mode: claiming one on a raw trace row
/// is CCS-rejected.
#[test]
fn ccs_rejects_gather_row_in_raw_mode() {
    let run = run_component();
    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let arg_row = raw
        .iter()
        .find(|row| row.row_kind == WasmRowKind::Aux(neo_wasm::WasmAuxOpcode::HostCallArg))
        .expect("arg row");
    let mut witness = build_witness_vector(arg_row);
    common::assert_satisfied(&witness, "untampered raw arg row");
    witness[COL_GATHER_ACTIVE] = neo_math::F::ONE;
    common::assert_rejected(&witness, "raw row claiming the gather kind");
}

/// The mode flag is a carried constant: flipping it mid-trace is rejected.
#[test]
fn ccs_rejects_mode_flip() {
    let trace = grammar_trace();
    let mut witness = build_witness_vector(&trace[0]);
    common::assert_satisfied(&witness, "untampered grammar row");
    witness[COL_GRAMMAR_MODE_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "row flipping the per-program mode constant");
}
