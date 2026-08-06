//! Shared grammar-mode fixture: a component with two grammar-bound host
//! imports (mul with a claim word, sink) and an export boundary template,
//! traced with grammar tables. Used by the F′ audit lifecycle test and the
//! Nebula proof test.

use neo_wasm::comm_chain::{COMM_CHAIN_BLOCK_WORDS, COMM_CHAIN_EVENT_ARGS};
use neo_wasm::event_grammar::{ExportTemplate, GrammarEvent, HostEventGrammar, ImportTemplate, Limb, SlotSource};
use neo_wasm::WasmVmStep;
use p3_field::PrimeCharacteristicRing;

pub const ENTRY_CLAIMS: [u64; 2] = [500, 501];

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
            events: vec![
                GrammarEvent::op(
                    10,
                    slots(&[(0, oracle(0)), (1, arg(0, Limb::Lo)), (2, arg(1, Limb::Lo))]),
                ),
                // The ResultElem Lo slot pushes the host result (atomic
                // import events; args gathered above, before the push); the
                // Hi slot binds the pushed hi lane (0 for the i32 result).
                GrammarEvent::op(
                    12,
                    slots(&[
                        (0, SlotSource::ResultElem { limb: Limb::Lo }),
                        (1, oracle(0)),
                        (2, SlotSource::ResultElem { limb: Limb::Hi }),
                    ]),
                ),
            ],
            claim_count: 1,
        },
    );
    grammar.imports.insert(
        sink_fref,
        ImportTemplate {
            events: vec![GrammarEvent::op(7, slots(&[(0, arg(0, Limb::Lo))]))],
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

/// The mul import is the two-event template; sink has one event.
pub fn mul_fref(grammar: &HostEventGrammar) -> u32 {
    *grammar
        .imports
        .iter()
        .find(|(_, t)| t.events.len() == 2)
        .expect("mul template")
        .0
}

pub fn sink_fref(grammar: &HostEventGrammar) -> u32 {
    *grammar
        .imports
        .iter()
        .find(|(_, t)| t.events.len() == 1)
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

pub struct GrammarLifecycleSetup {
    pub trace: Vec<WasmVmStep>,
    pub grammar: HostEventGrammar,
    pub run_fref: u32,
    pub component_bytes: Vec<u8>,
    pub initial_locals: Vec<u32>,
}

pub fn grammar_lifecycle_setup() -> GrammarLifecycleSetup {
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
        entry: ENTRY_CLAIMS.to_vec(),
        exit: vec![],
        ..Default::default()
    }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &turns,
        Default::default(),
    )
    .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    GrammarLifecycleSetup {
        trace,
        grammar,
        run_fref,
        component_bytes,
        initial_locals: run.initial_locals,
    }
}

/// The transcript the verifier expects for `inputs`: export entry (with the
/// claim inputs), the mul call (claim word 100, result 42), the sink call,
/// and the export exit carrying the output.
pub fn expected_transcript(
    grammar: &HostEventGrammar,
    run_fref: u32,
    inputs: &[u64],
) -> Vec<[p3_goldilocks::Goldilocks; COMM_CHAIN_BLOCK_WORDS]> {
    let template = grammar.exports.get(&run_fref).expect("export template");
    let mut blocks = neo_wasm::event_grammar::expand_export_entry(template, inputs).expect("entry");
    blocks.extend(
        neo_wasm::event_grammar::expand_import_events(
            &grammar.imports[&mul_fref(grammar)],
            &[(7, 0), (6, 0)],
            Some((42, 0)),
            &[100],
            &[],
        )
        .expect("mul events"),
    );
    blocks.extend(
        neo_wasm::event_grammar::expand_import_events(
            &grammar.imports[&sink_fref(grammar)],
            &[(42, 0)],
            None,
            &[],
            &[],
        )
        .expect("sink events"),
    );
    blocks.extend(neo_wasm::event_grammar::expand_export_exit(template, Some((42, 0)), &[], &[]).expect("exit"));
    blocks
        .into_iter()
        .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
        .collect()
}
