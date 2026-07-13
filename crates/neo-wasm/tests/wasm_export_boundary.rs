//! Export-boundary grammar templates: entry events (receiver-side
//! `Enter`/`Activation`/payload publication) absorb before the export's
//! first instruction, exit events (`Return` with the captured result) after
//! the halting row. Single-turn V1. Discriminants below are example
//! embedder data.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{ExportTemplate, GrammarEvent, HostEventGrammar, Limb, SlotSource};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::WasmVmStep;
use p3_field::PrimeCharacteristicRing;
use wasmtime::component::Val as ComponentVal;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// `run(x: s32, y: s32) -> s32 { x + y }`: a pure export, no host calls.
fn add_component_wat() -> &'static str {
    r#"
    (component
      (type $run-type (func (param "x" s32) (param "y" s32) (result s32)))
      (core module $m
        (func (export "run") (param i32 i32) (result i32)
          local.get 0
          local.get 1
          i32.add))
      (core instance $i (instantiate $m))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

/// Entry: Enter(f_id) + Activation(val=O0, caller=O1) + a payload event
/// publishing both params; exit: Return-ish event carrying the output.
fn export_template() -> ExportTemplate {
    let param = |arg, limb| SlotSource::ParamElem { arg, limb };
    let oracle = |idx| SlotSource::Oracle { idx };
    ExportTemplate {
        entry: vec![
            GrammarEvent::op(20, slots(&[(0, SlotSource::Const(55))])), // Enter(f_id)
            GrammarEvent::op(8, slots(&[(1, oracle(0)), (3, oracle(1))])), // Activation
            GrammarEvent::op(12, slots(&[(0, param(0, Limb::Lo)), (1, param(1, Limb::Lo))])),
        ],
        exit: vec![GrammarEvent::op(
            17,
            slots(&[(1, SlotSource::OutputElem { limb: Limb::Lo })]),
        )],
        oracle_count: 2,
    }
}

fn export_fref(trace: &[WasmVmStep]) -> u32 {
    trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row")
        .current_function_ref
}

fn boundary_trace() -> (Vec<WasmVmStep>, HostEventGrammar) {
    let component_bytes = wat::parse_str(add_component_wat()).expect("component wat");
    let args = [ComponentVal::S32(7), ComponentVal::S32(35)];
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, "run", &args, |_| Ok(()))
        .expect("component run");

    let raw = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("raw trace");
    let fref = export_fref(&raw);
    let mut grammar = HostEventGrammar::default();
    grammar.exports.insert(fref, export_template());

    let entry_oracles = [500u64, 501];
    let exit_oracles = [600u64, 601];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(&run.steps, &grammar, &entry_oracles, &exit_oracles)
        .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM + locals reads match");

    (trace, grammar)
}

#[test]
fn export_boundary_folds_entry_and_exit_events() {
    let (trace, _) = boundary_trace();

    // The trace opens with the entry gather rows (before any program row).
    assert!(trace[0].row_kind.is_host_event_gather());
    assert_eq!(trace[0].state_before.grammar.events_remaining, 3);

    let staged: Vec<[u64; 8]> = trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row.state_after.event_absorb.perm_pending
                && !row.state_before.event_absorb.perm_pending
        })
        .map(|row| row.state_after.event_absorb.evbuf)
        .collect();
    assert_eq!(
        staged,
        vec![
            [20, 55, 0, 0, 0, 0, 0, 0],   // Enter
            [8, 0, 500, 0, 501, 0, 0, 0], // Activation(val=O0, caller=O1)
            [12, 7, 35, 0, 0, 0, 0, 0],   // param publication
            [17, 0, 42, 0, 0, 0, 0, 0],   // Return(output)
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

    // The trace's initial state matches the verifier-side latch mirror.
    let first = &trace[0].state_before;
    assert_eq!(first.grammar.oracles, [500, 501, 0, 0]);
    assert!(first.grammar_mode);
}

/// Forging the exit event's output word is CCS-rejected: the word is bound
/// to the carried output-capture value.
#[test]
fn ccs_rejects_forged_exit_output() {
    let (trace, _) = boundary_trace();
    let output_slot_row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 5))
        .expect("output slot row");
    let mut witness = build_witness_vector(output_slot_row);
    common::assert_satisfied(&witness, "untampered output slot row");
    let cursor = usize::from(output_slot_row.state_before.grammar.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF0_AFTER + cursor] += neo_math::F::ONE;
    common::assert_rejected(&witness, "exit gather row staging a forged output word");
}

/// Forging a param word is CCS-rejected: the word is bound to the locals
/// read the ROM pins.
#[test]
fn ccs_rejects_forged_param_word() {
    let (trace, _) = boundary_trace();
    let param_slot_row = trace
        .iter()
        .find(|row| row.row_kind.is_host_event_gather() && row.grammar_rom_slot.is_some_and(|rom| rom.kind == 4))
        .expect("param slot row");
    let mut witness = build_witness_vector(param_slot_row);
    common::assert_satisfied(&witness, "untampered param slot row");
    let cursor = usize::from(param_slot_row.state_before.grammar.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF0_AFTER + cursor] += neo_math::F::ONE;
    common::assert_rejected(&witness, "entry gather row staging a forged param word");
}

/// The exit latch is forced: suppressing the exit schedule on the capture
/// row is CCS-rejected.
#[test]
fn ccs_rejects_suppressed_exit_schedule() {
    let (trace, _) = boundary_trace();
    let capture_row = trace
        .iter()
        .find(|row| row.output_captured)
        .expect("capture row");
    let mut witness = build_witness_vector(capture_row);
    common::assert_satisfied(&witness, "untampered capture row");
    witness[neo_wasm::layout::COL_GRAMMAR_EVREM_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "capture row suppressing the exit schedule");
}
