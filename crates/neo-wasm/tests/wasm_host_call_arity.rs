//! Host-call arity coverage: the call row pops all args (`sp' = args_base`,
//! plus the table index on `call_indirect`), arg gather rows read the popped
//! region above the live stack top, and the result-lo gather row pushes.
//! Fixtures here use 3+ params so gather addressing past the 1-2 arg
//! fixtures elsewhere is exercised, and `call_indirect` to a host import
//! covers the `CI_HOST_CALL` path.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{GrammarEvent, HostEventGrammar, ImportTemplate, Limb, SlotSource};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmBuildError, WasmGrammarSlotKind, WasmOpcode, WasmVmStep, WasmtimeTraceState};
use p3_field::PrimeCharacteristicRing;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

struct CheckedImportRun {
    trace: Vec<WasmVmStep>,
    import_fref: u32,
}

/// Run a single-import component under `template`, normalize with the
/// grammar, and put the trace through the full native check stack: per-row
/// CCS, lookup semantics, comm chain, and memory rows with the grammar ROM
/// preloaded.
fn checked_import_run(
    component_wat: &str,
    template: ImportTemplate,
    define_host: impl FnOnce(&mut wasmtime::component::Linker<WasmtimeTraceState>) -> Result<(), WasmBuildError>,
) -> CheckedImportRun {
    let component_bytes = wat::parse_str(component_wat).expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", define_host)
        .expect("component run");
    let mut import_frefs: Vec<u32> = run
        .steps
        .iter()
        .filter(|row| {
            matches!(row.opcode_decoded, Some(WasmOpcode::Call | WasmOpcode::CallIndirect))
                && !row.target_function_is_guest
        })
        .filter_map(|row| row.function_ref)
        .collect();
    import_frefs.dedup();
    let [import_fref] = import_frefs[..] else {
        panic!("expected exactly one host import call, got {import_frefs:?}");
    };
    let export_fref = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref");
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(import_fref, template);
    grammar
        .exports
        .insert(export_fref, neo_wasm::event_grammar::ExportTemplate::default());
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect("grammar trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &grammar);
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    for (row, witness) in trace.iter().zip(&witness_rows) {
        neo_wasm::sanity_check_lookup_row(&layout.auxiliary, witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
    }
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("grammar ROM contents match");
    CheckedImportRun { trace, import_fref }
}

fn host_call_row(trace: &[WasmVmStep]) -> &WasmVmStep {
    trace
        .iter()
        .find(|row| {
            row.row_kind.is_program()
                && matches!(row.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
                && !row.target_function_is_guest
        })
        .expect("host-call row")
}

fn arg_gather_rows(trace: &[WasmVmStep]) -> Vec<&WasmVmStep> {
    trace
        .iter()
        .filter(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .grammar_rom_slot
                    .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::Arg)
        })
        .collect()
}

fn five_arg_component_wat() -> &'static str {
    r#"
    (component
      (type $host-sum5 (func
        (param "a" s32) (param "b" s32) (param "c" s32) (param "d" s32) (param "e" s32)
        (result s32)))
      (type $run-type (func (result s32)))
      (import "host-sum5" (func $host-sum5 (type $host-sum5)))
      (core module $m
        (type $host-ty (func (param i32 i32 i32 i32 i32) (result i32)))
        (import "" "0" (func $host-sum5-core (type $host-ty)))
        (func (export "run") (result i32)
          i32.const 1
          i32.const 2
          i32.const 3
          i32.const 4
          i32.const 5
          call $host-sum5-core))
      (core func $lowered (canon lower (func $host-sum5)))
      (core instance $lowered-host
        (export "0" (func $lowered)))
      (core instance $i
        (instantiate $m
          (with "" (instance $lowered-host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

fn sum5_template() -> ImportTemplate {
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    ImportTemplate {
        events: vec![
            GrammarEvent::op(
                10,
                slots(&[
                    (0, arg(0, Limb::Lo)),
                    (1, arg(1, Limb::Lo)),
                    (2, arg(2, Limb::Lo)),
                    (3, arg(3, Limb::Lo)),
                    (4, arg(4, Limb::Lo)),
                ]),
            ),
            GrammarEvent::op(
                12,
                slots(&[
                    (0, SlotSource::ResultElem { limb: Limb::Lo }),
                    (1, SlotSource::ResultElem { limb: Limb::Hi }),
                ]),
            ),
        ],
        claim_count: 0,
    }
}

fn five_arg_run() -> CheckedImportRun {
    checked_import_run(five_arg_component_wat(), sum5_template(), |linker| {
        linker
            .root()
            .func_wrap("host-sum5", |_store, (a, b, c, d, e): (i32, i32, i32, i32, i32)| {
                Ok((a + b + c + d + e,))
            })
            .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
    })
}

/// Direct host call with five scalar args: the call row pops them all, and
/// each arg gather row reads its table-pinned slot above the live stack top.
#[test]
fn direct_host_call_with_five_scalar_args_is_provable() {
    let run = five_arg_run();

    let call_row = host_call_row(&run.trace);
    assert_eq!(call_row.opcode, WasmOpcode::Call);
    assert_eq!(
        call_row.state_after.sp,
        call_row.state_before.sp - 5,
        "the call row pops all five args"
    );

    let arg_rows = arg_gather_rows(&run.trace);
    assert_eq!(
        arg_rows
            .iter()
            .map(|row| row.stack_read0.expect("arg read").value_lo)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5],
        "arg gather rows read the popped region bottom-up"
    );
    assert!(
        arg_rows
            .iter()
            .all(|row| row.state_after.sp == row.state_before.sp),
        "arg reads never pop"
    );

    let events = neo_wasm::comm_chain::absorbed_event_blocks(&run.trace);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].words, [10, 1, 2, 3, 4, 5, 0, 0]);
    assert_eq!(events[1].words, [12, 15, 0, 0, 0, 0, 0, 0]);
    assert!(events
        .iter()
        .all(|event| event.metadata.attributed_fref == run.import_fref));

    let final_output = run.trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 15);
}

/// `call_indirect` to a host import: the call row also pops the table index
/// (the `CI_HOST_CALL` path) and falls through to the next instruction.
#[test]
fn indirect_host_call_with_three_args_is_provable() {
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let template = ImportTemplate {
        events: vec![
            GrammarEvent::op(
                3,
                slots(&[(0, arg(0, Limb::Lo)), (1, arg(1, Limb::Lo)), (2, arg(2, Limb::Lo))]),
            ),
            GrammarEvent::op(
                4,
                slots(&[
                    (0, SlotSource::ResultElem { limb: Limb::Lo }),
                    (1, SlotSource::ResultElem { limb: Limb::Hi }),
                ]),
            ),
        ],
        claim_count: 0,
    };
    let run = checked_import_run(
        r#"
        (component
          (type $host-sum3 (func (param "a" s32) (param "b" s32) (param "c" s32) (result s32)))
          (type $run-type (func (result s32)))
          (import "host-sum3" (func $host-sum3 (type $host-sum3)))
          (core module $m
            (type $host-ty (func (param i32 i32 i32) (result i32)))
            (import "" "0" (func $host-sum3-core (type $host-ty)))
            (table 1 funcref)
            (elem (i32.const 0) func $host-sum3-core)
            (func (export "run") (result i32)
              i32.const 10
              i32.const 20
              i32.const 30
              i32.const 0
              call_indirect (type $host-ty)))
          (core func $lowered (canon lower (func $host-sum3)))
          (core instance $lowered-host
            (export "0" (func $lowered)))
          (core instance $i
            (instantiate $m
              (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
        template,
        |linker| {
            linker
                .root()
                .func_wrap("host-sum3", |_store, (a, b, c): (i32, i32, i32)| Ok((a + b + c,)))
                .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
        },
    );

    let call_row = host_call_row(&run.trace);
    assert_eq!(call_row.opcode, WasmOpcode::CallIndirect);
    assert_eq!(
        call_row.state_after.sp,
        call_row.state_before.sp - 4,
        "three args plus the table index"
    );
    assert_eq!(
        call_row.pc_edge_kind,
        neo_wasm::WasmPcEdgeKind::DynamicCallIndirect,
        "the continuation pc is bound through the call site's return-pc slot"
    );

    let events = neo_wasm::comm_chain::absorbed_event_blocks(&run.trace);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].words, [3, 10, 20, 30, 0, 0, 0, 0]);
    assert_eq!(events[1].words, [4, 60, 0, 0, 0, 0, 0, 0]);
    assert!(events
        .iter()
        .all(|event| event.metadata.attributed_fref == run.import_fref));

    let final_output = run.trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 60);
}

/// The host-call row must latch the ROM/table-bound callee fref; claiming a
/// different import identity (which would redirect the event-schedule
/// lookup) is rejected.
#[test]
fn host_call_row_rejects_forged_callee_attribution() {
    let run = five_arg_run();
    let call_row = host_call_row(&run.trace);
    let mut witness = build_witness_vector(call_row);
    common::assert_satisfied(&witness, "untampered host-call row");
    witness[neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER] += neo_math::F::ONE;
    common::assert_rejected(&witness, "host-call row claiming a different callee");
}

/// Non-boundary rows preserve the latched callee fref; switching the
/// attribution mid-event is rejected.
#[test]
fn gather_row_rejects_switched_callee_attribution() {
    let run = five_arg_run();
    let arg_row = arg_gather_rows(&run.trace)[0];
    let mut witness = build_witness_vector(arg_row);
    common::assert_satisfied(&witness, "untampered arg gather row");
    witness[neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER] += neo_math::F::ONE;
    common::assert_rejected(&witness, "arg gather row switching the callee attribution");
}
