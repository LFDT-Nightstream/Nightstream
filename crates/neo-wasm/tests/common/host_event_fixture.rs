//! Shared event-bound fixture: a component with two template-bound host
//! imports (mul with a input word, sink) and an export boundary template,
//! traced with bindings tables. Used by the F′ audit lifecycle test and the
//! Nebula proof test.

use neo_wasm::comm_chain::COMM_CHAIN_BLOCK_WORDS;
use neo_wasm::host_event_bindings::{
    EventBlockBuilder, HostEventBindings, HostEventBindingsBuilder, Limb, SlotBinding,
};
use neo_wasm::{WasmBuildError, WasmProgramTables, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

pub const ENTRY_INPUTS: [u64; 2] = [500, 501];

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
fn test_bindings(
    program: &WasmProgramTables,
    mul_fref: u32,
    sink_fref: u32,
    run_fref: u32,
) -> Result<HostEventBindings, WasmBuildError> {
    let mul_args = EventBlockBuilder::op(10)
        .input_i32(0, 0)?
        .arg_i32(1, 0)?
        .arg_i32(2, 1)?
        .finish();
    // The ResultElem Lo slot pushes the host result; the Hi slot binds the
    // pushed hi lane (zero for this i32 result).
    let mul_result = EventBlockBuilder::op(12)
        .slot(0, SlotBinding::ResultElem { limb: Limb::Lo })?
        .input_i32(1, 0)?
        .slot(2, SlotBinding::ResultElem { limb: Limb::Hi })?
        .finish();
    let sink = EventBlockBuilder::op(7).arg_i32(0, 0)?.finish();
    let enter = EventBlockBuilder::op(20).constant_i32(0, 55)?.finish();
    let activation = EventBlockBuilder::op(8)
        .input_i32(1, 0)?
        .input_i32(3, 1)?
        .finish();
    let exit = EventBlockBuilder::op(17).output_i32(1)?.finish();

    let mut bindings = HostEventBindingsBuilder::new(program);
    bindings.import(mul_fref, vec![mul_args, mul_result])?;
    bindings.import(sink_fref, vec![sink])?;
    bindings.export(run_fref, vec![enter, activation], vec![exit])?;
    bindings.finish()
}

/// The mul import is the two-event template; sink has one event.
pub fn mul_fref(bindings: &HostEventBindings) -> u32 {
    *bindings
        .imports
        .iter()
        .find(|(_, t)| t.events.len() == 2)
        .expect("mul template")
        .0
}

pub fn sink_fref(bindings: &HostEventBindings) -> u32 {
    *bindings
        .imports
        .iter()
        .find(|(_, t)| t.events.len() == 1)
        .expect("sink template")
        .0
}

fn run_frefs(run: &neo_wasm::WasmtimeTraceRun) -> (Vec<u32>, u32) {
    let imports = run
        .steps
        .iter()
        .filter(|row| matches!(row.opcode_decoded, Some(neo_wasm::WasmOpcode::Call)) && !row.target_function_is_guest)
        .filter_map(|row| row.function_ref)
        .collect();
    let export = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref");
    (imports, export)
}

pub struct HostEventLifecycleSetup {
    pub trace: Vec<WasmVmStep>,
    pub bindings: HostEventBindings,
    pub run_fref: u32,
    pub component_bytes: Vec<u8>,
}

pub fn host_event_lifecycle_setup() -> HostEventLifecycleSetup {
    let component_bytes = wat::parse_str(mul_sink_component_wat()).expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-mul", |mut store, (x, y): (i32, i32)| {
                // The mul template consumes one input word; record it at
                // call time (the bindings hand-off path).
                store.data_mut().record_call_inputs(&[100])?;
                Ok((x * y,))
            })
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-mul: {err}")))?;
        linker
            .root()
            .func_wrap("host-sink", |_store, (_x,): (i32,)| Ok(()))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-sink: {err}")))
    })
    .expect("component run");

    let (frefs, run_fref) = run_frefs(&run);
    let bindings = test_bindings(&run.program_tables, frefs[0], frefs[1], run_fref).expect("build bindings");

    let turns = [neo_wasm::host_event_bindings::TurnInputs {
        entry: ENTRY_INPUTS.to_vec(),
        exit: vec![],
        ..Default::default()
    }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &turns,
        Default::default(),
    )
    .expect("bindings trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    HostEventLifecycleSetup {
        trace,
        bindings,
        run_fref,
        component_bytes,
    }
}

/// The transcript the verifier expects for `inputs`: export entry (with the
/// input words), the mul call (input word 100, result 42), the sink call,
/// and the export exit carrying the output.
pub fn expected_transcript(
    bindings: &HostEventBindings,
    run_fref: u32,
    inputs: &[u64],
) -> Vec<[p3_goldilocks::Goldilocks; COMM_CHAIN_BLOCK_WORDS]> {
    let template = bindings.exports.get(&run_fref).expect("export template");
    let mut blocks = neo_wasm::host_event_bindings::expand_export_entry(template, inputs).expect("entry");
    blocks.extend(
        neo_wasm::host_event_bindings::expand_import_events(
            &bindings.imports[&mul_fref(bindings)],
            &[(7, 0), (6, 0)],
            Some((42, 0)),
            &[100],
            &[],
        )
        .expect("mul events"),
    );
    blocks.extend(
        neo_wasm::host_event_bindings::expand_import_events(
            &bindings.imports[&sink_fref(bindings)],
            &[(42, 0)],
            None,
            &[],
            &[],
        )
        .expect("sink events"),
    );
    blocks.extend(neo_wasm::host_event_bindings::expand_export_exit(template, Some((42, 0)), &[], &[]).expect("exit"));
    blocks
        .into_iter()
        .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
        .collect()
}
