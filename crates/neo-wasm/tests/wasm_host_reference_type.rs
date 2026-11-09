mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, SlotBinding};
use wasmtime::component::Val as ComponentVal;

fn assert_host_scalar_rom_rejects(
    trace: &[neo_wasm::WasmVmStep],
    artifacts: &neo_wasm::WasmProgramArtifacts,
    bindings: &HostEventBindings,
) {
    common::ccs_check_trace(trace);
    let layout = neo_wasm::build_wasm_relation_layout();
    let witnesses: Vec<_> = trace
        .iter()
        .map(|row| neo_wasm::witness_builder::build_witness_vector(row).expect("witness"))
        .collect();
    for witness in &witnesses {
        neo_wasm::sanity_check_lookup_row(&layout.auxiliary, witness).expect("lookup semantics");
    }
    neo_application::check_continuity_rows(&layout.auxiliary.continuity, &witnesses)
        .expect("self-consistent forged trace");
    let mut preload = neo_wasm::preload_from_program_artifacts(artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, bindings, artifacts);
    let error = neo_wasm::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("the verifier-owned host-scalar type bit must reject the forged value");
    assert!(
        error.contains("host_event_slot_immediate_0") && error.contains("expected 0, got 1"),
        "unexpected rejection: {error}"
    );
}

fn result_bindings() -> HostEventBindings {
    let mut block = [SlotBinding::Const(0); 8];
    block[0] = SlotBinding::ResultElem { limb: Limb::Lo };
    block[1] = SlotBinding::ResultElem { limb: Limb::Hi };
    let mut bindings = HostEventBindings::default();
    bindings.imports.insert(
        1,
        ImportTemplate {
            events: vec![EventBlock::advice(block)],
            return_events: None,
            input_count: 0,
        },
    );
    bindings.exports.insert(2, ExportTemplate::default());
    bindings
}

fn input_bindings() -> HostEventBindings {
    let mut input = [SlotBinding::Const(0); 8];
    input[0] = SlotBinding::InputLocal {
        input: 0,
        local: 0,
        limb: Limb::Lo,
    };
    input[1] = SlotBinding::InputLocal {
        input: 1,
        local: 0,
        limb: Limb::Hi,
    };
    let mut output = [SlotBinding::Const(0); 8];
    output[0] = SlotBinding::OutputElem { limb: Limb::Lo };
    output[1] = SlotBinding::OutputElem { limb: Limb::Hi };
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        1,
        ExportTemplate {
            entry: vec![EventBlock::op(1, [SlotBinding::Const(0); 7]), EventBlock::advice(input)],
            exit: vec![EventBlock::advice(output)],
            entry_input_count: 2,
        },
    );
    bindings
}

#[test]
fn funcref_host_result_cannot_use_an_arbitrary_i32() {
    let i32_component = wat::parse_str(
        r#"(component
          (type $host-get (func (result s32)))
          (type $run-type (func (result s32)))
          (import "host-get" (func $host-get (type $host-get)))
          (core module $m
            (import "" "0" (func $get (result i32)))
            (func (export "run") (result i32) call $get))
          (core func $lowered (canon lower (func $host-get)))
          (core instance $lowered-host (export "0" (func $lowered)))
          (core instance $i (instantiate $m (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .expect("component wat");
    let funcref_module = wat::parse_str(
        r#"(module
          (import "" "0" (func $get (result funcref)))
          (func (export "run") (result funcref) call $get))"#,
    )
    .expect("funcref module wat");
    let bindings = result_bindings();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&i32_component, &bindings, "run", |linker| {
        linker
            .root()
            .func_wrap("host-get", |_store, (): ()| Ok((7i32,)))
            .map_err(|error| neo_wasm::WasmBuildError::Trace(error.to_string()))
    })
    .expect("i32 component run");
    let mut artifacts = neo_wasm::extract_wasm_program_artifacts(&funcref_module).expect("funcref program artifacts");
    artifacts.host_event_bindings = bindings.clone();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
        .expect("i32 trace");
    assert_eq!(
        trace
            .last()
            .expect("terminal row")
            .state_after
            .output
            .value_lo,
        7
    );
    common::sanity_check_trace_with_bindings(&trace, run.artifacts(), &bindings);

    let error = bindings
        .validate_against_program(&artifacts.tables)
        .expect_err("a host result needs a complete verifier-bound scalar encoding")
        .to_string();
    assert!(
        error.contains("returns a value") && error.contains("i32/i64"),
        "unexpected error: {error}"
    );
    let error = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &artifacts, Default::default())
        .expect_err("normalization must reject the unsupported host result")
        .to_string();
    assert!(error.contains("i32/i64"), "unexpected error: {error}");
    assert_host_scalar_rom_rejects(&trace, &artifacts, &bindings);
}

#[test]
fn funcref_export_input_cannot_use_an_arbitrary_i32() {
    let i32_component = wat::parse_str(
        r#"(component
          (type $run-type (func (param "x" s32) (result s32)))
          (core module $m
            (func (export "run") (param i32) (result i32) local.get 0))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .expect("component wat");
    let funcref_module = wat::parse_str(
        r#"(module
          (func (export "run") (param funcref) (result funcref) local.get 0))"#,
    )
    .expect("funcref module wat");
    let bindings = input_bindings();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &i32_component,
        &bindings,
        "run",
        &[ComponentVal::S32(7)],
        |_| Ok(()),
    )
    .expect("i32 component run");
    let mut artifacts = neo_wasm::extract_wasm_program_artifacts(&funcref_module).expect("funcref program artifacts");
    artifacts.host_event_bindings = bindings.clone();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
        .expect("i32 trace");
    assert_eq!(
        trace
            .last()
            .expect("terminal row")
            .state_after
            .output
            .value_lo,
        7
    );
    common::sanity_check_trace_with_bindings(&trace, run.artifacts(), &bindings);

    let error = bindings
        .validate_against_program(&artifacts.tables)
        .expect_err("an export input needs a complete verifier-bound scalar encoding")
        .to_string();
    assert!(
        error.contains("parameter 0") && error.contains("i32/i64"),
        "unexpected error: {error}"
    );
    let error = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &artifacts, Default::default())
        .expect_err("normalization must reject the unsupported export input")
        .to_string();
    assert!(error.contains("i32/i64"), "unexpected error: {error}");
    assert_host_scalar_rom_rejects(&trace, &artifacts, &bindings);
}
