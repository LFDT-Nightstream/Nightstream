mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, SlotBinding};
use neo_wasm::{WasmHostEventSlotKind, WasmOpcode};

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

fn result_high_row(trace: &[neo_wasm::WasmVmStep]) -> usize {
    trace
        .iter()
        .position(|row| {
            row.host_event_rom_slot
                .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::Result && slot.variant.is_high_limb())
        })
        .expect("result high-limb gather row")
}

#[test]
fn i32_host_result_cannot_forge_a_nonzero_high_limb() {
    let component = wat::parse_str(
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
    let bindings = result_bindings();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component, &bindings, "run", |linker| {
        linker
            .root()
            .func_wrap("host-get", |_store, (): ()| Ok((7i32,)))
            .map_err(|error| neo_wasm::WasmBuildError::Trace(error.to_string()))
    })
    .expect("component run");
    let mut trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
            .expect("normalized trace");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component).expect("artifacts");
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);

    let high = result_high_row(&trace);
    let slot = usize::from(trace[high].state_before.host_events.slot_cursor);
    assert_eq!(slot, 1);
    assert!(!trace[high].wide_values_enabled);
    trace[high].wide_values_enabled = true;
    trace[high]
        .host_event_rom_slot
        .as_mut()
        .expect("result slot ROM")
        .immediate1 = 1;
    trace[high]
        .stack_write0
        .as_mut()
        .expect("result high-limb write")
        .value_hi = Some(1);
    trace[high].state_after.event_absorb.evbuf[slot] = 1;
    for row in &mut trace[high + 1..] {
        row.state_before.event_absorb.evbuf[slot] = 1;
        row.state_after.event_absorb.evbuf[slot] = 1;
    }
    let terminal = trace.last_mut().expect("terminal row");
    assert_eq!(terminal.opcode, WasmOpcode::End);
    terminal.state_after.output.value_hi = 1;

    // The forged trace is self-consistent under the row relation, lookups,
    // and continuity. The verifier-owned, program-derived event ROM must reject it.
    common::ccs_check_trace(&trace);
    let layout = neo_wasm::build_wasm_relation_layout();
    let witnesses: Vec<_> = trace
        .iter()
        .map(|row| neo_wasm::witness_builder::build_witness_vector(row).expect("witness"))
        .collect();
    for witness in &witnesses {
        neo_wasm::sanity_check_operation_row(&layout.auxiliary, witness).expect("lookup semantics");
    }
    neo_application::check_continuity_rows(&layout.auxiliary.continuity, &witnesses)
        .expect("self-consistent forged trace");
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings, &artifacts);
    let error = neo_wasm::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("static i32 result type must reject a wide result row");
    assert!(
        error.contains("host_event_slot_immediate_1") && error.contains("expected 0, got 1"),
        "unexpected rejection: {error}"
    );
}

#[test]
fn i64_host_result_keeps_wide_mode_when_the_high_limb_is_zero() {
    let component = wat::parse_str(
        r#"(component
          (type $host-get (func (result s64)))
          (type $run-type (func (result s64)))
          (import "host-get" (func $host-get (type $host-get)))
          (core module $m
            (import "" "0" (func $get (result i64)))
            (func (export "run") (result i64) call $get))
          (core func $lowered (canon lower (func $host-get)))
          (core instance $lowered-host (export "0" (func $lowered)))
          (core instance $i (instantiate $m (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .expect("component wat");
    let bindings = result_bindings();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component, &bindings, "run", |linker| {
        linker
            .root()
            .func_wrap("host-get", |_store, (): ()| Ok((7i64,)))
            .map_err(|error| neo_wasm::WasmBuildError::Trace(error.to_string()))
    })
    .expect("component run");
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
        .expect("normalized trace");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component).expect("artifacts");
    let high = result_high_row(&trace);
    assert!(trace[high].wide_values_enabled);
    assert_eq!(
        trace[high]
            .stack_write0
            .expect("result high-limb write")
            .value_hi,
        Some(0)
    );
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);
}
