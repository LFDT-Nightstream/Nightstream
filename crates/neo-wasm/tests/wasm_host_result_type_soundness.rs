mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, SlotBinding};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    extract_first_component_core_program_artifacts, traces_from_wasmtime_steps_with_host_events, WasmHostEventSlotKind,
};

fn bindings() -> HostEventBindings {
    let mut bound_slots = [SlotBinding::Const(0); 7];
    bound_slots[0] = SlotBinding::ResultElem { limb: Limb::Lo };
    let mut advice_block = [SlotBinding::Const(0); 8];
    advice_block[1] = SlotBinding::ResultElem { limb: Limb::Hi };
    let mut bindings = HostEventBindings::default();
    bindings.imports.insert(
        1,
        ImportTemplate {
            events: vec![EventBlock::op(5, bound_slots), EventBlock::advice(advice_block)],
            return_events: None,
            input_count: 0,
        },
    );
    bindings.exports.insert(2, ExportTemplate::default());
    bindings
}

#[test]
fn i32_host_result_high_limb_is_zero() {
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
    .unwrap();
    let bindings = bindings();
    let reference = neo_wasm::collect_wasmtime_component_run_with_linker(&component, &bindings, "run", |linker| {
        linker
            .root()
            .func_wrap("host-get", |_store, (): ()| Ok((7i32,)))
            .map_err(|error| neo_wasm::WasmBuildError::Trace(error.to_string()))
    })
    .unwrap();
    let mut trace =
        traces_from_wasmtime_steps_with_host_events(&reference.steps, reference.artifacts(), Default::default())
            .unwrap();
    let artifacts = extract_first_component_core_program_artifacts(&component).unwrap();
    bindings
        .validate_against_program(&artifacts.tables)
        .unwrap();
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);
    let reference_last = trace.last().unwrap().state_after;
    assert_eq!(reference_last.output.value_lo, 7);
    assert_eq!(reference_last.output.value_hi, 0);

    let high = trace
        .iter()
        .position(|row| {
            row.host_event_rom_slot
                .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::Result && slot.variant.is_high_limb())
        })
        .unwrap();
    let slot = usize::from(trace[high].state_before.host_events.slot_cursor);
    assert_eq!(slot, 1);
    assert!(!trace[high].wide_values_enabled);
    trace[high].wide_values_enabled = true;
    trace[high].stack_write0.as_mut().unwrap().value_hi = Some(1);
    trace[high].state_after.event_absorb.evbuf[slot] = 1;
    for row in &mut trace[high + 1..] {
        row.state_before.event_absorb.evbuf[slot] = 1;
        row.state_after.event_absorb.evbuf[slot] = 1;
    }
    trace.last_mut().unwrap().state_after.output.value_hi = 1;

    let forged = build_witness_vector(&trace[high]);
    common::assert_rejected(&forged, "i32 host result cannot carry a high word");
}
