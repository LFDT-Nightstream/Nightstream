mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding};
use neo_wasm::{WasmHostEventSlotKind, WasmOpcode};
use wasmtime::component::Val as ComponentVal;

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

fn input_high_row(trace: &[neo_wasm::WasmVmStep]) -> usize {
    trace
        .iter()
        .position(|row| {
            row.host_event_rom_slot
                .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::InputLocal && slot.variant.is_high_limb())
        })
        .expect("input-local high-limb gather row")
}

#[test]
fn i32_export_input_cannot_forge_a_nonzero_high_limb() {
    let component = wat::parse_str(
        r#"(component
          (type $run-type (func (param "x" s32) (result s32)))
          (core module $m
            (func (export "run") (param i32) (result i32) local.get 0))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .expect("component wat");
    let bindings = input_bindings();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &component,
        &bindings,
        "run",
        &[ComponentVal::S32(7)],
        |_| Ok(()),
    )
    .expect("component run");
    let mut trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
            .expect("normalized trace");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component).expect("artifacts");
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);

    let input_hi = input_high_row(&trace);
    assert!(!trace[input_hi].wide_values_enabled);
    trace[input_hi].wide_values_enabled = true;
    trace[input_hi]
        .host_event_rom_slot
        .as_mut()
        .expect("input-local slot ROM")
        .immediate1 = 1;
    trace[input_hi].local_write_value_hi = Some(1);
    trace[input_hi].state_after.event_absorb.evbuf[1] = 1;
    for row in &mut trace[input_hi + 1..] {
        row.state_before.event_absorb.evbuf[1] = 1;
        row.state_after.event_absorb.evbuf[1] = 1;
    }

    let local_get = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::LocalGet)
        .expect("local.get row");
    trace[local_get].local_read_value_hi = Some(1);
    trace[local_get]
        .stack_write0
        .as_mut()
        .expect("local.get stack write")
        .value_hi = Some(1);

    let capture = trace
        .iter()
        .position(|row| row.output_captured)
        .expect("output capture row");
    trace[capture].state_after.output.value_hi = 1;
    for row in &mut trace[capture + 1..] {
        row.state_before.output.value_hi = 1;
        row.state_after.output.value_hi = 1;
    }

    // The forged trace is self-consistent under the row relation, lookups,
    // and continuity. The verifier-owned, program-derived event ROM must reject it.
    common::ccs_check_trace(&trace);
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
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings, &artifacts);
    let error = neo_wasm::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("static i32 local type must reject a wide input row");
    assert!(
        error.contains("host_event_slot_immediate_1") && error.contains("expected 0, got 1"),
        "unexpected rejection: {error}"
    );
}

#[test]
fn i64_export_input_keeps_wide_mode_when_the_high_limb_is_zero() {
    let component = wat::parse_str(
        r#"(component
          (type $run-type (func (param "x" s64) (result s64)))
          (core module $m
            (func (export "run") (param i64) (result i64) local.get 0))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .expect("component wat");
    let bindings = input_bindings();
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &component,
        &bindings,
        "run",
        &[ComponentVal::S64(7)],
        |_| Ok(()),
    )
    .expect("component run");
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
        .expect("normalized trace");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component).expect("artifacts");
    let input_hi = input_high_row(&trace);
    assert!(trace[input_hi].wide_values_enabled);
    assert_eq!(trace[input_hi].local_write_value_hi, Some(0));
    common::sanity_check_trace_with_bindings(&trace, &artifacts, &bindings);
    common::ccs_check_trace(&trace);
}

#[test]
fn declared_local_cannot_become_an_export_input() {
    let component = wat::parse_str(
        r#"(component
          (type $run-type (func (param "x" s32) (result s32)))
          (core module $m
            (func (export "run") (param i32) (result i32)
              (local i32)
              local.get 1))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .expect("component wat");
    let mut input = [SlotBinding::Const(0); 8];
    input[0] = SlotBinding::InputLocal {
        input: 0,
        local: 0,
        limb: Limb::Lo,
    };
    let mut output = [SlotBinding::Const(0); 8];
    output[0] = SlotBinding::OutputElem { limb: Limb::Lo };
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        1,
        ExportTemplate {
            entry: vec![EventBlock::op(1, [SlotBinding::Const(0); 7]), EventBlock::advice(input)],
            exit: vec![EventBlock::advice(output)],
            entry_input_count: 1,
        },
    );
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &component,
        &bindings,
        "run",
        &[ComponentVal::S32(0)],
        |_| Ok(()),
    )
    .expect("component run");
    let mut trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default())
            .expect("normalized trace");
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component).expect("artifacts");

    let mut invalid_bindings = bindings.clone();
    for slot in invalid_bindings
        .exports
        .get_mut(&1)
        .expect("export template")
        .entry
        .iter_mut()
        .flat_map(|event| &mut event.block)
    {
        if let SlotBinding::InputLocal { local, .. } = slot {
            *local = 1;
        }
    }
    let error = invalid_bindings
        .validate_against_program(&artifacts.tables)
        .expect_err("declared local is not an export input")
        .to_string();
    assert!(error.contains("declared local 1"), "unexpected error: {error}");

    let input_row = trace
        .iter()
        .position(|row| {
            row.host_event_rom_slot
                .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::InputLocal)
        })
        .expect("input-local gather row");
    let cursor = usize::from(trace[input_row].state_before.host_events.slot_cursor);
    let rom = trace[input_row]
        .host_event_rom_slot
        .as_mut()
        .expect("input-local slot ROM");
    rom.arg = 1;
    rom.immediate0 = 0;
    trace[input_row].local_index = Some(1);
    trace[input_row].local_write_value = Some(7);
    trace[input_row].state_after.event_absorb.evbuf[cursor] = 7;
    for row in &mut trace[input_row + 1..] {
        row.state_before.event_absorb.evbuf[cursor] = 7;
        row.state_after.event_absorb.evbuf[cursor] = 7;
    }
    let local_get = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::LocalGet)
        .expect("local.get row");
    trace[local_get].local_read_value = Some(7);
    trace[local_get]
        .stack_write0
        .as_mut()
        .expect("local.get stack write")
        .value_lo = 7;
    let capture = trace
        .iter()
        .position(|row| row.output_captured)
        .expect("output capture row");
    trace[capture].state_after.output.value_lo = 7;
    for row in &mut trace[capture + 1..] {
        row.state_before.output.value_lo = 7;
        row.state_after.output.value_lo = 7;
    }

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
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &invalid_bindings, &artifacts);
    neo_wasm::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect("forged trace is otherwise consistent with the invalid template");
    for (index, witness) in witnesses.iter().enumerate() {
        if index == input_row {
            common::assert_rejected(witness, "declared-local input row");
        } else {
            common::assert_satisfied(witness, "other forged trace row");
        }
    }
}
