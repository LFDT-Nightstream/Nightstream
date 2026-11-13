mod common;

use neo_math::F;
use neo_wasm::comm_chain::{fold_event_blocks, CommChainState};
use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, MemoryBase, SlotBinding};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    extract_first_component_core_program_artifacts, preload_from_program_artifacts,
    traces_from_wasmtime_steps_with_host_events, WasmHostEventSlotKind,
};
use p3_field::PrimeCharacteristicRing;
use wasmtime::component::Val as ComponentVal;

const WAT: &str = r#"
(component
  (type $run-type (func (param "x" s64) (result s32)))
  (core module $m
    (memory 1)
    (data (i32.const 16) "\00")
    (func (export "run") (param i64) (result i32)
      i32.const 16 i32.load8_u))
  (core instance $i (instantiate $m))
  (alias core export $i "run" (core func $run))
  (func (export "run") (type $run-type)
    (canon lift (core func $run))))
"#;

fn bindings(write: bool) -> HostEventBindings {
    let mut slots = [SlotBinding::Const(0); 7];
    slots[0] = SlotBinding::InputLocal {
        input: 0,
        local: 0,
        limb: Limb::Lo,
    };
    slots[1] = SlotBinding::InputLocal {
        input: 1,
        local: 0,
        limb: Limb::Hi,
    };
    slots[2] = if write {
        SlotBinding::MemoryWrite8 {
            input: 2,
            base: MemoryBase::Local(0),
            byte_offset: 0,
        }
    } else {
        SlotBinding::Const(42)
    };
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        1,
        ExportTemplate {
            entry: vec![EventBlock::op(13, slots)],
            exit: vec![],
            entry_input_count: if write { 3 } else { 2 },
        },
    );
    bindings
}

#[test]
fn high_i64_local_pointer_cannot_authorize_memory_write() {
    let component = wat::parse_str(WAT).expect("valid component WAT");
    let dummy_component = wat::parse_str(WAT.replace(r#"\00"#, r#"\2a"#)).expect("valid dummy component WAT");
    let selected = bindings(true);
    let dummy = bindings(false);
    let low_arg = [ComponentVal::S64(16)];
    let high_arg = [ComponentVal::S64((1i64 << 32) | 16)];
    let low_run =
        neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component, &selected, "run", &low_arg, |_| {
            Ok(())
        })
        .expect("selected low-pointer run");
    let low_trace =
        traces_from_wasmtime_steps_with_host_events(&low_run.steps, low_run.artifacts(), Default::default())
            .expect("selected low-pointer trace");
    let high_run =
        neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component, &selected, "run", &high_arg, |_| {
            Ok(())
        })
        .expect("valid selected module executes under Wasmtime");
    assert_eq!(high_run.results, vec!["0"]);
    let high_rejection =
        traces_from_wasmtime_steps_with_host_events(&high_run.steps, high_run.artifacts(), Default::default())
            .expect_err("selected high pointer must fail")
            .to_string();
    assert!(high_rejection.contains("not a wasm32 pointer"), "{high_rejection}");

    let high_dummy_run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &dummy_component,
        &dummy,
        "run",
        &high_arg,
        |_| Ok(()),
    )
    .expect("dummy high-pointer run");
    let mut forged = traces_from_wasmtime_steps_with_host_events(
        &high_dummy_run.steps,
        high_dummy_run.artifacts(),
        Default::default(),
    )
    .expect("dummy high-pointer trace");
    assert_eq!(low_run.results, vec!["0"]);
    assert_eq!(high_dummy_run.results, vec!["42"]);
    assert_eq!(low_trace.len(), forged.len());
    let selected_write = low_trace
        .iter()
        .find(|row| {
            row.host_event_rom_slot
                .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::MemoryWrite)
        })
        .expect("selected write row");
    let dummy_index = forged
        .iter()
        .position(|row| {
            row.host_event_rom_slot
                .is_some_and(|slot| slot.kind == WasmHostEventSlotKind::Const && slot.immediate0 == 42)
        })
        .expect("dummy constant row");
    assert_eq!(
        selected_write.state_before.host_events.slot_cursor,
        forged[dummy_index].state_before.host_events.slot_cursor
    );
    assert_eq!(selected_write.state_after.event_absorb.evbuf[3], 0);
    assert_eq!(forged[dummy_index].state_after.event_absorb.evbuf[3], 42);
    forged[dummy_index].host_event_rom_slot = selected_write.host_event_rom_slot;
    forged[dummy_index].local_index = selected_write.local_index;
    forged[dummy_index].local_read_value = selected_write.local_read_value;
    forged[dummy_index].linear_memory = selected_write.linear_memory;
    forged[dummy_index].linear_memory_offset = selected_write.linear_memory_offset;
    forged[dummy_index]
        .linear_memory
        .as_mut()
        .expect("forged selected memory write")
        .lane0
        .value_after = 42;

    let artifacts = extract_first_component_core_program_artifacts(&component).expect("selected module artifacts");
    let dummy_artifacts =
        extract_first_component_core_program_artifacts(&dummy_component).expect("dummy module artifacts");
    assert_eq!(artifacts.tables.linear_memory_init, vec![(16, 0)]);
    assert_eq!(dummy_artifacts.tables.linear_memory_init, vec![(16, 42)]);
    assert_eq!(artifacts.tables.program_decode, dummy_artifacts.tables.program_decode);
    assert_eq!(artifacts.tables.pc_rom, dummy_artifacts.tables.pc_rom);
    selected
        .validate_against_program(&artifacts.tables)
        .expect("selected template");
    let witnesses: Vec<Vec<F>> = forged
        .iter()
        .map(|row| build_witness_vector(row).unwrap())
        .collect();
    common::ccs_check_trace(&forged);
    for (index, witness) in witnesses.iter().enumerate() {
        common::assert_satisfied(witness, &format!("forged row {index}"));
        neo_wasm::audit_operation_witness(witness).unwrap_or_else(|error| panic!("operation row {index}: {error}"));
    }
    let expected_words = [13, 16, 1, 42, 0, 0, 0, 0].map(F::from_u64);
    let expected_chain = fold_event_blocks(CommChainState::default(), &[expected_words]).canonical_u64();
    assert_eq!(forged.last().expect("last row").state_after.comm_chain, expected_chain);
    let mut preload = preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &selected, &artifacts);
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    let rejection = neo_wasm::memory_semantics::sanity_check_memory_rows(layout, &witnesses, &preload);
    assert!(
        rejection.is_err(),
        "high local pointer must fail authenticated local high-lane read"
    );

    let mut explicit_high = forged[dummy_index].clone();
    explicit_high.local_read_value_hi = Some(1);
    explicit_high.wide_values_enabled = true;
    common::assert_rejected(
        &build_witness_vector(&explicit_high).unwrap(),
        "host memory event must reject a nonzero local high lane",
    );
}
