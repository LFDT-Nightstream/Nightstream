//! Export-boundary templates: entry events absorb before the export's first
//! instruction, with `InputLocal` slots bootstrapping the zero-initialized
//! entry frame. Exit events absorb after the halting row and may bind the
//! captured result. The verifier folds the claimed transcript natively and
//! compares it with the proof-carried final commitment chain. Tags below are
//! arbitrary embedder data.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, MemoryBase, SlotBinding};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmHostEventSlotKind, WasmVmStep};
use p3_field::PrimeCharacteristicRing;
use wasmtime::component::Val as ComponentVal;

const ZERO: SlotBinding = SlotBinding::Const(0);
const ENTRY_HEADER_TAG: u64 = 1;
const ENTRY_INPUT_TAG: u64 = 2;
const EXIT_OUTPUT_TAG: u64 = 3;
const ENTRY_HEADER_WORD: u64 = 9;

fn slots(entries: &[(usize, SlotBinding)]) -> [SlotBinding; COMM_CHAIN_EVENT_ARGS] {
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

/// Two entry blocks carry constants and bootstrap the parameter locals; one
/// exit block carries the output.
fn export_template() -> ExportTemplate {
    let write = |input, local| SlotBinding::InputLocal {
        input,
        local,
        limb: Limb::Lo,
    };
    ExportTemplate {
        entry: vec![
            EventBlock::op(ENTRY_HEADER_TAG, slots(&[(0, SlotBinding::Const(ENTRY_HEADER_WORD))])),
            EventBlock::op(ENTRY_INPUT_TAG, slots(&[(0, write(0, 0)), (1, write(1, 1))])),
        ],
        exit: vec![EventBlock::op(
            EXIT_OUTPUT_TAG,
            slots(&[(0, SlotBinding::OutputElem { limb: Limb::Lo })]),
        )],
        entry_input_count: 2,
    }
}

fn export_fref(steps: &[neo_wasm::WasmtimeTraceStep]) -> u32 {
    steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref")
}

fn boundary_trace() -> (Vec<WasmVmStep>, HostEventBindings) {
    let component_bytes = wat::parse_str(add_component_wat()).expect("component wat");
    let args = [ComponentVal::S32(7), ComponentVal::S32(35)];
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, "run", &args, |_| Ok(()))
        .expect("component run");

    let fref = export_fref(&run.steps);
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(fref, export_template());

    let turns = [neo_wasm::host_event_bindings::TurnInputs { entry: vec![7, 35] }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &turns,
        Default::default(),
    )
    .expect("bindings trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    // Input bootstrap: the RAM model's entry locals start all-zero; the
    // entry gather rows write the runtime inputs into them.
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("bindings ROM + locals reads match");

    (trace, bindings)
}

#[test]
fn export_boundary_folds_entry_and_exit_events() {
    let (trace, bindings) = boundary_trace();

    // The trace opens with the entry gather rows (before any program row).
    assert!(trace[0].row_kind.is_host_event_gather());
    assert_eq!(trace[0].state_before.host_events.events_remaining, 2);

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
            [ENTRY_HEADER_TAG, ENTRY_HEADER_WORD, 0, 0, 0, 0, 0, 0],
            [ENTRY_INPUT_TAG, 7, 35, 0, 0, 0, 0, 0],
            [EXIT_OUTPUT_TAG, 42, 0, 0, 0, 0, 0, 0],
        ],
    );

    // Expand the transcript from the template and runtime values, then fold
    // it natively. The proof-carried final chain must equal that fold; a
    // different entry input must not.
    let template = bindings.exports.values().next().expect("template");
    let mut expected = neo_wasm::host_event_bindings::expand_export_entry(template, &[7, 35]).expect("entry");
    expected.extend(neo_wasm::host_event_bindings::expand_export_exit(template, Some((42, 0)), &[]).expect("exit"));
    assert_eq!(expected, staged, "claimed transcript must match the staged blocks");
    let lift = |blocks: &[[u64; 8]]| -> Vec<[p3_goldilocks::Goldilocks; 8]> {
        blocks
            .iter()
            .map(|block| block.map(p3_goldilocks::Goldilocks::from_u64))
            .collect()
    };
    let final_chain = trace.last().expect("rows").state_after.comm_chain;
    assert_eq!(
        final_chain,
        neo_wasm::comm_chain::fold_event_blocks(Default::default(), &lift(&expected)).canonical_u64()
    );
    let wrong_inputs = neo_wasm::host_event_bindings::expand_export_entry(template, &[7, 36]).expect("wrong entry");
    assert_ne!(
        final_chain,
        neo_wasm::comm_chain::fold_event_blocks(
            Default::default(),
            &lift(&[wrong_inputs, expected[2..].to_vec()].concat())
        )
        .canonical_u64(),
        "a different entry input must fold to a different chain"
    );
}

/// Forging the exit event's output word is CCS-rejected: the word is bound
/// to the carried output-capture value.
#[test]
fn ccs_rejects_forged_exit_output() {
    let (trace, _) = boundary_trace();
    let output_slot_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::Output)
        })
        .expect("output slot row");
    let mut witness = build_witness_vector(output_slot_row);
    common::assert_satisfied(&witness, "untampered output slot row");
    let cursor = usize::from(output_slot_row.state_before.host_events.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF_AFTER[cursor]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "exit gather row staging a forged output word");

    let mut witness = build_witness_vector(output_slot_row);
    witness[neo_wasm::layout::COL_OUTPUT_ENABLED_BEFORE] = neo_math::F::ZERO;
    witness[neo_wasm::layout::COL_OUTPUT_ENABLED_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "output slot row with no captured output");
}

/// An input-local gather row must write the very word it stages: forging
/// the staged word away from the locals write is CCS-rejected.
#[test]
fn ccs_rejects_forged_input_word() {
    let (trace, _) = boundary_trace();
    let input_slot_row = trace
        .iter()
        .find(|row| {
            row.row_kind.is_host_event_gather()
                && row
                    .host_event_rom_slot
                    .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::InputLocal)
        })
        .expect("input-local slot row");
    let mut witness = build_witness_vector(input_slot_row);
    common::assert_satisfied(&witness, "untampered input-local slot row");
    let cursor = usize::from(input_slot_row.state_before.host_events.slot_cursor);
    witness[neo_wasm::layout::COL_EVBUF_AFTER[cursor]] += neo_math::F::ONE;
    common::assert_rejected(&witness, "entry gather row staging a word other than the locals write");
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
    witness[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = neo_math::F::ZERO;
    common::assert_rejected(&witness, "capture row suppressing the exit schedule");
}

/// `run(x: s64) -> s64 { x }`: an i64 export param bootstrapped lane by
/// lane (lo slot then hi slot), read back through the exit output limbs.
fn identity64_component_wat() -> &'static str {
    r#"
    (component
      (type $run-type (func (param "x" s64) (result s64)))
      (core module $m
        (func (export "run") (param i64) (result i64)
          local.get 0))
      (core instance $i (instantiate $m))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

#[test]
fn i64_param_bootstraps_both_lanes() {
    let component_bytes = wat::parse_str(identity64_component_wat()).expect("component wat");
    // x = 3·2^32 + 7.
    let args = [ComponentVal::S64((3i64 << 32) | 7)];
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, "run", &args, |_| Ok(()))
        .expect("component run");

    let fref = export_fref(&run.steps);
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        fref,
        ExportTemplate {
            entry: vec![EventBlock::op(
                21,
                slots(&[
                    (
                        0,
                        SlotBinding::InputLocal {
                            input: 0,
                            local: 0,
                            limb: Limb::Lo,
                        },
                    ),
                    (
                        1,
                        SlotBinding::InputLocal {
                            input: 1,
                            local: 0,
                            limb: Limb::Hi,
                        },
                    ),
                ]),
            )],
            exit: vec![EventBlock::op(
                18,
                slots(&[
                    (0, SlotBinding::OutputElem { limb: Limb::Lo }),
                    (1, SlotBinding::OutputElem { limb: Limb::Hi }),
                ]),
            )],
            entry_input_count: 2,
        },
    );

    let turns = [neo_wasm::host_event_bindings::TurnInputs { entry: vec![7, 3] }];
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &turns,
        Default::default(),
    )
    .expect("bindings trace");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);

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
            [21, 7, 3, 0, 0, 0, 0, 0], // entry: both param lanes
            [18, 7, 3, 0, 0, 0, 0, 0], // exit: both output lanes
        ],
    );

    // The locals RAM sees the zero-init + lane writes and the guest's i64
    // read back out of them.
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("bindings ROM + lane writes match");

    // The captured output is the full i64 round-tripped through the locals.
    let last = trace.last().expect("rows").state_after;
    assert_eq!((last.output.value_lo, last.output.value_hi), (7, 3));
}

#[test]
fn export_exit_memory_reads_use_the_captured_output_pointer() {
    let component_bytes = wat::parse_str(
        r#"
        (component
          (type $run-type (func (param "ptr" s32) (result s32)))
          (core module $m
            (memory 1)
            (func (export "run") (param i32) (result i32)
              local.get 0))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
    )
    .expect("component wat");
    let args = [ComponentVal::S32(16)];
    let run = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(&component_bytes, "run", &args, |_| Ok(()))
        .expect("component run");
    let fref = export_fref(&run.steps);

    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        fref,
        ExportTemplate {
            entry: vec![EventBlock::op(
                30,
                slots(&[
                    (
                        0,
                        SlotBinding::InputLocal {
                            input: 0,
                            local: 0,
                            limb: Limb::Lo,
                        },
                    ),
                    (
                        1,
                        SlotBinding::MemoryWrite32 {
                            input: 1,
                            base: MemoryBase::Local(0),
                            byte_offset: 0,
                        },
                    ),
                    (
                        2,
                        SlotBinding::MemoryWrite8 {
                            input: 2,
                            base: MemoryBase::Local(0),
                            byte_offset: 1,
                        },
                    ),
                    (
                        3,
                        SlotBinding::MemoryWrite16 {
                            input: 3,
                            base: MemoryBase::Local(0),
                            byte_offset: 2,
                        },
                    ),
                ]),
            )],
            exit: vec![EventBlock::op(
                31,
                slots(&[
                    (
                        0,
                        SlotBinding::MemoryRead32 {
                            base: MemoryBase::Output,
                            byte_offset: 0,
                        },
                    ),
                    (
                        1,
                        SlotBinding::MemoryRead8 {
                            base: MemoryBase::Output,
                            byte_offset: 1,
                        },
                    ),
                    (
                        2,
                        SlotBinding::MemoryRead16 {
                            base: MemoryBase::Output,
                            byte_offset: 2,
                        },
                    ),
                ]),
            )],
            entry_input_count: 4,
        },
    );
    let turns = [neo_wasm::host_event_bindings::TurnInputs {
        entry: vec![16, 77, 5, 0x1234],
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
    common::ccs_check_trace(&trace);

    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(&component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
    let witness_rows: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(&layout, &witness_rows, &preload)
        .expect("bindings output base and linear-memory accesses match");

    let staged_reads: Vec<u64> = trace
        .iter()
        .filter_map(|row| {
            let rom = row.host_event_rom_slot?;
            (rom.kind == WasmHostEventSlotKind::MemoryRead)
                .then(|| row.state_after.event_absorb.evbuf[usize::from(row.state_before.host_events.slot_cursor)])
        })
        .collect();
    assert_eq!(staged_reads, [77 | (5 << 8) | (0x1234 << 16), 5, 0x1234]);

    let read = trace
        .iter()
        .find(|row| {
            row.host_event_rom_slot
                .is_some_and(|rom| rom.kind == WasmHostEventSlotKind::MemoryRead)
        })
        .expect("memory read gather");
    let mut forged = build_witness_vector(read);
    common::assert_satisfied(&forged, "untampered output-base memory read");
    forged[neo_wasm::layout::COL_OUTPUT_VALUE_LO_BEFORE] += neo_math::F::ONE;
    forged[neo_wasm::layout::COL_OUTPUT_VALUE_LO_AFTER] += neo_math::F::ONE;
    common::assert_rejected(&forged, "memory read detached from the captured output pointer");
}

#[test]
fn export_exit_memory_rejects_an_oob_output_pointer_during_normalization() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 65536))"#,
    )
    .expect("valid wasm");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "run", &[]).expect("wasmtime trace");
    let fref = export_fref(&run.steps);
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        fref,
        ExportTemplate {
            exit: vec![EventBlock::op(
                1,
                slots(&[(
                    0,
                    SlotBinding::MemoryRead8 {
                        base: MemoryBase::Output,
                        byte_offset: 0,
                    },
                )]),
            )],
            ..Default::default()
        },
    );

    let err = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        &run.steps,
        &run.program_tables,
        &bindings,
        &[Default::default()],
        Default::default(),
    )
    .expect_err("OOB output pointer must fail during normalization");
    assert!(err.to_string().contains("out of bounds for 1 memory pages"));
}
