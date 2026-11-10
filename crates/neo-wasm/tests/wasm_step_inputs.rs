mod common;

use std::num::NonZeroUsize;
use std::panic::{catch_unwind, UnwindSafe};

use neo_wasm::batch::{batch_count, build_batched_witness, BatchError};
use neo_wasm::nebula::{is_active, step_inputs, WasmApplicationError, WasmLimits};
use neo_wasm::{build_witness_vector, opcode_code, WasmAuxOpcode, WasmOpcode, WasmRowKind, WasmVmStep};
use nightstream::relation::MemoryPortActivation;

fn row(opcode: WasmOpcode) -> WasmVmStep {
    common::step(0, 0, opcode_code(opcode), 0, 0, None, None, None, None, None, 0, false)
}

#[test]
fn state_machine_padding_refuses_unfinished_auxiliary_groups() {
    let mut cases = Vec::new();
    let mut moving = row(WasmOpcode::Nop);
    moving.state_after.frame_move.active = true;
    moving.state_after.frame_move.remaining = 1;
    cases.push(("frame move", moving));
    let mut zeroing = row(WasmOpcode::Nop);
    zeroing.state_after.local_zero.active = true;
    zeroing.state_after.local_zero.remaining = 1;
    cases.push(("local zero", zeroing));
    let mut suspending = row(WasmOpcode::Nop);
    suspending.state_after.host_events.suspend_pending = true;
    cases.push(("pending host suspension", suspending));
    let accepted: Vec<_> = cases
        .into_iter()
        .filter_map(|(name, step)| {
            step_inputs(&[step], 2, NonZeroUsize::MIN)
                .is_ok()
                .then_some(name)
        })
        .collect();
    assert!(accepted.is_empty(), "padding accepted unfinished states: {accepted:?}");
    assert!(step_inputs(&[row(WasmOpcode::Nop)], 2, NonZeroUsize::MIN).is_ok());
}

#[test]
fn state_machine_honest_trace_prefix_cannot_be_padded_inside_a_frame_move() {
    let checked = common::checked_main(
        "(module (func (export \"main\") (result i32) block (result i32) i32.const 99 i32.const 42 i32.const 1 br_if 0 drop end))",
    );
    let cut = checked
        .trace
        .iter()
        .position(|step| step.state_after.frame_move.active)
        .unwrap()
        + 1;
    assert!(
        step_inputs(&checked.trace[..cut], cut + 1, NonZeroUsize::MIN).is_err(),
        "an honest prefix with an open frame move must be refused before padding"
    );
    assert!(step_inputs(&checked.trace, checked.trace.len() + 1, NonZeroUsize::MIN).is_ok());
}

#[test]
fn state_machine_public_step_builders_do_not_panic_on_bad_inputs() {
    fn panics(action: impl FnOnce() + UnwindSafe) -> bool {
        catch_unwind(action).is_err()
    }
    let mut failures = Vec::new();
    let mut check = |name, failed| {
        if failed {
            failures.push(name);
        }
    };
    let mut gather = row(WasmOpcode::Nop);
    gather.row_kind = WasmRowKind::Aux(WasmAuxOpcode::HostEventGather);
    gather.state_before.host_events.slot_cursor = 8;
    check(
        "gather cursor",
        panics(|| {
            assert!(matches!(
                build_witness_vector(&gather),
                Err(neo_wasm::WasmBuildError::Trace(_))
            ));
        }),
    );
    let mut subword = row(WasmOpcode::Nop);
    subword.row_kind = WasmRowKind::Aux(WasmAuxOpcode::HostEventGather);
    subword.host_event_rom_slot = Some(neo_wasm::WasmHostEventRomEntry {
        kind: neo_wasm::WasmHostEventSlotKind::MemoryWrite,
        arg: 0,
        variant: neo_wasm::WasmHostEventRomVariant::Memory {
            base: neo_wasm::ir::WasmHostEventMemoryBase::Argument,
            width: neo_wasm::WasmHostEventMemoryWidth::Byte,
        },
        immediate0: 0,
        immediate1: 0,
        advice: false,
    });
    subword.linear_memory = Some(neo_wasm::LinearMemoryAccess {
        width_bytes: 1,
        byte_offset: 4,
        lane0: neo_wasm::LinearMemoryWordLane {
            word_addr: 0,
            value_before: 0,
            value_after: 0,
        },
        lane1: None,
        lane2: None,
    });
    check(
        "subword offset",
        panics(|| {
            assert!(matches!(
                build_witness_vector(&subword),
                Err(neo_wasm::WasmBuildError::Trace(_))
            ));
        }),
    );
    let mut grow = row(WasmOpcode::MemoryGrow);
    grow.state_before.memory_pages = Some(2);
    grow.state_before.max_memory_pages = Some(1);
    check(
        "memory pages",
        panics(|| {
            assert!(matches!(
                build_witness_vector(&grow),
                Err(neo_wasm::WasmBuildError::Trace(_))
            ));
        }),
    );
    check(
        "indirect-call fields",
        panics(|| {
            assert!(matches!(
                build_witness_vector(&row(WasmOpcode::CallIndirectResolve)),
                Err(neo_wasm::WasmBuildError::Trace(_))
            ));
        }),
    );
    check(
        "batch size",
        panics(|| {
            assert!(matches!(batch_count(1, 0), Err(BatchError::BatchSizeZero)));
        }),
    );
    check(
        "empty trace",
        panics(|| {
            assert!(matches!(batch_count(0, 1), Err(BatchError::EmptyTrace)));
        }),
    );
    check(
        "batch index",
        panics(|| {
            assert!(matches!(
                build_batched_witness(&[row(WasmOpcode::Nop)], 1, 1),
                Err(BatchError::Dimensions)
            ));
        }),
    );
    check(
        "batch capacity",
        panics(|| {
            assert!(matches!(
                step_inputs(&[row(WasmOpcode::Nop)], usize::MAX, NonZeroUsize::MIN),
                Err(WasmApplicationError::Relation(BatchError::Dimensions))
            ));
        }),
    );
    check(
        "activation columns",
        panics(|| {
            assert!(matches!(
                is_active(&[], MemoryPortActivation::When(0)),
                Err(WasmApplicationError::RowLength { .. })
            ));
        }),
    );
    check(
        "limit rows",
        panics(|| {
            assert!(matches!(
                WasmLimits::of(&[], &[Vec::new()]),
                Err(WasmApplicationError::RowLength { .. })
            ));
        }),
    );
    check(
        "schedule columns",
        panics(|| {
            assert!(matches!(
                neo_wasm::write_turn_schedule_guard_witness(&mut []),
                Err(neo_wasm::WasmBuildError::Trace(_))
            ));
        }),
    );
    let wasm = wat::parse_str("(module (func (export \"main\")))").unwrap();
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).unwrap();
    check(
        "entry pc",
        panics(|| {
            assert!(matches!(
                neo_wasm::top_level_initial_state(&artifacts.tables, u64::MAX),
                Err(neo_wasm::WasmBuildError::Trace(_))
            ));
        }),
    );
    assert!(failures.is_empty(), "public builders panicked: {failures:?}");
}
