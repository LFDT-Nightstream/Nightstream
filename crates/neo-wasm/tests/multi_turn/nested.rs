use futures::executor::block_on;
use neo_wasm::host_event_bindings::{
    EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding,
};
use neo_wasm::{
    traces_from_wasmtime_steps_with_host_events as normalize, WasmBuildError, WasmOpcode, WasmOutputState,
    WasmProgramArtifacts, WasmVmStep, WasmtimeTraceHandler, WasmtimeTraceRegistry, WasmtimeTraceStep,
};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, Store, Trap, Val};

use super::super::common;

const HOST: u32 = 1;
const RUN: u32 = 2;
const MIDDLE: u32 = 3;
const LEAF: u32 = 4;
const FAIL: u32 = 5;
const TERMINAL_TRAP: u32 = 6;
const DIVIDE: u32 = 7;
const WRITE_THEN_FAIL: u32 = 8;
const READ_UNBOUND: u32 = 9;
const READ_BOUND: u32 = 10;
const CALL_FAIL: u32 = 11;
const SCRATCH_ADDR: u32 = 32;
const SCRATCH_WORD: u32 = 0x1234_5678;
const SECOND_WORD: u32 = 0x89ab_cdef;

const MODULE: &str = r#"
    (module
      (import "host" "invoke" (func $invoke (param i32 i32) (result i32)))
      (memory (export "memory") 1 1)
      (data (i32.const 24) "\07\00\00\00")
      (func (export "run") (param $mode i32) (result i32)
        i32.const 16 local.get $mode call $invoke
        i32.const 16 i32.load i32.add
        i32.const 16 i32.const 0 i32.store)
      (func (export "middle") (param $mode i32) (result i32)
        i32.const 20 local.get $mode call $invoke
        i32.const 20 i32.load i32.add
        i32.const 20 i32.const 0 i32.store)
      (func (export "leaf") (result i32)
        i32.const 24 i32.load)
      (func $fail (export "fail") (result i32) unreachable)
      (func (export "terminal_trap") (result i32)
        i32.const 16 i32.const 0 call $invoke
        i32.const 16 i32.load i32.add drop
        i32.const 16 i32.const 0 i32.store
        unreachable)
      (func (export "divide") (result i32) i32.const 7 i32.const 0 i32.div_u)
      (func (export "write_then_fail") (param i32) (result i32)
        i32.const 32 local.get 0 i32.store
        unreachable)
      (func (export "read_unbound") (param i32) (result i32) local.get 0 i32.load)
      (func (export "read_bound") (param i32) (result i32) local.get 0 i32.load)
      (func (export "call_fail") (result i32) i32.const 7 call $fail i32.add))
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Success,
    CatchInner,
    CatchOuter,
    Propagate,
    WriteThenPropagate,
}

fn event(slots: &[SlotBinding]) -> EventBlock {
    let mut block = [SlotBinding::Const(0); 8];
    block[..slots.len()].copy_from_slice(slots);
    EventBlock { block, absorb: true }
}

fn bindings() -> HostEventBindings {
    use SlotBinding::*;
    let mut bindings = HostEventBindings::default();
    for fref in [
        RUN,
        MIDDLE,
        LEAF,
        FAIL,
        TERMINAL_TRAP,
        DIVIDE,
        WRITE_THEN_FAIL,
        READ_UNBOUND,
        READ_BOUND,
        CALL_FAIL,
    ] {
        let tag = u64::from(fref) * 10;
        let has_param = matches!(fref, RUN | MIDDLE | WRITE_THEN_FAIL | READ_UNBOUND | READ_BOUND);
        let binds_memory = fref == READ_BOUND;
        bindings.exports.insert(
            fref,
            ExportTemplate {
                entry: if has_param {
                    let mut slots = vec![
                        Const(tag),
                        InputLocal {
                            input: 0,
                            local: 0,
                            limb: Limb::Lo,
                        },
                    ];
                    if binds_memory {
                        slots.push(MemoryWrite32 {
                            input: 1,
                            base: MemoryBase::Local(0),
                            byte_offset: 0,
                        });
                    }
                    vec![event(&slots)]
                } else {
                    vec![]
                },
                exit: vec![event(&[Const(tag + 1), OutputElem { limb: Limb::Lo }])],
                entry_input_count: u8::from(has_param) + u8::from(binds_memory),
            },
        );
    }
    bindings.imports.insert(
        HOST,
        ImportTemplate {
            events: vec![event(&[Const(10), ArgElem { arg: 0, limb: Limb::Lo }])],
            return_events: Some(vec![event(&[
                Const(11),
                MemoryWrite32 {
                    input: 0,
                    base: MemoryBase::Arg(0),
                    byte_offset: 0,
                },
                ResultElem { limb: Limb::Lo },
                ResultElem { limb: Limb::Hi },
            ])]),
            input_count: 1,
        },
    );
    bindings
}

async fn callback(caller: &mut Caller<'_, WasmtimeTraceRegistry>, name: &str, args: &[Val]) -> wasmtime::Result<i32> {
    let function = caller.get_export(name).unwrap().into_func().unwrap();
    let mut result = [Val::I32(0)];
    function.call_async(caller, args, &mut result).await?;
    Ok(result[0].i32().unwrap())
}

fn runtime(trap: Trap, scratch_word: u32) -> (Store<WasmtimeTraceRegistry>, Instance) {
    let bytes = wat::parse_str(MODULE).unwrap();
    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(&bytes, bindings()).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap_async(
            "host",
            "invoke",
            move |mut caller: Caller<'_, WasmtimeTraceRegistry>, (address, mode): (i32, i32)| {
                Box::new(async move {
                    let value = match address {
                        16 => {
                            let middle = callback(&mut caller, "middle", &[Val::I32(mode)]).await;
                            let value = if mode == Mode::CatchOuter as i32 {
                                assert_eq!(middle.unwrap_err().downcast_ref::<Trap>(), Some(&trap));
                                callback(&mut caller, "middle", &[Val::I32(Mode::Success as i32)]).await?
                            } else {
                                middle?
                            };
                            value + 20
                        }
                        20 => {
                            if mode != Mode::Success as i32 {
                                let export = match (mode == Mode::WriteThenPropagate as i32, trap) {
                                    (true, Trap::UnreachableCodeReached) => "write_then_fail",
                                    (false, Trap::IntegerDivisionByZero) => "divide",
                                    (false, Trap::UnreachableCodeReached) => "fail",
                                    _ => unreachable!(),
                                };
                                let args = [Val::I32(scratch_word as i32)];
                                let args = if mode == Mode::WriteThenPropagate as i32 {
                                    &args[..]
                                } else {
                                    &[]
                                };
                                let error = callback(&mut caller, export, args).await.unwrap_err();
                                assert_eq!(error.downcast_ref::<Trap>(), Some(&trap));
                                if mode != Mode::CatchInner as i32 {
                                    return Err(error);
                                }
                            }
                            callback(&mut caller, "leaf", &[]).await? + 10
                        }
                        _ => panic!("unexpected import address {address}"),
                    };
                    let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                    memory.write(&mut caller, address as usize, &(address + 3).to_le_bytes())?;
                    Ok((value,))
                })
            },
        )
        .unwrap();
    let module = Module::new(&engine, &bytes).unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    (store, instance)
}

fn invoke(
    store: &mut Store<WasmtimeTraceRegistry>,
    instance: Instance,
    export: &str,
    args: &[Val],
) -> wasmtime::Result<i32> {
    let function = instance.get_func(&mut *store, export).unwrap();
    let memory = instance.get_memory(&mut *store, "memory").unwrap();
    let mut result = [Val::I32(0)];
    let call = block_on(function.call_async(&mut *store, args, &mut result));
    assert_eq!(&memory.data(&*store)[16..28], &[0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0]);
    store.data().check_errors().unwrap();
    call.map(|()| result[0].i32().unwrap())
}

fn check_capture(rows: &[WasmtimeTraceStep], export: u32, mode: Mode, trap: Trap) {
    let mut callbacks = Vec::new();
    for row in rows {
        let frame = (row.current_function_ref.unwrap(), row.activation_depth);
        if callbacks.last() != Some(&frame) {
            callbacks.push(frame);
        }
    }
    let mut expected = vec![(export, 1), (MIDDLE, 2)];
    if mode != Mode::Success {
        expected.push((
            if mode == Mode::WriteThenPropagate {
                WRITE_THEN_FAIL
            } else if trap == Trap::IntegerDivisionByZero {
                DIVIDE
            } else {
                FAIL
            },
            3,
        ));
    }
    if mode == Mode::CatchOuter {
        expected.push((MIDDLE, 2));
    }
    if !matches!(mode, Mode::Propagate | Mode::WriteThenPropagate) {
        expected.extend([(LEAF, 3), (MIDDLE, 2), (export, 1)]);
    }
    assert_eq!(callbacks, expected, "Wasmtime callback order for {mode:?}");

    let calls: Vec<_> = rows
        .iter()
        .filter(|row| row.function_ref == Some(HOST) && row.opcode_decoded == Some(WasmOpcode::Call))
        .collect();
    let returned: &[Option<u32>] = match mode {
        Mode::Success | Mode::CatchInner => &[Some(16), Some(20)],
        Mode::CatchOuter => &[Some(16), None, Some(20)],
        Mode::Propagate | Mode::WriteThenPropagate => &[None, None],
    };
    assert_eq!(calls.len(), returned.len());
    for (call, &address) in calls.into_iter().zip(returned) {
        if let Some(address) = address {
            let bytes = call.host_call_memory.as_ref().unwrap().as_ref().unwrap();
            assert_eq!(bytes.len(), 4);
            let word = u32::from_le_bytes(std::array::from_fn(|index| bytes[&(address + index as u32)]));
            assert_eq!(word, address + 3);
        } else {
            assert_eq!(call.host_call_memory, None);
        }
    }
}

fn check_turn(trace: &[WasmVmStep], export: u32) {
    let end = common::final_state(trace);
    assert!(end.is_terminal());
    assert_eq!((end.call_stack_depth, end.host_events.turn_export_fref), (0, export));
    let trapped = export == TERMINAL_TRAP;
    assert_eq!(end.trapped, trapped);
    let output = if trapped {
        WasmOutputState::ZERO
    } else {
        WasmOutputState {
            enabled: true,
            value_lo: 79,
            value_hi: 0,
        }
    };
    assert_eq!(end.output, output);
    let transitions: Vec<_> = trace
        .iter()
        .filter_map(|row| {
            let kind = row.row_kind;
            let state = row.state_after.host_events;
            if kind.is_host_suspend()
                || kind.is_host_resume()
                || (kind.is_turn_boundary() && state.host_frame_depth != 0)
            {
                Some((kind.is_turn_boundary(), state.turn_export_fref, state.host_frame_depth))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        transitions,
        [
            (false, export, 1),
            (true, MIDDLE, 1),
            (false, MIDDLE, 2),
            (true, LEAF, 2),
            (false, MIDDLE, 1),
            (false, export, 0),
        ]
    );

    let events: Vec<_> = common::absorbed_blocks(trace)
        .into_iter()
        .map(|block| (block[0], block[1], block[2]))
        .collect();
    let mut expected = vec![
        (10, 16, 0),
        (30, 0, 0),
        (10, 20, 0),
        (41, 7, 0),
        (11, 23, 17),
        (31, 40, 0),
        (11, 19, 60),
    ];
    if !trapped {
        expected.insert(0, (20, 0, 0));
        expected.push((21, 79, 0));
    }
    assert_eq!(events, expected);

    assert_eq!(
        memory_effects(trace),
        [
            (24, 7, 7),
            (20, 0, 23),
            (20, 23, 23),
            (20, 23, 0),
            (16, 0, 19),
            (16, 19, 19),
            (16, 19, 0)
        ]
    );
}

fn memory_effects(trace: &[WasmVmStep]) -> Vec<(u64, u32, u32)> {
    trace
        .iter()
        .filter_map(|row| {
            row.linear_memory.map(|access| {
                assert_eq!(
                    (access.width_bytes, access.byte_offset, access.lane1, access.lane2),
                    (4, 0, None, None)
                );
                let lane = access.lane0;
                (lane.word_addr * 4, lane.value_before, lane.value_after)
            })
        })
        .collect()
}

fn check_rows(trace: &[WasmVmStep], artifacts: &WasmProgramArtifacts, export: u32) {
    let bindings = &artifacts.host_event_bindings;
    let pc = common::entry_pc_for_function_ref(artifacts, export.into());
    let initial =
        neo_wasm::host_event_top_level_initial_state(&artifacts.tables, pc, bindings, export, Default::default())
            .unwrap();
    assert_eq!(trace[0].state_before, initial);
    common::sanity_check_trace_with_bindings(trace, artifacts, bindings);
    common::ccs_check_trace(trace);
}

fn successful_followup(
    store: &mut Store<WasmtimeTraceRegistry>,
    instance: Instance,
    start: usize,
    trap: Trap,
) -> (Vec<WasmVmStep>, WasmProgramArtifacts) {
    assert_eq!(invoke(store, instance, "run", &[Val::I32(0)]).unwrap(), 79);
    let captured = store.data().single_instance().unwrap();
    let suffix = &captured.steps()[start..];
    check_capture(suffix, RUN, Mode::Success, trap);
    let artifacts = captured.artifacts();
    let trace = normalize(suffix, artifacts, Default::default()).expect("normalize the successful recovery invocation");
    check_turn(&trace, RUN);
    check_rows(&trace, artifacts, RUN);
    (trace, artifacts.clone())
}

fn check_recovery(mode: Mode, trap: Trap) -> (Vec<WasmVmStep>, WasmProgramArtifacts) {
    let (mut store, instance) = runtime(trap, SCRATCH_WORD);
    let result = invoke(&mut store, instance, "run", &[Val::I32(mode as i32)]);
    match mode {
        Mode::CatchInner | Mode::CatchOuter => assert_eq!(result.unwrap(), 79),
        Mode::Propagate => assert_eq!(result.unwrap_err().downcast_ref::<Trap>(), Some(&trap)),
        Mode::Success | Mode::WriteThenPropagate => unreachable!(),
    }
    let captured = store.data().single_instance().unwrap();
    let prefix_len = captured.steps().len();
    check_capture(captured.steps(), RUN, mode, trap);
    let error = normalize(captured.steps(), captured.artifacts(), Default::default())
        .expect_err("caught trap or incomplete nested unwind is unsupported");
    let reason = match mode {
        Mode::CatchInner | Mode::CatchOuter => "is not the final step",
        Mode::Propagate => "suspended host activation",
        Mode::Success | Mode::WriteThenPropagate => unreachable!(),
    };
    assert!(error.to_string().contains(reason), "{mode:?} {trap:?}: {error}");
    let recovered = successful_followup(&mut store, instance, prefix_len, trap);
    let captured = store.data().single_instance().unwrap();
    check_capture(&captured.steps()[..prefix_len], RUN, mode, trap);
    let combined = normalize(captured.steps(), captured.artifacts(), Default::default());
    if mode == Mode::Propagate {
        let trace = combined.expect("normalize propagated trap and recovery together");
        assert_eq!(common::observe_turns(&trace), 1);
        check_rows(&trace, captured.artifacts(), RUN);
        return (trace, captured.artifacts().clone());
    }
    let combined = combined.expect_err("caught callback trap is unsupported");
    assert!(
        combined.to_string().contains("is not the final step"),
        "{mode:?} {trap:?}: {combined}"
    );
    recovered
}

fn check_raw_scratch_access(row: &WasmtimeTraceStep, opcode: WasmOpcode, before: u32, after: u32) {
    assert_eq!(row.opcode_decoded, Some(opcode));
    let access = row.memory.as_ref().unwrap();
    assert_eq!(
        (access.effective_address, access.width_bytes),
        (u64::from(SCRATCH_ADDR), 4)
    );
    assert_eq!((access.lane0.value_before, access.lane0.value_after), (before, after));
}

fn check_persisted_word_recovery(word: u32) -> (Vec<WasmVmStep>, Vec<WasmVmStep>, WasmProgramArtifacts) {
    let trap = Trap::UnreachableCodeReached;
    let (mut store, instance) = runtime(trap, word);
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let word_bytes = word.to_le_bytes();
    let address = SCRATCH_ADDR as usize;
    let scratch = address..address + word_bytes.len();
    assert_eq!(&memory.data(&store)[scratch.clone()], &[0; 4]);
    let mode = Mode::WriteThenPropagate;
    let error = invoke(&mut store, instance, "run", &[Val::I32(mode as i32)]).unwrap_err();
    assert_eq!(error.downcast_ref::<Trap>(), Some(&trap));
    assert_eq!(&memory.data(&store)[scratch.clone()], &word_bytes);

    let captured = store.data().single_instance().unwrap();
    let trapped_steps = captured.steps().to_vec();
    check_capture(&trapped_steps, RUN, mode, trap);
    let (fail, before_trap) = trapped_steps.split_last().unwrap();
    let write = before_trap.last().unwrap();
    check_raw_scratch_access(write, WasmOpcode::I32Store, 0, word);
    assert_eq!(fail.opcode_decoded, Some(WasmOpcode::Unreachable));
    for step in [write, fail] {
        assert_eq!(
            (step.current_function_ref, step.activation_depth),
            (Some(WRITE_THEN_FAIL), 3)
        );
    }
    let error = normalize(&trapped_steps, captured.artifacts(), Default::default()).unwrap_err();
    assert!(error.to_string().contains("suspended host activation"), "{error}");

    let args = [Val::I32(SCRATCH_ADDR as i32)];
    assert_eq!(
        invoke(&mut store, instance, "read_unbound", &args).unwrap(),
        word as i32
    );
    assert_eq!(&memory.data(&store)[scratch.clone()], &word_bytes);
    let captured = store.data().single_instance().unwrap();
    assert_eq!(&captured.steps()[..trapped_steps.len()], trapped_steps.as_slice());
    let unbound = &captured.steps()[trapped_steps.len()..];
    let load = unbound.iter().find(|row| row.memory.is_some()).unwrap();
    assert_eq!(
        (load.current_function_ref, load.activation_depth),
        (Some(READ_UNBOUND), 1)
    );
    check_raw_scratch_access(load, WasmOpcode::I32Load, word, word);
    let error = normalize(unbound, captured.artifacts(), Default::default()).unwrap_err();
    let reason = format!("word {}: trace observed {word}, replay has 0", SCRATCH_ADDR / 4);
    assert!(
        matches!(&error, WasmBuildError::Trace(message) if message.contains("linear-memory replay mismatch") && message.ends_with(&reason)),
        "{error}"
    );
    let artifacts = captured.artifacts();
    let combined = normalize(captured.steps(), artifacts, Default::default()).expect("normalize both top-level calls");
    check_rows(&combined, artifacts, RUN);
    assert_eq!(common::observe_turns(&combined), 1);
    let unwinds: Vec<_> = combined
        .iter()
        .filter(|row| row.row_kind.is_turn_boundary() && row.state_before.trapped)
        .collect();
    let [unwind] = unwinds.as_slice() else {
        panic!("expected one trap unwind")
    };
    let before = unwind.state_before;
    let after = unwind.state_after;
    assert_eq!((before.call_stack_depth, before.host_events.host_frame_depth), (2, 2));
    assert!(before.activation_finished);
    assert_eq!(before.output, WasmOutputState::ZERO);
    assert_eq!((after.call_stack_depth, after.host_events.host_frame_depth), (0, 0));
    assert_eq!(after.host_events.turn_export_fref, READ_UNBOUND);
    assert!(!after.trapped && !after.activation_finished && !after.host_events.awaiting_host);
    assert_eq!(after.output, WasmOutputState::ZERO);
    assert!(combined.iter().all(|row| !row.row_kind.is_host_resume()));
    assert_eq!(
        memory_effects(&combined),
        [
            (u64::from(SCRATCH_ADDR), 0, word),
            (u64::from(SCRATCH_ADDR), word, word),
        ]
    );
    let events = [
        (20, mode as u64),
        (10, 16),
        (30, mode as u64),
        (10, 20),
        (80, u64::from(word)),
        (90, u64::from(SCRATCH_ADDR)),
        (91, u64::from(word)),
    ]
    .map(|(tag, argument)| [tag, argument, 0, 0, 0, 0, 0, 0]);
    assert_eq!(common::absorbed_blocks(&combined), events);
    let end = common::final_state(&combined);
    assert!(end.is_terminal() && !end.trapped);
    assert_eq!(
        end.output,
        WasmOutputState {
            enabled: true,
            value_lo: word,
            value_hi: 0
        }
    );
    assert_eq!(
        end.comm_chain,
        events
            .into_iter()
            .fold([0; 4], neo_wasm::comm_chain::absorb_block)
    );
    let before_bound = captured.steps().to_vec();

    assert_eq!(invoke(&mut store, instance, "read_bound", &args).unwrap(), word as i32);
    assert_eq!(&memory.data(&store)[scratch], &word_bytes);
    let captured = store.data().single_instance().unwrap();
    assert_eq!(&captured.steps()[..before_bound.len()], before_bound.as_slice());
    let bound = &captured.steps()[before_bound.len()..];
    let entry_bytes = bound[0].entry_memory.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(entry_bytes.len(), 4);
    for (offset, byte) in word_bytes.into_iter().enumerate() {
        assert_eq!(entry_bytes[&(SCRATCH_ADDR + offset as u32)], byte);
    }
    let load = bound.iter().find(|row| row.memory.is_some()).unwrap();
    assert_eq!(
        (load.current_function_ref, load.activation_depth),
        (Some(READ_BOUND), 1)
    );
    check_raw_scratch_access(load, WasmOpcode::I32Load, word, word);
    let artifacts = captured.artifacts();
    let trace = normalize(bound, artifacts, Default::default()).expect("normalize the bound recovery invocation");
    let end = common::final_state(&trace);
    assert!(end.is_terminal());
    assert!(!end.trapped);
    assert_eq!(
        end.output,
        WasmOutputState {
            enabled: true,
            value_lo: word,
            value_hi: 0,
        }
    );
    let tag = u64::from(READ_BOUND) * 10;
    assert_eq!(
        common::absorbed_blocks(&trace),
        [
            [tag, u64::from(SCRATCH_ADDR), u64::from(word), 0, 0, 0, 0, 0],
            [tag + 1, u64::from(word), 0, 0, 0, 0, 0, 0],
        ]
    );
    let address = u64::from(SCRATCH_ADDR);
    assert_eq!(memory_effects(&trace), [(address, 0, word), (address, word, word)]);
    check_rows(&trace, artifacts, READ_BOUND);
    (combined, trace, artifacts.clone())
}

#[test]
fn nested_callbacks_across_top_level_turns() {
    let (mut store, instance) = runtime(Trap::UnreachableCodeReached, SCRATCH_WORD);
    let mut start = 0;
    for _ in 0..2 {
        assert_eq!(invoke(&mut store, instance, "run", &[Val::I32(0)]).unwrap(), 79);
        let captured = store.data().single_instance().unwrap();
        check_capture(
            &captured.steps()[start..],
            RUN,
            Mode::Success,
            Trap::UnreachableCodeReached,
        );
        start = captured.steps().len();
    }
    let captured = store.data().single_instance().unwrap();
    let artifacts = captured.artifacts();
    let trace = normalize(captured.steps(), artifacts, Default::default()).unwrap();
    assert_eq!(common::observe_turns(&trace), 1);
    let turns: Vec<_> = trace
        .split_inclusive(|row| row.state_after.is_terminal())
        .collect();
    assert_eq!(turns.len(), 2);
    for turn in turns {
        check_turn(turn, RUN);
    }
    check_rows(&trace, artifacts, RUN);
}

#[test]
fn trapped_nested_callbacks_allow_recovery() {
    for trap in [Trap::UnreachableCodeReached, Trap::IntegerDivisionByZero] {
        for mode in [Mode::CatchInner, Mode::CatchOuter, Mode::Propagate] {
            check_recovery(mode, trap);
        }
    }
}

#[test]
fn nested_trap_preserves_a_guest_write_across_top_level_unwind() {
    for word in [SCRATCH_WORD, SECOND_WORD] {
        check_persisted_word_recovery(word);
    }
}

#[test]
fn unwind_discards_guest_frames_and_retained_operands() {
    let trap = Trap::UnreachableCodeReached;
    let (mut store, instance) = runtime(trap, SCRATCH_WORD);
    let error = invoke(&mut store, instance, "call_fail", &[]).unwrap_err();
    assert_eq!(error.downcast_ref::<Trap>(), Some(&trap));
    assert_eq!(
        invoke(&mut store, instance, "read_unbound", &[Val::I32(24)]).unwrap(),
        7
    );
    let captured = store.data().single_instance().unwrap();
    let trace = normalize(captured.steps(), captured.artifacts(), Default::default()).unwrap();
    check_rows(&trace, captured.artifacts(), CALL_FAIL);
    let boundary = trace
        .iter()
        .find(|row| row.row_kind.is_turn_boundary())
        .unwrap();
    assert!(boundary.state_before.trapped);
    assert_eq!(boundary.state_before.call_stack_depth, 1);
    assert_eq!(boundary.state_before.host_events.host_frame_depth, 0);
    assert!(boundary.state_before.sp > 0);
    assert_eq!(boundary.state_after.call_stack_depth, 0);
    assert_eq!(boundary.state_after.locals_fbp, 0);
    assert_eq!(common::final_state(&trace).output.value_lo, 7);
    assert_eq!(common::observe_turns(&trace), 1);
}

#[test]
fn top_level_trap_after_nested_callbacks() {
    let (mut store, instance) = runtime(Trap::UnreachableCodeReached, SCRATCH_WORD);
    let error = invoke(&mut store, instance, "terminal_trap", &[]).unwrap_err();
    assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::UnreachableCodeReached));
    let captured = store.data().single_instance().unwrap();
    let artifacts = captured.artifacts();
    check_capture(
        captured.steps(),
        TERMINAL_TRAP,
        Mode::Success,
        Trap::UnreachableCodeReached,
    );
    let trace = normalize(captured.steps(), artifacts, Default::default()).expect("normalize terminal trap");
    check_turn(&trace, TERMINAL_TRAP);
    check_rows(&trace, artifacts, TERMINAL_TRAP);
    let start = captured.steps().len();
    successful_followup(&mut store, instance, start, Trap::UnreachableCodeReached);
    let captured = store.data().single_instance().unwrap();
    let combined =
        normalize(captured.steps(), captured.artifacts(), Default::default()).expect("recover after top-level trap");
    check_rows(&combined, captured.artifacts(), TERMINAL_TRAP);
    assert_eq!(common::observe_turns(&combined), 1);
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_nested_callback_recovery_replay() {
    let (trace, artifacts) = check_recovery(Mode::Propagate, Trap::IntegerDivisionByZero);
    common::shared_host::prove(&trace, &artifacts);
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_nested_trap_bound_read_replay() {
    let (_, trace, artifacts) = check_persisted_word_recovery(SCRATCH_WORD);
    common::shared_host::prove(&trace, &artifacts);
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_nested_unwind_reads_callback_write() {
    prove_callback_write(SCRATCH_WORD);
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_nested_unwind_reads_second_callback_word() {
    prove_callback_write(SECOND_WORD);
}

fn prove_callback_write(word: u32) {
    let (trace, _, artifacts) = check_persisted_word_recovery(word);
    let preload = common::shared_host::program_preload(&artifacts);
    common::shared_host::prove_with_limits(&trace, &preload, neo_wasm::nebula::WasmSize::Small.limits(), 16);
}
