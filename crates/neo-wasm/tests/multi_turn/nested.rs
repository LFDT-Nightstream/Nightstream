use futures::executor::block_on;
use neo_wasm::host_event_bindings::{
    EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding,
};
use neo_wasm::{
    traces_from_wasmtime_steps_with_host_events as normalize, WasmOpcode, WasmOutputState, WasmProgramArtifacts,
    WasmVmStep, WasmtimeTraceHandler, WasmtimeTraceRegistry, WasmtimeTraceStep,
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
      (func (export "fail") (result i32) unreachable)
      (func (export "terminal_trap") (result i32)
        i32.const 16 i32.const 0 call $invoke
        i32.const 16 i32.load i32.add drop
        i32.const 16 i32.const 0 i32.store
        unreachable)
      (func (export "divide") (result i32) i32.const 7 i32.const 0 i32.div_u))
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Success,
    CatchInner,
    CatchOuter,
    Propagate,
}

fn event(slots: &[SlotBinding]) -> EventBlock {
    let mut block = [SlotBinding::Const(0); 8];
    block[..slots.len()].copy_from_slice(slots);
    EventBlock { block, absorb: true }
}

fn bindings() -> HostEventBindings {
    use SlotBinding::*;
    let mut bindings = HostEventBindings::default();
    for fref in [RUN, MIDDLE, LEAF, FAIL, TERMINAL_TRAP, DIVIDE] {
        let tag = u64::from(fref) * 10;
        let has_mode = matches!(fref, RUN | MIDDLE);
        bindings.exports.insert(
            fref,
            ExportTemplate {
                entry: if has_mode {
                    vec![event(&[
                        Const(tag),
                        InputLocal {
                            input: 0,
                            local: 0,
                            limb: Limb::Lo,
                        },
                    ])]
                } else {
                    vec![]
                },
                exit: vec![event(&[Const(tag + 1), OutputElem { limb: Limb::Lo }])],
                entry_input_count: u8::from(has_mode),
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

fn runtime(trap: Trap) -> (Store<WasmtimeTraceRegistry>, Instance) {
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
                                let export = match trap {
                                    Trap::IntegerDivisionByZero => "divide",
                                    Trap::UnreachableCodeReached => "fail",
                                    _ => unreachable!(),
                                };
                                let error = callback(&mut caller, export, &[]).await.unwrap_err();
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
            if trap == Trap::IntegerDivisionByZero {
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
    if mode != Mode::Propagate {
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
        Mode::Propagate => &[None, None],
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

    let memory: Vec<_> = trace
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
        .collect();
    assert_eq!(
        memory,
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
    let (mut store, instance) = runtime(trap);
    let result = invoke(&mut store, instance, "run", &[Val::I32(mode as i32)]);
    match mode {
        Mode::CatchInner | Mode::CatchOuter => assert_eq!(result.unwrap(), 79),
        Mode::Propagate => assert_eq!(result.unwrap_err().downcast_ref::<Trap>(), Some(&trap)),
        Mode::Success => unreachable!(),
    }
    let captured = store.data().single_instance().unwrap();
    let prefix_len = captured.steps().len();
    check_capture(captured.steps(), RUN, mode, trap);
    let error = normalize(captured.steps(), captured.artifacts(), Default::default())
        .expect_err("nested trap unwinding is outside the trace relation");
    let reason = match mode {
        Mode::CatchInner | Mode::CatchOuter => "is not the final step",
        Mode::Propagate => "suspended host activation",
        Mode::Success => unreachable!(),
    };
    assert!(error.to_string().contains(reason), "{mode:?} {trap:?}: {error}");
    let recovered = successful_followup(&mut store, instance, prefix_len, trap);
    let captured = store.data().single_instance().unwrap();
    check_capture(&captured.steps()[..prefix_len], RUN, mode, trap);
    let combined = normalize(captured.steps(), captured.artifacts(), Default::default())
        .expect_err("a later invocation cannot continue the trapped row");
    assert!(
        combined.to_string().contains("is not the final step"),
        "{mode:?} {trap:?}: {combined}"
    );
    recovered
}

#[test]
fn nested_callbacks_across_top_level_turns() {
    let (mut store, instance) = runtime(Trap::UnreachableCodeReached);
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
fn top_level_trap_after_nested_callbacks() {
    let (mut store, instance) = runtime(Trap::UnreachableCodeReached);
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
}

#[test]
#[ignore = "runs the public memory prover and verifier"]
fn public_nested_callback_recovery_replay() {
    let (trace, artifacts) = check_recovery(Mode::Propagate, Trap::IntegerDivisionByZero);
    common::shared_host::prove(&trace, &artifacts);
}
