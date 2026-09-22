//! Import output buffers recovered from real host writes at the return boundary.

mod common;

use futures::executor::block_on;
use neo_wasm::host_event_bindings::{EventBlock, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding};
use neo_wasm::{WasmProgramArtifacts, WasmVmStep, WasmtimeTraceHandler, WasmtimeTraceRegistry, WasmtimeTraceStep};
use std::sync::atomic::{AtomicU32, Ordering};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, Store, Val};

fn bindings(slots: &[SlotBinding], inputs: u8, export: u32) -> HostEventBindings {
    let mut block = [SlotBinding::Const(0); 8];
    block[..slots.len()].copy_from_slice(slots);
    block[6] = SlotBinding::ResultElem { limb: Limb::Lo };
    block[7] = SlotBinding::ResultElem { limb: Limb::Hi };
    let mut bindings = HostEventBindings::import_free(export);
    let mut exit = [SlotBinding::Const(0); 8];
    exit[0] = SlotBinding::OutputElem { limb: Limb::Lo };
    exit[1] = SlotBinding::OutputElem { limb: Limb::Hi };
    bindings.exports.get_mut(&export).unwrap().exit = vec![EventBlock {
        block: exit,
        absorb: true,
    }];
    bindings.imports.insert(
        1,
        ImportTemplate {
            events: vec![EventBlock { block, absorb: true }],
            input_count: inputs,
        },
    );
    bindings
}

fn word(input: u8, offset: u32) -> SlotBinding {
    SlotBinding::MemoryWrite32 {
        input,
        base: MemoryBase::Arg(0),
        byte_offset: offset,
    }
}

fn store() -> Store<WasmtimeTraceRegistry> {
    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    let engine = Engine::new(&config).unwrap();
    let mut store = Store::new(&engine, WasmtimeTraceRegistry::default());
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    store
}

fn invoke(store: &mut Store<WasmtimeTraceRegistry>, instance: Instance) {
    let run = instance.get_func(&mut *store, "run").unwrap();
    block_on(run.call_async(store, &[], &mut [Val::I32(0)])).unwrap();
}

fn capture(
    wat: &str,
    bindings: HostEventBindings,
    calls: usize,
    host: impl Fn(Caller<'_, WasmtimeTraceRegistry>, i32) -> wasmtime::Result<i32> + Send + Sync + 'static,
) -> (Vec<WasmtimeTraceStep>, WasmProgramArtifacts) {
    let bytes = wat::parse_str(wat).unwrap();
    let mut store = store();
    store.data_mut().register_module(&bytes, bindings).unwrap();
    let module = Module::new(store.engine(), &bytes).unwrap();
    let mut linker = Linker::new(store.engine());
    linker.func_wrap("host", "write", host).unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    for _ in 0..calls {
        invoke(&mut store, instance);
    }
    let state = store.data().single_instance().unwrap();
    (state.steps().to_vec(), state.artifacts().clone())
}

fn normalize(
    steps: &[WasmtimeTraceStep],
    artifacts: &WasmProgramArtifacts,
) -> Result<Vec<WasmVmStep>, neo_wasm::WasmBuildError> {
    neo_wasm::traces_from_wasmtime_steps_with_host_events(steps, artifacts, Default::default())
}

fn check(steps: &[WasmtimeTraceStep], artifacts: &WasmProgramArtifacts) -> Vec<WasmVmStep> {
    let trace = normalize(steps, artifacts).unwrap();
    common::sanity_check_trace_with_bindings(&trace, artifacts, &artifacts.host_event_bindings);
    common::ccs_check_trace(&trace);
    trace
}

#[test]
fn import_buffers_are_captured_before_resume_and_preserve_memory_across_turns() {
    for indirect in [false, true] {
        let call = if indirect {
            "i32.const 0 call_indirect (type $host)"
        } else {
            "call $write"
        };
        let wat = format!(
            r#"(module
            (type $host (func (param i32) (result i32)))
            (import "host" "write" (func $write (type $host)))
            (memory (export "memory") 1)
            (data (i32.const 16) "\01\00\00\00")
            (table 1 funcref) (elem (i32.const 0) $write)
            (func $nested (result i32)
                i32.const 16 i32.const 16 {call}
                i32.store
                i32.const 20 i32.load16_u
                i32.const 22 i32.load8_u i32.add)
            (func (export "run") (result i32) call $nested))"#
        );
        let bindings = bindings(
            &[
                word(0, 0),
                SlotBinding::MemoryWrite16 {
                    input: 1,
                    base: MemoryBase::Arg(0),
                    byte_offset: 4,
                },
                SlotBinding::MemoryWrite8 {
                    input: 2,
                    base: MemoryBase::Arg(0),
                    byte_offset: 6,
                },
                SlotBinding::MemoryWrite8 {
                    input: 2,
                    base: MemoryBase::Arg(0),
                    byte_offset: 7,
                },
            ],
            3,
            3,
        );
        let counter = AtomicU32::new(0);
        let (steps, artifacts) = capture(&wat, bindings, 2, move |mut caller, ptr| {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
            memory.write(&mut caller, ptr as usize, &(50 + i).to_le_bytes())?;
            memory.write(&mut caller, ptr as usize + 4, &(500 + i as u16).to_le_bytes())?;
            memory.write(&mut caller, ptr as usize + 6, &[7 + i as u8; 2])?;
            Ok(9)
        });
        let captured: Vec<_> = steps
            .iter()
            .filter_map(|row| row.host_call_memory.as_ref())
            .collect();
        assert_eq!(captured.len(), 2);
        for (i, bytes) in captured.iter().enumerate() {
            let bytes = bytes.as_ref().unwrap();
            assert_eq!(bytes.len(), 8);
            assert_eq!(bytes[&16], 50 + i as u8);
            assert_eq!(bytes[&22], 7 + i as u8);
        }
        let trace = check(&steps, &artifacts);
        let writes: Vec<_> = trace
            .iter()
            .filter_map(|row| {
                let rom = row.host_event_rom_slot?;
                let access = row.linear_memory?;
                (rom.kind == neo_wasm::WasmHostEventSlotKind::MemoryWrite && access.width_bytes == 4)
                    .then_some((access.lane0.value_before, access.lane0.value_after))
            })
            .collect();
        assert_eq!(writes, [(1, 50), (9, 51)]);
    }
}

const SIMPLE: &str = r#"(module
    (import "host" "write" (func $write (param i32) (result i32)))
    (memory (export "memory") 1)
    (func (export "run") (result i32) i32.const 16 call $write))"#;

#[test]
fn import_recovery_checks_argument_pointer_and_requires_configured_capture() {
    // The full address matrix is covered by capture_and_replay_share_host_event_address_rules.
    // Here, check that capture and replay both resolve the pointer from the import argument.
    let wat = SIMPLE.replace("i32.const 16", "i32.const 17");
    let (steps, artifacts) = capture(&wat, bindings(&[word(0, 0)], 1, 2), 1, |_caller, _ptr| Ok(9));
    let error = steps
        .iter()
        .find_map(|row| row.host_call_memory.as_ref())
        .unwrap()
        .as_ref()
        .unwrap_err();
    assert!(error.contains("not naturally aligned"), "{error}");
    assert!(normalize(&steps, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("not naturally aligned"));

    let (steps, mut artifacts) = capture(SIMPLE, HostEventBindings::default(), 1, |_caller, _ptr| Ok(9));
    assert!(steps.iter().all(|row| row.host_call_memory.is_none()));
    artifacts.host_event_bindings = bindings(&[word(0, 0)], 1, 2);
    assert!(normalize(&steps, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("missing import return memory capture"));
}

#[test]
fn import_capture_waits_for_the_caller_after_cross_instance_reentry() {
    let caller_bytes = wat::parse_str(SIMPLE).unwrap();
    let callee_bytes = wat::parse_str(
        r#"(module
        (import "host" "write" (func $write (param i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "run") (result i32) i32.const 32 call $write))"#,
    )
    .unwrap();
    let mut store = store();
    for bytes in [&caller_bytes, &callee_bytes] {
        store
            .data_mut()
            .register_module(bytes, bindings(&[word(0, 0)], 1, 2))
            .unwrap();
    }
    let mut linker = Linker::new(store.engine());
    linker
        .func_wrap(
            "host",
            "write",
            |mut caller: Caller<'_, WasmtimeTraceRegistry>, ptr: i32| {
                let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                memory.write(&mut caller, ptr as usize, &17u32.to_le_bytes())?;
                Ok(17i32)
            },
        )
        .unwrap();
    let module = Module::new(store.engine(), &callee_bytes).unwrap();
    let callee_instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let callee = callee_instance.get_func(&mut store, "run").unwrap();
    let mut linker = Linker::new(store.engine());
    linker
        .func_wrap_async(
            "host",
            "write",
            move |mut caller: Caller<'_, WasmtimeTraceRegistry>, (ptr,): (i32,)| {
                Box::new(async move {
                    let mut result = [Val::I32(0)];
                    callee.call_async(&mut caller, &[], &mut result).await?;
                    let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                    memory.write(&mut caller, ptr as usize, &99u32.to_le_bytes())?;
                    Ok((result[0].i32().unwrap(),))
                })
            },
        )
        .unwrap();
    let module = Module::new(store.engine(), &caller_bytes).unwrap();
    let caller_instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    invoke(&mut store, caller_instance);
    for (instance, address, value) in [(caller_instance, 16, 99), (callee_instance, 32, 17)] {
        let state = store
            .data()
            .instance(instance.debug_index_in_store())
            .unwrap();
        let bytes = state
            .steps()
            .iter()
            .find_map(|row| row.host_call_memory.as_ref())
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[&address], value);
        check(state.steps(), state.artifacts());
    }
}

#[test]
fn same_instance_reentry_captures_each_return_but_cannot_be_normalized_as_an_atomic_call() {
    let bytes = wat::parse_str(
        r#"(module
        (import "host" "write" (func $write (param i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "run") (result i32) i32.const 16 call $write)
        (func (export "callback") (result i32) i32.const 32 call $write))"#,
    )
    .unwrap();
    let mut store = store();
    store
        .data_mut()
        .register_module(&bytes, bindings(&[word(0, 0)], 1, 2))
        .unwrap();
    let mut linker = Linker::new(store.engine());
    linker
        .func_wrap_async(
            "host",
            "write",
            |mut caller: Caller<'_, WasmtimeTraceRegistry>, (ptr,): (i32,)| {
                Box::new(async move {
                    if ptr == 16 {
                        let callback = caller.get_export("callback").unwrap().into_func().unwrap();
                        callback
                            .call_async(&mut caller, &[], &mut [Val::I32(0)])
                            .await?;
                    }
                    let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                    memory.write(&mut caller, ptr as usize, &(ptr as u32 + 1).to_le_bytes())?;
                    Ok((ptr + 1,))
                })
            },
        )
        .unwrap();
    let module = Module::new(store.engine(), &bytes).unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    invoke(&mut store, instance);
    let state = store.data().single_instance().unwrap();
    let captured: Vec<_> = state
        .steps()
        .iter()
        .filter_map(|row| row.host_call_memory.as_ref())
        .collect();
    assert_eq!(captured.len(), 2);
    assert_eq!(captured[0].as_ref().unwrap()[&16], 17);
    assert_eq!(captured[1].as_ref().unwrap()[&32], 33);
    assert!(normalize(state.steps(), state.artifacts())
        .unwrap_err()
        .to_string()
        .contains("same-instance reentry"));
}

#[test]
fn import_memory_growth_is_rejected() {
    let (steps, artifacts) = capture(SIMPLE, bindings(&[word(0, 0)], 1, 2), 1, |mut caller, _ptr| {
        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
        memory.grow(&mut caller, 1)?;
        Ok(9)
    });
    assert!(normalize(&steps, &artifacts)
        .unwrap_err()
        .to_string()
        .contains("host memory growth"));
}

#[test]
fn consecutive_imports_keep_distinct_return_buffers() {
    let wat = SIMPLE.replace("call $write", "call $write call $write");
    let count = AtomicU32::new(0);
    let (steps, artifacts) = capture(&wat, bindings(&[word(0, 0)], 1, 2), 1, move |mut caller, ptr| {
        let value = 50 + count.fetch_add(1, Ordering::Relaxed);
        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
        memory.write(&mut caller, ptr as usize, &value.to_le_bytes())?;
        Ok(ptr)
    });
    let captured: Vec<_> = steps
        .iter()
        .filter_map(|row| row.host_call_memory.as_ref())
        .map(|bytes| bytes.as_ref().unwrap()[&16])
        .collect();
    assert_eq!(captured, [50, 51]);
    check(&steps, &artifacts);
}

#[test]
fn failed_import_does_not_poison_followup_invocations() {
    // Wasmtime reports ordinary errors as HostcallError and trap-valued errors
    // as Trap. Both unwind the currently executing activation.
    for trap in [false, true] {
        let bytes = wat::parse_str(SIMPLE).unwrap();
        let mut store = store();
        store
            .data_mut()
            .register_module(&bytes, bindings(&[word(0, 0)], 1, 2))
            .unwrap();
        let mut linker = Linker::new(store.engine());
        let calls = AtomicU32::new(0);
        linker
            .func_wrap(
                "host",
                "write",
                move |mut caller: Caller<'_, WasmtimeTraceRegistry>, ptr: i32| -> wasmtime::Result<i32> {
                    if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                        return Err(if trap {
                            wasmtime::Trap::UnreachableCodeReached.into()
                        } else {
                            wasmtime::Error::msg("host failed")
                        });
                    }
                    let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                    memory.write(&mut caller, ptr as usize, &99u32.to_le_bytes())?;
                    Ok(17)
                },
            )
            .unwrap();
        let module = Module::new(store.engine(), &bytes).unwrap();
        let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
        let run = instance.get_func(&mut store, "run").unwrap();
        assert!(block_on(run.call_async(&mut store, &[], &mut [Val::I32(0)])).is_err());
        let state = store.data().single_instance().unwrap();
        let failed_rows = state.steps().len();
        assert!(state
            .steps()
            .iter()
            .all(|row| row.host_call_memory.is_none()));
        assert!(normalize(state.steps(), state.artifacts())
            .unwrap_err()
            .to_string()
            .contains("continuation"));

        invoke(&mut store, instance);
        let state = store
            .data()
            .single_instance()
            .expect("a host error must not poison later capture");
        assert!(state.steps()[..failed_rows]
            .iter()
            .all(|row| row.host_call_memory.is_none()));
        let returned = state.steps()[failed_rows..]
            .iter()
            .find_map(|row| row.host_call_memory.as_ref())
            .unwrap();
        assert_eq!(returned.as_ref().unwrap()[&16], 99);
        check(&state.steps()[failed_rows..], state.artifacts());
        // Retrieving the capture is allowed, but its failed prefix remains invalid.
        assert!(normalize(state.steps(), state.artifacts()).is_err());
    }
}

#[test]
fn callback_error_only_retires_the_activations_that_unwind() {
    let bytes = wat::parse_str(
        r#"(module
        (import "host" "write" (func $write (param i32) (result i32)))
        (memory (export "memory") 1)
        (func (export "run") (result i32) i32.const 16 call $write)
        (func (export "callback") (result i32) i32.const 32 call $write))"#,
    )
    .unwrap();
    for catch in [false, true] {
        let mut store = store();
        store
            .data_mut()
            .register_module(&bytes, bindings(&[word(0, 0)], 1, 2))
            .unwrap();
        let mut linker = Linker::new(store.engine());
        linker
            .func_wrap_async(
                "host",
                "write",
                move |mut caller: Caller<'_, WasmtimeTraceRegistry>, (ptr,): (i32,)| {
                    Box::new(async move {
                        if ptr == 32 {
                            return Err(wasmtime::Error::msg("callback host failed"));
                        }
                        let callback = caller.get_export("callback").unwrap().into_func().unwrap();
                        let result = callback
                            .call_async(&mut caller, &[], &mut [Val::I32(0)])
                            .await;
                        if catch {
                            assert!(result.is_err());
                        } else {
                            result?;
                        }
                        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                        memory.write(&mut caller, ptr as usize, &99u32.to_le_bytes())?;
                        Ok((17i32,))
                    })
                },
            )
            .unwrap();
        let module = Module::new(store.engine(), &bytes).unwrap();
        let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
        let run = instance.get_func(&mut store, "run").unwrap();
        for _ in 0..2 {
            assert_eq!(
                block_on(run.call_async(&mut store, &[], &mut [Val::I32(0)])).is_ok(),
                catch
            );
            store
                .data()
                .check_errors()
                .expect("unwinding a callback must not poison the registry");
        }
        let state = store.data().single_instance().unwrap();
        let calls: Vec<_> = state
            .steps()
            .iter()
            .filter(|row| row.opcode_decoded == Some(neo_wasm::WasmOpcode::Call))
            .collect();
        assert_eq!(calls.len(), 4);
        for pair in calls.chunks_exact(2) {
            assert!(pair[1].host_call_memory.is_none(), "the callback never returned");
            if catch {
                assert_eq!(pair[0].host_call_memory.as_ref().unwrap().as_ref().unwrap()[&16], 99);
            } else {
                assert!(pair[0].host_call_memory.is_none(), "the outer call also unwound");
            }
        }
        assert!(normalize(state.steps(), state.artifacts()).is_err());
    }
}
