mod common;

use futures::executor::block_on;
use neo_wasm::host_event_bindings::{
    EventBlock, ExportTemplate, HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding,
};
use neo_wasm::{WasmtimeTraceHandler, WasmtimeTraceRegistry};
use wasmtime::{Caller, Config, Engine, Linker, Module, Store, Val};

fn event(slots: &[SlotBinding]) -> EventBlock {
    let mut block = [SlotBinding::Const(0); 8];
    block[..slots.len()].copy_from_slice(slots);
    EventBlock { block, absorb: true }
}

fn bindings() -> HostEventBindings {
    use SlotBinding::*;
    let mut bindings = HostEventBindings::import_free(2);
    bindings.exports.get_mut(&2).unwrap().exit = vec![event(&[Const(20), OutputElem { limb: Limb::Lo }])];
    bindings.exports.insert(
        3,
        ExportTemplate {
            entry: vec![event(&[
                Const(30),
                InputLocal {
                    input: 0,
                    local: 0,
                    limb: Limb::Lo,
                },
            ])],
            exit: vec![event(&[Const(31), OutputElem { limb: Limb::Lo }])],
            entry_input_count: 1,
        },
    );
    bindings.imports.insert(
        1,
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

#[test]
fn split_templates_separate_phases_and_preserve_slot_order() {
    use SlotBinding::*;
    let valid = bindings().imports.remove(&1).unwrap();
    valid.validate(1, 1).unwrap();
    let resultless = ImportTemplate {
        events: vec![],
        return_events: Some(vec![]),
        input_count: 0,
    };
    resultless.validate(0, 0).unwrap();
    assert!(resultless.is_split());
    assert!(!ImportTemplate::default().is_split());
    for invalid in [
        ImportTemplate {
            events: vec![event(&[ResultElem { limb: Limb::Lo }])],
            ..valid.clone()
        },
        ImportTemplate {
            events: vec![event(&[MemoryWrite8 {
                input: 0,
                base: MemoryBase::Arg(0),
                byte_offset: 0,
            }])],
            ..valid.clone()
        },
        ImportTemplate {
            return_events: Some(vec![event(&[
                ResultElem { limb: Limb::Lo },
                ResultElem { limb: Limb::Hi },
                MemoryWrite32 {
                    input: 0,
                    base: MemoryBase::Arg(0),
                    byte_offset: 0,
                },
            ])]),
            ..valid
        },
    ] {
        assert!(invalid.validate(1, 1).is_err());
    }
}

#[derive(Default)]
struct EffectRuntime {
    trace: WasmtimeTraceRegistry,
    utxo: Option<wasmtime::Func>,
    handler: Option<wasmtime::Func>,
}

impl neo_wasm::WasmTraceSink for EffectRuntime {
    fn wasm_trace_registry(&self) -> &WasmtimeTraceRegistry {
        &self.trace
    }
    fn wasm_trace_registry_mut(&mut self) -> &mut WasmtimeTraceRegistry {
        &mut self.trace
    }
}

fn scalar_export(entry: u64, exit: u64) -> ExportTemplate {
    ExportTemplate {
        entry: vec![event(&[
            SlotBinding::Const(entry),
            SlotBinding::InputLocal {
                input: 0,
                local: 0,
                limb: Limb::Lo,
            },
        ])],
        exit: vec![event(&[
            SlotBinding::Const(exit),
            SlotBinding::OutputElem { limb: Limb::Lo },
        ])],
        entry_input_count: 1,
    }
}

fn scalar_import(call: u64, ret: u64) -> ImportTemplate {
    ImportTemplate {
        events: vec![event(&[
            SlotBinding::Const(call),
            SlotBinding::ArgElem { arg: 0, limb: Limb::Lo },
        ])],
        return_events: Some(vec![event(&[
            SlotBinding::Const(ret),
            SlotBinding::ResultElem { limb: Limb::Lo },
            SlotBinding::ResultElem { limb: Limb::Hi },
        ])]),
        input_count: 0,
    }
}

#[test]
fn coordinator_utxo_effect_handler_preserves_each_instances_event_order() {
    let a = wat::parse_str(
        r#"(module
        (import "host" "invoke" (func $invoke (param i32) (result i32)))
        (func (export "coord") (param $x i32) (result i32)
            local.get $x i32.const 1 i32.add call $invoke local.get $x i32.add)
        (func (export "handler") (param i32) (result i32) local.get 0 i32.const 3 i32.add))"#,
    )
    .unwrap();
    let b = wat::parse_str(
        r#"(module
        (import "host" "effect" (func $effect (param i32) (result i32)))
        (func (export "utxo") (param $x i32) (result i32)
            local.get $x i32.const 2 i32.mul call $effect local.get $x i32.add))"#,
    )
    .unwrap();
    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).unwrap();
    let mut runtime = EffectRuntime::default();
    let mut ab = HostEventBindings::default();
    ab.imports.insert(1, scalar_import(110, 111));
    ab.exports.insert(2, scalar_export(100, 101));
    ab.exports.insert(3, scalar_export(120, 121));
    runtime.trace.register_module(&a, ab).unwrap();
    let mut bb = HostEventBindings::default();
    bb.imports.insert(1, scalar_import(210, 211));
    bb.exports.insert(2, scalar_export(200, 201));
    runtime.trace.register_module(&b, bb).unwrap();
    let mut store = Store::new(&engine, runtime);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let mut linker = Linker::new(&engine);
    for (name, handler) in [("invoke", false), ("effect", true)] {
        linker
            .func_wrap_async(
                "host",
                name,
                move |mut caller: Caller<'_, EffectRuntime>, (x,): (i32,)| {
                    Box::new(async move {
                        let function = if handler {
                            caller.data().handler
                        } else {
                            caller.data().utxo
                        }
                        .unwrap();
                        let mut result = [Val::I32(0)];
                        function
                            .call_async(&mut caller, &[Val::I32(x)], &mut result)
                            .await?;
                        Ok((result[0].i32().unwrap(),))
                    })
                },
            )
            .unwrap();
    }
    let ai = block_on(linker.instantiate_async(&mut store, &Module::new(&engine, &a).unwrap())).unwrap();
    let bi = block_on(linker.instantiate_async(&mut store, &Module::new(&engine, &b).unwrap())).unwrap();
    store.data_mut().handler = Some(ai.get_func(&mut store, "handler").unwrap());
    store.data_mut().utxo = Some(bi.get_func(&mut store, "utxo").unwrap());
    let coord = ai.get_func(&mut store, "coord").unwrap();
    let mut result = [Val::I32(0)];
    block_on(coord.call_async(&mut store, &[Val::I32(10)], &mut result)).unwrap();
    assert_eq!(result[0].i32(), Some(46));
    for (instance, expected) in [
        (
            ai,
            vec![(100, 10), (110, 11), (120, 22), (121, 25), (111, 36), (101, 46)],
        ),
        (bi, vec![(200, 11), (210, 22), (211, 25), (201, 36)]),
    ] {
        let captured = store
            .data()
            .trace
            .instance(instance.debug_index_in_store())
            .unwrap();
        let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
            captured.steps(),
            captured.artifacts(),
            Default::default(),
        )
        .unwrap();
        common::sanity_check_trace_with_bindings(
            &trace,
            captured.artifacts(),
            &captured.artifacts().host_event_bindings,
        );
        common::ccs_check_trace(&trace);
        let events: Vec<_> = trace
            .iter()
            .filter(|row| row.row_kind.is_host_event_perm() && row.state_before.event_absorb.perm_pending)
            .map(|row| {
                (
                    row.state_before.event_absorb.evbuf[0],
                    row.state_before.event_absorb.evbuf[1],
                )
            })
            .collect();
        assert_eq!(events, expected);
    }
}

#[test]
fn resultless_import_resumes_without_gather_rows() {
    use p3_field::PrimeCharacteristicRing;
    let bytes = wat::parse_str(
        r#"(module
        (import "host" "yield" (func $yield))
        (global $count (mut i32) (i32.const 0))
        (func (export "run") (result i32)
            i32.const 5 call $yield global.get $count i32.add)
        (func (export "callback")
            global.get $count i32.const 1 i32.add global.set $count))"#,
    )
    .unwrap();
    let mut bindings = HostEventBindings::import_free(2);
    bindings.imports.insert(
        1,
        ImportTemplate {
            events: vec![],
            return_events: Some(vec![]),
            input_count: 0,
        },
    );
    bindings.exports.get_mut(&2).unwrap().exit = vec![event(&[SlotBinding::Const(20)])];
    bindings.exports.insert(
        3,
        ExportTemplate {
            entry: vec![],
            exit: vec![event(&[SlotBinding::Const(30)])],
            entry_input_count: 0,
        },
    );
    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(&bytes, bindings).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap_async(
            "host",
            "yield",
            |mut caller: Caller<'_, WasmtimeTraceRegistry>, (): ()| {
                Box::new(async move {
                    let callback = caller.get_export("callback").unwrap().into_func().unwrap();
                    callback.call_async(&mut caller, &[], &mut []).await?;
                    Ok(())
                })
            },
        )
        .unwrap();
    let module = Module::new(&engine, &bytes).unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let run = instance.get_func(&mut store, "run").unwrap();
    block_on(run.call_async(&mut store, &[], &mut [Val::I32(0)])).unwrap();
    let captured = store.data().single_instance().unwrap();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
        captured.steps(),
        captured.artifacts(),
        Default::default(),
    )
    .unwrap();
    let witnesses = common::sanity_check_trace_with_bindings(
        &trace,
        captured.artifacts(),
        &captured.artifacts().host_event_bindings,
    );
    common::ccs_check_trace(&trace);
    let resume = trace
        .iter()
        .position(|row| row.row_kind.is_host_resume())
        .unwrap();
    assert_eq!(trace[resume].host_event_return_schedule_count, Some(1));
    assert_eq!(trace[resume].state_after.host_events.events_remaining, 0);
    assert!(
        trace[resume + 1].row_kind.is_program(),
        "empty return phase has no gather rows"
    );
    let mut wrong_count = witnesses[resume].clone();
    wrong_count[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = neo_math::F::ONE;
    common::assert_rejected(&wrong_count, "return schedule must remove the ROM count bias");
    assert_eq!(trace.last().unwrap().state_after.output.value_lo, 6);
}

#[test]
fn split_import_preserves_caller_arguments_across_callbacks() {
    for (callbacks, indirect, recursive) in [(0, false, false), (2, false, false), (1, true, false), (1, false, true)] {
        // With reentry, the outer buffer becomes valid only after guest growth.
        let pointer = if callbacks == 0 { 16 } else { 65536 };
        let call = if indirect {
            "i32.const 0 call_indirect (type $host)"
        } else {
            "call $effect"
        };
        let bytes = wat::parse_str(format!(
            r#"(module
            (type $host (func (param i32) (result i32)))
            (import "host" "effect" (func $effect (param i32) (result i32)))
            (memory (export "memory") 1)
            (table 1 funcref) (elem (i32.const 0) $effect)
            (func (export "run") (result i32) call $worker)
            (func (export "callback") (param $x i32) (result i32) (local $saved i32)
                i32.const 1 memory.grow drop
                local.get $x i32.const 100 i32.add local.set $saved
                i32.const 24 local.get $saved i32.store
                local.get $x if i32.const 32 call $effect drop end
                local.get $saved)
            (func $worker (result i32) return_call $body)
            (func $body (result i32) (local $saved i32)
                i32.const 7 local.set $saved
                i32.const 7 i32.const {pointer} {call}
                local.get $saved i32.const 7 i32.sub i32.add i32.add))"#,
        ))
        .unwrap();
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
                "effect",
                move |mut caller: Caller<'_, WasmtimeTraceRegistry>, (ptr,): (i32,)| {
                    Box::new(async move {
                        for _ in 0..callbacks {
                            let callback = caller.get_export("callback").unwrap().into_func().unwrap();
                            callback
                                .call_async(
                                    &mut caller,
                                    &[Val::I32(if recursive && ptr == pointer { 1 } else { 0 })],
                                    &mut [Val::I32(0)],
                                )
                                .await?;
                        }
                        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                        memory.write(&mut caller, ptr as usize, &77u32.to_le_bytes())?;
                        Ok((9i32,))
                    })
                },
            )
            .unwrap();
        let module = Module::new(&engine, &bytes).unwrap();
        let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
        for _ in 0..2 {
            let run = instance.get_func(&mut store, "run").unwrap();
            let mut result = [Val::I32(0)];
            block_on(run.call_async(&mut store, &[], &mut result)).unwrap();
            assert_eq!(result[0].i32(), Some(16));
        }
        let captured = store.data().single_instance().unwrap();
        let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(
            captured.steps(),
            captured.artifacts(),
            Default::default(),
        )
        .unwrap();
        let witnesses = common::sanity_check_trace_with_bindings(
            &trace,
            captured.artifacts(),
            &captured.artifacts().host_event_bindings,
        );
        common::ccs_check_trace(&trace);
        use neo_application::MemoryPortActivation;
        use p3_field::PrimeCharacteristicRing;
        let memories = neo_wasm::build_wasm_relation_layout()
            .auxiliary
            .memory
            .entries();
        let ports: Vec<_> = memories.iter().flat_map(|memory| &memory.ports).collect();
        let slots = neo_wasm::plan_wasm_memory_slots(memories);
        let peak = witnesses
            .iter()
            .map(|w| {
                let active = |index: usize| match ports[index].activation {
                    MemoryPortActivation::Always => true,
                    MemoryPortActivation::When(col) => w[col] == neo_math::F::ONE,
                    MemoryPortActivation::Unless(col) => w[col] == neo_math::F::ZERO,
                };
                for slot in &slots.slots {
                    assert!(
                        slot.iter().filter(|&&index| active(index)).count() <= 1,
                        "physical slot has multiple live candidates"
                    );
                }
                (0..ports.len()).filter(|&index| active(index)).count()
            })
            .max()
            .unwrap();
        println!("callbacks={callbacks} indirect={indirect} recursive={recursive}: peak active ports={peak}");
        let suspended = trace
            .iter()
            .find(|row| row.state_after.host_events.awaiting_host)
            .unwrap()
            .state_after;
        let mut omitted = suspended;
        omitted.host_events.host_frame_depth = 0;
        omitted.host_events.awaiting_host = false;
        assert_ne!(
            neo_wasm::semantic_state_digest(suspended),
            neo_wasm::semantic_state_digest(omitted)
        );
        let resume = trace
            .iter()
            .position(|row| row.row_kind.is_host_resume())
            .unwrap();
        let layout = neo_wasm::build_wasm_relation_layout();
        let mut preload = neo_wasm::preload_from_program_artifacts(captured.artifacts());
        neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &captured.artifacts().host_event_bindings);
        if callbacks != 0 {
            let nested_exit = trace[resume].state_before;
            assert!(nested_exit.activation_finished);
            assert_eq!(nested_exit.host_events.events_remaining, 0);
            assert!(!nested_exit.is_terminal(), "nested exit leaves its caller suspended");
            // Each field independently rules out completion, even for a claimed
            // state whose other suspended-activation field was cleared.
            let mut claimed = nested_exit;
            claimed.host_events.host_frame_depth = 0;
            assert!(!claimed.is_terminal());
            claimed = nested_exit;
            claimed.host_events.awaiting_host = false;
            assert!(!claimed.is_terminal());

            let entry = trace
                .iter()
                .position(|row| row.row_kind.is_turn_boundary() && row.state_before.host_events.awaiting_host)
                .unwrap();
            // The entered frame starts at the saved sp; shifting it together
            // with every frame register keeps the row satisfied.
            let mut forged = witnesses.clone();
            for column in [
                neo_wasm::layout::COL_HOST_CONTEXT[7],
                neo_wasm::layout::COL_SP_BEFORE,
                neo_wasm::layout::COL_SP_AFTER,
                neo_wasm::layout::COL_LOCALS_FBP_AFTER,
                neo_wasm::layout::COL_STACK_FRAME_BASE_AFTER,
            ] {
                forged[entry][column] -= neo_math::F::ONE;
            }
            common::assert_satisfied(&forged[entry], "allocation needs context RAM authority");
            assert!(neo_wasm::sanity_check_memory_rows(layout, &forged, &preload).is_err());

            // Hand-built captures must not bypass the registry's atomic guard.
            let mut atomic_artifacts = captured.artifacts().clone();
            let import = atomic_artifacts
                .host_event_bindings
                .imports
                .get_mut(&1)
                .unwrap();
            import.events.extend(import.return_events.take().unwrap());
            let err = neo_wasm::traces_from_wasmtime_steps_with_host_events(
                captured.steps(),
                &atomic_artifacts,
                Default::default(),
            )
            .unwrap_err();
            assert!(err
                .to_string()
                .contains("same-instance reentry under an atomic import"));
        }
        for (field, column) in [
            (0, neo_wasm::layout::COL_PC_AFTER),
            (3, neo_wasm::layout::COL_TURN_EXPORT_FREF_AFTER),
            (4, neo_wasm::layout::COL_HOST_CALLEE_FREF_AFTER),
        ] {
            let mut forged = witnesses.clone();
            forged[resume][neo_wasm::layout::COL_HOST_CONTEXT[field]] += neo_math::F::ONE;
            forged[resume][column] += neo_math::F::ONE;
            common::assert_satisfied(&forged[resume], "restoration needs context RAM authority");
            assert!(neo_wasm::sanity_check_memory_rows(layout, &forged, &preload).is_err());
        }
        let mut skipped = witnesses[resume].clone();
        skipped[neo_wasm::layout::COL_HOST_EVENTS_REMAINING_AFTER] = neo_math::F::ZERO;
        common::assert_rejected(&skipped, "cannot skip import return phase");
        let call = trace
            .iter()
            .position(|row| row.host_event_return_schedule_count.is_some_and(|n| n > 0) && row.row_kind.is_program())
            .unwrap();
        // Context saving is a mandatory separate row, even with no reentry.
        use neo_wasm::layout::{COL_HOST_SUSPEND_PENDING_AFTER, COL_HOST_SUSPEND_PENDING_BEFORE};
        let suspend = call + 1;
        assert!(trace[suspend].row_kind.is_host_suspend());
        assert!(trace[call].host_context.is_none());
        assert_eq!(
            trace[call].state_before.call_stack_depth,
            trace[call].state_after.call_stack_depth
        );
        assert_eq!(
            trace[suspend].state_after.call_stack_depth,
            trace[suspend].state_before.call_stack_depth + 1
        );
        let mut forged = witnesses[call].clone();
        forged[COL_HOST_SUSPEND_PENDING_AFTER] = neo_math::F::ZERO;
        common::assert_rejected(&forged, "split call must schedule suspension");
        let mut forged = witnesses[suspend].clone();
        forged[COL_HOST_SUSPEND_PENDING_BEFORE] = neo_math::F::ZERO;
        common::assert_rejected(&forged, "cannot insert an unsolicited suspend row");
        let mut forged = witnesses[suspend + 1].clone();
        forged[COL_HOST_SUSPEND_PENDING_BEFORE] = neo_math::F::ONE;
        common::assert_rejected(&forged, "cannot execute the successor instead of suspension");
        let mut skipped = witnesses.clone();
        skipped.remove(suspend);
        assert!(neo_application::check_continuity_rows(&layout.auxiliary.continuity, &skipped).is_err());
        let pending = trace[call].state_after;
        let mut cleared = pending;
        cleared.host_events.suspend_pending = false;
        assert_ne!(
            neo_wasm::semantic_state_digest(pending),
            neo_wasm::semantic_state_digest(cleared)
        );
        // A u32 ROM word does not by itself bound each half: borrowing from
        // the return half can leave the packed value unchanged.
        use neo_wasm::layout::{COL_HOST_IMPORT_SCHEDULE_COUNTS, COL_HOST_INITIAL_COUNT};
        let mut forged = witnesses[suspend].clone();
        forged[COL_HOST_INITIAL_COUNT] = neo_math::F::from_u64(1 << 16);
        forged[COL_HOST_IMPORT_SCHEDULE_COUNTS] = neo_math::F::from_u64(1 << 16);
        common::assert_rejected(&forged, "call half must fit 16 bits even when the packed word fits u32");
        for row in [call, resume] {
            let mut forged = witnesses[row].clone();
            forged[COL_HOST_IMPORT_SCHEDULE_COUNTS] += neo_math::F::ONE;
            common::assert_rejected(&forged, "packed ROM word must equal its decoded counts");
            let mut forged = witnesses.clone();
            forged[row][COL_HOST_IMPORT_SCHEDULE_COUNTS] += neo_math::F::ONE;
            assert!(neo_wasm::sanity_check_memory_rows(layout, &forged, &preload).is_err());
        }
        let mut atomic = witnesses.clone();
        atomic[call][neo_wasm::layout::COL_HOST_RETURN_COUNT] = neo_math::F::ZERO;
        atomic[call][neo_wasm::layout::COL_HOST_IMPORT_SCHEDULE_COUNTS] =
            atomic[call][neo_wasm::layout::COL_HOST_INITIAL_COUNT];
        assert!(neo_wasm::sanity_check_memory_rows(layout, &atomic, &preload).is_err());
        let cutoff = captured
            .steps()
            .iter()
            .position(|row| row.opcode_decoded == Some(neo_wasm::WasmOpcode::Call) && !row.target_function_is_guest);
        if let Some(cutoff) = cutoff {
            assert!(neo_wasm::traces_from_wasmtime_steps_with_host_events(
                &captured.steps()[..=cutoff],
                captured.artifacts(),
                Default::default()
            )
            .is_err());
        }

        let final_state = trace.last().unwrap().state_after;
        assert!(final_state.is_terminal());
        let mut pending_completion = final_state;
        pending_completion.host_events.suspend_pending = true;
        assert!(!pending_completion.is_terminal(), "completion cannot owe suspension");
        assert_eq!(final_state.output.value_lo, 16);
        assert_eq!(final_state.call_stack_depth, 0);
        assert_eq!(final_state.host_events.host_frame_depth, 0);
        assert!(!final_state.host_events.awaiting_host);
    }
}
