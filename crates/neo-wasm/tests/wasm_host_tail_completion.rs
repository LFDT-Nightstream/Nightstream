mod common;

use futures::executor::block_on;
use neo_wasm::host_event_bindings::{EventBlock, HostEventBindings, ImportTemplate, Limb, SlotBinding};
use neo_wasm::{WasmOpcode, WasmtimeTraceHandler, WasmtimeTraceRegistry};
use wasmtime::{Caller, Config, Engine, Linker, Module, Store, Val};

fn bindings() -> HostEventBindings {
    let mut bindings = HostEventBindings::import_free(2);
    let mut event = [SlotBinding::Const(0); 8];
    event[0] = SlotBinding::ResultElem { limb: Limb::Lo };
    event[1] = SlotBinding::ResultElem { limb: Limb::Hi };
    bindings.imports.insert(
        1,
        ImportTemplate {
            events: vec![EventBlock::advice(event)],
            input_count: 0,
        },
    );
    bindings.exports.get_mut(&2).unwrap().exit = vec![EventBlock {
        block: [SlotBinding::Const(17); 8],
        absorb: true,
    }];
    bindings
}

fn store(bytes: &[u8], bindings: HostEventBindings) -> Store<WasmtimeTraceRegistry> {
    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(bytes, bindings).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    store
}

#[test]
fn terminal_host_tail_commits_arguments_and_keeps_results_as_advice_across_turns() {
    for width in ["i32", "i64"] {
        let bytes = wat::parse_str(format!(
            r#"(module
            (import "host" "identity" (func $identity (param {width}) (result {width})))
            (func (export "run") (result {width})
                {width}.const -7
                return_call $identity))"#
        ))
        .unwrap();
        let mut bindings = bindings();
        let mut arguments = [SlotBinding::Const(0); 8];
        arguments[0] = SlotBinding::ArgElem { arg: 0, limb: Limb::Lo };
        arguments[1] = SlotBinding::ArgElem { arg: 0, limb: Limb::Hi };
        bindings.imports.get_mut(&1).unwrap().events.insert(
            0,
            EventBlock {
                block: arguments,
                absorb: true,
            },
        );
        let mut store = store(&bytes, bindings.clone());
        let module = Module::new(store.engine(), &bytes).unwrap();
        let mut linker = Linker::new(store.engine());
        if width == "i32" {
            linker.func_wrap("host", "identity", |x: i32| x).unwrap();
        } else {
            linker.func_wrap("host", "identity", |x: i64| x).unwrap();
        }
        let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
        let run = instance.get_func(&mut store, "run").unwrap();
        for _ in 0..2 {
            let mut results = [Val::I32(0)];
            block_on(run.call_async(&mut store, &[], &mut results)).unwrap();
            match results[0] {
                Val::I32(value) => assert_eq!(value, -7),
                Val::I64(value) => assert_eq!(value, -7),
                _ => panic!("expected integer result"),
            }
        }
        let state = store.data().single_instance().unwrap();
        let trace =
            neo_wasm::traces_from_wasmtime_steps_with_host_events(state.steps(), state.artifacts(), Default::default())
                .unwrap();
        common::sanity_check_trace_with_bindings(&trace, state.artifacts(), &bindings);
        common::ccs_check_trace(&trace);
        let mut expected_arguments = [0; 8];
        expected_arguments[0] = u64::from((-7i32) as u32);
        expected_arguments[1] = if width == "i64" { u64::from(u32::MAX) } else { 0 };
        let events: Vec<_> = neo_wasm::comm_chain::absorbed_event_blocks(&trace)
            .iter()
            .map(|event| event.words)
            .collect();
        assert_eq!(events, [expected_arguments, [17; 8], expected_arguments, [17; 8]]);
        assert_eq!(trace.last().unwrap().state_after.output.value_lo, 0);
        assert_eq!(trace.last().unwrap().state_after.output.value_hi, 0);
        assert_eq!(
            trace
                .iter()
                .filter(|row| row.opcode == WasmOpcode::Return)
                .count(),
            2
        );
        assert!(trace.iter().all(|row| row.call_stack_push.is_none()));

        // The synthetic return reads the result written by the import gather;
        // forging it must fail memory consistency, even without new constraints.
        let mut forged = trace.clone();
        let returned = forged
            .iter_mut()
            .find(|row| row.opcode == WasmOpcode::Return)
            .unwrap();
        returned.state_after.output.value_lo ^= 1;
        let witnesses: Vec<_> = forged
            .iter()
            .map(neo_wasm::witness_builder::build_witness_vector)
            .collect();
        let mut preload = neo_wasm::preload_from_program_artifacts(state.artifacts());
        neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings);
        assert!(
            neo_wasm::sanity_check_memory_rows(neo_wasm::build_wasm_relation_layout(), &witnesses, &preload).is_err()
        );

        let mut extra_operands = state.steps().to_vec();
        extra_operands
            .last_mut()
            .unwrap()
            .operand_stack_words
            .insert(0, 99);
        assert!(neo_wasm::traces_from_wasmtime_steps_with_host_events(
            &extra_operands,
            state.artifacts(),
            Default::default(),
        )
        .unwrap_err()
        .to_string()
        .contains("residual operands"));
    }
}

fn resource_component(post_return: bool, result: &str) -> Vec<u8> {
    let post_return = if post_return { "(post-return $cleanup)" } else { "" };
    wat::parse_str(format!(
        r#"(component
        (type $r (resource (rep i32)))
        (export $exported_r "r" (type $r))
        (core func $new (canon resource.new $r))
        (core module $m
            (import "host" "new" (func $new (param i32) (result i32)))
            (func (export "run") (result i32)
                i32.const 42
                return_call $new)
            (func (export "cleanup") (param i32)))
        (core instance $host (export "new" (func $new)))
        (core instance $i (instantiate $m (with "host" (instance $host))))
        (alias core export $i "run" (core func $run))
        (alias core export $i "cleanup" (core func $cleanup))
        (func (export "run") (result {result}) (canon lift (core func $run) {post_return})))"#
    ))
    .unwrap()
}

#[test]
fn component_scalar_advice_and_post_return_rejection() {
    // Even a lift that hides the raw core value is fine for unobserved advice.
    let run = neo_wasm::collect_wasmtime_component_run(&resource_component(false, "u8"), &bindings(), "run").unwrap();
    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default()).unwrap();
    common::sanity_check_trace_with_bindings(&trace, run.artifacts(), &bindings());
    common::ccs_check_trace(&trace);
    assert_eq!(trace.last().unwrap().state_after.output.value_lo, 0);

    let error =
        neo_wasm::collect_wasmtime_component_run(&resource_component(true, "u32"), &bindings(), "run").unwrap_err();
    assert!(error.to_string().contains("post-return"), "{error}");
}

#[test]
fn host_tail_capture_rejects_nested_returns_and_failed_host_calls() {
    for nested in [false, true] {
        let bytes = wat::parse_str(if nested {
            r#"(module
                (import "host" "identity" (func $identity (param i32) (result i32)))
                (func $tail (result i32) i32.const 7 return_call $identity)
                (func (export "run") (result i32) call $tail))"#
        } else {
            r#"(module
                (import "host" "identity" (func $identity (param i32) (result i32)))
                (func (export "run") (result i32) i32.const 7 return_call $identity))"#
        })
        .unwrap();
        let mut store = store(&bytes, Default::default());
        let module = Module::new(store.engine(), &bytes).unwrap();
        let mut linker = Linker::new(store.engine());
        linker
            .func_wrap(
                "host",
                "identity",
                move |_caller: Caller<'_, WasmtimeTraceRegistry>, x: i32| -> wasmtime::Result<i32> {
                    if nested {
                        Ok(x)
                    } else {
                        wasmtime::bail!("host failed")
                    }
                },
            )
            .unwrap();
        let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
        let run = instance.get_func(&mut store, "run").unwrap();
        let result = block_on(run.call_async(&mut store, &[], &mut [Val::I32(0)]));
        if nested {
            assert!(store
                .data()
                .check_errors()
                .unwrap_err()
                .to_string()
                .contains("directly to the host"));
        } else {
            assert!(result.is_err());
            assert!(store
                .data()
                .check_errors()
                .unwrap_err()
                .to_string()
                .contains("did not complete"));
        }
    }
}

#[test]
fn resultless_host_tail_commits_arguments_and_allows_later_execution() {
    let bytes = wat::parse_str(
        r#"(module
        (import "host" "sink" (func $sink (param i32)))
        (func (export "run") i32.const 42 return_call $sink))"#,
    )
    .unwrap();
    let mut bindings = HostEventBindings::import_free(2);
    let mut arguments = [SlotBinding::Const(0); 8];
    arguments[0] = SlotBinding::ArgElem { arg: 0, limb: Limb::Lo };
    bindings.imports.insert(
        1,
        ImportTemplate {
            events: vec![EventBlock {
                block: arguments,
                absorb: true,
            }],
            input_count: 0,
        },
    );
    let mut store = store(&bytes, bindings.clone());
    let module = Module::new(store.engine(), &bytes).unwrap();
    let mut linker = Linker::new(store.engine());
    linker
        .func_wrap("host", "sink", |x: i32| assert_eq!(x, 42))
        .unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let run = instance
        .get_typed_func::<(), ()>(&mut store, "run")
        .unwrap();
    block_on(run.call_async(&mut store, ())).unwrap();
    let state = store.data().single_instance().unwrap();
    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(state.steps(), state.artifacts(), Default::default())
            .unwrap();
    common::sanity_check_trace_with_bindings(&trace, state.artifacts(), &bindings);
    common::ccs_check_trace(&trace);
    assert!(!trace.last().unwrap().state_after.output.enabled);
    let events = neo_wasm::comm_chain::absorbed_event_blocks(&trace);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].words, [42, 0, 0, 0, 0, 0, 0, 0]);

    let other = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let other_run = other.get_typed_func::<(), ()>(&mut store, "run").unwrap();
    // A completed tail does not block execution in another instance.
    block_on(other_run.call_async(&mut store, ())).unwrap();
    let states: Vec<_> = store.data().instances().unwrap().collect();
    assert_eq!(states.len(), 2);
    for (_, state) in states {
        neo_wasm::traces_from_wasmtime_steps_with_host_events(state.steps(), state.artifacts(), Default::default())
            .unwrap();
    }
}

#[test]
fn host_tail_lowering_accepts_polymorphic_stacks_in_unreachable_code() {
    for body in [
        "i32.const 1 return return_call $identity",
        "block (result i32) i32.const 1 br 0 return_call $identity end",
        "unreachable return_call $identity",
    ] {
        let bytes = wat::parse_str(format!(
            r#"(module
            (import "host" "identity" (func $identity (param i32) (result i32)))
            (func (export "run") (result i32) {body}))"#,
        ))
        .unwrap();
        // No argument is needed for the host call: execution cannot reach it.
        wasmparser::Validator::new().validate_all(&bytes).unwrap();
        neo_wasm::extract_wasm_program_artifacts(&bytes).unwrap();
    }
}

#[test]
fn host_tail_lowering_rejects_residual_operands_from_the_module_without_a_trace() {
    let bytes = wat::parse_str(
        r#"(module
        (import "host" "identity" (func $identity (param i32) (result i32)))
        (func (export "run") (result i32)
            i32.const 99
            i32.const 7
            return_call $identity))"#,
    )
    .unwrap();
    // Valid Wasm, but Call/Return in the current relation cannot discard 99.
    wasmparser::Validator::new().validate_all(&bytes).unwrap();
    let error = neo_wasm::extract_wasm_program_artifacts(&bytes).unwrap_err();
    assert!(error.to_string().contains("residual operands"), "{error}");
}

#[test]
fn owned_resource_tail_uses_unobserved_advice_and_preserves_the_real_resource() {
    let bytes = resource_component(false, "(own $exported_r)");
    let bindings = bindings();
    let run = neo_wasm::collect_wasmtime_component_run(&bytes, &bindings, "run").unwrap();
    assert_eq!(run.results, ["own<resource>"]);
    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default()).unwrap();
    common::sanity_check_trace_with_bindings(&trace, run.artifacts(), &bindings);
    common::ccs_check_trace(&trace);

    // The caller-owned store path must leave a usable own<T> in the result buffer.
    let core_bytes = wasmparser::Parser::new(0)
        .parse_all(&bytes)
        .find_map(|payload| match payload.unwrap() {
            wasmparser::Payload::ModuleSection { unchecked_range, .. } => Some(&bytes[unchecked_range]),
            _ => None,
        })
        .unwrap();
    let mut store = store(core_bytes, bindings.clone());
    let component = wasmtime::component::Component::new(store.engine(), &bytes).unwrap();
    let linker = wasmtime::component::Linker::new(store.engine());
    let first = block_on(linker.instantiate_async(&mut store, &component)).unwrap();
    let second = block_on(linker.instantiate_async(&mut store, &component)).unwrap();
    for instance in [first, second, first] {
        let run = instance.get_func(&mut store, "run").unwrap();
        let mut results = [wasmtime::component::Val::Bool(false)];
        block_on(run.call_async(&mut store, &[], &mut results)).unwrap();
        let wasmtime::component::Val::Resource(resource) = &results[0] else {
            panic!("expected resource")
        };
        assert!(resource.owned());
        assert_eq!(resource.ty(), instance.get_resource(&mut store, "r").unwrap());
        block_on(resource.resource_drop_async(&mut store)).unwrap();
    }
    let states: Vec<_> = store.data().instances().unwrap().collect();
    assert_eq!(states.len(), 2);
    for (_, state) in states {
        let trace =
            neo_wasm::traces_from_wasmtime_steps_with_host_events(state.steps(), state.artifacts(), Default::default())
                .unwrap();
        common::sanity_check_trace_with_bindings(&trace, state.artifacts(), &bindings);
        common::ccs_check_trace(&trace);
        assert_eq!(
            trace.last().unwrap().state_after.output.value_lo,
            0,
            "canonical advice, not a captured handle"
        );
    }
}

#[test]
fn terminal_advice_rejects_committed_results_and_output_observers() {
    let bytes = resource_component(false, "u32");
    let bindings = bindings();
    let run = neo_wasm::collect_wasmtime_component_run(&bytes, &bindings, "run").unwrap();
    for case in 0..4 {
        let mut artifacts = run.artifacts().clone();
        match case {
            0 => {
                artifacts
                    .host_event_bindings
                    .imports
                    .get_mut(&1)
                    .unwrap()
                    .events[0]
                    .absorb = true
            }
            1 => {
                artifacts
                    .host_event_bindings
                    .exports
                    .get_mut(&2)
                    .unwrap()
                    .exit[0]
                    .block[0] = SlotBinding::OutputElem { limb: Limb::Lo }
            }
            2 => {
                artifacts
                    .host_event_bindings
                    .exports
                    .get_mut(&2)
                    .unwrap()
                    .exit[0]
                    .block[0] = SlotBinding::MemoryRead32 {
                    base: neo_wasm::host_event_bindings::MemoryBase::Output,
                    byte_offset: 0,
                }
            }
            _ => {
                artifacts
                    .host_event_bindings
                    .imports
                    .get_mut(&1)
                    .unwrap()
                    .events = EventBlock {
                    block: [SlotBinding::Const(0); 8],
                    absorb: true,
                }
                .with_opaque(
                    0,
                    [0; 4],
                    vec![
                        SlotBinding::ResultElem { limb: Limb::Lo },
                        SlotBinding::ResultElem { limb: Limb::Hi },
                    ],
                )
                .unwrap();
            }
        }
        let error = neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, &artifacts, Default::default())
            .unwrap_err();
        assert!(
            error.to_string().contains(if case == 0 || case == 3 {
                "results cannot appear in absorbing events"
            } else {
                "observed export output"
            }),
            "{error}"
        );
    }
}
