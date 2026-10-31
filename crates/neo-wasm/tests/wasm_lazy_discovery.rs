use futures::executor::block_on;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::{WasmOpcode, WasmtimeTraceHandler, WasmtimeTraceRegistry};
use wasmtime::{Config, Engine, Instance, Linker, Module, Store, Val};

fn store(registry: WasmtimeTraceRegistry) -> Store<WasmtimeTraceRegistry> {
    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    config.wasm_component_model(true);
    let engine = Engine::new(&config).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    store
}

fn instantiate(store: &mut Store<WasmtimeTraceRegistry>, module: &Module) -> Instance {
    block_on(Linker::new(store.engine()).instantiate_async(store, module)).unwrap()
}

fn call(store: &mut Store<WasmtimeTraceRegistry>, instance: Instance, name: &str) -> i32 {
    let function = instance.get_func(&mut *store, name).unwrap();
    let mut results = [Val::I32(0)];
    block_on(function.call_async(store, &[], &mut results)).unwrap();
    results[0].i32().unwrap()
}

#[test]
fn start_discovers_function_references_before_instantiation_returns() {
    let bytes = wat::parse_str(
        r#"(module
        (global $value (mut i32) (i32.const 0))
        (func $target)
        (elem declare func $target)
        (func $start
            ref.func $target drop
            i32.const 42 global.set $value)
        (start $start)
        (func (export "run") (result i32) global.get $value))"#,
    )
    .unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(&bytes, HostEventBindings::default())
        .unwrap();
    let mut store = store(registry);
    let module = Module::new(store.engine(), &bytes).unwrap();
    let instance = instantiate(&mut store, &module);
    let state = store
        .data()
        .instance(instance.debug_index_in_store())
        .unwrap();
    assert_eq!(state.steps()[0].opcode_decoded, Some(WasmOpcode::RefFunc));
    assert_eq!(state.steps()[1].operand_stack_words, [1]);
    let start_rows = state.steps().len();
    assert!(start_rows >= 5);
    assert_eq!(call(&mut store, instance, "run"), 42);
    let state = store.data().single_instance().unwrap();
    assert_eq!(state.steps()[start_rows].step, start_rows as u64);
    assert_eq!(state.steps()[start_rows].opcode_decoded, Some(WasmOpcode::GlobalGet));
    assert_eq!(state.artifacts().tables.globals_init, [(0, 0, 0)]);
    // Discovery retains the start prefix; it does not adopt live globals as initialization.
}

#[test]
fn same_module_shares_artifacts_but_keeps_instance_state_separate() {
    let bytes = wat::parse_str(
        r#"(module
        (global $value (mut i32) (i32.const 0))
        (func (export "run") (result i32)
            global.get $value i32.const 1 i32.add global.set $value global.get $value))"#,
    )
    .unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(&bytes, HostEventBindings::default())
        .unwrap();
    let mut store = store(registry);
    let module = Module::new(store.engine(), &bytes).unwrap();
    let a = instantiate(&mut store, &module);
    let b = instantiate(&mut store, &module);
    assert_eq!(call(&mut store, a, "run"), 1);
    assert_eq!(call(&mut store, b, "run"), 1);
    assert_eq!(call(&mut store, a, "run"), 2);
    let a = store.data().instance(a.debug_index_in_store()).unwrap();
    let b = store.data().instance(b.debug_index_in_store()).unwrap();
    assert!(std::ptr::eq(a.artifacts(), b.artifacts()));
    assert_eq!(a.steps().len(), b.steps().len() * 2);
    assert_eq!((a.steps()[0].step, b.steps()[0].step), (0, 0));
    assert_eq!(store.data().instances().unwrap().count(), 2);
    assert!(store
        .data()
        .single_instance()
        .unwrap_err()
        .to_string()
        .contains("found 2"));
}

#[test]
fn unconfigured_bytes_are_ignored_and_cannot_be_registered_after_execution() {
    let selected = wat::parse_str(r#"(module (func (export "run") (result i32) i32.const 1))"#).unwrap();
    let other = wat::parse_str(r#"(module (func (export "run") (result i32) i32.const 2))"#).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(&selected, HostEventBindings::default())
        .unwrap();
    let mut store = store(registry);
    let module = Module::new(store.engine(), &other).unwrap();
    let instance = instantiate(&mut store, &module);
    assert_eq!(call(&mut store, instance, "run"), 2);
    assert!(store
        .data()
        .single_instance()
        .unwrap_err()
        .to_string()
        .contains("found 0"));
    assert!(store
        .data_mut()
        .register_module(&other, HostEventBindings::default())
        .unwrap_err()
        .to_string()
        .contains("after untraced execution"));
    let module = Module::new(store.engine(), &selected).unwrap();
    let instance = instantiate(&mut store, &module);
    assert_eq!(call(&mut store, instance, "run"), 1);
    assert_eq!(
        store.data().single_instance().unwrap().steps()[0].immediate_i32,
        Some(1)
    );
}

#[test]
fn configured_bindings_cannot_be_replaced() {
    let bytes = wat::parse_str(r#"(module (func (export "run") (result i32) i32.const 7))"#).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(&bytes, HostEventBindings::default())
        .unwrap();
    registry
        .register_module(&bytes, HostEventBindings::default())
        .unwrap();
    let mut store = store(registry);
    let module = Module::new(store.engine(), &bytes).unwrap();
    let instance = instantiate(&mut store, &module);
    assert_eq!(call(&mut store, instance, "run"), 7);
    assert!(store
        .data_mut()
        .register_module(&bytes, HostEventBindings::import_free(1))
        .unwrap_err()
        .to_string()
        .contains("different bindings"));
    assert_eq!(
        store
            .data()
            .single_instance()
            .unwrap()
            .artifacts()
            .host_event_bindings,
        HostEventBindings::default()
    );
}

#[test]
fn discovery_and_frame_errors_prevent_partial_trace_retrieval() {
    let bytes = wat::parse_str(r#"(module (func (export "run") (result i32) i32.const 7))"#).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(&bytes, HostEventBindings::import_free(2))
        .unwrap();
    let mut first = store(registry);
    let module = Module::new(first.engine(), &bytes).unwrap();
    let instance = instantiate(&mut first, &module);
    assert_eq!(call(&mut first, instance, "run"), 7);
    let error = first.data().check_errors().unwrap_err().to_string();
    assert!(
        error.contains("binding fref 2 has no function-call metadata"),
        "{error}"
    );
    assert!(first.data().instances().is_err());
    assert!(first
        .data()
        .instance(instance.debug_index_in_store())
        .is_err());
    assert!(first.data().single_instance().is_err());

    // i32 instructions are supported, but the frame's f32 local cannot be
    // represented in our witness lanes. The failure must not disappear.
    let bytes = wat::parse_str(
        r#"(module
        (func $bad (result i32) (local f32) i32.const 9)
        (func (export "run") (result i32) i32.const 1 drop call $bad))"#,
    )
    .unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(&bytes, HostEventBindings::default())
        .unwrap();
    let mut second = store(registry);
    let module = Module::new(second.engine(), &bytes).unwrap();
    let instance = instantiate(&mut second, &module);
    assert_eq!(call(&mut second, instance, "run"), 9);
    assert!(second
        .data()
        .check_errors()
        .unwrap_err()
        .to_string()
        .contains("unsupported Wasmtime operand stack value"));
    assert!(second.data().single_instance().is_err());
}

#[test]
fn component_selection_skips_unsupported_adapter_and_discovers_guest_start() {
    let bytes = wat::parse_str(
        r#"(component
        (core module $adapter
            (global f32 (f32.const 1))
            (func $start f32.const 1 drop)
            (start $start))
        (core module $guest
            (global $value (mut i32) (i32.const 0))
            (func $start i32.const 23 global.set $value)
            (start $start)
            (func (export "run") (result i32) global.get $value))
        (core instance (instantiate $adapter))
        (core instance $guest (instantiate $guest))
        (func (export "run") (result s32) (canon lift (core func $guest "run"))))"#,
    )
    .unwrap();
    let modules = wasmparser::Parser::new(0)
        .parse_all(&bytes)
        .filter_map(|payload| match payload.unwrap() {
            wasmparser::Payload::ModuleSection { unchecked_range, .. } => Some(&bytes[unchecked_range]),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(neo_wasm::extract_wasm_program_artifacts(modules[0]).is_err());
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(modules[1], HostEventBindings::default())
        .unwrap();
    let mut store = store(registry);
    let component = wasmtime::component::Component::new(store.engine(), &bytes).unwrap();
    let instance =
        block_on(wasmtime::component::Linker::new(store.engine()).instantiate_async(&mut store, &component)).unwrap();
    let function = instance.get_func(&mut store, "run").unwrap();
    let mut results = [wasmtime::component::Val::S32(0)];
    block_on(function.call_async(&mut store, &[], &mut results)).unwrap();
    assert_eq!(results, [wasmtime::component::Val::S32(23)]);
    let state = store.data().single_instance().unwrap();
    assert_eq!(state.steps()[0].immediate_i32, Some(23));
    assert_eq!(state.steps().len(), 5);
    assert!(
        neo_wasm::collect_wasmtime_component_run(&bytes, &HostEventBindings::default(), "run")
            .unwrap_err()
            .to_string()
            .contains("requires exactly one core module")
    );
}
