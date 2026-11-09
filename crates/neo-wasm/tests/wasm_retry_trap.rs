//! A caller-owned Wasmtime store retains failed rows. A later successful
//! invocation must not make that failed prefix one valid proof trace.

use futures::executor::block_on;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::{traces_from_wasmtime_steps, WasmOpcode, WasmtimeTraceHandler, WasmtimeTraceRegistry};
use wasmtime::{Config, Engine, Instance, Linker, Module, Store, Trap, Val};

const RECURSIVE: &str = r#"(module
    (global $remaining (export "remaining") (mut i32) (i32.const 1000))
    (func $run (export "run") (result i32)
        global.get $remaining
        i32.eqz
        if (result i32)
            i32.const 7
        else
            global.get $remaining
            i32.const 1
            i32.sub
            global.set $remaining
            call $run
        end))"#;

fn runtime(bytes: &[u8]) -> (Store<WasmtimeTraceRegistry>, Instance) {
    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    config.max_wasm_stack(16 * 1024);
    let engine = Engine::new(&config).unwrap();
    let module = Module::new(&engine, bytes).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry
        .register_module(bytes, HostEventBindings::default())
        .unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let instance = block_on(Linker::new(&engine).instantiate_async(&mut store, &module)).unwrap();
    (store, instance)
}

fn invoke(store: &mut Store<WasmtimeTraceRegistry>, instance: Instance) -> wasmtime::Result<i32> {
    let function = instance.get_func(&mut *store, "run").unwrap();
    let mut result = [Val::I32(0)];
    block_on(function.call_async(store, &[], &mut result))?;
    let Val::I32(value) = result[0] else {
        panic!("i32 result");
    };
    Ok(value)
}

#[test]
fn retry_after_stack_overflow_cannot_form_one_valid_trace() {
    let bytes = wat::parse_str(RECURSIVE).unwrap();
    let (mut store, instance) = runtime(&bytes);
    let mut remaining = 1000;
    let mut failures = 0;
    let output = loop {
        match invoke(&mut store, instance) {
            Ok(output) => break output,
            Err(error) => {
                assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::StackOverflow));
                failures += 1;
                let Val::I32(next) = instance
                    .get_global(&mut store, "remaining")
                    .unwrap()
                    .get(&mut store)
                else {
                    panic!("i32 remaining global");
                };
                assert!(next < remaining, "stack-overflow retry must make progress");
                remaining = next;
            }
        }
    };
    assert_eq!(output, 7);
    assert!(failures > 0);

    let state = store.data().single_instance().unwrap();
    assert!(state
        .steps()
        .iter()
        .any(|row| row.opcode_decoded == Some(WasmOpcode::Call)));
    let error = traces_from_wasmtime_steps(state.steps())
        .expect_err("failed prefix and successful retry must not normalize together");
    eprintln!("combined retry rejected: {error}");
}
