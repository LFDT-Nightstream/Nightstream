//! Component fixtures using the caller-owned store API for entry-memory capture.

use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::{WasmBuildError, WasmtimeTraceHandler, WasmtimeTraceRun, WasmtimeTraceState};
use wasmtime::component::{Component, Linker, Val};
use wasmtime::{Config, Engine, Store};

pub fn component_i32(
    bytes: &[u8],
    export: &str,
    args: &[Val],
    bindings: &HostEventBindings,
    configure_linker: impl FnOnce(&mut Linker<WasmtimeTraceState>) -> Result<(), WasmBuildError>,
) -> WasmtimeTraceRun {
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(bytes).unwrap();
    bindings
        .validate_against_program(&artifacts.tables)
        .unwrap();
    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_component_model(true);
    let engine = Engine::new(&config).unwrap();
    let trace = WasmtimeTraceState::from_program_artifacts(&artifacts, bindings);
    let mut store = Store::new(&engine, trace);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let component = Component::new(&engine, bytes).unwrap();
    let mut linker = Linker::new(&engine);
    configure_linker(&mut linker).unwrap();
    let instance = futures::executor::block_on(linker.instantiate_async(&mut store, &component)).unwrap();
    let mut ids = std::collections::BTreeMap::new();
    for core in store.debug_all_instances() {
        ids.extend(neo_wasm::build_debug_function_id_map(&core, &mut store).unwrap());
    }
    store.data_mut().set_func_ref_ids(ids);
    let function = instance.get_func(&mut store, export).unwrap();
    let mut results = [Val::S32(0)];
    futures::executor::block_on(function.call_async(&mut store, args, &mut results)).unwrap();
    let [Val::S32(result)] = results else {
        panic!("expected i32 component result")
    };
    WasmtimeTraceRun {
        program_tables: artifacts.tables,
        steps: store.data_mut().take_steps(),
        results: vec![result.to_string()],
    }
}
