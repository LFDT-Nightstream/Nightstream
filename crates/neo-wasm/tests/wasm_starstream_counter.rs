mod common;

use futures::executor::block_on;
use neo_wasm::comm_chain::absorb_block;
use neo_wasm::host_event_bindings::{EventSequenceBuilder, HostEventBindingsBuilder, MemoryBase, SlotBinding};
use neo_wasm::{
    extract_wasm_program_artifacts, host_event_top_level_initial_state, traces_from_wasmtime_steps_with_host_events,
    WasmOutputState, WasmtimeTraceHandler, WasmtimeTraceRegistry,
};
use wasmtime::{Config, Engine, Linker, Module, Store};

const COUNTER: &str = include_str!("fixtures/starstream_utxo_public.wat");
const GET_STORAGE_FREF: u32 = 32;
const SENTINEL: u64 = 0xcccc_cccc;

#[test]
fn proves_counter_get_storage() {
    let wasm = wat::parse_str(COUNTER).unwrap();
    let program = extract_wasm_program_artifacts(&wasm).unwrap();
    let mut entry = EventSequenceBuilder::absorbing()
        .input_local_i32(0, 0)
        .unwrap()
        .push(SlotBinding::MemoryWrite16 {
            input: 1,
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })
        .unwrap();
    for address in (12..64).step_by(4) {
        entry = entry
            .memory_write_i32(2, MemoryBase::Local(0), address - 10)
            .unwrap();
    }
    let mut exit = EventSequenceBuilder::absorbing().output_i32().unwrap();
    for offset in (0..56).step_by(4) {
        exit = exit.memory_read_i32(MemoryBase::Output, offset).unwrap();
    }
    let mut builder = HostEventBindingsBuilder::new(&program.tables);
    builder
        .export(GET_STORAGE_FREF, entry.finish().unwrap(), exit.finish().unwrap())
        .unwrap();
    let bindings = builder.finish().unwrap();

    let mut config = Config::new();
    config.guest_debug(true).wasm_tail_call(true);
    let engine = Engine::new(&config).unwrap();
    let module = Module::from_binary(&engine, &wasm).unwrap();
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(&wasm, bindings.clone()).unwrap();
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store.edit_breakpoints().unwrap().single_step(true).unwrap();
    let mut linker = Linker::new(&engine);
    linker.define_unknown_imports_as_traps(&module).unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    memory.write(&mut store, 10, &[0xcc; 54]).unwrap();
    let getter = instance
        .get_typed_func::<i32, i32>(&mut store, "counter#get-storage")
        .unwrap();
    assert_eq!(block_on(getter.call_async(&mut store, 10)).unwrap(), 8);
    let mut expected = [0; 56];
    expected[4..8].fill(0xcc);
    assert_eq!(&memory.data(&store)[8..64], &expected);

    let captured = store.data().single_instance().unwrap();
    let trace = traces_from_wasmtime_steps_with_host_events(captured.steps(), captured.artifacts(), Default::default())
        .unwrap();
    let entry_pc = common::entry_pc_for_function_ref(&program, u64::from(GET_STORAGE_FREF));
    let initial = host_event_top_level_initial_state(
        &program.tables,
        entry_pc,
        &bindings,
        GET_STORAGE_FREF,
        Default::default(),
    )
    .unwrap();
    assert_eq!(trace[0].state_before, initial);
    let last = common::final_state(&trace);
    assert!(last.activation_finished && !last.trapped);
    assert_eq!(
        last.output,
        WasmOutputState {
            enabled: true,
            value_lo: 8,
            value_hi: 0
        }
    );
    let blocks = [
        [10, 0xcccc, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL],
        [SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL, 0],
        [8, 0, SENTINEL, 0, 0, 0, 0, 0],
        [0; 8],
    ];
    assert_eq!(last.comm_chain, blocks.into_iter().fold([0; 4], absorb_block));
    common::shared_host::prove(&trace, captured.artifacts());
}
