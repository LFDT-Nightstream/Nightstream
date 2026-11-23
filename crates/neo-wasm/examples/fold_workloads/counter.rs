use std::error::Error;
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use neo_wasm::host_event_bindings::{EventSequenceBuilder as Seq, HostEventBindings, HostEventBindingsBuilder};
use neo_wasm::memory_semantics::preload_host_event_tables;
use neo_wasm::{
    WasmMemoryPreload, WasmOpcode, WasmProgramArtifacts, WasmVmStep, WasmtimeTraceHandler, WasmtimeTraceRegistry,
};
use wasmtime::{Config, Engine, Linker, Module, Store};

use super::{prepare_run, Case};

const TEST17: u32 = 36;
const OBSERVATIONS: [i64; 3] = [15, 122, 1130];

pub(super) fn prepare() -> Result<(WasmMemoryPreload, Case), Box<dyn Error>> {
    let wasm = wat::parse_str(include_str!("../../tests/fixtures/starstream_utxo_public.wat"))?;
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm)?;
    let bindings = transaction_bindings(&artifacts)?;
    let mut preload = neo_wasm::preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(&mut preload, &bindings, &artifacts);
    let mut config = Config::new();
    config.guest_debug(true).wasm_tail_call(true);
    let engine = Engine::new(&config)?;
    let module = Module::from_binary(&engine, &wasm)?;
    let mut registry = WasmtimeTraceRegistry::default();
    registry.register_module(&wasm, bindings.clone())?;
    let mut store = Store::new(&engine, registry);
    store.set_debug_handler(WasmtimeTraceHandler::new());
    store
        .edit_breakpoints()
        .expect("debug enabled")
        .single_step(true)?;
    let mut linker = Linker::new(&engine);
    let observations = Arc::new(Mutex::new(Vec::new()));
    transaction_host(&mut linker, observations.clone())?;
    linker.define_unknown_imports_as_traps(&module)?;
    let instance = block_on(linker.instantiate_async(&mut store, &module))?;
    let transaction = instance.get_typed_func::<(), ()>(&mut store, "test17")?;
    block_on(transaction.call_async(&mut store, ()))?;
    assert_eq!(observations.lock().unwrap().as_slice(), OBSERVATIONS);
    let captured = store.data().single_instance()?;
    let artifacts = captured.artifacts();
    let trace = neo_wasm::traces_from_wasmtime_steps_with_host_events(captured.steps(), artifacts, Default::default())?;
    let last = trace.last().unwrap().state_after;
    assert!(!last.output.enabled);
    check_transaction(&trace);
    let run = prepare_run(&trace, artifacts, &bindings, TEST17)?;
    Ok((
        preload,
        Case {
            name: "test17",
            rounds: 1,
            region_words: 0,
            touched: 0,
            run,
        },
    ))
}

const CTX: i32 = 9;
const HANDLE: i32 = 1;
const COUNTER_IMPORTS: &str = "starstream:self/counter";
const CONTEXT: &str = "starstream:std/utxo-context";
type TraceCaller<'a> = wasmtime::Caller<'a, WasmtimeTraceRegistry>;
const METHODS: [[i64; 4]; 3] = [
    [
        -4992487955248509468,
        -2486664750538924298,
        6169022919517591603,
        -8827430911239586701,
    ],
    [
        -209942881528799618,
        3273890352309065231,
        3729828498835407705,
        7442122955733531148,
    ],
    [
        7802059248105938632,
        -7464406540535523192,
        -7930903170017996614,
        -7603060425757492419,
    ],
];

fn transaction_host(
    linker: &mut Linker<WasmtimeTraceRegistry>,
    observations: Arc<Mutex<Vec<i64>>>,
) -> wasmtime::Result<()> {
    linker.func_wrap("[export]counter", "[resource-new]utxo", |rep: i32| {
        assert_eq!(rep, 0);
        HANDLE
    })?;
    linker.func_wrap(CONTEXT, "[method]utxo-context.resume", |ctx: i32| assert_eq!(ctx, CTX))?;
    linker.func_wrap(
        CONTEXT,
        "[method]utxo-context.implements-method",
        |ctx: i32, a: i64, b: i64, c: i64, d: i64| {
            assert_eq!(ctx, CTX);
            assert!(METHODS.contains(&[a, b, c, d]));
        },
    )?;
    linker.func_wrap("starstream:events/observations", "value", move |value: i64| {
        observations.lock().unwrap().push(value)
    })?;
    linker.func_wrap_async(
        COUNTER_IMPORTS,
        "[static]utxo.new",
        |mut caller: TraceCaller<'_>, (): ()| {
            Box::new(async move {
                let callback = caller
                    .get_export("counter#[static]utxo.new")
                    .unwrap()
                    .into_func()
                    .unwrap()
                    .typed::<i32, i32>(&caller)?;
                let handle = callback.call_async(&mut caller, CTX).await?;
                assert_eq!(handle, HANDLE);
                Ok((handle,))
            })
        },
    )?;
    linker.func_wrap_async(
        COUNTER_IMPORTS,
        "[method]utxo.advance",
        |mut caller: TraceCaller<'_>, (handle,): (i32,)| {
            Box::new(async move {
                assert_eq!(handle, HANDLE);
                let callback = caller
                    .get_export("counter#[method]utxo.advance")
                    .unwrap()
                    .into_func()
                    .unwrap()
                    .typed::<i32, ()>(&caller)?;
                callback.call_async(&mut caller, 0).await
            })
        },
    )?;
    linker.func_wrap_async(
        COUNTER_IMPORTS,
        "[method]utxo.add",
        |mut caller: TraceCaller<'_>, (handle, amount): (i32, i64)| {
            Box::new(async move {
                assert_eq!(handle, HANDLE);
                let callback = caller
                    .get_export("counter#[method]utxo.add")
                    .unwrap()
                    .into_func()
                    .unwrap()
                    .typed::<(i32, i64), i64>(&caller)?;
                Ok((callback.call_async(&mut caller, (0, amount)).await?,))
            })
        },
    )?;
    Ok(())
}

fn transaction_bindings(artifacts: &WasmProgramArtifacts) -> Result<HostEventBindings, Box<dyn Error>> {
    let mut bindings = HostEventBindingsBuilder::new(&artifacts.tables);
    bindings.export(TEST17, Seq::op(u64::from(TEST17)).finish()?, vec![])?;
    bindings.export(
        22,
        Seq::op(22).input_local_i32(0, 0)?.finish()?,
        Seq::op(122).output_i32()?.finish()?,
    )?;
    bindings.export(20, Seq::op(20).input_local_i32(0, 0)?.finish()?, vec![])?;
    bindings.export(
        29,
        Seq::op(29)
            .input_local_i32(0, 0)?
            .input_local_i64(1, 1)?
            .finish()?,
        Seq::op(129).output_i64()?.finish()?,
    )?;
    bindings.split_import(16, Seq::op(16).finish()?, Seq::op(116).result()?.finish()?)?;
    bindings.split_import(15, Seq::op(15).arg_i32(0)?.finish()?, vec![])?;
    bindings.split_import(
        18,
        Seq::op(18).arg_i32(0)?.arg_i64(1)?.finish()?,
        Seq::op(118).result()?.finish()?,
    )?;
    bindings.import(13, Seq::op(13).arg_i32(0)?.result()?.finish()?)?;
    bindings.import(8, Seq::op(8).arg_i32(0)?.finish()?)?;
    let mut method = Seq::op(9).arg_i32(0)?;
    for arg in 1..=4 {
        method = method.arg_i64(arg)?;
    }
    bindings.import(9, method.finish()?)?;
    bindings.import(11, Seq::op(11).arg_i64(0)?.finish()?)?;
    Ok(bindings.finish()?)
}

fn check_transaction(trace: &[WasmVmStep]) {
    let writes: Vec<_> = trace
        .iter()
        .filter(|s| s.opcode == WasmOpcode::GlobalSet && s.global_index == Some(2))
        .map(|s| (s.global_write_value.unwrap(), s.global_write_value_hi.unwrap_or(0)))
        .collect();
    assert_eq!(writes, [(10, 0), (15, 0), (115, 0), (122, 0), (1122, 0), (1130, 0)]);
    let mut events = vec![
        vec![u64::from(TEST17)],
        vec![16],
        vec![22, CTX as u64],
        vec![13, 0, HANDLE as u64, 0],
    ];
    let method = |i: usize| {
        let mut words = vec![9, CTX as u64];
        for lane in METHODS[i] {
            words.extend([lane as u32 as u64, (lane as u64) >> 32]);
        }
        words
    };
    events.extend([method(0), method(1), vec![122, 1], vec![116, 1, 0]]);
    let add = |arg, value| {
        [
            vec![18, 1, arg, 0],
            vec![29, 0, arg, 0],
            vec![129, value, 0],
            vec![118, value, 0],
            vec![11, value, 0],
        ]
    };
    let advance = || [vec![15, 1], vec![20, 0], vec![8, CTX as u64]];
    events.extend(add(5, 15));
    events.extend(advance());
    events.extend([method(2), method(0), method(1)]);
    events.extend(add(7, 122));
    events.extend(advance());
    events.extend([method(0), method(1)]);
    events.extend(add(8, 1130));
    let mut digest = [0; 4];
    for event in events {
        for words in event.chunks(8) {
            let mut block = [0; 8];
            block[..words.len()].copy_from_slice(words);
            digest = neo_wasm::comm_chain::absorb_block(digest, block);
        }
    }
    assert_eq!(trace.last().unwrap().state_after.comm_chain, digest);
}
