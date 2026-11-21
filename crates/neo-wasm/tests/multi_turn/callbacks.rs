use futures::executor::block_on;
use neo_wasm::host_event_bindings::{HostEventBindings, ImportTemplate, Limb, MemoryBase, SlotBinding};
use neo_wasm::{traces_from_wasmtime_steps_with_host_events, WasmtimeTraceRegistry};
use wasmtime::{Caller, Linker};

use super::{event, export_bindings, traced_module};
use crate::common;

const EFFECT: u32 = 1;
const RUN: u32 = 2;
const CALLBACK: u32 = 3;
const READ: u32 = 4;

const MODULE: &str = r#"
    (module
      (import "host" "effect" (func $effect (param i32) (result i32)))
      (memory (export "memory") 1)
      (data (i32.const 16) "\01\02\03\05\06\07\08\09")
      (func (export "run") (param $ptr i32) (result i32)
        local.get $ptr call $effect local.get $ptr i32.load8_u i32.add)
      (func (export "callback") (param $ptr i32) (param $increment i32) (result i32) (local $sum i32)
        local.get $ptr local.get $ptr i32.load8_u local.get $increment i32.add
        local.tee $sum i32.store8 local.get $sum)
      (func (export "read") (param $ptr i32) (result i32)
        local.get $ptr i32.load8_u))
"#;

fn bindings() -> HostEventBindings {
    use SlotBinding::*;
    let mut bindings = export_bindings(&[(RUN, 1), (CALLBACK, 2), (READ, 1)]);
    bindings.imports.insert(
        EFFECT,
        ImportTemplate {
            events: vec![event(10, [ArgElem { arg: 0, limb: Limb::Lo }])],
            return_events: Some(vec![event(
                11,
                [
                    MemoryWrite8 {
                        input: 0,
                        base: MemoryBase::Arg(0),
                        byte_offset: 0,
                    },
                    MemoryWrite8 {
                        input: 1,
                        base: MemoryBase::Arg(0),
                        byte_offset: 1,
                    },
                    ResultElem { limb: Limb::Lo },
                    ResultElem { limb: Limb::Hi },
                ],
            )]),
            input_count: 2,
        },
    );
    bindings
}

#[test]
fn successful_callbacks_preserve_byte_write_order_across_turns() {
    let (mut store, module) = traced_module(MODULE, bindings());
    let mut linker = Linker::new(store.engine());
    linker
        .func_wrap_async(
            "host",
            "effect",
            |mut caller: Caller<'_, WasmtimeTraceRegistry>, (ptr,): (i32,)| {
                Box::new(async move {
                    let callback = caller
                        .get_export("callback")
                        .unwrap()
                        .into_func()
                        .unwrap()
                        .typed::<(i32, i32), i32>(&caller)?;
                    let first = callback.call_async(&mut caller, (ptr, 7)).await?;
                    let second = callback.call_async(&mut caller, (ptr, 8)).await?;
                    let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                    memory.write(&mut caller, ptr as usize, &[40, 90])?;
                    Ok((first + second,))
                })
            },
        )
        .unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    let run = instance
        .get_typed_func::<i32, i32>(&mut store, "run")
        .unwrap();
    let read = instance
        .get_typed_func::<i32, i32>(&mut store, "read")
        .unwrap();
    assert_eq!(block_on(run.call_async(&mut store, 19)).unwrap(), 12 + 20 + 40);
    assert_eq!(block_on(read.call_async(&mut store, 20)).unwrap(), 90);

    let captured = store.data().single_instance().unwrap();
    let artifacts = captured.artifacts();
    let trace = traces_from_wasmtime_steps_with_host_events(captured.steps(), artifacts, Default::default()).unwrap();
    common::sanity_check_trace_with_bindings(&trace, artifacts, &artifacts.host_event_bindings);
    common::ccs_check_trace(&trace);
    let returns: Vec<_> = trace
        .iter()
        .filter(|row| row.state_after.is_terminal())
        .map(|row| row.state_after.output.value_lo)
        .collect();
    assert_eq!(returns, [12 + 20 + 40, 90]);

    let mut image = [1, 2, 3, 5, 6, 7, 8, 9];
    let expected = [
        (19, None),
        (19, Some(12)),
        (19, None),
        (19, Some(20)),
        (19, Some(40)),
        (20, Some(90)),
        (19, None),
        (20, None),
    ];
    let mut accesses = trace.iter().filter_map(|row| row.linear_memory);
    for (address, write) in expected {
        let access = accesses.next().expect("missing byte access");
        assert_eq!(access.width_bytes, 1);
        assert_eq!(
            4 * access.lane0.word_addr as usize + usize::from(access.byte_offset),
            address
        );
        let cell = address - 16;
        let word = cell / 4 * 4;
        assert_eq!(access.lane0.value_before.to_le_bytes(), image[word..word + 4]);
        if let Some(byte) = write {
            image[cell] = byte;
        }
        assert_eq!(access.lane0.value_after.to_le_bytes(), image[word..word + 4]);
    }
    assert!(accesses.next().is_none());
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    assert_eq!(memory.data(&store)[16..24], image);
}
