use futures::executor::block_on;
use neo_wasm::host_event_bindings::{HostEventBindings, ImportTemplate, Limb, SlotBinding};
use neo_wasm::{
    traces_from_wasmtime_steps_with_host_events as normalize, WasmBuildError, WasmOutputState, WasmProgramArtifacts,
    WasmVmStep, WasmtimeTraceRegistry,
};
use wasmtime::{Caller, Instance, Linker, Store, Trap, Val};

use super::{event, export_bindings, traced_module};
use crate::common;

const ADDRESS: i32 = 32;
const INITIAL: u32 = 0x0102_0304;

const HOST: u32 = 1;
const RUN: u32 = 2;
const MIDDLE: u32 = 3;
const CALLBACK: u32 = 4;
const READ: u32 = 5;
const CALL_FAIL: u32 = 6;

const MODULE: &str = r#"
    (module
      (import "host" "invoke" (func $invoke (param i32) (result i32)))
      (memory 1 1)
      (data (i32.const 32) "\04\03\02\01")
      (func (export "run") (result i32) i32.const 1 call $invoke)
      (func (export "middle") (result i32) i32.const 2 call $invoke)
      (func $callback (export "callback") (param $word i32) (param $write i32) (result i32)
        local.get $write i32.eqz if unreachable end
        i32.const 32 local.get $word i32.store
        unreachable)
      (func (export "read") (param i32) (result i32) local.get 0 i32.load)
      (func (export "call_fail") (result i32)
        i32.const 7 i32.const 42 i32.const 0 call $callback i32.add))
"#;

#[derive(Clone, Copy)]
enum Callback {
    TrapBeforeStore,
    TrapAfterStore,
    CatchAndRetry,
}

fn bindings() -> HostEventBindings {
    use SlotBinding::*;
    let mut bindings = export_bindings(&[(RUN, 0), (MIDDLE, 0), (CALLBACK, 2), (READ, 1), (CALL_FAIL, 0)]);
    bindings.imports.insert(
        HOST,
        ImportTemplate {
            events: vec![event(10, [ArgElem { arg: 0, limb: Limb::Lo }])],
            return_events: Some(vec![event(
                11,
                [ResultElem { limb: Limb::Lo }, ResultElem { limb: Limb::Hi }],
            )]),
            input_count: 0,
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

fn runtime(behavior: Callback, word: u32) -> (Store<WasmtimeTraceRegistry>, Instance) {
    let (mut store, module) = traced_module(MODULE, bindings());
    let mut linker = Linker::new(store.engine());
    linker
        .func_wrap_async(
            "host",
            "invoke",
            move |mut caller: Caller<'_, WasmtimeTraceRegistry>, (depth,): (i32,)| {
                Box::new(async move {
                    if depth == 1 {
                        return Ok((callback(&mut caller, "middle", &[]).await?,));
                    }
                    assert_eq!(depth, 2);
                    let write = !matches!(behavior, Callback::TrapBeforeStore);
                    let error = callback(
                        &mut caller,
                        "callback",
                        &[Val::I32(word as i32), Val::I32(i32::from(write))],
                    )
                    .await
                    .unwrap_err();
                    assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::UnreachableCodeReached));
                    if matches!(behavior, Callback::CatchAndRetry) {
                        Ok((callback(&mut caller, "read", &[Val::I32(ADDRESS)]).await?,))
                    } else {
                        Err(error)
                    }
                })
            },
        )
        .unwrap();
    let instance = block_on(linker.instantiate_async(&mut store, &module)).unwrap();
    (store, instance)
}

fn invoke(
    store: &mut Store<WasmtimeTraceRegistry>,
    instance: Instance,
    name: &str,
    args: &[Val],
) -> wasmtime::Result<u32> {
    let function = instance.get_func(&mut *store, name).unwrap();
    let mut result = [Val::I32(0)];
    let call = block_on(function.call_async(&mut *store, args, &mut result));
    store.data().check_errors().unwrap();
    call.map(|()| result[0].i32().unwrap() as u32)
}

fn recovery(behavior: Callback, word: u32, expected: u32) -> (Vec<WasmVmStep>, WasmProgramArtifacts) {
    let (mut store, instance) = runtime(behavior, word);
    let error = invoke(&mut store, instance, "run", &[]).unwrap_err();
    assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::UnreachableCodeReached));
    assert_eq!(
        invoke(&mut store, instance, "read", &[Val::I32(ADDRESS)]).unwrap(),
        expected
    );
    let captured = store.data().single_instance().unwrap();
    let artifacts = captured.artifacts();
    let trace = normalize(captured.steps(), artifacts, Default::default()).unwrap();
    let bindings = &artifacts.host_event_bindings;
    let pc = common::entry_pc_for_function_ref(artifacts, RUN.into());
    let initial =
        neo_wasm::host_event_top_level_initial_state(&artifacts.tables, pc, bindings, RUN, Default::default()).unwrap();
    assert_eq!(trace[0].state_before, initial);
    let write = u64::from(matches!(behavior, Callback::TrapAfterStore));
    let events = [
        (20, 0, 0),
        (10, 1, 0),
        (30, 0, 0),
        (10, 2, 0),
        (40, u64::from(word), write),
        (50, ADDRESS as u64, 0),
        (51, u64::from(expected), 0),
    ]
    .map(|(tag, arg, flag)| [tag, arg, flag, 0, 0, 0, 0, 0]);
    let end = common::final_state(&trace);
    assert!(end.is_terminal() && !end.trapped);
    assert_eq!(
        end.output,
        WasmOutputState {
            enabled: true,
            value_lo: expected,
            value_hi: 0
        }
    );
    assert_eq!(
        end.comm_chain,
        events
            .into_iter()
            .fold([0; 4], neo_wasm::comm_chain::absorb_block)
    );
    common::sanity_check_trace_with_bindings(&trace, artifacts, bindings);
    common::ccs_check_trace(&trace);
    (trace, artifacts.clone())
}

#[test]
fn trap_before_and_after_store_return_different_words() {
    for word in [0x1234_5678, 0x89ab_cdef] {
        recovery(Callback::TrapBeforeStore, word, INITIAL);
        recovery(Callback::TrapAfterStore, word, word);
    }
}

#[test]
fn public_nested_unwind_reads_callback_write() {
    let word = 0x89ab_cdef;
    let (trace, artifacts) = recovery(Callback::TrapAfterStore, word, word);
    let preload = common::shared_host::program_preload(&artifacts);
    common::shared_host::prove_with_limits(&trace, &preload, neo_wasm::nebula::WasmSize::Small.limits(), 16);
}

#[test]
fn caught_callback_retry_is_not_a_top_level_unwind() {
    let word = 0x1234_5678;
    let (mut store, instance) = runtime(Callback::CatchAndRetry, word);
    assert_eq!(invoke(&mut store, instance, "run", &[]).unwrap(), word);
    let captured = store.data().single_instance().unwrap();
    let error = normalize(captured.steps(), captured.artifacts(), Default::default()).unwrap_err();
    assert!(
        matches!(error, WasmBuildError::Trace(ref msg) if msg.contains("is not the final step")),
        "{error}"
    );
}

#[test]
fn unwind_discards_guest_frames_and_retained_operands() {
    let (mut store, instance) = runtime(Callback::TrapBeforeStore, 42);
    let error = invoke(&mut store, instance, "call_fail", &[]).unwrap_err();
    assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::UnreachableCodeReached));
    assert_eq!(
        invoke(&mut store, instance, "read", &[Val::I32(ADDRESS)]).unwrap(),
        INITIAL
    );
    let captured = store.data().single_instance().unwrap();
    let artifacts = captured.artifacts();
    let trace = normalize(captured.steps(), artifacts, Default::default()).unwrap();
    let boundary = trace
        .iter()
        .find(|row| row.row_kind.is_turn_boundary())
        .unwrap();
    assert!(boundary.state_before.trapped && boundary.state_before.sp > 0);
    assert_eq!(boundary.state_before.call_stack_depth, 1);
    assert_eq!(boundary.state_before.host_events.host_frame_depth, 0);
    assert_eq!(boundary.state_after.call_stack_depth, 0);
    assert_eq!(common::final_state(&trace).output.value_lo, INITIAL);
    common::sanity_check_trace_with_bindings(&trace, artifacts, &artifacts.host_event_bindings);
    common::ccs_check_trace(&trace);
}
