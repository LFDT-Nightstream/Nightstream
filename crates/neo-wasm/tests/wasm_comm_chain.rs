//! Cross-repo parity fixtures for the host-event commitment chain.
//!
//! The expected digests below are protocol constants shared bit-for-bit with
//! `starstream-interleaving-proof` (its `LedgerEffectsCommitment` chain); the
//! same vectors are pinned by a test there. If either side changes the
//! permutation instantiation or the compression layout, both tests must be
//! updated together.

use neo_wasm::comm_chain::{self, commit_event, COMM_CHAIN_EVENT_ARGS, COMM_CHAIN_STATE_LEN};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn f(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

#[test]
fn comm_chain_fixture_vectors() {
    // Vector 1: genesis state, discriminant 1, args 1..=7.
    let prev = [Goldilocks::ZERO; COMM_CHAIN_STATE_LEN];
    let args: [Goldilocks; COMM_CHAIN_EVENT_ARGS] = core::array::from_fn(|i| f(i as u64 + 1));
    let state1 = commit_event(prev, f(1), args);
    assert_eq!(
        state1,
        [
            f(16060384774117980274),
            f(6217562501851223455),
            f(9809238410420041413),
            f(4191298748431046296),
        ]
    );

    // Vector 2: chained on vector 1, discriminant 16, distinctive args.
    let args2: [Goldilocks; COMM_CHAIN_EVENT_ARGS] =
        [f(0xffff_ffff), f(0xffff_ffff_0000_0000), f(0), f(42), f(7), f(0), f(1)];
    let state2 = commit_event(state1, f(16), args2);
    assert_eq!(
        state2,
        [
            f(2581777910110991851),
            f(4248944502313846729),
            f(3337412769805346927),
            f(12455009736376722043),
        ]
    );
}

/// Two chained host-call events: the trace's carried chain must equal the
/// native recompute over the known event data.
#[test]
fn trace_carries_host_event_chain() {
    let component_wat = r#"
    (component
      (type $host-mul (func (param "x" s32) (param "y" s32) (result s32)))
      (type $host-sink (func (param "x" s32)))
      (type $run-type (func (result s32)))
      (import "host-mul" (func $host-mul (type $host-mul)))
      (import "host-sink" (func $host-sink (type $host-sink)))
      (core module $m
        (import "" "0" (func $mul (param i32 i32) (result i32)))
        (import "" "1" (func $sink (param i32)))
        (func (export "run") (result i32)
          (local i32)
          i32.const 7
          i32.const 6
          call $mul
          local.tee 0
          call $sink
          local.get 0))
      (core func $lowered-mul (canon lower (func $host-mul)))
      (core func $lowered-sink (canon lower (func $host-sink)))
      (core instance $lowered-host
        (export "0" (func $lowered-mul))
        (export "1" (func $lowered-sink)))
      (core instance $i
        (instantiate $m
          (with "" (instance $lowered-host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#;
    let component_bytes = wat::parse_str(component_wat).expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-mul", |_store, (x, y): (i32, i32)| Ok((x * y,)))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-mul: {err}")))?;
        linker
            .root()
            .func_wrap("host-sink", |_store, (_x,): (i32,)| Ok(()))
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-sink: {err}")))
    })
    .expect("component trace run");
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("trace normalization");

    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");

    let call_frefs: Vec<u32> = trace
        .iter()
        .filter(|row| {
            row.row_kind.is_program()
                && matches!(row.opcode, neo_wasm::WasmOpcode::Call)
                && !row.target_function_is_guest
        })
        .map(|row| row.state_after.host_callee_fref)
        .collect();
    assert_eq!(call_frefs.len(), 2, "two host calls");

    let after_mul =
        comm_chain::commit_host_call_event_u64([0; 4], call_frefs[0], 2, 1, &[(7, 0), (6, 0)], Some((42, 0)));
    let after_sink = comm_chain::commit_host_call_event_u64(after_mul, call_frefs[1], 1, 0, &[(42, 0)], None);
    assert_eq!(trace.last().expect("final row").state_after.comm_chain, after_sink);
}

/// The debug checker must reject a forged carried chain state.
#[test]
fn comm_chain_checker_rejects_forged_state() {
    let wasm = wat::parse_str(r#"(module (func (export "main") (result i32) i32.const 20 i32.const 22 i32.add))"#)
        .expect("wat");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("trace");
    let mut trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalize");
    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("untampered chain");
    let mid = trace.len() / 2;
    trace[mid].state_after.comm_chain[0] ^= 1;
    assert!(
        neo_wasm::comm_chain::sanity_check_comm_chain(&trace).is_err(),
        "checker must reject a forged chain state"
    );
}
