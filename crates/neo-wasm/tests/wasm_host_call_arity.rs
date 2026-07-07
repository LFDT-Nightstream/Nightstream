//! Host-call arity coverage: call rows pop only the indirect table index.
//! `HostCallArg` rows pop arguments, and `HostCallResult` pushes a single
//! result.

mod common;

use neo_math::F;
use neo_wasm::layout::{
    COL_HOST_ARGS_ACTIVE_AFTER, COL_HOST_ARGS_REMAINING_AFTER, COL_HOST_ARGS_REMAINING_AFTER_INV,
    COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO, COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE,
    COL_STACK_WRITE0_ADDR_HI, COL_STACK_WRITE0_ADDR_LO,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    collect_wasmtime_component_run_with_linker, extract_first_component_core_program_artifacts,
    traces_from_wasmtime_steps, WasmAuxOpcode, WasmBuildError, WasmOpcode, WasmRowKind, WasmStepTrace,
    WasmtimeTraceState,
};
use p3_field::PrimeCharacteristicRing;

fn checked_component_run(
    component_wat: &str,
    define_host: impl FnOnce(&mut wasmtime::component::Linker<WasmtimeTraceState>) -> Result<(), WasmBuildError>,
) -> Vec<WasmStepTrace> {
    let component_bytes = wat::parse_str(component_wat).expect("component wat");
    let run =
        collect_wasmtime_component_run_with_linker(&component_bytes, "run", define_host).expect("component trace run");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("component trace normalization");
    let artifacts = extract_first_component_core_program_artifacts(&component_bytes).expect("program artifacts");
    common::sanity_check_trace(&trace, &artifacts, &run.initial_locals);
    common::ccs_check_trace(&trace);
    trace
}

fn host_arg_rows(trace: &[WasmStepTrace]) -> Vec<&WasmStepTrace> {
    trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::HostCallArg))
        .collect()
}

fn host_result_rows(trace: &[WasmStepTrace]) -> Vec<&WasmStepTrace> {
    trace
        .iter()
        .filter(|row| row.row_kind == WasmRowKind::Aux(WasmAuxOpcode::HostCallResult))
        .collect()
}

fn five_arg_component_wat() -> &'static str {
    r#"
    (component
      (type $host-sum5 (func
        (param "a" s32) (param "b" s32) (param "c" s32) (param "d" s32) (param "e" s32)
        (result s32)))
      (type $run-type (func (result s32)))
      (import "host-sum5" (func $host-sum5 (type $host-sum5)))
      (core module $m
        (type $host-ty (func (param i32 i32 i32 i32 i32) (result i32)))
        (import "" "0" (func $host-sum5-core (type $host-ty)))
        (func (export "run") (result i32)
          i32.const 1
          i32.const 2
          i32.const 3
          i32.const 4
          i32.const 5
          call $host-sum5-core))
      (core func $lowered (canon lower (func $host-sum5)))
      (core instance $lowered-host
        (export "0" (func $lowered)))
      (core instance $i
        (instantiate $m
          (with "" (instance $lowered-host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

fn five_arg_trace() -> Vec<WasmStepTrace> {
    checked_component_run(five_arg_component_wat(), |linker| {
        linker
            .root()
            .func_wrap("host-sum5", |_store, (a, b, c, d, e): (i32, i32, i32, i32, i32)| {
                Ok((a + b + c + d + e,))
            })
            .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
    })
}

/// Direct host calls can have more scalar arguments than the on-row read lanes.
#[test]
fn direct_host_call_with_five_scalar_args_is_provable() {
    let trace = five_arg_trace();

    let args = host_arg_rows(&trace);
    assert_eq!(args.len(), 5, "one HostCallArg aux row per argument");
    // Args pop top-down and the remaining counter walks 5 -> 0.
    for (i, row) in args.iter().enumerate() {
        assert_eq!(row.state_before.host_args.remaining, 5 - i as u32);
        assert_eq!(row.state_after.host_args.remaining, 4 - i as u32);
        assert_eq!(row.state_after.sp, row.state_before.sp - 1);
        assert!(row.state_before.host_result_pending);
    }
    assert_eq!(
        args.iter()
            .map(|row| row.stack_read0.expect("arg pop").value_lo)
            .collect::<Vec<_>>(),
        vec![5, 4, 3, 2, 1],
    );

    let results = host_result_rows(&trace);
    assert_eq!(results.len(), 1, "one HostCallResult aux row");
    let result = results[0];
    assert_eq!(result.stack_write0.expect("result push").value_lo, 15);
    assert_eq!(result.state_after.sp, result.state_before.sp + 1);
    assert!(!result.state_after.host_result_pending);

    let final_output = trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 15);
}

/// i64 host args carry their high limbs through the arg aux rows, exactly
/// like guest param-init does.
#[test]
fn host_call_with_i64_args_is_provable() {
    let trace = checked_component_run(
        r#"
        (component
          (type $host-add64 (func (param "x" s64) (param "y" s64) (result s64)))
          (type $run-type (func (result s64)))
          (import "host-add64" (func $host-add64 (type $host-add64)))
          (core module $m
            (type $host-ty (func (param i64 i64) (result i64)))
            (import "" "0" (func $host-add64-core (type $host-ty)))
            (func (export "run") (result i64)
              i64.const 4294967296
              i64.const 8589934592
              call $host-add64-core))
          (core func $lowered (canon lower (func $host-add64)))
          (core instance $lowered-host
            (export "0" (func $lowered)))
          (core instance $i
            (instantiate $m
              (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
        |linker| {
            linker
                .root()
                .func_wrap("host-add64", |_store, (x, y): (i64, i64)| Ok((x + y,)))
                .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
        },
    );

    let args = host_arg_rows(&trace);
    assert_eq!(args.len(), 2);
    // 2^32 and 2^33: both args live entirely in the high limb.
    assert_eq!(
        args.iter()
            .map(|row| row.stack_read0.expect("arg pop").value_hi)
            .collect::<Vec<_>>(),
        vec![Some(2), Some(1)],
    );
    assert!(args.iter().all(|row| row.wide_values_enabled));

    let results = host_result_rows(&trace);
    assert_eq!(results.len(), 1);
    let write = results[0].stack_write0.expect("result push");
    assert_eq!((write.value_hi, write.value_lo), (Some(3), 0));
}

/// Indirect host calls keep the table index on the call row and pop arguments
/// through aux rows.
#[test]
fn indirect_host_call_with_three_args_is_provable() {
    let trace = checked_component_run(
        r#"
        (component
          (type $host-sum3 (func (param "a" s32) (param "b" s32) (param "c" s32) (result s32)))
          (type $run-type (func (result s32)))
          (import "host-sum3" (func $host-sum3 (type $host-sum3)))
          (core module $m
            (type $host-ty (func (param i32 i32 i32) (result i32)))
            (import "" "0" (func $host-sum3-core (type $host-ty)))
            (table 1 funcref)
            (elem (i32.const 0) func $host-sum3-core)
            (func (export "run") (result i32)
              i32.const 10
              i32.const 20
              i32.const 30
              i32.const 0
              call_indirect (type $host-ty)))
          (core func $lowered (canon lower (func $host-sum3)))
          (core instance $lowered-host
            (export "0" (func $lowered)))
          (core instance $i
            (instantiate $m
              (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
        |linker| {
            linker
                .root()
                .func_wrap("host-sum3", |_store, (a, b, c): (i32, i32, i32)| Ok((a + b + c,)))
                .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
        },
    );

    let args = host_arg_rows(&trace);
    assert_eq!(args.len(), 3, "table index pops on the call row, args on aux rows");
    assert_eq!(
        args.iter()
            .map(|row| row.stack_read0.expect("arg pop").value_lo)
            .collect::<Vec<_>>(),
        vec![30, 20, 10],
    );
    assert_eq!(host_result_rows(&trace).len(), 1);

    let final_output = trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 60);
}

#[test]
fn zero_arg_host_call_emits_only_a_result_row() {
    let trace = checked_component_run(
        r#"
        (component
          (type $host-const (func (result s32)))
          (type $run-type (func (result s32)))
          (import "host-const" (func $host-const (type $host-const)))
          (core module $m
            (type $host-ty (func (result i32)))
            (import "" "0" (func $host-const-core (type $host-ty)))
            (func (export "run") (result i32)
              call $host-const-core))
          (core func $lowered (canon lower (func $host-const)))
          (core instance $lowered-host
            (export "0" (func $lowered)))
          (core instance $i
            (instantiate $m
              (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
        |linker| {
            linker
                .root()
                .func_wrap("host-const", |_store, (): ()| Ok((42,)))
                .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
        },
    );

    assert!(host_arg_rows(&trace).is_empty());
    let results = host_result_rows(&trace);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].stack_write0.expect("result push").value_lo, 42);
}

#[test]
fn host_call_without_results_emits_no_result_row() {
    let trace = checked_component_run(
        r#"
        (component
          (type $host-sink (func (param "x" s32)))
          (type $run-type (func (result s32)))
          (import "host-sink" (func $host-sink (type $host-sink)))
          (core module $m
            (type $host-ty (func (param i32)))
            (import "" "0" (func $host-sink-core (type $host-ty)))
            (func (export "run") (result i32)
              i32.const 7
              call $host-sink-core
              i32.const 1))
          (core func $lowered (canon lower (func $host-sink)))
          (core instance $lowered-host
            (export "0" (func $lowered)))
          (core instance $i
            (instantiate $m
              (with "" (instance $lowered-host))))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type)
            (canon lift (core func $run))))
        "#,
        |linker| {
            linker
                .root()
                .func_wrap("host-sink", |_store, (_x,): (i32,)| Ok(()))
                .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
        },
    );

    let args = host_arg_rows(&trace);
    assert_eq!(args.len(), 1);
    assert!(!args[0].state_before.host_result_pending);
    assert!(host_result_rows(&trace).is_empty());
}

/// Skipping the remaining-counter decrement on an arg row must not satisfy
/// the CCS: the counter is what forces exactly `param_count` pops.
#[test]
fn host_arg_row_rejects_skipped_remaining_decrement() {
    let trace = five_arg_trace();
    let arg_row = host_arg_rows(&trace)[0];
    let mut witness = build_witness_vector(arg_row);
    common::assert_satisfied(&witness, "untampered host arg row");
    witness[COL_HOST_ARGS_REMAINING_AFTER] = witness[COL_HOST_ARGS_REMAINING_AFTER] + F::ONE;
    common::assert_rejected(&witness, "host arg row with skipped remaining decrement");
}

/// The host-call row must load the ROM-declared param count into the
/// remaining counter; a prover claiming fewer pending pops is rejected.
#[test]
fn host_call_row_rejects_tampered_remaining_count() {
    let trace = five_arg_trace();
    let call_row = trace
        .iter()
        .find(|row| row.row_kind.is_program() && row.state_after.host_args.active)
        .expect("host call row");
    let mut witness = build_witness_vector(call_row);
    common::assert_satisfied(&witness, "untampered host call row");
    witness[COL_HOST_ARGS_REMAINING_AFTER] = F::from_u64(1);
    common::assert_rejected(&witness, "host call row with understated arg count");
}

/// The result row must consume the owed-result flag; keeping it pending
/// would let a prover push a second value.
#[test]
fn host_result_row_rejects_unconsumed_pending_flag() {
    let trace = five_arg_trace();
    let result_row = host_result_rows(&trace)[0];
    let mut witness = build_witness_vector(result_row);
    common::assert_satisfied(&witness, "untampered host result row");
    witness[COL_HOST_RESULT_PENDING_AFTER] = F::ONE;
    common::assert_rejected(&witness, "host result row keeping the pending flag");
}

#[test]
fn host_result_row_rejects_redirected_write_address() {
    let trace = five_arg_trace();
    let result_row = host_result_rows(&trace)[0];
    let mut witness = build_witness_vector(result_row);
    witness[COL_STACK_WRITE0_ADDR_LO] = witness[COL_STACK_WRITE0_ADDR_LO] + F::from_u64(2);
    witness[COL_STACK_WRITE0_ADDR_HI] = witness[COL_STACK_WRITE0_ADDR_HI] + F::from_u64(2);
    common::assert_rejected(&witness, "host result row writing to a redirected slot");
}

/// A program row cannot execute while a host result push is still owed: the
/// row-kind one-hot forces the pending flag to be consumed first.
#[test]
fn program_row_rejects_pending_host_result() {
    let trace = five_arg_trace();
    let program_row = trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .expect("program row");
    let mut witness = build_witness_vector(program_row);
    witness[COL_HOST_RESULT_PENDING_BEFORE] = F::ONE;
    common::assert_rejected(&witness, "program row with an unconsumed host result");
}

/// Only host-call rows may enter host-arg mode. A guest call to a zero-param
/// callee never activates param-init, so without the full
/// `is_program − call − ci_not_trap + push_present` gate a prover could enter
/// host-arg mode there and append phantom arg pops that shift sp inside the
/// callee. The forgery keeps the exit-mode identity and zero-test gadget
/// satisfied (remaining = 1, inv = 1, is_zero = 0), so only the enter-mode
/// gate can reject it.
#[test]
fn guest_call_row_rejects_forged_host_arg_mode() {
    let checked = common::checked_wasm_run(
        r#"(module
            (func $noargs (result i32)
                i32.const 7)
            (func (export "run") (result i32)
                call $noargs))
        "#,
        "run",
        &[],
    );
    let call_row = checked
        .trace
        .iter()
        .find(|row| row.row_kind.is_program() && row.opcode == WasmOpcode::Call)
        .expect("guest call row");
    assert!(call_row.call_stack_push.is_some(), "call targets a guest callee");
    let mut witness = build_witness_vector(call_row);
    common::assert_satisfied(&witness, "untampered guest call row");
    witness[COL_HOST_ARGS_ACTIVE_AFTER] = F::ONE;
    witness[COL_HOST_ARGS_REMAINING_AFTER] = F::ONE;
    witness[COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO] = F::ZERO;
    witness[COL_HOST_ARGS_REMAINING_AFTER_INV] = F::ONE;
    common::assert_rejected(&witness, "guest call row entering host-arg mode");
}
