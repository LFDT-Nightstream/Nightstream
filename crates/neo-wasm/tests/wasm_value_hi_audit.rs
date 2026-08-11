//! High-limb tamper audits for locals/globals cells logs.

mod common;

use common::assert_satisfied;
use neo_math::F;
use neo_wasm::layout::{COL_GLOBAL_VALUE_HI, COL_LOCAL_VALUE_HI, COL_STACK_READ_VALUE_HI, COL_STACK_WRITE0_VALUE_HI};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    build_wasm_relation_layout, collect_wasmtime_steps, extract_wasm_program_artifacts, preload_from_program_artifacts,
    sanity_check_memory_rows, traces_from_wasmtime_steps, WasmOpcode,
};
use p3_field::PrimeCharacteristicRing;

/// Keep downstream stack reads consistent with a tampered high limb.
fn zero_downstream_hi_reads(witnesses: &mut [Vec<F>], from: usize, honest_hi: F) {
    for downstream in (from + 1)..witnesses.len() {
        let w = &mut witnesses[downstream];
        for col in [
            COL_STACK_READ_VALUE_HI[0],
            COL_STACK_READ_VALUE_HI[1],
            COL_STACK_READ_VALUE_HI[2],
        ] {
            if w[col] == honest_hi {
                w[col] = F::ZERO;
            }
        }
    }
}

#[test]
fn i64_local_get_after_set_rejects_tampered_hi() {
    let wasm = wat::parse_str(
        r#"(module
             (func (export "main") (result i64)
               (local i64)
               i64.const 0x0000_0001_0000_0001
               local.set 0
               local.get 0))"#,
    )
    .expect("valid WAT");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");

    let (get_idx, _) = trace
        .iter()
        .enumerate()
        .find(|(_, row)| row.opcode == WasmOpcode::LocalGet)
        .expect("local.get row");

    let mut witnesses: Vec<Vec<F>> = trace.iter().map(build_witness_vector).collect();

    let w = &mut witnesses[get_idx];
    w[COL_LOCAL_VALUE_HI] = F::ZERO;
    w[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;
    assert_satisfied(
        w,
        "local.get row-local CCS unchanged by tampering value_hi — the test premise",
    );

    zero_downstream_hi_reads(&mut witnesses, get_idx, F::ONE);

    let layout = build_wasm_relation_layout();
    let preload = preload_from_program_artifacts(&artifacts, &run.initial_locals);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("locals_hi must reject a tampered i64 local.get hi limb (cells log mismatch)");
    assert!(
        err.contains("locals_hi"),
        "expected locals_hi to fire on the hi-limb tamper, got: {err}",
    );
}

#[test]
fn i64_local_get_uninitialized_rejects_nonzero_hi() {
    let wasm = wat::parse_str(
        r#"(module
             (func (export "main") (result i64)
               (local i64)
               local.get 0))"#,
    )
    .expect("valid WAT");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");

    let (get_idx, _) = trace
        .iter()
        .enumerate()
        .find(|(_, row)| row.opcode == WasmOpcode::LocalGet)
        .expect("local.get row");

    let mut witnesses: Vec<Vec<F>> = trace.iter().map(build_witness_vector).collect();

    // Keep the row-local constraints consistent while changing the high limb.
    let w = &mut witnesses[get_idx];
    w[COL_LOCAL_VALUE_HI] = F::ONE;
    w[COL_STACK_WRITE0_VALUE_HI] = F::ONE;
    assert_satisfied(w, "uninitialized local.get row-local CCS unaffected by hi-limb tamper");

    for downstream in (get_idx + 1)..witnesses.len() {
        let w = &mut witnesses[downstream];
        for col in [
            COL_STACK_READ_VALUE_HI[0],
            COL_STACK_READ_VALUE_HI[1],
            COL_STACK_READ_VALUE_HI[2],
        ] {
            if w[col] == F::ZERO {
                w[col] = F::ONE;
            }
        }
    }

    let layout = build_wasm_relation_layout();
    let preload = preload_from_program_artifacts(&artifacts, &run.initial_locals);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("locals_hi must reject a non-zero hi on a first-read of an uninitialized local");
    assert!(
        err.contains("locals_hi") && (err.contains("read mismatch") || err.contains("zero-default")),
        "expected the locals_hi cells log to reject the hi-limb tamper, got: {err}",
    );
}

#[test]
fn i64_global_get_after_set_rejects_tampered_hi() {
    let wasm = wat::parse_str(
        r#"(module
             (global $g (mut i64) (i64.const 0))
             (func (export "main") (result i64)
               i64.const 0x0000_0001_0000_0001
               global.set $g
               global.get $g))"#,
    )
    .expect("valid WAT");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");

    let (get_idx, _) = trace
        .iter()
        .enumerate()
        .find(|(_, row)| row.opcode == WasmOpcode::GlobalGet)
        .expect("global.get row");

    let mut witnesses: Vec<Vec<F>> = trace.iter().map(build_witness_vector).collect();

    let w = &mut witnesses[get_idx];
    w[COL_GLOBAL_VALUE_HI] = F::ZERO;
    w[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;
    assert_satisfied(
        w,
        "global.get row-local CCS unchanged by tampering value_hi — the test premise",
    );

    zero_downstream_hi_reads(&mut witnesses, get_idx, F::ONE);

    let layout = build_wasm_relation_layout();
    let preload = preload_from_program_artifacts(&artifacts, &run.initial_locals);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("globals_hi must reject a tampered i64 global.get hi limb (cells log mismatch)");
    assert!(
        err.contains("globals_hi"),
        "expected globals_hi to fire on the hi-limb tamper, got: {err}",
    );
}

#[test]
fn i64_global_get_first_read_rejects_tampered_initializer() {
    let wasm = wat::parse_str(
        r#"(module
             (global $g (mut i64) (i64.const 0x0000_0001_0000_00FF))
             (func (export "main") (result i64)
               global.get $g))"#,
    )
    .expect("valid WAT");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");

    let (get_idx, _) = trace
        .iter()
        .enumerate()
        .find(|(_, row)| row.opcode == WasmOpcode::GlobalGet)
        .expect("global.get row");

    let mut witnesses: Vec<Vec<F>> = trace.iter().map(build_witness_vector).collect();

    let w = &mut witnesses[get_idx];
    w[COL_GLOBAL_VALUE_HI] = F::ZERO;
    w[COL_STACK_WRITE0_VALUE_HI] = F::ZERO;
    assert_satisfied(
        w,
        "global.get row-local CCS unchanged by tampering value_hi — the test premise",
    );

    zero_downstream_hi_reads(&mut witnesses, get_idx, F::ONE);

    let layout = build_wasm_relation_layout();
    let preload = preload_from_program_artifacts(&artifacts, &run.initial_locals);
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("globals_hi preload must reject a first-read tamper of an initializer hi limb");
    assert!(
        err.contains("globals_hi") && err.contains("read mismatch"),
        "expected the globals_hi preload to drive a cells-log read mismatch, got: {err}",
    );
}
