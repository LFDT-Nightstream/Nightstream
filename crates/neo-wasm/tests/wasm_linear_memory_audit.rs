//! Linear-memory tamper audits split out from `wasm_row_ccs.rs`.

mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{
    COL_LINEAR_MEM_LANE0_ADDR, COL_LINEAR_MEM_LANE0_BYTE0_BEFORE, COL_LINEAR_MEM_LANE0_BYTE1,
    COL_LINEAR_MEM_LANE0_BYTE1_BEFORE, COL_LINEAR_MEM_LANE0_BYTE2, COL_LINEAR_MEM_LANE0_BYTE2_BEFORE,
    COL_LINEAR_MEM_LANE0_BYTE3, COL_LINEAR_MEM_LANE0_BYTE3_BEFORE, COL_LINEAR_MEM_LANE0_VALUE,
    COL_LINEAR_MEM_LANE0_VALUE_BEFORE, COL_LINEAR_MEM_LANE1_VALUE, COL_LINEAR_MEM_LANE2_VALUE, COL_STACK_WRITE0_VALUE,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, preload_from_wasmtime_run, sanity_check_memory_rows,
    traces_from_wasmtime_steps, traces_from_wasmtime_wasm_bytes, WasmOpcode, WasmStepTrace,
};
use p3_field::PrimeCharacteristicRing;

fn trace_from_wat(wat_src: &str) -> Vec<WasmStepTrace> {
    let wasm = wat::parse_str(wat_src).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    traces_from_wasmtime_steps(&run.steps).expect("normalize trace")
}

#[test]
fn i32_store8_row_rejects_tampered_unselected_byte() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 255
               i32.store8
               i32.const 0
               i32.load))"#,
    );
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Store8)
        .expect("store8 row");
    let mut witness = build_witness_vector(store);
    witness[COL_LINEAR_MEM_LANE0_BYTE1] = F::from_u64(0x42);
    witness[COL_LINEAR_MEM_LANE0_VALUE] = F::from_u64(0x42FF);
    assert_rejected(
        &witness,
        "i32.store8 must reject prover-chosen bytes outside the written byte slot",
    );
}

#[test]
fn i32_store8_memory_check_rejects_tampered_consistent_prior_state() {
    let wasm = wat::parse_str(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 255
               i32.store8
               i32.const 0
               i32.load8_u))"#,
    )
    .expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
    let (store_idx, store_row) = trace
        .iter()
        .enumerate()
        .find(|(_, row)| row.opcode == WasmOpcode::I32Store8)
        .expect("store8 row");
    let mut witnesses: Vec<Vec<F>> = trace.iter().map(build_witness_vector).collect();

    // Preserve untouched bytes against the fake prior so row-local CCS passes.
    let claimed_prior: u32 = 0xDEAD_BEEF;
    let stored_byte: u8 = 0xFF;
    let prior_bytes = claimed_prior.to_le_bytes();
    let mut after_bytes = prior_bytes;
    after_bytes[0] = stored_byte;
    let claimed_after = u32::from_le_bytes(after_bytes);

    let w = &mut witnesses[store_idx];
    w[COL_LINEAR_MEM_LANE0_VALUE_BEFORE] = F::from_u64(u64::from(claimed_prior));
    w[COL_LINEAR_MEM_LANE0_BYTE0_BEFORE] = F::from_u64(u64::from(prior_bytes[0]));
    w[COL_LINEAR_MEM_LANE0_BYTE1_BEFORE] = F::from_u64(u64::from(prior_bytes[1]));
    w[COL_LINEAR_MEM_LANE0_BYTE2_BEFORE] = F::from_u64(u64::from(prior_bytes[2]));
    w[COL_LINEAR_MEM_LANE0_BYTE3_BEFORE] = F::from_u64(u64::from(prior_bytes[3]));
    w[COL_LINEAR_MEM_LANE0_VALUE] = F::from_u64(u64::from(claimed_after));
    w[COL_LINEAR_MEM_LANE0_BYTE1] = F::from_u64(u64::from(after_bytes[1]));
    w[COL_LINEAR_MEM_LANE0_BYTE2] = F::from_u64(u64::from(after_bytes[2]));
    w[COL_LINEAR_MEM_LANE0_BYTE3] = F::from_u64(u64::from(after_bytes[3]));

    assert_satisfied(
        w,
        "consistent-prior tamper must still pass every row-local CCS constraint",
    );

    let layout = build_wasm_lookup_binding_layout();
    let preload = preload_from_wasmtime_run(&run, &run.initial_locals);
    let result = sanity_check_memory_rows(layout, &witnesses, &preload);
    assert!(
        result.is_err(),
        "memory sanity check must reject consistent-prior tamper; instead accepted",
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("linear_memory") && (err.contains("zero-default") || err.contains("RMW")),
        "expected the linear_memory RMW / zero-default check to fire, got: {err}",
    );
    let _ = store_row;
}

#[test]
fn i64_store_row_rejects_tampered_high_lane() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 8
                i64.const 0x1122334455667788
                i64.store
                i32.const 0)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Store)
        .expect("i64.store row");
    let mut row = build_witness_vector(store);
    row[COL_LINEAR_MEM_LANE1_VALUE] += F::ONE;
    assert_rejected(&row, "tampered i64.store high lane");
}

#[test]
fn i64_unaligned_load_row_rejects_tampered_lane2() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 9
                i64.const 0x1122334455667788
                i64.store
                i32.const 9
                i64.load
                i64.const 0x1122334455667788
                i64.sub
                i64.eqz)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Load)
        .expect("i64.load row");
    let mut row = build_witness_vector(load);
    row[COL_LINEAR_MEM_LANE2_VALUE] += F::ONE;
    assert_rejected(&row, "tampered unaligned i64.load lane2");
}

#[test]
fn i64_unaligned_store_row_rejects_tampered_lane2() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 9
                i64.const 0x1122334455667788
                i64.store
                i32.const 0)
        )"#,
    )
    .expect("wat");

    let trace = traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("normalize");
    let store = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I64Store)
        .expect("i64.store row");
    let mut row = build_witness_vector(store);
    row[COL_LINEAR_MEM_LANE2_VALUE] += F::ONE;
    assert_rejected(&row, "tampered unaligned i64.store lane2");
}

#[test]
fn i32_load_row_rejects_wrong_linear_memory_word_addr() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 77
               i32.store
               i32.const 0
               i32.load))"#,
    );
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load)
        .expect("load row");
    let mut row = build_witness_vector(load);
    row[COL_LINEAR_MEM_LANE0_ADDR] = F::from_u64(2);
    assert_rejected(&row, "tampered i32.load linear memory word addr");
}

#[test]
fn i32_load_row_accepts_unaligned_word_recomposition() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 1144201745
               i32.store
               i32.const 1
               i32.load))"#,
    );

    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load)
        .expect("unaligned load row");

    let row = build_witness_vector(load);

    assert_satisfied(&row, "unaligned i32.load row");
}

#[test]
fn i32_load_row_rejects_tampered_unaligned_output() {
    let trace = trace_from_wat(
        r#"(module
             (memory 1)
             (func (export "main") (result i32)
               i32.const 0
               i32.const 1144201745
               i32.store
               i32.const 1
               i32.load))"#,
    );
    let load = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Load)
        .expect("unaligned load row");
    let mut row = build_witness_vector(load);
    row[COL_STACK_WRITE0_VALUE] = F::from_u64(0);
    assert_rejected(&row, "tampered unaligned i32.load output");
}
