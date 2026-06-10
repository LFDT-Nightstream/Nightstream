use neo_math::F;
use neo_wasm::layout::{COL_MEMORY_PAGES_BEFORE, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    build_wasm_lookup_binding_layout, sanity_check_lookup_row, traces_from_wasmtime_wasm_bytes, WasmOpcode,
};
use p3_field::PrimeCharacteristicRing;

fn trace_rows() -> Vec<neo_wasm::WasmStepTrace> {
    let wasm = wat::parse_str(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 0
                i32.const 9
                i32.store
                i32.const 6
                i32.const 7
                i32.mul
                drop
                i64.const 3
                i64.const 5
                i64.mul
                drop
                i32.const 0
                i32.load)
        )"#,
    )
    .expect("wat");
    traces_from_wasmtime_wasm_bytes(&wasm, "run").expect("trace")
}

#[test]
fn lookup_semantics_accept_real_wasm_rows() {
    let layout = build_wasm_lookup_binding_layout();
    for row in trace_rows() {
        let witness = build_witness_vector(&row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("expected lookup semantics to accept {:?}: {err}", row.opcode));
    }
}

#[test]
fn lookup_semantics_reject_tampered_shout_output() {
    let layout = build_wasm_lookup_binding_layout();
    let row = trace_rows()
        .into_iter()
        .find(|row| row.opcode == WasmOpcode::I32Mul)
        .expect("i32.mul row");
    let mut witness = build_witness_vector(&row);
    witness[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(1234);
    let err = sanity_check_lookup_row(layout, &witness).expect_err("tampered op_table output should fail");
    assert!(err.contains("i32_mul"));
}

#[test]
fn lookup_semantics_reject_tampered_linear_memory_bounds() {
    let layout = build_wasm_lookup_binding_layout();
    let row = trace_rows()
        .into_iter()
        .find(|row| row.opcode == WasmOpcode::I32Load)
        .expect("i32.load row");
    let mut witness = build_witness_vector(&row);
    witness[COL_MEMORY_PAGES_BEFORE] = F::ZERO;
    let err = sanity_check_lookup_row(layout, &witness).expect_err("tampered bounds witness should fail");
    assert!(err.contains("linear_memory_bounds"));
}
