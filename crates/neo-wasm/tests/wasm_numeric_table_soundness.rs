mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{audit_compact_lookup_witness, traces_from_wasmtime_wasm_bytes, WasmOpcode};
use p3_field::PrimeCharacteristicRing;

fn rows(wat: &str) -> Vec<neo_wasm::WasmVmStep> {
    let wasm = wat::parse_str(wat).expect("valid WAT");
    traces_from_wasmtime_wasm_bytes(&wasm, "main").expect("Wasmtime trace")
}

#[test]
fn i64_counts_reject_goldilocks_alias() {
    let trace = rows(
        r#"(module (func (export "main") (result i32)
            i64.const 0 i64.clz drop
            i64.const 0 i64.ctz drop
            i64.const -1 i64.popcnt drop
            i32.const 7))"#,
    );
    for opcode in [WasmOpcode::I64Clz, WasmOpcode::I64Ctz, WasmOpcode::I64Popcnt] {
        let row = trace
            .iter()
            .find(|row| row.opcode == opcode)
            .expect("count row");
        let honest = build_witness_vector(row);
        audit_compact_lookup_witness(&honest).expect("honest count");
        let mut alias = honest;
        alias[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(65);
        alias[COL_STACK_WRITE0_VALUE_HI] = F::from_u64(u64::from(u32::MAX));
        neo_wasm::write_range_check_bits(&mut alias);
        assert!(
            audit_compact_lookup_witness(&alias).is_err(),
            "{opcode:?} accepted q + 64 as a count of 64"
        );
    }
}

#[test]
fn signed_byte_source_rejects_low_seven_bits_equal_to_128() {
    let trace = rows(r#"(module (func (export "main") (result i32) i32.const 128 i32.extend8_s))"#);
    let row = trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32Extend8S)
        .expect("sign extension row");
    let honest = build_witness_vector(row);
    assert_satisfied(&honest, "honest i32.extend8_s");
    let mut forged = honest;
    forged[COL_SIGN_EXT_LOW7] = F::from_u64(128);
    forged[COL_SIGN_EXT_BIT] = F::ZERO;
    forged[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(128);
    assert_rejected(&forged, "i32.extend8_s cannot treat 0x80 as positive");
}

#[test]
fn table_access_rejects_index_equal_to_size() {
    let trace = rows(
        r#"(module
            (type (func))
            (func $f)
            (elem declare funcref (ref.func $f))
            (table 1 funcref)
            (func (export "main") (result i32)
                i32.const 0 ref.func $f table.set 0
                i32.const 0 table.get 0 drop
                i32.const 7))"#,
    );
    for opcode in [WasmOpcode::TableGet, WasmOpcode::TableSet] {
        let row = trace
            .iter()
            .find(|row| row.opcode == opcode)
            .expect("table row");
        let honest = build_witness_vector(row);
        assert_satisfied(&honest, "honest table access");
        let mut forged_row = row.clone();
        forged_row.table_index = Some(1);
        forged_row
            .stack_read0
            .as_mut()
            .expect("table index stack slot")
            .value_lo = 1;
        let forged = build_witness_vector(&forged_row);
        assert_rejected(&forged, "table access at the table size must trap");
    }
}
