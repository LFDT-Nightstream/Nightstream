mod common;

use common::{assert_rejected, assert_satisfied};
use neo_math::F;
use neo_wasm::layout::{COL_SIGN_EXT_BIT, COL_SIGN_EXT_LOW7, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{audit_operation_witness, traces_from_wasmtime_wasm_bytes, WasmOpcode};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn rows(wat: &str) -> Vec<neo_wasm::WasmVmStep> {
    let wasm = wat::parse_str(wat).expect("valid WAT");
    traces_from_wasmtime_wasm_bytes(&wasm, "main").expect("Wasmtime trace")
}

/// A count `c` has the alias `q + c` in two 32-bit limbs only for
/// `c < 2^64 − q = 59`, so each count here is below 59.
#[test]
fn i64_counts_reject_a_modulus_alias() {
    let trace = rows(
        r#"(module (func (export "main") (result i32)
            i64.const 32 i64.clz drop
            i64.const 32 i64.ctz drop
            i64.const 7 i64.popcnt drop
            i32.const 7))"#,
    );
    for opcode in [WasmOpcode::I64Clz, WasmOpcode::I64Ctz, WasmOpcode::I64Popcnt] {
        let row = trace
            .iter()
            .find(|row| row.opcode == opcode)
            .expect("count row");
        let honest = build_witness_vector(row).unwrap();
        audit_operation_witness(&honest).expect("honest count");
        let count = honest[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64();
        let aliased = F::ORDER_U64
            .checked_add(count)
            .expect("the count has an alias below 2^64");
        let mut alias = honest;
        alias[COL_STACK_WRITE0_VALUE_LO] = F::from_u64(aliased & 0xFFFF_FFFF);
        alias[COL_STACK_WRITE0_VALUE_HI] = F::from_u64(aliased >> 32);
        neo_wasm::write_range_check_bits(&mut alias);
        assert!(
            audit_operation_witness(&alias).is_err(),
            "{opcode:?} accepted q + {count} as a count of {count}"
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
    let honest = build_witness_vector(row).unwrap();
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
        let honest = build_witness_vector(row).unwrap();
        assert_satisfied(&honest, "honest table access");
        let mut forged_row = row.clone();
        forged_row.table_index = Some(1);
        forged_row
            .stack_read0
            .as_mut()
            .expect("table index stack slot")
            .value_lo = 1;
        let forged = build_witness_vector(&forged_row).unwrap();
        assert_rejected(&forged, "table access at the table size must trap");
    }
}
