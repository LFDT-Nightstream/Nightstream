mod common;

use common::{assert_rejected, assert_satisfied, step};
use neo_math::F;
use neo_wasm::layout::{
    COL_ACTIVATION_FINISHED_AFTER, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_INDIRECT_IS_TRAP, COL_CI_ENTRY_IS_NULL,
    COL_DIV_OVERFLOW, COL_DIV_TRAP, COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_REF, COL_MEM_OOB, COL_TABLE_VALUE,
    COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE,
};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{opcode_code, StackValueAccess, WasmOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn unreachable_row_requires_activation_finished_boundary() {
    let accepted = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        true,
    ))
    .unwrap();
    assert_satisfied(&accepted, "unreachable row");

    let rejected = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ))
    .unwrap();
    assert_rejected(&rejected, "unreachable row with activation_finished=0");
}

#[test]
fn unreachable_row_rejects_clean_trapped_flag() {
    let mut witness = build_witness_vector(&step(
        0,
        14,
        opcode_code(WasmOpcode::Unreachable),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        true,
    ))
    .unwrap();
    assert_satisfied(&witness, "unreachable row with trapped_after=1");
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    assert_rejected(&witness, "unreachable row claiming a clean (non-trapped) exit");
}

#[test]
fn non_trap_row_rejects_fake_trapped_flag() {
    let mut witness = build_witness_vector(&step(
        0,
        1,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ))
    .unwrap();
    assert_satisfied(&witness, "nop row");
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "nop row claiming a trap");
}

#[test]
fn program_row_rejects_execution_after_trap() {
    let mut witness = build_witness_vector(&step(
        0,
        1,
        opcode_code(WasmOpcode::Nop),
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        false,
    ))
    .unwrap();
    witness[COL_TRAPPED_BEFORE] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "program row executing after a trap");
}

#[test]
fn div_by_zero_row_rejects_clean_claim() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivU),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 0)),
        None,
        Some(StackValueAccess::new(0, 0)),
        None,
        0,
        true,
    ))
    .unwrap();
    assert_satisfied(&witness, "i32.div_u row trapping on a zero divisor");
    witness[COL_DIV_TRAP] = F::ZERO;
    assert_rejected(&witness, "div-by-zero row claiming no trap");
}

#[test]
fn div_row_rejects_fake_trap_on_nonzero_divisor() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivU),
        2,
        1,
        Some(StackValueAccess::new(0, 7)),
        Some(StackValueAccess::new(1, 2)),
        None,
        Some(StackValueAccess::new(0, 3)),
        None,
        0,
        false,
    ))
    .unwrap();
    assert_satisfied(&witness, "i32.div_u row with a nonzero divisor");
    witness[COL_DIV_TRAP] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "div row faking a trap on a nonzero divisor");
}

#[test]
fn signed_div_overflow_row_rejects_clean_claim() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivS),
        2,
        1,
        Some(StackValueAccess::new(0, 0x8000_0000)),
        Some(StackValueAccess::new(1, u32::MAX)),
        None,
        Some(StackValueAccess::new(0, 0)),
        None,
        0,
        true,
    ))
    .unwrap();
    assert_satisfied(&witness, "i32.div_s row trapping on MIN / -1");
    witness[COL_DIV_TRAP] = F::ZERO;
    assert_rejected(&witness, "signed-overflow row claiming no trap");
}

#[test]
fn div_s_row_rejects_fake_overflow_trap() {
    let mut witness = build_witness_vector(&step(
        0,
        5,
        opcode_code(WasmOpcode::I32DivS),
        2,
        1,
        Some(StackValueAccess::new(0, 0x8000_0000)),
        Some(StackValueAccess::new(1, 2)),
        None,
        Some(StackValueAccess::new(0, 0xC000_0000)),
        None,
        0,
        false,
    ))
    .unwrap();
    assert_satisfied(&witness, "i32.div_s row with MIN dividend and divisor 2");
    witness[COL_DIV_OVERFLOW] = F::ONE;
    witness[COL_DIV_TRAP] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    assert_rejected(&witness, "div_s row faking an overflow trap");
}

const VALID_CALL_INDIRECT_WAT: &str = r#"(module
    (type $t (func (param i32) (result i32)))
    (func $add_one (type $t)
        local.get 0
        i32.const 1
        i32.add)
    (table 1 funcref)
    (elem (i32.const 0) func $add_one)
    (func (export "main") (result i32)
        i32.const 5
        i32.const 0
        call_indirect (type $t)))"#;

const NULL_ENTRY_CALL_INDIRECT_WAT: &str = r#"(module
    (type $t (func (param i32) (result i32)))
    (func $add_one (type $t)
        local.get 0
        i32.const 1
        i32.add)
    (table 2 funcref)
    (elem (i32.const 0) func $add_one)
    (func (export "main") (result i32)
        i32.const 5
        i32.const 1
        call_indirect (type $t)))"#;

#[test]
fn healthy_call_indirect_row_rejects_faked_trap() {
    let checked = common::checked_main(VALID_CALL_INDIRECT_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_CALL_INDIRECT_IS_TRAP] = F::ONE;
    witness[COL_TRAPPED_AFTER] = F::ONE;
    witness[COL_ACTIVATION_FINISHED_AFTER] = F::ONE;
    witness[COL_CALL_INDIRECT_IS_NOT_TRAP] = F::ZERO;
    assert_rejected(&witness, "faked trap on a valid call_indirect");
}

#[test]
fn null_entry_call_indirect_row_rejects_hidden_trap() {
    let checked = common::checked_main(NULL_ENTRY_CALL_INDIRECT_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_CALL_INDIRECT_IS_TRAP] = F::ZERO;
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    witness[COL_ACTIVATION_FINISHED_AFTER] = F::ZERO;
    witness[COL_CALL_INDIRECT_IS_NOT_TRAP] = F::ONE;
    assert_rejected(&witness, "hidden null-entry call_indirect trap");
}

#[test]
fn null_entry_call_indirect_row_rejects_forged_null_flag() {
    let checked = common::checked_main(NULL_ENTRY_CALL_INDIRECT_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_CI_ENTRY_IS_NULL] = F::ZERO;
    witness[COL_CALL_INDIRECT_IS_TRAP] = F::ZERO;
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    witness[COL_ACTIVATION_FINISHED_AFTER] = F::ZERO;
    witness[COL_CALL_INDIRECT_IS_NOT_TRAP] = F::ONE;
    witness[COL_FUNCTION_CALL_TYPE_LOOKUP_GATE] = F::ONE;
    assert_rejected(&witness, "forged null flag on a null table entry");
}

const OOB_CALL_INDIRECT_WAT: &str = r#"(module
    (type $t (func (param i32) (result i32)))
    (func $add_one (type $t)
        local.get 0
        i32.const 1
        i32.add)
    (table 1 funcref)
    (elem (i32.const 0) func $add_one)
    (func (export "main") (result i32)
        i32.const 9
        i32.const 5
        call_indirect (type $t)))"#;

#[test]
fn oob_call_indirect_row_rejects_hidden_trap() {
    let checked = common::checked_main(OOB_CALL_INDIRECT_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_CALL_INDIRECT_IS_TRAP] = F::ZERO;
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    witness[COL_ACTIVATION_FINISHED_AFTER] = F::ZERO;
    witness[COL_CALL_INDIRECT_IS_NOT_TRAP] = F::ONE;
    assert_rejected(&witness, "hidden OOB-index call_indirect trap");
}

const TYPE_MISMATCH_CALL_INDIRECT_WAT: &str = r#"(module
    (type $t (func (param i32) (result i32)))
    (type $u (func (result i32)))
    (func $add_one (type $t)
        local.get 0
        i32.const 1
        i32.add)
    (table 1 funcref)
    (elem (i32.const 0) func $add_one)
    (func (export "main") (result i32)
        i32.const 0
        call_indirect (type $u)))"#;

#[test]
fn type_mismatch_call_indirect_row_binds_looked_up_reference() {
    let checked = common::checked_main(TYPE_MISMATCH_CALL_INDIRECT_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    assert_eq!(witness[COL_CALL_INDIRECT_IS_TRAP], F::ONE);
    assert_eq!(witness[COL_FUNCTION_CALL_TYPE_LOOKUP_GATE], F::ONE);
    witness[COL_FUNCTION_REF] = witness[COL_TABLE_VALUE] + F::ONE;
    assert_rejected(&witness, "type lookup through a reference other than the table entry");
}

const OOB_LOAD_WAT: &str = r#"(module
    (memory 1)
    (func (export "main") (result i32)
        i32.const 1000000
        i32.load))"#;

#[test]
fn oob_load_row_rejects_hidden_trap() {
    let checked = common::checked_main(OOB_LOAD_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::I32Load)
        .expect("i32.load row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_MEM_OOB] = F::ZERO;
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    witness[COL_ACTIVATION_FINISHED_AFTER] = F::ZERO;
    assert_rejected(&witness, "hidden OOB linear-memory trap");
}

const OOB_STORE_WAT: &str = r#"(module
    (memory 1)
    (func (export "main") (result i32)
        i32.const 1000000
        i32.const 42
        i32.store
        i32.const 0))"#;

#[test]
fn oob_store_row_rejects_hidden_trap() {
    let checked = common::checked_main(OOB_STORE_WAT);
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::I32Store)
        .expect("i32.store row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_MEM_OOB] = F::ZERO;
    witness[COL_TRAPPED_AFTER] = F::ZERO;
    witness[COL_ACTIVATION_FINISHED_AFTER] = F::ZERO;
    assert_rejected(&witness, "hidden OOB linear-memory store trap");
}
