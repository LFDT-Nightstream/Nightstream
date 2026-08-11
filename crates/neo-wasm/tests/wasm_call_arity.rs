//! Call-row stack-arity regressions: guest calls defer arg pops to the
//! param-init aux rows, so callee arity is not capped by the 3 read lanes,
//! and `call_indirect` binds its table index to the popped stack operand.

mod common;

use neo_math::F;
use neo_wasm::layout::{COL_STACK_READ_ADDR_LO, COL_TABLE_INDEX};
use neo_wasm::WasmOpcode;
use p3_field::PrimeCharacteristicRing;

/// Used to be unprovable: with on-row arg pops, `stack_reads = 4` contradicted
/// the 3-lane activation sum. Aux rows now pop one arg each.
#[test]
fn direct_call_with_four_params_is_provable() {
    let checked = common::checked_wasm_run(
        r#"(module
            (func $sum4 (param i32 i32 i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
                local.get 2
                i32.add
                local.get 3
                i32.add)
            (func (export "run") (result i32)
                i32.const 1
                i32.const 2
                i32.const 3
                i32.const 4
                call $sum4))
        "#,
        "run",
        &[],
    );
    let final_output = checked.trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 10);
}

/// Used to be unprovable: `stack_reads = param_count + 1 = 4` exceeded the
/// lane budget. The call row now pops only the table index.
#[test]
fn call_indirect_with_three_params_is_provable() {
    let checked = common::checked_wasm_run(
        r#"(module
            (type $t (func (param i32 i32 i32) (result i32)))
            (func $sum3 (type $t)
                local.get 0
                local.get 1
                i32.add
                local.get 2
                i32.add)
            (table 1 funcref)
            (elem (i32.const 0) func $sum3)
            (func (export "run") (result i32)
                i32.const 1
                i32.const 2
                i32.const 3
                i32.const 0
                call_indirect (type $t)))
        "#,
        "run",
        &[],
    );
    let final_output = checked.trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 6);
}

/// `table.index` must equal the index operand popped off the stack. Before
/// the lane-0 index binding, the CCS accepted a witness whose table read was
/// redirected to a different validly preloaded slot than the stack operand
/// selected.
#[test]
fn call_indirect_rejects_table_index_decoupled_from_stack_operand() {
    let checked = common::checked_wasm_run(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (func $add_one (type $t)
                local.get 0
                i32.const 1
                i32.add)
            (table 2 funcref)
            (elem (i32.const 0) func $add_one $add_one)
            (func (export "run") (result i32)
                i32.const 5
                i32.const 0
                call_indirect (type $t)))
        "#,
        "run",
        &[],
    );
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    let mut witness = checked.witnesses[row_index].clone();
    // Slot 1 holds the same funcref, so the tables-memory read stays
    // consistent with the preload; only the index↔stack binding can reject.
    assert_eq!(witness[COL_TABLE_INDEX], F::ZERO);
    witness[COL_TABLE_INDEX] = F::ONE;
    common::assert_rejected(&witness, "table index decoupled from popped stack operand");
}

/// The lane-0 index read must point at the operand-stack top (slot sp − 1);
/// otherwise the prover could satisfy the value binding against an
/// non-top stack slot.
#[test]
fn call_indirect_rejects_index_read_redirected_to_other_slot() {
    let checked = common::checked_wasm_run(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (func $add_one (type $t)
                local.get 0
                i32.const 1
                i32.add)
            (table 1 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "run") (result i32)
                i32.const 0
                i32.const 0
                call_indirect (type $t)))
        "#,
        "run",
        &[],
    );
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    let mut witness = checked.witnesses[row_index].clone();
    witness[COL_STACK_READ_ADDR_LO[0]] = F::ZERO;
    common::assert_rejected(&witness, "index read redirected away from sp - 1");
}

/// Wasmtime snapshots the operand stack per frame; the normalizer rebases
/// each frame onto the global stack address space (callee slots start above
/// the caller's residual operands), so an operand held across a call must
/// survive the callee's pushes. The folding-pipeline counterpart is
/// `folding_proof_covers_operand_held_across_call`.
#[test]
fn operand_held_across_call_survives() {
    common::checked_wasm_run(
        r#"(module
            (func $one (param i32) (result i32)
                i32.const 1)
            (func (export "run") (result i32)
                i32.const 10
                i32.const 5
                call $one
                i32.add))
        "#,
        "run",
        &[],
    );
}
