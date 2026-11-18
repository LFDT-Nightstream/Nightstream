mod common;

use neo_math::F;
use neo_wasm::host_event_bindings::HostEventBindings;
use neo_wasm::layout::{
    COL_FUNCTION_REF, COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_VALUE_LO, COL_TABLE_INDEX,
    COL_TABLE_VALUE,
};
use neo_wasm::{collect_wasmtime_steps, extract_wasm_program_artifacts, WasmBuildError, WasmOpcode, WasmProgramError};
use p3_field::PrimeCharacteristicRing;

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
    );
    let final_output = checked.trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 10);
}

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
    );
    let final_output = checked.trace.last().expect("final row").state_after.output;
    assert!(final_output.enabled);
    assert_eq!(final_output.value_lo, 6);
}

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
    );
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    assert_eq!(witness[COL_TABLE_INDEX], F::ZERO);
    witness[COL_TABLE_INDEX] = F::ONE;
    common::assert_rejected(&witness, "table index decoupled from popped stack operand");
}

const CALL_INDIRECT_WAT: &str = r#"(module
    (type $t (func (param i32) (result i32)))
    (func $add_one (type $t)
        local.get 0
        i32.const 1
        i32.add)
    (table 1 funcref)
    (elem (i32.const 0) func $add_one)
    (func (export "run") (result i32)
        i32.const 5
        i32.const 0
        call_indirect (type $t)))
"#;

#[test]
fn call_indirect_resolve_rejects_a_handoff_other_than_the_checked_entry() {
    let checked = common::checked_wasm_run(CALL_INDIRECT_WAT, "run");
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirectResolve)
        .expect("call_indirect resolve row");
    let mut witness = checked.witnesses[row_index].clone();
    assert_eq!(witness[COL_STACK_WRITE0_VALUE_LO], witness[COL_TABLE_VALUE]);
    witness[COL_STACK_WRITE0_VALUE_LO] += F::ONE;
    common::assert_rejected(&witness, "resolve step handing off an unchecked reference");
}

#[test]
fn call_indirect_invocation_rejects_a_callee_other_than_the_popped_reference() {
    let checked = common::checked_wasm_run(CALL_INDIRECT_WAT, "run");
    let row_index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect invocation row");
    let mut witness = checked.witnesses[row_index].clone();
    assert_eq!(witness[COL_FUNCTION_REF], witness[COL_STACK_READ_VALUE_LO[0]]);
    witness[COL_FUNCTION_REF] += F::ONE;
    common::assert_rejected(&witness, "invocation calling a reference it did not pop");
}

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
    );
}

#[test]
fn an_internal_call_with_two_results_is_provable() {
    let checked = common::checked_wasm_run(
        r#"(module
            (func $pair (result i32 i32)
                i32.const 9
                i32.const 2)
            (func (export "run") (result i32)
                call $pair
                i32.sub))
        "#,
        "run",
    );
    assert_eq!(common::final_state(&checked.trace).output.value_lo, 7);
}

#[test]
fn an_export_with_two_results_is_refused() {
    let wasm = wat::parse_str(r#"(module (func (export "main") (result i32 i32) i32.const 1 i32.const 7))"#).unwrap();
    let prover = collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "main", &[]).map(|run| run.results);
    assert!(
        matches!(prover, Err(WasmBuildError::Program(WasmProgramError::Unsupported(_)))),
        "the prover path accepts a two-result export: {prover:?}"
    );
    let verifier = extract_wasm_program_artifacts(&wasm).map(|artifacts| artifacts.tables.function_entries);
    assert!(
        matches!(verifier, Err(WasmProgramError::Unsupported(_))),
        "the verifier path accepts a two-result export: {verifier:?}"
    );
}
