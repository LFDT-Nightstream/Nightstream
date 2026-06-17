//! Trapped executions as a provable terminal state.
//!
//! Modeled trap causes (`unreachable`, div/rem by zero, signed division
//! overflow, `call_indirect` OOB index / null entry / callee type mismatch) end the
//! trace at the faulting row, the carried `trapped` flag enters the
//! semantic-state digest, and `verify` authenticates a prover-disclosed
//! final state with `trapped: true` and no captured output. Unmodeled
//! causes (e.g. OOB linear-memory or non-call_indirect table access) stay loud
//! trace-collection errors.

mod common;

use neo_wasm::{
    preprocess_seeded_batched, prove_batched, top_level_initial_state_digest, verify, WasmOpcode, WasmProveError,
};

#[test]
fn unreachable_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7
                drop
                unreachable))"#,
    );

    // A trapped run has no reference results and no captured output.
    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::Unreachable);
    assert!(last.state_after.trapped);
    assert!(last.state_after.halted);
    assert!(!last.state_after.output.enabled);

    // batch_size 2 forces one padding row after the trap row, covering
    // trapped-flag preservation across padding.
    let batch_size = 2;
    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");

    let final_state = common::final_state(&checked.trace);
    assert!(final_state.trapped);
    verify(&prep, &proof, final_state).expect("verify trapped final state");

    // The trap outcome is bound: claiming a clean (non-trapped) final state
    // for the same proof must fail.
    let mut clean_claim = final_state;
    clean_claim.trapped = false;
    assert!(matches!(
        verify(&prep, &proof, clean_claim),
        Err(WasmProveError::FinalStateMismatch)
    ));
}

#[test]
fn div_by_zero_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7
                i32.const 0
                i32.div_u))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I32DivU);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);

    let batch_size = 2;
    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");

    let final_state = common::final_state(&checked.trace);
    verify(&prep, &proof, final_state).expect("verify trapped final state");

    let mut clean_claim = final_state;
    clean_claim.trapped = false;
    assert!(matches!(
        verify(&prep, &proof, clean_claim),
        Err(WasmProveError::FinalStateMismatch)
    ));
}

#[test]
fn i64_rem_by_zero_traps_on_the_wide_divisor() {
    // The divisor zero-test sums both limbs, so an i64 divisor with a zero
    // low limb but nonzero high limb must NOT trap, and a fully-zero one must.
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 9
                i64.const 0
                i64.rem_u
                i32.wrap_i64))"#,
    );
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I64RemU);
    assert!(last.state_after.trapped);

    let not_trapping = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 9
                i64.const 0x100000000
                i64.rem_u
                i32.wrap_i64))"#,
    );
    assert_eq!(not_trapping.run.results.as_slice(), &["9".to_string()]);
    assert!(!common::final_state(&not_trapping.trace).trapped);
}

#[test]
fn i32_signed_division_overflow_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const -2147483648
                i32.const -1
                i32.div_s))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I32DivS);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);

    let batch_size = 2;
    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");

    let final_state = common::final_state(&checked.trace);
    verify(&prep, &proof, final_state).expect("verify trapped final state");

    let mut clean_claim = final_state;
    clean_claim.trapped = false;
    assert!(matches!(
        verify(&prep, &proof, clean_claim),
        Err(WasmProveError::FinalStateMismatch)
    ));
}

#[test]
fn i64_signed_division_overflow_traps_only_on_exact_min_and_neg1() {
    // The dividend-is-MIN test packs both limbs, so i64::MIN / -1 must trap
    // while the near-miss (i64::MIN + 1) / -1 must divide cleanly.
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const -9223372036854775808
                i64.const -1
                i64.div_s
                i32.wrap_i64))"#,
    );
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I64DivS);
    assert!(last.state_after.trapped);

    let not_trapping = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const -9223372036854775807
                i64.const -1
                i64.div_s
                i32.wrap_i64))"#,
    );
    // (MIN + 1) / -1 = i64::MAX, wrapped to i32 = -1.
    assert_eq!(not_trapping.run.results.as_slice(), &["-1".to_string()]);
    assert!(!common::final_state(&not_trapping.trace).trapped);
}

#[test]
fn call_indirect_null_entry_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
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
                call_indirect (type $t)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::CallIndirect);
    assert_eq!(last.table_value, Some(0), "slot 1 is a null funcref");
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
    assert!(last.call_stack_push.is_none(), "a trapping call_indirect never calls");

    let batch_size = 2;
    // "main" is the second defined function in this fixture.
    let entry_pc = common::entry_pc_for_function_ref(&checked.artifacts, 2);
    let digest = top_level_initial_state_digest(&checked.artifacts.tables, entry_pc);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");

    let final_state = common::final_state(&checked.trace);
    assert!(final_state.trapped);
    verify(&prep, &proof, final_state).expect("verify trapped final state");

    let mut clean_claim = final_state;
    clean_claim.trapped = false;
    assert!(matches!(
        verify(&prep, &proof, clean_claim),
        Err(WasmProveError::FinalStateMismatch)
    ));
}

#[test]
fn call_indirect_oob_index_trap_is_a_provable_terminal_state() {
    // The table holds one slot; calling through index 5 is out of bounds and
    // traps before any entry is read.
    let checked = common::checked_main(
        r#"(module
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
                call_indirect (type $t)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::CallIndirect);
    assert_eq!(last.table_index, Some(5));
    assert_eq!(last.table_size, Some(1), "index 5 is past the one-slot table");
    assert_eq!(last.table_value, None, "no entry is read on an OOB index");
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
    assert!(last.call_stack_push.is_none(), "a trapping call_indirect never calls");

    let batch_size = 2;
    // "main" is the second defined function in this fixture.
    let entry_pc = common::entry_pc_for_function_ref(&checked.artifacts, 2);
    let digest = top_level_initial_state_digest(&checked.artifacts.tables, entry_pc);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");

    let final_state = common::final_state(&checked.trace);
    assert!(final_state.trapped);
    verify(&prep, &proof, final_state).expect("verify trapped final state");

    let mut clean_claim = final_state;
    clean_claim.trapped = false;
    assert!(matches!(
        verify(&prep, &proof, clean_claim),
        Err(WasmProveError::FinalStateMismatch)
    ));
}

#[test]
fn call_indirect_type_mismatch_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (type $u (func (result i64)))
            (func $wide (type $u)
                i64.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $wide)
            (func (export "main") (result i32)
                i32.const 5
                i32.const 0
                call_indirect (type $t)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::CallIndirect);
    assert!(last.table_value.is_some_and(|funcref| funcref != 0));
    assert_ne!(last.function_type_id, last.expected_type_id);
    assert!(last.state_after.trapped);
    assert!(last.call_stack_push.is_none(), "a trapping call_indirect never calls");

    let batch_size = 2;
    // "main" is the second defined function in this fixture.
    let entry_pc = common::entry_pc_for_function_ref(&checked.artifacts, 2);
    let digest = top_level_initial_state_digest(&checked.artifacts.tables, entry_pc);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");

    let final_state = common::final_state(&checked.trace);
    verify(&prep, &proof, final_state).expect("verify trapped final state");

    let mut clean_claim = final_state;
    clean_claim.trapped = false;
    assert!(matches!(
        verify(&prep, &proof, clean_claim),
        Err(WasmProveError::FinalStateMismatch)
    ));
}
