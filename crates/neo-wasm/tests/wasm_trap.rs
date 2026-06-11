//! Trapped executions as a provable terminal state.
//!
//! Modeled trap causes (`unreachable`, div/rem by zero) end the trace at the
//! faulting row, the carried `trapped` flag enters the semantic-state digest,
//! and `verify` authenticates a prover-disclosed final state with
//! `trapped: true` and no captured output. Unmodeled causes (e.g. signed
//! division overflow) stay loud trace-collection errors.

mod common;

use neo_wasm::{preprocess_seeded_batched, prove_batched, verify, WasmOpcode, WasmProveError};

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
fn signed_division_overflow_remains_unprovable() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "main") (result i32)
                i32.const -2147483648
                i32.const -1
                i32.div_s))"#,
    )
    .expect("wat");
    let err =
        neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect_err("signed overflow trap must stay an error");
    assert!(
        err.to_string()
            .contains("failed to execute Wasmtime export"),
        "expected the wasmtime trap to surface as a collection error, got: {err}"
    );
}
