//! Trapped executions as a provable terminal state.
//!
//! `unreachable` is the only modeled trap cause today: the trace ends at the
//! faulting row, the carried `trapped` flag enters the semantic-state digest,
//! and `verify` authenticates a prover-disclosed final state with
//! `trapped: true` and no captured output.

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
fn non_unreachable_traps_remain_unprovable() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "main") (result i32)
                i32.const 1
                i32.const 0
                i32.div_u))"#,
    )
    .expect("wat");
    let err = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect_err("div-by-zero trap must stay an error");
    assert!(
        err.to_string()
            .contains("failed to execute Wasmtime export"),
        "expected the wasmtime trap to surface as a collection error, got: {err}"
    );
}
