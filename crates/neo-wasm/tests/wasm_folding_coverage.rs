//! End-to-end folding-proof coverage for cross-step state dimensions
//! fibonacci does not exercise:
//!
//! - **Nested calls** — exercises the `call_stack_depth`, `locals_fbp`,
//!   and `param_init` columns the semantic-state digest carries through
//!   call/return boundaries.
//! - **Memory mutation** — `memory.grow` plus a tight `i32.store` loop;
//!   exercises `memory_pages` digest carrying and linear-memory writes
//!   across batch boundaries.
//! - **i64 arithmetic** — keeps the wide-value (`*_hi`) witness columns
//!   warm; fibonacci is i32-only and never writes them.

mod common;

use neo_wasm::{preprocess_seeded_batched, prove_batched, top_level_initial_state_digest, verify};

#[test]
fn folding_proof_covers_nested_calls() {
    let checked = common::checked_main(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "main") (result i32)
                i32.const 5
                call $add_one
                call $add_one))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["7".to_string()]);

    // "main" is the second defined function in this module; passing its
    // entry PC is the verifier's claim about which export is being proven.
    let entry_pc = common::entry_pc_for_function_ref(&checked.artifacts, 2);
    let digest = top_level_initial_state_digest(&checked.artifacts.tables, entry_pc);

    // batch_size 4 with two call boundaries guarantees at least one
    // call/return crosses a batch edge, so the semantic-state digest must
    // carry `call_stack_depth` / `locals_fbp` / `param_init` correctly.
    let batch_size = 4;
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");
    verify(&prep, &proof).expect("verify");
}

#[test]
fn folding_proof_covers_memory_mutation() {
    let checked = common::checked_main(
        r#"(module
            (memory 1)
            (func (export "main") (result i32)
                (local $i i32)
                i32.const 1
                memory.grow
                drop
                (loop $loop
                    local.get $i
                    i32.const 4
                    i32.mul
                    local.get $i
                    i32.store
                    local.get $i
                    i32.const 1
                    i32.add
                    local.tee $i
                    i32.const 4
                    i32.lt_s
                    br_if $loop)
                i32.const 12
                i32.load))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["3".to_string()]);

    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let batch_size = 6;
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");
    verify(&prep, &proof).expect("verify");
}

#[test]
fn folding_proof_covers_i64_arithmetic() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i64)
                i64.const 0x100000000
                i64.const 7
                i64.add))"#,
    );
    assert_eq!(checked.run.results.as_slice(), &["4294967303".to_string()]);

    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let batch_size = 2;
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");
    verify(&prep, &proof).expect("verify");
}
