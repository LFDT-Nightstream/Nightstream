mod common;

use common::audit::{prove_batched, verify};
use neo_wasm::preprocess::preprocess_seeded_batched;
use std::time::Instant;

// Iterative fibonacci (do-while loop, valid for n >= 1) with the iteration
// count baked in: local 0 = n (set from a constant; the locals RAM starts
// all-zero, so plain-mode inputs cannot arrive as entry params), locals 1
// and 2 = a and b. fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, ...
fn fib_wat(n: u32) -> String {
    format!(
        r#"
(module
  (func (export "main") (result i32)
    (local i32 i32 i32)
    i32.const {n}
    local.set 0
    i32.const 1
    local.set 1
    i32.const 0
    local.set 2
    (loop $L
      local.get 1
      local.get 2
      i32.add
      local.get 2
      local.set 1
      local.set 2
      local.get 0
      i32.const 1
      i32.sub
      local.tee 0
      br_if $L
    )
    local.get 2
  )
)
"#
    )
}

#[test]
fn wasm_fibonacci_sanity_roundtrip() {
    for (n, expected) in [(1u32, 1u32), (2, 1), (3, 2), (4, 3), (5, 5), (10, 55)] {
        let checked = common::checked_wasm_run(&fib_wat(n), "main");
        assert_eq!(
            checked.run.results.as_slice(),
            &[expected.to_string()],
            "fib({n}) should be {expected}"
        );
    }
}

#[test]
fn wasm_fibonacci_folding_proof_covers_control_flow() {
    let checked = common::checked_wasm_run(&fib_wat(3), "main");
    assert_eq!(checked.run.results.as_slice(), &["2".to_string()]);

    let batch_size = 20;
    let digest = common::verifier_initial_state_digest(&checked.artifacts);
    let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
    let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove fibonacci run");
    verify(&prep, &proof, common::final_state(&checked.trace)).expect("verify fibonacci run");
}

/// Batched-folding timing demo: prove fib traces at multiple fold counts
/// and print per-stage timings.
///
/// The fib(5) cases use exact divisors of its 65-row trace, so they fold
/// without padding and the timing difference is purely fold count. The
/// fib(7) case needs padding (its trace isn't a multiple of 9), which adds
/// at most one extra padded fold.
#[test]
fn wasm_fibonacci_folding_proof_batched() {
    // (fib_n, batch_size, expected fold count via div_ceil).
    let cases = [(5u32, 65usize, 1usize), (5, 13, 5), (5, 5, 13), (7, 9, 10)];

    for &(n, batch_size, expected_folds) in &cases {
        let checked = common::checked_wasm_run(&fib_wat(n), "main");
        let trace_len = checked.trace.len();
        assert_eq!(trace_len.div_ceil(batch_size), expected_folds);

        let t0 = Instant::now();
        let digest = common::verifier_initial_state_digest(&checked.artifacts);
        let prep = preprocess_seeded_batched(batch_size, digest).expect("prep");
        let t_prep = t0.elapsed();
        let t0 = Instant::now();
        let proof = prove_batched(&prep, &checked.trace, batch_size).expect("prove");
        let t_prove = t0.elapsed();
        let t0 = Instant::now();
        verify(&prep, &proof, common::final_state(&checked.trace)).expect("verify");
        let t_verify = t0.elapsed();

        eprintln!(
            "fib({n}) trace_len={trace_len} N={batch_size} ({expected_folds} folds): prep {:.2?}, prove {:.2?}, verify {:.2?}",
            t_prep, t_prove, t_verify
        );
    }
}
