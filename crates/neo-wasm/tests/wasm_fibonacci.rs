mod common;

// Iterative fibonacci (do-while loop, valid for n >= 1).
// param 0 = n (iteration counter), locals 1 and 2 = a and b.
// fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, ...
const FIB_WAT: &str = r#"
(module
  (func (export "main") (param i32) (result i32)
    (local i32 i32)
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
"#;

#[test]
fn wasm_fibonacci_sanity_roundtrip() {
    for (n, expected) in [(1u32, 1u32), (2, 1), (3, 2), (4, 3), (5, 5), (10, 55)] {
        let checked = common::checked_wasm_run(FIB_WAT, "main", &[n as i32]);
        assert_eq!(
            checked.run.results.as_slice(),
            &[expected.to_string()],
            "fib({n}) should be {expected}"
        );
    }
}

/// Check representative batch counts without repeating the full proof path.
#[test]
fn wasm_fibonacci_batch_count_profiles() {
    // (fib_n, batch_size, expected fold count via div_ceil).
    let cases = [(5u32, 63usize, 1usize), (5, 21, 3), (5, 9, 7), (7, 9, 10)];

    for &(n, batch_size, expected_folds) in &cases {
        let checked = common::checked_wasm_run(FIB_WAT, "main", &[n as i32]);
        let trace_len = checked.trace.len();
        assert_eq!(trace_len.div_ceil(batch_size), expected_folds);
    }
}
