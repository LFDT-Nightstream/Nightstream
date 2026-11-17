mod common;

fn fib_wat(n: u32) -> String {
    assert!(n >= 1);
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
