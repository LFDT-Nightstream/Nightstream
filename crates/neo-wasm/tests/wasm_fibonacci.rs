mod common;

use neo_wasm::preprocess::preprocess_seeded;
use neo_wasm::{prove, verify, WasmVmSpec};

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
fn wasm_fibonacci_kernel_roundtrip() {
    for (n, expected) in [(1u32, 1u32), (2, 1), (3, 2), (4, 3), (5, 5), (10, 55)] {
        let checked = common::checked_wasm_run(FIB_WAT, "main", &[n as i32]);
        assert_eq!(
            checked.run.results.as_slice(),
            &[expected.to_string()],
            "fib({n}) should be {expected}"
        );
    }
}

#[test]
fn wasm_fibonacci_trace_rows_satisfy_ccs() {
    let checked = common::checked_wasm_run(FIB_WAT, "main", &[3]);
    assert!(checked
        .trace
        .iter()
        .any(|row| row.opcode == neo_wasm::WasmOpcode::End && !row.halted));
}

#[test]
#[ignore = "folding proof is currently too slow for the normal wasm test suite"]
fn wasm_fibonacci_folding_proof_covers_control_flow() {
    let checked = common::checked_wasm_run(FIB_WAT, "main", &[3]);
    assert_eq!(checked.run.results.as_slice(), &["2".to_string()]);

    let prep = preprocess_seeded(&WasmVmSpec::default()).expect("prep");
    let proof = prove(&prep, &checked.trace).expect("prove fibonacci run");
    verify(&prep, &proof).expect("verify fibonacci run");
}
