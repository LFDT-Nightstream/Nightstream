use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, preload_from_wasmtime_run, sanity_check_lookup_row,
    sanity_check_memory_rows, traces_from_wasmtime_steps,
};

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
    let wasm = wat::parse_str(FIB_WAT).expect("valid WAT");
    let layout = build_wasm_lookup_binding_layout();

    for (n, expected) in [(1u32, 1u32), (2, 1), (3, 2), (4, 3), (5, 5), (10, 55)] {
        let run = collect_wasmtime_steps(&wasm, "main", &[n as i32]).expect("wasmtime trace");
        assert_eq!(
            run.results.as_slice(),
            &[expected.to_string()],
            "fib({n}) should be {expected}"
        );

        let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
        let mut witnesses = Vec::with_capacity(trace.len());
        for row in &trace {
            let witness = neo_wasm::builder::build_witness_vector(row);
            sanity_check_lookup_row(layout, &witness)
                .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
            witnesses.push(witness);
        }
        let preload = preload_from_wasmtime_run(&run, &run.initial_locals);
        sanity_check_memory_rows(layout, &witnesses, &preload)
            .unwrap_or_else(|err| panic!("memory semantics rejected fib({n}): {err}"));
    }
}
