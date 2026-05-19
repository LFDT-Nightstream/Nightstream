use neo_wasm::{
    collect_wasmtime_steps, prove_simple_kernel, traces_from_wasmtime_steps, verify_simple_kernel,
    WasmKernelProverInput, WasmKernelPublicInput, WasmKernelVerifierInput,
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

    for (n, expected) in [(1u32, 1u32), (2, 1), (3, 2), (4, 3), (5, 5), (10, 55)] {
        let run = collect_wasmtime_steps(&wasm, "main", &[n as i32]).expect("wasmtime trace");
        assert_eq!(
            run.results.as_slice(),
            &[expected.to_string()],
            "fib({n}) should be {expected}"
        );

        let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
        let public = WasmKernelPublicInput {
            transcript_seed: format!("wasm-fib-{n}").into_bytes(),
            initial_locals: run.initial_locals.clone(),
        };
        let prover_input = WasmKernelProverInput {
            public: public.clone(),
            trace: &trace,
            pc_rom: run.pc_rom.clone(),
            pc_edge_kinds: run.pc_edge_kinds.clone(),
            function_entries: run.function_entries.clone(),
        };
        let (output, proof) = prove_simple_kernel(&prover_input).expect("prove");

        let verifier_input = WasmKernelVerifierInput {
            public,
            trace: &trace,
            pc_rom: run.pc_rom.clone(),
            pc_edge_kinds: run.pc_edge_kinds.clone(),
            function_entries: run.function_entries.clone(),
        };
        let verified = verify_simple_kernel(&verifier_input, &proof).expect("verify");
        assert_eq!(output.prepared_steps.len(), trace.len());
        assert_eq!(verified.prepared_steps.len(), output.prepared_steps.len());
        assert_eq!(verified.opening_summary, output.opening_summary);
    }
}
