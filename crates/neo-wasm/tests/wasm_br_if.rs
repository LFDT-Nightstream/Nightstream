use neo_wasm::{
    collect_wasmtime_steps, prove_simple_kernel, traces_from_wasmtime_steps, verify_simple_kernel,
    WasmKernelProverInput, WasmKernelPublicInput, WasmKernelVerifierInput,
};

// Loop that counts down: local[0] starts at 3, decrements to 0.
// br_if $L branches back while local[0] > 0, then falls through.
const COUNTDOWN_WAT: &str = r#"
(module
  (func (export "main") (result i32)
    (local i32)
    i32.const 3
    local.set 0
    (loop $L
      local.get 0
      i32.const 1
      i32.sub
      local.set 0
      local.get 0
      br_if $L
    )
    local.get 0
  )
)
"#;

fn compile_and_trace(
    wat_src: &str,
) -> (
    Vec<u8>,
    Vec<neo_wasm::WasmStepTrace>,
    Vec<(u64, u64, u64)>,
    Vec<(u64, u64)>,
    Vec<(u64, u64)>,
) {
    let wasm = wat::parse_str(wat_src).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
    let pc_rom = run.pc_rom.clone();
    (wasm, trace, pc_rom, run.pc_edge_kinds, run.function_entries)
}

#[test]
fn print_br_if_raw_trace() {
    let wasm = wat::parse_str(COUNTDOWN_WAT).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");

    println!("=== raw wasmtime steps ({}) ===", run.steps.len());
    for step in &run.steps {
        println!("  step={} pc={:?} opcode={:?}", step.step, step.pc, step.opcode);
    }

    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");

    println!("\n=== normalized IR steps ({}) ===", trace.len());
    for step in &trace {
        println!(
            "  cycle={} pc={}->{} opcode={} sp={}->{}",
            step.cycle, step.pc_before, step.pc_after, step.info.name, step.sp_before, step.sp_after,
        );
    }
}

#[test]
fn wasm_br_if_kernel_roundtrip() {
    let (_, trace, pc_rom, pc_edge_kinds, function_entries) = compile_and_trace(COUNTDOWN_WAT);
    let public = WasmKernelPublicInput {
        transcript_seed: b"wasm-br-if".to_vec(),
        initial_locals: vec![],
    };
    let prover_input = WasmKernelProverInput {
        public: public.clone(),
        trace: &trace,
        pc_rom: pc_rom.clone(),
        pc_edge_kinds: pc_edge_kinds.clone(),
        function_entries: function_entries.clone(),
    };
    let (output, proof) = prove_simple_kernel(&prover_input).expect("prove");

    let verifier_input = WasmKernelVerifierInput {
        public,
        trace: &trace,
        pc_rom,
        pc_edge_kinds,
        function_entries,
    };
    let verified = verify_simple_kernel(&verifier_input, &proof).expect("verify");
    assert_eq!(output.prepared_steps.len(), trace.len());
    assert_eq!(verified.prepared_steps.len(), output.prepared_steps.len());
}
