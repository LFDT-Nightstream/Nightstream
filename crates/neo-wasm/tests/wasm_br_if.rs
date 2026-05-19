use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, preload_from_wasmtime_run, sanity_check_lookup_row,
    sanity_check_memory_rows, traces_from_wasmtime_steps,
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
    let wasm = wat::parse_str(COUNTDOWN_WAT).expect("valid WAT");
    let run = collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize trace");
    let layout = build_wasm_lookup_binding_layout();
    let mut witnesses = Vec::with_capacity(trace.len());
    for row in &trace {
        let witness = neo_wasm::builder::build_witness_vector(row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
        witnesses.push(witness);
    }
    let preload = preload_from_wasmtime_run(&run, &run.initial_locals);
    sanity_check_memory_rows(layout, &witnesses, &preload)
        .unwrap_or_else(|err| panic!("memory semantics rejected trace: {err}"));
}
