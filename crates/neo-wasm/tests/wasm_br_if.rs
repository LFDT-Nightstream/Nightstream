mod common;

use neo_wasm::{collect_wasmtime_steps, traces_from_wasmtime_steps};

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
#[ignore = "debug dump for raw br_if trace collection"]
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
            step.cycle,
            step.state_before.pc,
            step.state_after.pc,
            step.info.name,
            step.state_before.sp,
            step.state_after.sp,
        );
    }
}

#[test]
fn wasm_br_if_kernel_roundtrip() {
    common::checked_main(COUNTDOWN_WAT);
}
