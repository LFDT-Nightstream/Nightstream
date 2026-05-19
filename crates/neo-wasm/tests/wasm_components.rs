use neo_wasm::{
    build_wasm_lookup_binding_layout, builder::build_witness_vector, collect_wasmtime_component_run,
    collect_wasmtime_component_run_with_linker, prove_simple_kernel, sanity_check_lookup_row,
    traces_from_wasmtime_component, traces_from_wasmtime_steps, verify_simple_kernel, WasmBuildError,
    WasmKernelProverInput, WasmKernelPublicInput, WasmKernelVerifierInput, WasmOpcode,
};
use wasmtime::{
    component::{Component, Linker},
    Config, Engine, Store,
};

fn component_wat() -> &'static str {
    r#"
    (component
      (core module $m
        (func (export "run")
          i32.const 7
          drop
          i32.const 9
          drop))
      (core instance $i (instantiate $m))
      (func (export "run")
        (canon lift (core func $i "run"))))
    "#
}

fn component_import_wat() -> &'static str {
    r#"
    (component
      (type $host-double (func (param "x" s32) (result s32)))
      (type $run-type (func (result s32)))
      (import "host-double" (func $host-double (type $host-double)))
      (core module $m
        (type $host-ty (func (param i32) (result i32)))
        (import "" "0" (func $host-double-core (type $host-ty)))
        (func (export "run") (result i32)
          i32.const 21
          call $host-double-core))
      (core func $lowered-host-double (canon lower (func $host-double)))
      (core instance $lowered-host
        (export "0" (func $lowered-host-double)))
      (core instance $i
        (instantiate $m
          (with "" (instance $lowered-host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

#[test]
fn wasmtime_component_debug_trace_captures_core_frames() {
    let component_bytes = wat::parse_str(component_wat()).expect("component wat");
    let run = collect_wasmtime_component_run(&component_bytes, "run").expect("component debug trace");
    let steps = &run.steps;
    assert!(
        !steps.is_empty(),
        "expected at least one debug step from component execution"
    );
    assert!(
        steps.iter().any(|step| step.pc.is_some()),
        "expected at least one core wasm frame with a concrete pc"
    );
    assert!(
        steps.iter().any(|step| step.function_index.is_some()),
        "expected at least one core wasm frame with a concrete function index"
    );
    assert!(
        steps.iter().any(|step| step.opcode_decoded.is_some()),
        "expected at least one normalized opcode from the embedded core module"
    );
}

#[test]
fn wasmtime_component_traces_extract_core_wasm_steps() {
    let component_bytes = wat::parse_str(component_wat()).expect("component wat");
    let traces = traces_from_wasmtime_component(&component_bytes, "run").expect("component trace normalization");
    let opcodes: Vec<_> = traces.iter().map(|row| row.opcode).collect();
    assert!(
        opcodes.starts_with(&[
            WasmOpcode::I32Const,
            WasmOpcode::Drop,
            WasmOpcode::I32Const,
            WasmOpcode::Drop
        ]),
        "unexpected opcode prefix from embedded core module: {opcodes:?}"
    );
}

#[test]
fn wasmtime_component_import_can_lower_into_core_import() {
    // Fixture smoke test: keep the component WAT executable independently of the tracing path.
    let component_bytes = wat::parse_str(component_import_wat()).expect("component wat");

    let mut config = Config::new();
    config.wasm_component_model(true);
    let engine = Engine::new(&config).expect("engine");
    let component = Component::new(&engine, &component_bytes).expect("component compile");
    let mut linker = Linker::new(&engine);
    linker
        .root()
        .func_wrap("host-double", |_store, (x,): (i32,)| Ok((x * 2,)))
        .expect("host func");
    let mut store = Store::new(&engine, ());
    let instance = linker
        .instantiate(&mut store, &component)
        .expect("instantiate component");
    let run = instance
        .get_typed_func::<(), (i32,)>(&mut store, "run")
        .expect("typed export")
        .call(&mut store, ())
        .expect("call component");
    assert_eq!(run.0, 42);
}

#[test]
fn wasm_component_import_kernel_roundtrip_for_embedded_core_trace() {
    let component_bytes = wat::parse_str(component_import_wat()).expect("component wat");
    let run = collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-double", |_store, (x,): (i32,)| Ok((x * 2,)))
            .map_err(|err| WasmBuildError::Trace(format!("failed to define component import: {err}")))
    })
    .expect("component trace run");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("component trace normalization");
    let layout = build_wasm_lookup_binding_layout();
    for row in &trace {
        let witness = build_witness_vector(row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
    }
    let public = WasmKernelPublicInput {
        transcript_seed: b"wasm-component-import-kernel".to_vec(),
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
        pc_rom: run.pc_rom,
        pc_edge_kinds: run.pc_edge_kinds,
        function_entries: run.function_entries,
    };
    let verified = verify_simple_kernel(&verifier_input, &proof).expect("verify");
    assert_eq!(verified.prepared_steps.len(), trace.len());
    assert_eq!(verified.opening_summary, output.opening_summary);
}

#[test]
fn wasm_component_kernel_roundtrip_for_embedded_core_trace() {
    let component_bytes = wat::parse_str(component_wat()).expect("component wat");
    let run = collect_wasmtime_component_run(&component_bytes, "run").expect("component trace run");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("component trace normalization");
    let layout = build_wasm_lookup_binding_layout();
    for row in &trace {
        let witness = build_witness_vector(row);
        sanity_check_lookup_row(layout, &witness)
            .unwrap_or_else(|err| panic!("lookup semantics rejected {:?}: {err}", row.opcode));
    }
    let public = WasmKernelPublicInput {
        transcript_seed: b"wasm-component-kernel".to_vec(),
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
        pc_rom: run.pc_rom,
        pc_edge_kinds: run.pc_edge_kinds,
        function_entries: run.function_entries,
    };
    let verified = verify_simple_kernel(&verifier_input, &proof).expect("verify");
    assert_eq!(verified.prepared_steps.len(), trace.len());
    assert_eq!(verified.opening_summary, output.opening_summary);
}

#[test]
#[ignore = "debug dump for wasm component execution under Wasmtime"]
fn dump_wasmtime_component_debug_trace() {
    let component_bytes = wat::parse_str(component_wat()).expect("component wat");
    let run = collect_wasmtime_component_run(&component_bytes, "run").expect("component debug trace");
    let steps = run.steps;
    println!("component wasmtime steps: {}", steps.len());
    for step in steps {
        println!(
            "step={} depth={} func_index={:?} pc={:?} opcode={:?} stack_words={:?}",
            step.step, step.frame_depth, step.function_index, step.pc, step.opcode_decoded, step.operand_stack_words
        );
    }
}
