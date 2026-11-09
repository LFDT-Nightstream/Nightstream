use super::*;
use nightstream::nebula::Circuit;

/// Wasmtime's own instantiation of `wasm`, with no imports.
fn instantiate(wasm: &[u8]) -> Result<(wasmtime::Store<()>, wasmtime::Instance), wasmtime::Error> {
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, wasm)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    Ok((store, instance))
}

/// The preload and entry state of the last function of `artifacts`.
fn preload_and_initial(artifacts: &neo_wasm::WasmProgramArtifacts) -> (WasmMemoryPreload, WasmStepState) {
    let (export_fref, entry_pc) = artifacts
        .tables
        .function_entries
        .iter()
        .copied()
        .max_by_key(|(_, pc)| *pc)
        .unwrap();
    let mut preload = preload_from_program_artifacts(artifacts);
    preload_host_event_tables(
        &mut preload,
        &HostEventBindings::import_free(u32::try_from(export_fref).unwrap()),
        artifacts,
    );
    (preload, top_level_initial_state(&artifacts.tables, entry_pc).unwrap())
}

#[test]
fn zero_page_memory_presence_is_bound_by_the_public_verifier() {
    let setup = setup_of(
        r#"(module (memory 0 0) (func (export "main") (result i32) i32.const 7))"#,
        NonZeroUsize::MIN,
        false,
    );
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    assert_eq!(traced.initial.memory_pages, Some(0));
    assert_eq!(traced.last.memory_pages, Some(0));
    assert_eq!(traced.last.max_memory_pages, Some(0));

    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let program = app.program(&traced.preload).unwrap();
    let mut run = prover.start(&program, traced.initial).unwrap();
    run.segment(&traced.inputs).unwrap();
    let proof = run.finish().unwrap();
    verifier
        .verify(&program, &traced.initial, &traced.last, &proof)
        .unwrap();

    let mut accepted = Vec::new();
    let mut no_memory = traced.initial;
    no_memory.memory_pages = None;
    if verifier
        .verify(&program, &no_memory, &traced.last, &proof)
        .is_ok()
    {
        accepted.push("initial.memory_pages");
    }
    no_memory = traced.last;
    no_memory.memory_pages = None;
    if verifier
        .verify(&program, &traced.initial, &no_memory, &proof)
        .is_ok()
    {
        accepted.push("last.memory_pages");
    }
    let mut no_maximum = traced.initial;
    no_maximum.max_memory_pages = None;
    if verifier
        .verify(&program, &no_maximum, &traced.last, &proof)
        .is_ok()
    {
        accepted.push("initial.max_memory_pages");
    }
    no_maximum = traced.last;
    no_maximum.max_memory_pages = None;
    if verifier
        .verify(&program, &traced.initial, &no_maximum, &proof)
        .is_ok()
    {
        accepted.push("last.max_memory_pages");
    }
    let no_memory_module = super::traced(
        r#"(module (func (export "main") (result i32) i32.const 7))"#,
        setup.n,
        NonZeroUsize::MIN,
    );
    let no_memory_program = app.program(&no_memory_module.preload).unwrap();
    if verifier
        .verify(
            &no_memory_program,
            &no_memory_module.initial,
            &no_memory_module.last,
            &proof,
        )
        .is_ok()
    {
        accepted.push("program with no memory");
    }
    assert!(
        accepted.is_empty(),
        "unproved public WASM state changes were accepted: {accepted:?}"
    );
}

/// The adapter may refuse the module; if it builds a program, the public
/// verifier must reject the claimed execution.
#[test]
fn a_module_that_cannot_instantiate_is_rejected_by_the_public_verifier() {
    let valid =
        r#"(module (memory 0 0) (table 0 funcref) (func $target) (func (export "main") (result i32) i32.const 7))"#;
    let impossible = [
        (
            "data segment beyond memory",
            r#"(module (memory 0 0) (table 0 funcref) (data (i32.const 0) "\2a") (func $target) (func (export "main") (result i32) i32.const 7))"#,
        ),
        (
            "element segment beyond table",
            r#"(module (memory 0 0) (table 0 funcref) (func $target) (elem (i32.const 0) func $target) (func (export "main") (result i32) i32.const 7))"#,
        ),
    ];
    let mut accepted = Vec::new();
    for (kind, invalid_wat) in impossible {
        let invalid = wat::parse_str(invalid_wat).unwrap();
        assert!(
            instantiate(&invalid).is_err(),
            "{kind}: Wasmtime instantiated the module"
        );
        let Ok(artifacts) = neo_wasm::extract_wasm_program_artifacts(&invalid) else {
            continue;
        };
        let (preload, initial) = preload_and_initial(&artifacts);
        let mut setup = setup_of(valid, NonZeroUsize::MIN, false);
        setup.programs[0].preload = preload;
        let setup = Setup::of(setup.programs, setup.n, setup.per_step);
        let (app, geometry) = setup.complete();
        let traced = &setup.programs[0];
        assert_eq!(initial, traced.initial);

        let engine = nightstream::Engine::Optimized;
        let circuit = Circuit::compile(geometry, app.clone()).unwrap();
        let prover = circuit.prover(engine, 109).unwrap();
        let verifier = circuit.verifier(engine, 109).unwrap();
        let program = app.program(&traced.preload).unwrap();
        let mut run = prover.start(&program, traced.initial).unwrap();
        run.segment(&traced.inputs).unwrap();
        let proof = run.finish().unwrap();
        if verifier
            .verify(&program, &traced.initial, &traced.last, &proof)
            .is_ok()
        {
            accepted.push(kind);
        }
    }
    assert!(
        accepted.is_empty(),
        "the public verifier accepted executions of modules that cannot instantiate: {accepted:?}"
    );
}

/// The adapter may refuse the module; if it builds a program, the public
/// verifier must reject the output that omits the start function.
#[test]
fn a_start_section_that_changes_output_is_bound_by_the_public_verifier() {
    let valid = r#"(module (global $value (mut i32) (i32.const 0)) (func $start i32.const 42 global.set $value) (func (export "main") (result i32) global.get $value))"#;
    let invalid = wat::parse_str(
        r#"(module (global $value (mut i32) (i32.const 0)) (func $start i32.const 42 global.set $value) (start $start) (func (export "main") (result i32) global.get $value))"#,
    )
    .unwrap();
    let (mut store, instance) = instantiate(&invalid).unwrap();
    let main = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .unwrap();
    assert_eq!(main.call(&mut store, ()).unwrap(), 42);
    let Ok(artifacts) = neo_wasm::extract_wasm_program_artifacts(&invalid) else {
        return;
    };

    let setup = setup_of(valid, NonZeroUsize::MIN, false);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    assert_eq!(traced.last.output.value_lo, 0);
    let (invalid_preload, invalid_initial) = preload_and_initial(&artifacts);
    assert_eq!(invalid_initial, traced.initial);
    let invalid_program = app.program(&invalid_preload).unwrap();

    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let valid_program = app.program(&traced.preload).unwrap();
    let mut run = prover.start(&valid_program, traced.initial).unwrap();
    run.segment(&traced.inputs).unwrap();
    let proof = run.finish().unwrap();
    verifier
        .verify(&valid_program, &traced.initial, &traced.last, &proof)
        .unwrap();
    assert!(
        verifier
            .verify(&invalid_program, &invalid_initial, &traced.last, &proof)
            .is_err(),
        "the public verifier accepted output 0 for a module whose start function makes the real output 42"
    );
}
