//! Imported Wasm state and the public memory verifier (SPEC.md section 7.6).

use super::*;
use nightstream::nebula::Circuit;

fn assert_imported_state_is_bound(local: &str, imported_bytes: &[u8], kind: &str) {
    let artifacts = neo_wasm::extract_wasm_program_artifacts(imported_bytes).unwrap();
    let setup = setup_of(local, NonZeroUsize::MIN, false);
    let (app, geometry) = setup.complete();
    let traced = &setup.programs[0];
    let (fref, entry_pc) = artifacts.tables.function_entries[0];
    let mut imported_preload = preload_from_program_artifacts(&artifacts);
    preload_host_event_tables(
        &mut imported_preload,
        &HostEventBindings::import_free(u32::try_from(fref).unwrap()),
    );
    let local_program = app.program(&traced.preload).unwrap();
    let imported_program = app.program(&imported_preload).unwrap();
    assert_eq!(local_program.d_init(), imported_program.d_init());
    assert_eq!(traced.initial, top_level_initial_state(&artifacts.tables, entry_pc));
    assert_eq!(traced.last.output.value_lo, 0);

    let engine = nightstream::Engine::Optimized;
    let circuit = Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let mut run = prover.start(&local_program, traced.initial).unwrap();
    run.segment(&traced.inputs).unwrap();
    let proof = run.finish().unwrap();
    verifier
        .verify(&local_program, &traced.initial, &traced.last, &proof)
        .unwrap();
    assert!(
        verifier
            .verify(&imported_program, &traced.initial, &traced.last, &proof)
            .is_err(),
        "the public verifier accepted output 0 when the host supplied {kind} value 42"
    );
}

#[test]
fn an_imported_global_must_be_bound_to_host_state_before_public_verification() {
    let local = r#"(module (global $g i32 (i32.const 0)) (func (export "main") (result i32) global.get $g))"#;
    let imported = r#"(module (import "env" "g" (global $g i32)) (func (export "main") (result i32) global.get $g))"#;
    let bytes = wat::parse_str(imported).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let global = wasmtime::Global::new(
        &mut store,
        wasmtime::GlobalType::new(wasmtime::ValType::I32, wasmtime::Mutability::Const),
        wasmtime::Val::I32(42),
    )
    .unwrap();
    let mut linker = wasmtime::Linker::new(&engine);
    linker.define(&store, "env", "g", global).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let result = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .unwrap()
        .call(&mut store, ())
        .unwrap();
    assert_eq!(result, 42);
    assert_imported_state_is_bound(local, &bytes, "global");
}

#[test]
fn an_imported_memory_must_be_bound_to_host_state_before_public_verification() {
    let local = r#"(module (memory 1 1) (func (export "main") (result i32) i32.const 0 i32.load))"#;
    let imported =
        r#"(module (import "env" "mem" (memory 1 1)) (func (export "main") (result i32) i32.const 0 i32.load))"#;
    let bytes = wat::parse_str(imported).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let memory = wasmtime::Memory::new(&mut store, wasmtime::MemoryType::new(1, Some(1))).unwrap();
    memory.write(&mut store, 0, &42u32.to_le_bytes()).unwrap();
    let mut linker = wasmtime::Linker::new(&engine);
    linker.define(&store, "env", "mem", memory).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let result = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .unwrap()
        .call(&mut store, ())
        .unwrap();
    assert_eq!(result, 42);
    assert_imported_state_is_bound(local, &bytes, "memory");
}
