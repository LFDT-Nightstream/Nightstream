mod common;

#[test]
fn imported_memory_requires_verifier_bound_initial_state() {
    let imported = wat::parse_str(
        r#"(module
          (import "env" "memory" (memory 1))
          (func (export "main") (result i32) i32.const 0 i32.load))"#,
    )
    .expect("wat");
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &imported).expect("valid module");
    let mut store = wasmtime::Store::new(&engine, ());
    let memory_type = wasmtime::MemoryType::new(1, None).expect("memory type");
    let memory = wasmtime::Memory::new(&mut store, memory_type).expect("memory");
    memory
        .write(&mut store, 0, &7i32.to_le_bytes())
        .expect("initialize memory");
    let mut linker = wasmtime::Linker::new(&engine);
    linker
        .define(&store, "env", "memory", memory)
        .expect("define memory");
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate");
    let main = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .expect("main export");
    assert_eq!(main.call(&mut store, ()).expect("main call"), 7);

    let error = neo_wasm::extract_wasm_program_artifacts(&imported)
        .expect_err("host memory is not a verifier-bound initial state")
        .to_string();
    assert!(
        error.contains("imported memory") && error.contains("verifier-bound"),
        "{error}"
    );

    let defined = common::checked_main(
        r#"(module
          (memory 1)
          (func (export "main") (result i32) i32.const 0 i32.load))"#,
    );
    assert_eq!(
        defined
            .trace
            .last()
            .expect("trace")
            .state_after
            .output
            .value_lo,
        0
    );
}

#[test]
fn imported_global_requires_verifier_bound_initial_state() {
    let imported = wat::parse_str(
        r#"(module
          (import "env" "value" (global i32))
          (func (export "main") (result i32) global.get 0))"#,
    )
    .expect("wat");
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &imported).expect("valid module");
    let mut store = wasmtime::Store::new(&engine, ());
    let global = wasmtime::Global::new(
        &mut store,
        wasmtime::GlobalType::new(wasmtime::ValType::I32, wasmtime::Mutability::Const),
        wasmtime::Val::I32(7),
    )
    .expect("global");
    let mut linker = wasmtime::Linker::new(&engine);
    linker
        .define(&store, "env", "value", global)
        .expect("define global");
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate");
    let main = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .expect("main export");
    assert_eq!(main.call(&mut store, ()).expect("main call"), 7);

    let error = neo_wasm::extract_wasm_program_artifacts(&imported)
        .expect_err("host global is not a verifier-bound initial state")
        .to_string();
    assert!(
        error.contains("imported globals") && error.contains("verifier-bound"),
        "{error}"
    );

    let defined = common::checked_main(
        r#"(module
          (global i32 (i32.const 0))
          (func (export "main") (result i32) global.get 0))"#,
    );
    assert_eq!(
        defined
            .trace
            .last()
            .expect("trace")
            .state_after
            .output
            .value_lo,
        0
    );
}
