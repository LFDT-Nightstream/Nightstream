mod common;

fn module(start: bool) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
          (global $value (mut i32) (i32.const 0))
          (func $start i32.const 42 global.set $value)
          {}
          (func (export "main") (result i32) global.get $value))"#,
        if start { "(start $start)" } else { "" }
    ))
    .expect("wat")
}

#[test]
fn start_function_is_not_silently_omitted_from_proof_initialization() {
    let with_start = module(true);
    let engine = wasmtime::Engine::default();
    let compiled = wasmtime::Module::new(&engine, &with_start).expect("valid module");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &compiled, &[]).expect("start function runs");
    let main = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .expect("main export");
    assert_eq!(main.call(&mut store, ()).expect("main call"), 42);

    let error = neo_wasm::extract_wasm_program_artifacts(&with_start)
        .expect_err("proof initialization does not model start execution")
        .to_string();
    assert!(error.contains("start function"), "{error}");

    let without_start = module(false);
    let checked = common::checked_main(
        r#"(module
          (global $value (mut i32) (i32.const 0))
          (func $start i32.const 42 global.set $value)
          (func (export "main") (result i32) global.get $value))"#,
    );
    assert_eq!(
        checked
            .trace
            .last()
            .expect("trace")
            .state_after
            .output
            .value_lo,
        0
    );
    neo_wasm::extract_wasm_program_artifacts(&without_start).expect("module without start is supported");
}
