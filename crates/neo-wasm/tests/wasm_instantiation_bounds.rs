use neo_wasm::host_event_bindings::HostEventBindings;

#[test]
fn active_data_segment_must_fit_initial_memory() {
    let wasm = wat::parse_str(
        r#"(module
          (memory 0)
          (data (i32.const 0) "\01")
          (func (export "main") (result i32) i32.const 7))"#,
    )
    .expect("wat");
    assert!(neo_wasm::collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "main", &[]).is_err());
    let error = neo_wasm::extract_wasm_program_artifacts(&wasm)
        .expect_err("non-instantiable data segment must not produce proof tables")
        .to_string();
    assert!(
        error.contains("active data segment") && error.contains("initial size"),
        "{error}"
    );
}

#[test]
fn active_element_segment_must_fit_initial_table() {
    let wasm = wat::parse_str(
        r#"(module
          (table 0 funcref)
          (func $main (export "main") (result i32) i32.const 7)
          (elem (i32.const 0) $main))"#,
    )
    .expect("wat");
    assert!(neo_wasm::collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "main", &[]).is_err());
    let error = neo_wasm::extract_wasm_program_artifacts(&wasm)
        .expect_err("non-instantiable element segment must not produce proof tables")
        .to_string();
    assert!(
        error.contains("active element segment") && error.contains("initial size"),
        "{error}"
    );
}

#[test]
fn segments_at_the_initial_boundary_are_accepted() {
    let wasm = wat::parse_str(
        r#"(module
          (memory 1)
          (data (i32.const 65535) "\01")
          (table 1 funcref)
          (func $main (export "main") (result i32) i32.const 7)
          (elem (i32.const 0) $main))"#,
    )
    .expect("wat");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("bounded segments");
    assert!(artifacts.tables.linear_memory_init.contains(&(65_535, 1)));
    assert!(artifacts.tables.tables_init.contains(&(0, 0, 1)));
    neo_wasm::collect_wasmtime_steps(&wasm, &HostEventBindings::default(), "main", &[]).expect("module instantiates");
}
