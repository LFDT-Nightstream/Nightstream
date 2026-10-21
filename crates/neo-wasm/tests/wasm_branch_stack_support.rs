use neo_wasm::{extract_wasm_program_artifacts, WasmProgramError};

fn reject_unwinding(body: &str) {
    let wasm = wat::parse_str(format!("(module (func (export \"main\") {body}))")).unwrap();
    wasmparser::Validator::new().validate_all(&wasm).unwrap();
    let error =
        extract_wasm_program_artifacts(&wasm).expect_err("branch stack unwinding has no circuit constraints yet");
    assert!(matches!(error, WasmProgramError::Unsupported(_)), "{error}");
    assert!(error.to_string().contains("branch stack unwinding"), "{error}");
}

#[test]
fn reject_branches_that_discard_temporaries() {
    reject_unwinding("block i32.const 99 i32.const 1 br_if 0 drop end");
    reject_unwinding("block i32.const 99 br 0 end");
    reject_unwinding("block i32.const 99 i32.const 0 br_table 0 end");
    reject_unwinding("block i32.const 99 block i32.const 1 br_if 1 end drop end");
    reject_unwinding("block (result i32) i32.const 99 i32.const 42 i32.const 1 br_if 0 drop end drop");
    reject_unwinding("i32.const 99 return");
    reject_unwinding("block i32.const 99 return end");
}

#[test]
fn allow_branches_without_stack_unwinding() {
    for body in [
        "block i32.const 1 br_if 0 end",
        "block block i32.const 1 br_if 1 end end",
        "block (result i32) i32.const 42 i32.const 1 br_if 0 end drop",
        "block br 0 end",
        "block i32.const 0 br_table 0 end",
        "loop i32.const 0 br_if 0 end",
        "return",
        "block return end",
        "i32.const 99 drop return",
    ] {
        let wasm = wat::parse_str(format!("(module (func (export \"main\") {body}))")).unwrap();
        extract_wasm_program_artifacts(&wasm).unwrap();
    }
}

#[test]
fn return_discarding_temporaries_below_results_is_rejected() {
    let wasm = wat::parse_str(
        r#"(module
            (func $f (result i32) i32.const 7 i32.const 9 return)
            (func (export "main") (result i32) call $f))"#,
    )
    .unwrap();
    let error = extract_wasm_program_artifacts(&wasm).expect_err("return unwinding has no circuit constraints yet");
    assert!(matches!(error, WasmProgramError::Unsupported(_)), "{error}");
}

#[test]
fn return_with_exact_results_is_allowed() {
    let wasm = wat::parse_str(
        r#"(module
            (func $f (result i32) i32.const 9 return)
            (func (export "main") (result i32) call $f))"#,
    )
    .unwrap();
    extract_wasm_program_artifacts(&wasm).unwrap();
}
