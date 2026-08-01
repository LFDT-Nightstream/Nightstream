//! Rust conformance checks for the Lean-owned 42-times-6 WASM module.

use neo_wasm::{collect_wasmtime_steps, WasmApplicationModule};
use serde_json::Value;

const MANIFEST: &[u8] = include_bytes!("fixtures/wasm_benchmark_42x6.module.json");

const INDEPENDENT_WAT: &str = r#"
(module
  (memory 1 1)
  (data (i32.const 0) "\2a\00\00\00")
  (func (export "main") (result i32)
    i32.const 0
    i32.load
    i32.const 6
    i32.mul))
"#;

#[test]
fn lean_module_matches_wat_parser_and_computes_252() {
    let module = WasmApplicationModule::from_json_slice(MANIFEST).expect("Lean module manifest");
    assert_eq!(module.module_id(), "wasm-benchmark-42x6");
    assert_eq!(module.entrypoint(), "main");

    let independent_bytes = wat::parse_str(INDEPENDENT_WAT).expect("independent WAT");
    assert_eq!(module.bytes(), independent_bytes);
    assert_eq!(module.artifacts().tables.initial_memory_pages, Some(1));
    assert_eq!(module.artifacts().tables.max_memory_pages, Some(1));
    assert_eq!(
        module.artifacts().tables.linear_memory_init,
        vec![(0, 42), (1, 0), (2, 0), (3, 0)]
    );

    let run = collect_wasmtime_steps(module.bytes(), module.entrypoint(), &[]).expect("exact module run");
    assert_eq!(run.results, vec!["252"]);
}

#[test]
fn lean_module_manifest_fails_closed_on_mutations() {
    let original: Value = serde_json::from_slice(MANIFEST).expect("fixture JSON");
    for (field, replacement) in [
        ("schema", Value::from(2)),
        ("entrypoint_hex", Value::from("6e6f7065")),
        ("memory_minimum_pages", Value::from(2)),
        ("data_hex", Value::from("2b000000")),
        ("module_hex", Value::from("00")),
    ] {
        let mut mutated = original.clone();
        mutated[field] = replacement;
        let encoded = serde_json::to_vec(&mutated).expect("mutated manifest");
        assert!(
            WasmApplicationModule::from_json_slice(&encoded).is_err(),
            "mutation of {field} must fail"
        );
    }

    let mut unknown = original;
    unknown["claimed_result"] = Value::from(252);
    let encoded = serde_json::to_vec(&unknown).expect("unknown-field manifest");
    assert!(WasmApplicationModule::from_json_slice(&encoded).is_err());
}
