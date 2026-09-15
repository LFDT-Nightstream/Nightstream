mod common;

use neo_wasm::{
    collect_wasmtime_steps, extract_wasm_program_artifacts, preload_from_program_artifacts, traces_from_wasmtime_steps,
};

const PROGRAM: &str = r#"(module
    (func $unused nop)
    (func (export "main") (result i32)
        block
            i32.const 1000
            drop
        end
        call $answer)
    (func $answer (result i32) i32.const 7))"#;

#[test]
fn dense_pcs_cover_unexecuted_functions_and_structural_operators() {
    let checked = common::checked_main(PROGRAM);
    let tables = &checked.artifacts.tables;
    assert_eq!(
        tables
            .program_decode
            .iter()
            .map(|entry| entry.pc)
            .collect::<Vec<_>>(),
        (0..10).collect::<Vec<_>>()
    );
}

#[test]
fn shifting_byte_offsets_preserves_program_tables_and_trace() {
    let wasm = wat::parse_str(PROGRAM).unwrap();
    // Insert a valid custom section before the code to shift every byte PC.
    let mut shifted = wasm[..8].to_vec();
    shifted.extend_from_slice(&[0, 123, 2, b'p', b'c']);
    shifted.extend_from_slice(&[0; 120]);
    shifted.extend_from_slice(&wasm[8..]);
    let artifacts = extract_wasm_program_artifacts(&wasm).unwrap();
    let shifted_artifacts = extract_wasm_program_artifacts(&shifted).unwrap();
    assert_eq!(
        preload_from_program_artifacts(&artifacts).entries(),
        preload_from_program_artifacts(&shifted_artifacts).entries(),
        "byte layout must not affect any proof-bound memory table"
    );

    let run = collect_wasmtime_steps(&wasm, "main", &[]).unwrap();
    let shifted_run = collect_wasmtime_steps(&shifted, "main", &[]).unwrap();
    assert_eq!(run.results, shifted_run.results);
    assert_eq!(run.steps, shifted_run.steps);
    let trace = traces_from_wasmtime_steps(&shifted_run.steps).unwrap();
    common::sanity_check_trace(&trace, &artifacts);
    common::ccs_check_trace(&trace);
}
