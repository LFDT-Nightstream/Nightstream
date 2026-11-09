mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, SlotBinding};
use neo_wasm::{collect_wasmtime_steps, extract_wasm_program_artifacts, traces_from_wasmtime_steps};

fn wasm(wat: &str) -> Vec<u8> {
    wat::parse_str(wat).unwrap()
}

#[test]
fn start_execution_must_not_be_omitted_from_the_program() {
    let bytes = wasm(
        r#"(module
            (global $value (mut i32) (i32.const 0))
            (func $start i32.const 42 global.set $value)
            (start $start)
            (func (export "run") (result i32) global.get $value))"#,
    );
    assert!(extract_wasm_program_artifacts(&bytes).is_err());
}

#[test]
fn active_segments_must_fit_the_declared_initial_store() {
    for source in [
        r#"(module (memory 0 0) (data (i32.const 0) "\2a")
            (func (export "run") (result i32) i32.const 7))"#,
        r#"(module (table 0 funcref) (func $target)
            (elem (i32.const 0) func $target)
            (func (export "run") (result i32) i32.const 7))"#,
    ] {
        assert!(extract_wasm_program_artifacts(&wasm(source)).is_err(), "{source}");
    }
}

#[test]
fn extended_constant_expressions_must_not_be_truncated() {
    for source in [
        r#"(module (global $g i32 (i32.add (i32.const 1) (i32.const 2)))
            (func (export "run") (result i32) global.get $g))"#,
        r#"(module (memory 1) (data (offset (i32.add (i32.const 1) (i32.const 2))) "\2a")
            (func (export "run") (result i32) i32.const 7))"#,
        r#"(module (table 4 funcref) (func $target)
            (elem (offset (i32.add (i32.const 1) (i32.const 2))) func $target)
            (func (export "run") (result i32) i32.const 7))"#,
    ] {
        assert!(extract_wasm_program_artifacts(&wasm(source)).is_err(), "{source}");
    }
}

#[test]
fn missing_final_function_end_is_invalid() {
    let mut bytes = wasm(r#"(module (func (export "run") (result i32) i32.const 7 return))"#);
    assert_eq!(bytes.last(), Some(&0x0b));
    *bytes.last_mut().unwrap() = 0x01;
    assert!(extract_wasm_program_artifacts(&bytes).is_err());
}

#[test]
fn missing_branch_table_metadata_returns_an_error() {
    let bytes = wasm(
        r#"(module (func (export "run") (result i32)
            block (result i32) i32.const 7 i32.const 0 br_table 0 end))"#,
    );
    let mut run = collect_wasmtime_steps(&bytes, &HostEventBindings::default(), "run", &[]).unwrap();
    let branch = run
        .steps
        .iter_mut()
        .find(|row| row.opcode_decoded == Some(neo_wasm::WasmOpcode::BrTable))
        .unwrap();
    branch.immediate_i32 = None;
    let result = std::panic::catch_unwind(|| traces_from_wasmtime_steps(&run.steps));
    assert!(result.is_ok(), "trace normalization panicked");
    assert!(result.unwrap().is_err());
}

#[test]
fn export_entry_events_keep_trace_cycles_consecutive() {
    let bytes = wasm(r#"(module (func (export "run") i32.const 7 drop))"#);
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        1,
        ExportTemplate {
            entry: vec![EventBlock::op(1, [SlotBinding::Const(0); 7])],
            exit: vec![],
            entry_input_count: 0,
        },
    );
    let run = collect_wasmtime_steps(&bytes, &bindings, "run", &[]).unwrap();
    let trace =
        neo_wasm::traces_from_wasmtime_steps_with_host_events(&run.steps, run.artifacts(), Default::default()).unwrap();
    assert_eq!(
        trace.iter().map(|row| row.cycle).collect::<Vec<_>>(),
        (0..trace.len() as u64).collect::<Vec<_>>()
    );
}

#[test]
fn branch_to_function_label_returns_the_value() {
    for source in [
        r#"(module (func (export "run") (result i32) i32.const 7 br 0))"#,
        r#"(module (func (export "run") (result i32) (local i32)
            i32.const 9 local.set 0
            block (result i32) i32.const 7 br 1 end))"#,
    ] {
        let bytes = wasm(source);
        wasmparser::Validator::new().validate_all(&bytes).unwrap();
        let run = collect_wasmtime_steps(&bytes, &HostEventBindings::default(), "run", &[]).unwrap();
        assert_eq!(run.results, vec!["7".to_string()]);
        let trace = traces_from_wasmtime_steps(&run.steps).unwrap();
        assert!(trace.last().unwrap().state_after.is_terminal());
        assert_eq!(trace.last().unwrap().state_after.output.value_lo, 7);
        common::checked_wasm_run(source, "run");
    }
}
