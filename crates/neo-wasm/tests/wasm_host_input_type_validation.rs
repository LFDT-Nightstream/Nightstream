use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding};
use neo_wasm::{extract_first_component_core_program_artifacts, extract_wasm_program_artifacts};
use wasmtime::component::Val as ComponentVal;

fn entry_template() -> ExportTemplate {
    let mut input = [SlotBinding::Const(0); 8];
    input[0] = SlotBinding::InputLocal {
        input: 0,
        local: 0,
        limb: Limb::Lo,
    };
    input[1] = SlotBinding::InputLocal {
        input: 1,
        local: 0,
        limb: Limb::Hi,
    };
    ExportTemplate {
        entry: vec![EventBlock::op(1, [SlotBinding::Const(0); 7]), EventBlock::advice(input)],
        exit: vec![],
        entry_input_count: 2,
    }
}

#[test]
fn i32_export_input_high_slot_is_rejected_before_capture() {
    let component = wat::parse_str(
        r#"(component
          (type $run-type (func (param "x" s32) (result s32)))
          (core module $m
            (func (export "run") (param i32) (result i32) local.get 0))
          (core instance $i (instantiate $m))
          (alias core export $i "run" (core func $run))
          (func (export "run") (type $run-type) (canon lift (core func $run))))"#,
    )
    .unwrap();
    let artifacts = extract_first_component_core_program_artifacts(&component).unwrap();
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(1, entry_template());
    let rejection = bindings
        .validate_against_program(&artifacts.tables)
        .unwrap_err();
    assert!(rejection.to_string().contains("i32 local 0 high lane"), "{rejection}");
    let rejection = neo_wasm::collect_wasmtime_component_run_with_linker_and_args(
        &component,
        &bindings,
        "run",
        &[ComponentVal::S32(7)],
        |_| Ok(()),
    )
    .unwrap_err();
    assert!(rejection.to_string().contains("i32 local 0 high lane"), "{rejection}");
}

#[test]
fn i64_export_input_high_slot_remains_valid() {
    let wasm = wat::parse_str(r#"(module (func (export "run") (param i64) (result i64) local.get 0))"#).unwrap();
    let artifacts = extract_wasm_program_artifacts(&wasm).unwrap();
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(1, entry_template());
    bindings
        .validate_against_program(&artifacts.tables)
        .unwrap();
}
