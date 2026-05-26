use neo_math::F;
use neo_wasm::layout::{COL_CURRENT_FUNCTION_NUM_LOCALS, COL_CURRENT_FUNCTION_REF, COL_LOCALS_FBP_AFTER};
use neo_wasm::{
    build_wasm_lookup_binding_layout, collect_wasmtime_steps, preload_from_wasmtime_run, sanity_check_memory_rows,
    traces_from_wasmtime_steps, witness_builder::build_witness_vector, WasmMemoryPreload, WasmOpcode,
};
use p3_field::PrimeCharacteristicRing;

fn witness_run(wat_src: &str) -> (Vec<neo_wasm::WasmStepTrace>, Vec<Vec<F>>, WasmMemoryPreload) {
    let wasm = wat::parse_str(wat_src).expect("wat");
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let witnesses = trace.iter().map(build_witness_vector).collect();
    let preload = preload_from_wasmtime_run(&run, &run.initial_locals);
    (trace, witnesses, preload)
}

#[test]
fn memory_semantics_accept_real_direct_call_trace() {
    let (_, witnesses, preload) = witness_run(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    );
    let layout = build_wasm_lookup_binding_layout();
    sanity_check_memory_rows(layout, &witnesses, &preload).expect("memory sanity");
}

#[test]
fn memory_semantics_reject_missing_pc_rom_edge() {
    let (trace, witnesses, mut preload) = witness_run(
        r#"(module
            (func $add_one (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 5
                call $add_one))
        "#,
    );
    let call_row = trace
        .iter()
        .find(|row| matches!(row.opcode, WasmOpcode::Call))
        .expect("call row");
    preload.remove("pc_rom", &[call_row.pc_before, u64::from(call_row.control_choice)]);
    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("missing edge must fail");
    assert!(err.contains("memory `pc_rom` ROM read before initialization"));
}

#[test]
fn memory_semantics_rejects_broken_locals_fbp_cross_step_link() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func $sum2 (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add)
            (func (export "run") (result i32)
                i32.const 4
                i32.const 7
                call $sum2))
        "#,
    );
    let call_index = trace
        .iter()
        .position(|row| matches!(row.opcode, WasmOpcode::Call))
        .expect("call row");
    witnesses[call_index][COL_LOCALS_FBP_AFTER] += F::ONE;

    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("broken fbp link must fail");
    assert!(err.contains("cross-step link `locals_fbp_continuity`"));
}

#[test]
fn memory_semantics_rejects_wrong_current_function_local_count() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func (export "run") (result i32)
                (local i32 i32)
                i32.const 5))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.row_kind.is_program())
        .expect("program row");
    witnesses[row_index][COL_CURRENT_FUNCTION_NUM_LOCALS] += F::ONE;

    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong local count must fail");
    assert!(err.contains("memory `function_local_counts` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_wrong_current_function_ref() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func $same_shape (result i32)
                i32.const 7)
            (func (export "run") (result i32)
                i32.const 5))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| matches!(row.opcode, WasmOpcode::I32Const))
        .expect("program row");
    let wrong_existing_function_ref = if trace[row_index].current_function_ref == 1 {
        2
    } else {
        1
    };
    witnesses[row_index][COL_CURRENT_FUNCTION_REF] = F::from_u64(wrong_existing_function_ref);

    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong function ref must fail");
    assert!(err.contains("memory `pc_function_refs` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_wrong_call_indirect_target() {
    let (trace, witnesses, mut preload) = witness_run(
        r#"(module
            (type $t (func (result i32)))
            (func $f (type $t) (result i32)
                i32.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $f)
            (func (export "run") (result i32)
                i32.const 0
                call_indirect (type $t)))
        "#,
    );
    let row = trace
        .iter()
        .find(|row| matches!(row.opcode, WasmOpcode::CallIndirect))
        .expect("call_indirect row");
    let function_ref = u64::from(row.table_value.expect("call_indirect function ref"));
    preload.remove("function_entries", &[function_ref]);
    preload.insert("function_entries", vec![function_ref], row.pc_after.saturating_add(1));
    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong call_indirect target must fail");
    assert!(err.contains("memory `function_entries` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_missing_if_taken_edge() {
    let (trace, witnesses, mut preload) = witness_run(
        r#"(module
            (func (export "run") (result i32)
                i32.const 1
                if
                    i32.const 7
                    drop
                end
                i32.const 0))
        "#,
    );
    let row = trace
        .iter()
        .find(|row| matches!(row.opcode, WasmOpcode::If))
        .expect("if row");
    assert_eq!(row.control_choice, 1);
    preload.remove("pc_rom", &[row.pc_before, u64::from(row.control_choice)]);
    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("missing if edge must fail");
    assert!(err.contains("memory `pc_rom` ROM read before initialization"));
}

#[test]
fn memory_semantics_rejects_missing_br_if_taken_edge() {
    let (trace, witnesses, mut preload) = witness_run(
        r#"(module
            (func (export "run") (result i32)
                block
                    i32.const 1
                    br_if 0
                    unreachable
                end
                i32.const 0))
        "#,
    );
    let row = trace
        .iter()
        .find(|row| matches!(row.opcode, WasmOpcode::BrIf))
        .expect("br_if row");
    assert_eq!(row.control_choice, 1);
    preload.remove("pc_rom", &[row.pc_before, u64::from(row.control_choice)]);
    let layout = build_wasm_lookup_binding_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("missing br_if edge must fail");
    assert!(err.contains("memory `pc_rom` ROM read before initialization"));
}
