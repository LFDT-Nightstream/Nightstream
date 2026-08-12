use neo_math::F;
use neo_wasm::layout::{
    COL_CALL_STACK_RETURN_PC_VALUE, COL_CALL_TARGET_METADATA, COL_CURRENT_FUNCTION_NUM_LOCALS,
    COL_CURRENT_FUNCTION_REF, COL_EXPECTED_TYPE_ID, COL_FUNCTION_TYPE_ID, COL_LINEAR_MEM_IMM_OFFSET, COL_LOCAL_INDEX,
    COL_OPCODE_CODE, COL_STACK_READ_VALUE_HI, COL_STACK_WRITE0_VALUE_HI, COL_TABLE_INDEX, COL_TABLE_SIZE,
    COL_TABLE_VALUE, PC_ROM_CALL_RETURN_CHOICE,
};
use neo_wasm::{
    build_wasm_relation_layout, collect_wasmtime_steps, extract_wasm_program_artifacts, preload_from_program_artifacts,
    sanity_check_memory_rows, traces_from_wasmtime_steps, witness_builder::build_witness_vector, WasmMemoryPreload,
    WasmOpcode,
};
use p3_field::PrimeCharacteristicRing;

fn witness_run(wat_src: &str) -> (Vec<neo_wasm::WasmVmStep>, Vec<Vec<F>>, WasmMemoryPreload) {
    let wasm = wat::parse_str(wat_src).expect("wat");
    let artifacts = extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = collect_wasmtime_steps(&wasm, "run", &[]).expect("trace");
    let trace = traces_from_wasmtime_steps(&run.steps).expect("normalize");
    let witnesses = trace.iter().map(build_witness_vector).collect();
    let mut preload = preload_from_program_artifacts(&artifacts);
    // Import-free traces run under the canonical single-shot grammar; the
    // exit latch reads its (biased) export count cells.
    let export_fref = trace[0].state_before.grammar.turn_export_fref;
    neo_wasm::memory_semantics::preload_grammar_tables(
        &mut preload,
        &neo_wasm::event_grammar::HostEventGrammar::import_free(export_fref),
    );
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
    let layout = build_wasm_relation_layout();
    sanity_check_memory_rows(layout, &witnesses, &preload).expect("memory sanity");
}

#[test]
fn memory_semantics_rejects_tampered_function_call_metadata() {
    let (trace, mut witnesses, preload) = witness_run(
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
    let call_idx = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::Call)
        .expect("call row");
    witnesses[call_idx][COL_CALL_TARGET_METADATA] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("tampered metadata must fail");
    assert!(
        err.contains("memory `function_call_metadata` ROM mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn memory_semantics_rejects_tampered_i64_stack_high_limb() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func (export "run") (result i64)
                i64.const 0x1_0000_0000
                i64.const 1
                i64.add)
        )"#,
    );
    let add_idx = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::I64Add)
        .expect("i64.add row");

    witnesses[add_idx][COL_STACK_READ_VALUE_HI[0]] = F::from_u64(9);

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .err()
        .expect("must reject tampered i64 stack high limb");
    assert!(err.contains("memory `stack`"), "unexpected error: {err}");
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
    preload.remove("pc_rom", &[call_row.state_before.pc as u32, call_row.control_choice]);
    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("missing edge must fail");
    assert!(err.contains("memory `pc_rom` ROM read before initialization"));
}

#[test]
fn memory_semantics_rejects_wrong_program_opcode() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func (export "run") (result i32)
                i32.const 5))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.row_kind.is_program())
        .expect("program row");
    witnesses[row_index][COL_OPCODE_CODE] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong opcode must fail");
    assert!(err.contains("memory `program_opcodes` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_wrong_program_local_index() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func (export "run") (result i32)
                (local i32)
                local.get 0))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::LocalGet)
        .expect("local.get row");
    witnesses[row_index][COL_LOCAL_INDEX] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong local index must fail");
    assert!(err.contains("memory `program_local_indices` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_wrong_program_memory_offset() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (memory 1)
            (func (export "run") (result i32)
                i32.const 0
                i32.load offset=8))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::I32Load)
        .expect("i32.load row");
    witnesses[row_index][COL_LINEAR_MEM_IMM_OFFSET] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong memory offset must fail");
    assert!(err.contains("memory `program_memory_offsets` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_wrong_program_i64_const_high_limb() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (func (export "run") (result i64)
                i64.const 0x0000_0001_0000_0002))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::I64Const)
        .expect("i64.const row");
    witnesses[row_index][COL_STACK_WRITE0_VALUE_HI] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("wrong i64 const hi must fail");
    assert!(err.contains("memory `program_i64_const_values_hi` ROM mismatch"));
}

#[test]
fn memory_semantics_rejects_wrong_program_call_indirect_expected_type() {
    let (trace, mut witnesses, preload) = witness_run(
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
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    witnesses[row_index][COL_EXPECTED_TYPE_ID] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("wrong call_indirect expected type must fail");
    assert!(err.contains("memory `program_call_indirect_expected_type_ids` ROM mismatch"));
}

#[test]
fn memory_semantics_reject_missing_call_return_pc_rom_edge() {
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
        .find(|row| row.call_stack_push.is_some())
        .expect("call row");
    preload.remove(
        "pc_rom",
        &[call_row.state_before.pc as u32, PC_ROM_CALL_RETURN_CHOICE as u32],
    );
    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("missing return edge must fail");
    assert!(err.contains("memory `pc_rom` ROM read before initialization"));
}

#[test]
fn memory_semantics_rejects_return_pc_not_read_from_call_stack() {
    let (trace, mut witnesses, preload) = witness_run(
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
    let return_index = trace
        .iter()
        .position(|row| row.call_stack_pop.is_some())
        .expect("non-final return row");
    witnesses[return_index][COL_CALL_STACK_RETURN_PC_VALUE] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("broken return pc read must fail");
    assert!(err.contains("memory `call_stack_return_pcs` read mismatch"));
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

    let layout = build_wasm_relation_layout();
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

    let layout = build_wasm_relation_layout();
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
    let function_ref = row.table_value.expect("call_indirect function ref");
    preload.remove("function_entries", &[function_ref]);
    preload.insert(
        "function_entries",
        vec![function_ref],
        row.state_after.pc.saturating_add(1) as u32,
    );
    let layout = build_wasm_relation_layout();
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
    preload.remove("pc_rom", &[row.state_before.pc as u32, row.control_choice]);
    let layout = build_wasm_relation_layout();
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
    preload.remove("pc_rom", &[row.state_before.pc as u32, row.control_choice]);
    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("missing br_if edge must fail");
    assert!(err.contains("memory `pc_rom` ROM read before initialization"));
}

/// Table with a non-zero element-segment offset plus a `table.size` read;
/// exercises both authoritative init sets (`tables_init` / `table_sizes_init`).
const TABLE_INIT_WAT: &str = r#"(module
    (type $t (func (param i32) (result i32)))
    (func $add_one (type $t)
        local.get 0
        i32.const 1
        i32.add)
    (table 4 funcref)
    (elem (i32.const 2) func $add_one)
    (func (export "run") (result i32)
        i32.const 5
        i32.const 2
        call_indirect (type $t)
        table.size
        i32.add))
"#;

#[test]
fn memory_semantics_accept_element_segment_table_init() {
    let (_, witnesses, preload) = witness_run(TABLE_INIT_WAT);
    let layout = build_wasm_relation_layout();
    sanity_check_memory_rows(layout, &witnesses, &preload).expect("memory sanity");
}

#[test]
fn memory_semantics_rejects_tampered_table_funcref() {
    let (trace, mut witnesses, preload) = witness_run(TABLE_INIT_WAT);
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    witnesses[row_index][COL_TABLE_VALUE] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("tampered table funcref must fail");
    assert!(err.contains("memory `tables` read mismatch"), "unexpected error: {err}");
}

#[test]
fn memory_semantics_rejects_nonnull_read_of_uninitialized_table_entry() {
    let (trace, mut witnesses, preload) = witness_run(TABLE_INIT_WAT);
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    // Redirect the read to index 0 — in bounds but outside the element
    // segment, so it must be a null funcref (0) at instantiation. Claiming a
    // live funcref there was exactly what `FirstReadDefines` allowed.
    witnesses[row_index][COL_TABLE_INDEX] = F::ZERO;

    let layout = build_wasm_relation_layout();
    let err =
        sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("non-null uninitialized read must fail");
    assert!(err.contains("memory `tables`"), "unexpected error: {err}");
}

#[test]
fn memory_semantics_rejects_tampered_table_size() {
    let (trace, mut witnesses, preload) = witness_run(TABLE_INIT_WAT);
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::TableSize)
        .expect("table.size row");
    witnesses[row_index][COL_TABLE_SIZE] += F::ONE;

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("tampered table size must fail");
    assert!(
        err.contains("memory `table_sizes` read mismatch"),
        "unexpected error: {err}"
    );
}

/// On a type-mismatch trap row the `function_types` ROM read stays active
/// (the entry is non-null), so forging the callee type id to match the
/// expected type — hiding the mismatch from the CCS zero-test — must fail
/// against the ROM.
#[test]
fn memory_semantics_rejects_forged_callee_type_on_mismatch_trap() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (type $u (func (result i64)))
            (func $wide (type $u)
                i64.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $wide)
            (func (export "run") (result i32)
                i32.const 5
                i32.const 0
                call_indirect (type $t)))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    assert!(trace[row_index].state_after.trapped);
    witnesses[row_index][COL_FUNCTION_TYPE_ID] = witnesses[row_index][COL_EXPECTED_TYPE_ID];

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("forged callee type must fail");
    assert!(
        err.contains("memory `function_types` ROM mismatch"),
        "unexpected error: {err}"
    );
}

/// The OOB-index trap compares the table index against the table size, so the
/// size must be authoritative on a `call_indirect` row. Forging it (to claim
/// an OOB index is in bounds) must fail against the `table_sizes` read, which
/// is now gated on for call_indirect, not just `table.size`.
#[test]
fn memory_semantics_rejects_forged_table_size_on_oob_trap() {
    let (trace, mut witnesses, preload) = witness_run(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (func $add_one (type $t)
                local.get 0
                i32.const 1
                i32.add)
            (table 1 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "run") (result i32)
                i32.const 9
                i32.const 5
                call_indirect (type $t)))
        "#,
    );
    let row_index = trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::CallIndirect)
        .expect("call_indirect row");
    assert!(trace[row_index].state_after.trapped);
    // Claim the table is large enough to make index 5 in bounds.
    witnesses[row_index][COL_TABLE_SIZE] = F::from_u64(6);

    let layout = build_wasm_relation_layout();
    let err = sanity_check_memory_rows(layout, &witnesses, &preload).expect_err("forged table size must fail");
    assert!(
        err.contains("memory `table_sizes` read mismatch"),
        "unexpected error: {err}"
    );
}
