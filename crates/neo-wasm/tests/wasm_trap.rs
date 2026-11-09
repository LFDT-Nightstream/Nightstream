//! Trapped executions as a provable terminal state.
//!
//! Modeled trap causes (`unreachable`, div/rem by zero, signed division
//! overflow, indirect-call OOB index / null entry / callee type mismatch,
//! table access OOB (SPEC.md section 7.7), and linear-memory load/store OOB)
//! end the trace at the faulting row with
//! `trapped: true` and no captured output. Unmodeled causes still stay loud
//! trace-collection errors.

mod common;

use neo_wasm::host_event_bindings::{EventBlock, ExportTemplate, HostEventBindings, Limb, SlotBinding};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{
    build_wasm_relation_layout, collect_wasmtime_steps, preload_from_program_artifacts,
    traces_from_wasmtime_steps_with_host_events, WasmOpcode,
};

#[test]
fn unreachable_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7
                drop
                unreachable))"#,
    );

    // A trapped run has no reference results and no captured output.
    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::Unreachable);
    assert!(last.state_after.trapped);
    assert!(last.state_after.activation_finished);
    assert!(!last.state_after.output.enabled);
}

#[test]
fn div_by_zero_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7
                i32.const 0
                i32.div_u))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I32DivU);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
}

#[test]
fn i64_rem_by_zero_traps_on_the_wide_divisor() {
    // The divisor zero-test sums both limbs, so an i64 divisor with a zero
    // low limb but nonzero high limb must NOT trap, and a fully-zero one must.
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 9
                i64.const 0
                i64.rem_u
                i32.wrap_i64))"#,
    );
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I64RemU);
    assert!(last.state_after.trapped);

    let not_trapping = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const 9
                i64.const 0x100000000
                i64.rem_u
                i32.wrap_i64))"#,
    );
    assert_eq!(not_trapping.run.results.as_slice(), &["9".to_string()]);
    assert!(!common::final_state(&not_trapping.trace).trapped);
}

#[test]
fn i32_signed_division_overflow_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const -2147483648
                i32.const -1
                i32.div_s))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I32DivS);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
}

#[test]
fn i64_signed_division_overflow_traps_only_on_exact_min_and_neg1() {
    // The dividend-is-MIN test packs both limbs, so i64::MIN / -1 must trap
    // while the near-miss (i64::MIN + 1) / -1 must divide cleanly.
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const -9223372036854775808
                i64.const -1
                i64.div_s
                i32.wrap_i64))"#,
    );
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I64DivS);
    assert!(last.state_after.trapped);

    let not_trapping = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i64.const -9223372036854775807
                i64.const -1
                i64.div_s
                i32.wrap_i64))"#,
    );
    // (MIN + 1) / -1 = i64::MAX, wrapped to i32 = -1.
    assert_eq!(not_trapping.run.results.as_slice(), &["-1".to_string()]);
    assert!(!common::final_state(&not_trapping.trace).trapped);
}

#[test]
fn linear_memory_oob_load_trap_is_a_provable_terminal_state() {
    // The memory is one page (65536 bytes); loading at byte 1_000_000 is far
    // out of bounds and traps before any byte is read.
    let checked = common::checked_main(
        r#"(module
            (memory 1)
            (func (export "main") (result i32)
                i32.const 1000000
                i32.load))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I32Load);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
}

#[test]
fn linear_memory_oob_store_trap_is_a_provable_terminal_state() {
    // Storing at byte 1_000_000 is out of bounds and traps before any byte is
    // written. Exercises the store lane de-gating (store_live = 0).
    let checked = common::checked_main(
        r#"(module
            (memory 1)
            (func (export "main") (result i32)
                i32.const 1000000
                i32.const 42
                i32.store
                i32.const 0))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::I32Store);
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
}

#[test]
fn call_indirect_null_entry_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (func $add_one (type $t)
                local.get 0
                i32.const 1
                i32.add)
            (table 2 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "main") (result i32)
                i32.const 5
                i32.const 1
                call_indirect (type $t)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::CallIndirectResolve);
    assert_eq!(last.table_value, Some(0), "slot 1 is a null funcref");
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
    assert!(last.call_stack_push.is_none(), "a trapping resolve step never calls");
}

#[test]
fn call_indirect_oob_index_trap_is_a_provable_terminal_state() {
    // The table holds one slot; calling through index 5 is out of bounds and
    // traps before any entry is read.
    let checked = common::checked_main(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (func $add_one (type $t)
                local.get 0
                i32.const 1
                i32.add)
            (table 1 funcref)
            (elem (i32.const 0) func $add_one)
            (func (export "main") (result i32)
                i32.const 9
                i32.const 5
                call_indirect (type $t)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::CallIndirectResolve);
    assert_eq!(last.table_index, Some(5));
    assert_eq!(last.table_size, Some(1), "index 5 is past the one-slot table");
    assert_eq!(last.table_value, None, "no entry is read on an OOB index");
    assert!(last.state_after.trapped);
    assert!(!last.state_after.output.enabled);
    assert!(last.call_stack_push.is_none(), "a trapping resolve step never calls");
}

fn assert_hidden_table_oob_is_rejected(wat: &str, export_fref: u32, opcode: WasmOpcode, index_from_stack_end: usize) {
    let wasm = wat::parse_str(wat).expect("valid WAT");
    let mut slots = [SlotBinding::Const(0); 8];
    slots[0] = SlotBinding::InputLocal {
        input: 0,
        local: 0,
        limb: Limb::Lo,
    };
    let mut bindings = HostEventBindings::default();
    bindings.exports.insert(
        export_fref,
        ExportTemplate {
            entry: vec![EventBlock {
                block: slots,
                absorb: true,
            }],
            exit: vec![],
            entry_input_count: 1,
        },
    );

    let honest = collect_wasmtime_steps(&wasm, &bindings, "main", &[0]).expect("in-bounds reference run");
    assert_eq!(honest.results, ["7"]);
    let native = collect_wasmtime_steps(&wasm, &bindings, "main", &[5]).expect("the table trap is modeled");
    assert!(native.results.is_empty());
    assert_eq!(native.steps.last().and_then(|row| row.opcode_decoded), Some(opcode));

    let mut forged_steps = honest.steps.clone();
    let local_get = forged_steps
        .iter_mut()
        .find(|row| row.opcode_decoded == Some(WasmOpcode::LocalGet))
        .expect("local.get row");
    local_get.locals_words[0].0 = 5;
    let table_op = forged_steps
        .iter_mut()
        .find(|row| row.opcode_decoded == Some(opcode))
        .expect("table operation row");
    let index_position = table_op
        .operand_stack_words
        .len()
        .checked_sub(index_from_stack_end)
        .expect("table index operand position");
    table_op.operand_stack_words[index_position] = 5;
    table_op.table_index = Some(5);
    table_op.table_size = Some(6);

    let forged = traces_from_wasmtime_steps_with_host_events(&forged_steps, honest.artifacts(), Default::default())
        .expect("self-consistent forged trace");
    let table_op = forged
        .iter()
        .position(|row| row.opcode == opcode)
        .expect("normalized table operation row");
    let mut hidden = forged[table_op].clone();
    hidden.table_size = Some(1);
    common::assert_rejected(
        &build_witness_vector(&hidden).expect("witness"),
        "an out-of-bounds table operation that does not trap",
    );

    // A prover also cannot hide the OOB index by increasing the claimed
    // table size. The TableSize RAM read binds the real declared size.
    common::ccs_check_trace(&forged);
    let layout = build_wasm_relation_layout();
    let witnesses: Vec<_> = forged
        .iter()
        .map(|row| build_witness_vector(row).expect("witness"))
        .collect();
    for witness in &witnesses {
        neo_wasm::sanity_check_lookup_row(&layout.auxiliary, witness).expect("lookup semantics");
    }
    neo_application::check_continuity_rows(&layout.auxiliary.continuity, &witnesses).expect("forged trace continuity");
    let mut preload = preload_from_program_artifacts(honest.artifacts());
    neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings, honest.artifacts());
    let error = neo_wasm::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect_err("the verifier-owned table size must reject the forged bound");
    assert!(error.contains("table_size"), "unexpected rejection: {error}");
}

#[test]
fn table_get_and_set_oob_cannot_be_hidden_as_ram_accesses() {
    assert_hidden_table_oob_is_rejected(
        r#"(module
            (table 1 funcref)
            (func (export "main") (param i32) (result i32)
                local.get 0
                table.get 0
                drop
                i32.const 7))"#,
        1,
        WasmOpcode::TableGet,
        1,
    );
    assert_hidden_table_oob_is_rejected(
        r#"(module
            (type $t (func))
            (func $f (type $t))
            (elem declare funcref (ref.func $f))
            (table 1 funcref)
            (func (export "main") (param i32) (result i32)
                local.get 0
                ref.func $f
                table.set 0
                i32.const 7))"#,
        2,
        WasmOpcode::TableSet,
        2,
    );
}

#[test]
fn call_indirect_type_mismatch_trap_is_a_provable_terminal_state() {
    let checked = common::checked_main(
        r#"(module
            (type $t (func (param i32) (result i32)))
            (type $u (func (result i64)))
            (func $wide (type $u)
                i64.const 7)
            (table 1 funcref)
            (elem (i32.const 0) func $wide)
            (func (export "main") (result i32)
                i32.const 5
                i32.const 0
                call_indirect (type $t)))"#,
    );

    assert!(checked.run.results.is_empty());
    let last = checked.trace.last().expect("non-empty trace");
    assert_eq!(last.opcode, WasmOpcode::CallIndirectResolve);
    assert!(last.table_value.is_some_and(|funcref| funcref != 0));
    assert_ne!(last.function_type_id, last.expected_type_id);
    assert!(last.state_after.trapped);
    assert!(last.call_stack_push.is_none(), "a trapping resolve step never calls");
}
