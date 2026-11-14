mod common;

use neo_application::MemoryPortKind;
use neo_math::F;
use neo_wasm::{build_wasm_relation_layout, layout::*, WasmMemoryId, WasmOpcode};
use p3_field::PrimeCharacteristicRing;

fn program(index: u32, callee: &str) -> String {
    format!(
        r#"(module
      (type $t (func (result i32)))
      (table 2 funcref) (elem (i32.const 0) $f)
      (func $f {callee})
      (func (export "main") (result i32) i32.const {index} call_indirect (type $t)))"#
    )
}

fn rejects_forged_replacement(run: &common::CheckedWasmRun, i: usize) {
    let z = &run.witnesses[i];
    assert_eq!(z[COL_STACK_RMW], F::ONE);
    let rmw_ports = build_wasm_relation_layout()
        .auxiliary
        .memory
        .entries()
        .iter()
        .filter(|m| m.id == WasmMemoryId::Stack)
        .flat_map(|m| &m.ports)
        .filter(|p| neo_wasm::nebula::is_active(z, p.activation).unwrap())
        .filter(|p| {
            matches!(
                p.kind,
                MemoryPortKind::Write {
                    value_before_column: Some(_)
                }
            )
        })
        .count();
    assert_eq!(rmw_ports, 2);
    let mut preload = neo_wasm::preload_from_program_artifacts(&run.artifacts);
    neo_wasm::memory_semantics::preload_host_event_tables(
        &mut preload,
        &neo_wasm::host_event_bindings::HostEventBindings::import_free(
            run.trace[0].state_before.host_events.turn_export_fref,
        ),
        &run.artifacts,
    );
    neo_wasm::sanity_check_memory_rows(build_wasm_relation_layout(), &run.witnesses, &preload).unwrap();
    for prior in [COL_STACK_READ_VALUE_LO[0], COL_STACK_READ_VALUE_HI[0]] {
        let mut rows = run.witnesses.clone();
        rows[i][prior] += F::ONE;
        assert!(neo_wasm::sanity_check_memory_rows(build_wasm_relation_layout(), &rows, &preload).is_err());
    }
    for gate in COL_STACK_SEPARATE {
        let mut z = run.witnesses[i].clone();
        z[gate] = F::ONE;
        common::assert_rejected(&z, "replacement cannot enable a second stack access");
    }
    let mut z = run.witnesses[i].clone();
    z[COL_STACK_WRITE0_ADDR_LO] += F::from_u64(2);
    common::assert_rejected(&z, "read-modify-write must use the operand address");
    let mut z = run.witnesses[i].clone();
    z[COL_STACK_RMW] = F::ZERO;
    for (separate, original) in COL_STACK_SEPARATE.into_iter().zip([
        COL_STACK_READ_ACTIVE[0],
        COL_STACK_WRITE0_ACTIVE,
        COL_STACK_WRITE0_HI_ACTIVE,
    ]) {
        z[separate] = z[original];
    }
    common::assert_rejected(&z, "replacement cannot switch to unbound writes");
}

#[test]
fn resolve_binds_both_prior_limbs_on_success_and_traps() {
    for source in [
        program(0, "(result i32) i32.const 7"),
        program(1, "(result i32) i32.const 7"),
        program(2, "(result i32) i32.const 7"),
        program(0, "(result i64) i64.const 7"),
    ] {
        let run = common::checked_main(&source);
        let i = run
            .trace
            .iter()
            .position(|r| r.opcode == WasmOpcode::CallIndirectResolve)
            .unwrap();
        let layout = build_wasm_relation_layout();
        let active: Vec<_> = layout
            .auxiliary
            .memory
            .entries()
            .iter()
            .flat_map(|m| m.ports.iter().map(move |p| (m.id, p)))
            .filter(|(_, p)| neo_wasm::nebula::is_active(&run.witnesses[i], p.activation).unwrap())
            .collect();
        let stack: Vec<_> = active
            .iter()
            .filter(|(id, _)| *id == WasmMemoryId::Stack)
            .collect();
        assert_eq!(stack.len(), 2);
        assert!(stack.iter().all(|(_, p)| matches!(
            p.kind,
            MemoryPortKind::Write {
                value_before_column: Some(_)
            }
        )));
        if !run.trace[i].state_after.trapped {
            assert_eq!(active.len(), 8);
        }
        rejects_forged_replacement(&run, i);
    }
}

#[test]
fn fixed_arity_replacements_save_two_accesses_and_bind_both_prior_limbs() {
    for (source, opcode, expected_accesses) in [
        (
            "(module (func (export \"main\") (result i32) i32.const 7 i32.clz))",
            WasmOpcode::I32Clz,
            3,
        ),
        (
            "(module (func (export \"main\") (result i32) i32.const 7 i32.const 3 i32.add))",
            WasmOpcode::I32Add,
            5,
        ),
        (
            "(module (func (export \"main\") (result i64) i64.const 0x123456789abcdef0 i64.const 0xfedcba9876543210 i32.const 0 select))",
            WasmOpcode::Select,
            7,
        ),
        (
            "(module (memory 1) (func (export \"main\") (result i64) i32.const 1 i64.load))",
            WasmOpcode::I64Load,
            7,
        ),
        (
            "(module (memory 1 2) (func (export \"main\") (result i32) i32.const 1 memory.grow))",
            WasmOpcode::MemoryGrow,
            3,
        ),
        (
            "(module (table 1 funcref) (func (export \"main\") i32.const 0 table.get drop))",
            WasmOpcode::TableGet,
            6,
        ),
        (
            "(module (func (export \"main\") (result i64) (local i64) i64.const 0x123456789abcdef0 local.tee 0 drop local.get 0))",
            WasmOpcode::LocalTee,
            6,
        ),
    ] {
        let run = common::checked_main(source);
        let i = run.trace.iter().position(|r| r.opcode == opcode).unwrap();
        let z = &run.witnesses[i];
        assert_eq!(z[COL_STACK_READ_ADDR_LO[0]], z[COL_STACK_WRITE0_ADDR_LO], "{opcode:?}");
        assert_eq!(z[COL_STACK_READ_ADDR_HI[0]], z[COL_STACK_WRITE0_ADDR_HI], "{opcode:?}");
        let accesses = build_wasm_relation_layout()
            .auxiliary
            .memory
            .entries()
            .iter()
            .flat_map(|m| &m.ports)
            .filter(|p| neo_wasm::nebula::is_active(z, p.activation).unwrap())
            .count();
        assert_eq!(accesses, expected_accesses, "{opcode:?}");
        rejects_forged_replacement(&run, i);
    }
}

#[test]
fn trapping_replacements_still_bind_the_old_stack_cell() {
    for source in [
        "(module (func (export \"main\") (result i32) i32.const 7 i32.const 0 i32.div_u))",
        "(module (func (export \"main\") (result i64) i64.const -9223372036854775808 i64.const -1 i64.div_s))",
        "(module (memory 1) (func (export \"main\") (result i64) i32.const 65535 i64.load))",
        "(module (table 1 funcref) (func (export \"main\") i32.const 1 table.get drop))",
    ] {
        let run = common::checked_main(source);
        let i = run.trace.len() - 1;
        assert!(run.trace[i].state_after.trapped);
        rejects_forged_replacement(&run, i);
    }
}
