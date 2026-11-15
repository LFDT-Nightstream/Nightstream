mod common;

use neo_math::F;
use neo_wasm::layout::{COL_EXPECTED_TYPE_ID, COL_TABLE_DESCRIPTOR, COL_TABLE_ID};
use neo_wasm::{build_witness_vector, WasmOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn table_operations_share_the_descriptor_with_indirect_calls() {
    let run = common::checked_main(
        r#"(module
          (type $t (func (result i32)))
          (table $a 1 funcref) (table $b 1 funcref)
          (func $f (type $t) i32.const 7)
          (elem (table $b) (i32.const 0) func $f)
          (func (export "main") (result i32)
            i32.const 0 i32.const 0 table.get $b table.set $a
            table.size $b drop
            i32.const 0 call_indirect $a (type $t)))"#,
    );
    let preload = neo_wasm::preload_from_program_artifacts(&run.artifacts);
    for opcode in [
        WasmOpcode::TableGet,
        WasmOpcode::TableSet,
        WasmOpcode::TableSize,
        WasmOpcode::CallIndirectResolve,
    ] {
        let i = run
            .trace
            .iter()
            .position(|row| row.opcode == opcode)
            .unwrap();
        let row = &run.trace[i];
        let table = row.table_id.unwrap();
        let expected = row.expected_type_id.unwrap_or(0);
        let descriptor = table + 256 * expected;
        assert_eq!(run.witnesses[i][COL_TABLE_DESCRIPTOR], F::from_u32(descriptor));
        assert!(preload.entries().into_iter().any(|(id, address, value)| {
            id == neo_wasm::WasmMemoryId::ProgramTableDescriptor
                && address == vec![row.state_before.pc as u32]
                && value == descriptor
        }));
        if opcode == WasmOpcode::CallIndirectResolve {
            assert_ne!(expected, 0);
        } else {
            assert_eq!(expected, 0);
        }
        let mut forged = run.witnesses[i].clone();
        forged[COL_TABLE_DESCRIPTOR] += F::ONE;
        common::assert_rejected(&forged, "the descriptor binds both decoded fields");
    }
}

#[test]
fn table_descriptor_bounds_are_checked_by_builder_and_circuit() {
    let run = common::checked_main("(module (func (export \"main\") nop))");
    let mut row = run
        .trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::Nop)
        .unwrap()
        .clone();
    for (table, expected) in [(0, 0), (255, 0), (0, (1 << 24) - 1), (255, (1 << 24) - 1)] {
        row.table_id = Some(table);
        row.expected_type_id = Some(expected);
        let witness = build_witness_vector(&row).unwrap();
        common::assert_satisfied(&witness, "8/24 descriptor endpoints");
    }
    for (table, expected, message) in [
        (256, 0, "table id 256 exceeds the 8-bit"),
        (0, 1 << 24, "expected type id 16777216 exceeds the 24-bit"),
    ] {
        row.table_id = Some(table);
        row.expected_type_id = Some(expected);
        let error = build_witness_vector(&row).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
        row.table_id = Some(0);
        row.expected_type_id = Some(0);
        // Rocq `TableDescriptor.full_table_descriptor`: the packing equation alone permits oversized fields.
        let mut forged = build_witness_vector(&row).unwrap();
        forged[COL_TABLE_ID] = F::from_u32(table);
        forged[COL_EXPECTED_TYPE_ID] = F::from_u32(expected);
        forged[COL_TABLE_DESCRIPTOR] = F::from_u64(u64::from(table) + 256 * u64::from(expected));
        neo_wasm::write_range_check_bits(&mut forged);
        common::assert_rejected(
            &forged,
            "decoded fields cannot carry into each other or overflow the word",
        );
    }
}
