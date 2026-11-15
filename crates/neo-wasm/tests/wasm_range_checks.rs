mod common;

use neo_application::decomposition_bit_count;
use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::ccs::host_event_chain::{AUX_COLUMN_FAMILIES, AUX_WIDTH};
use neo_wasm::layout::{
    column_families, named_column_family, ColumnWidth, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_GLOBAL_INDEX,
    NAMED_COLUMN_COUNT,
};
use neo_wasm::range_check::range_checked_bit_columns;
use neo_wasm::{
    build_wasm_relation, write_range_check_bits, WasmOpcode, RANGE_BITS_REGION, RANGE_CHECKED_WITNESS_WIDTH,
};
use p3_field::PrimeCharacteristicRing;

fn expected_aux_bits() -> usize {
    column_families()
        .chain(AUX_COLUMN_FAMILIES.iter())
        .map(|family| decomposition_bit_count(family.width) * family.len)
        .sum()
}

#[test]
fn range_checked_width_bookkeeping() {
    assert_eq!(
        RANGE_CHECKED_WITNESS_WIDTH,
        NAMED_COLUMN_COUNT + AUX_WIDTH + expected_aux_bits()
    );

    let relation = build_wasm_relation().expect("valid WASM relation");
    assert_eq!(relation.r1cs().column_count(), RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(relation.r1cs().structure().m(), RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(relation.r1cs().catalog().len(), relation.r1cs().structure().n());
    assert!(relation
        .r1cs()
        .structure()
        .matrices()
        .iter()
        .all(|matrix| matrix.as_csc().is_some()));
    assert_eq!(relation.columns().column_count(), RANGE_CHECKED_WITNESS_WIDTH);

    let range_bits = relation
        .columns()
        .family_for_column(NAMED_COLUMN_COUNT + AUX_WIDTH)
        .expect("range-bit suffix must have registry metadata");
    assert_eq!(range_bits.region, RANGE_BITS_REGION);
    assert_eq!(range_bits.start, NAMED_COLUMN_COUNT + AUX_WIDTH);
    assert_eq!(range_bits.end(), RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(range_bits.width, ColumnWidth::Boolean);
}

#[test]
fn host_event_auxiliary_registry_preserves_the_existing_layout() {
    let mut next = NAMED_COLUMN_COUNT;
    for family in AUX_COLUMN_FAMILIES {
        assert_eq!(family.region, "host_event_chain_aux");
        assert_eq!(family.start, next, "{}", family.name);
        next = family.end();
    }
    assert_eq!(next, NAMED_COLUMN_COUNT + AUX_WIDTH);
}

#[test]
fn range_bit_lookup_exactly_partitions_the_auxiliary_suffix() {
    let mut next = NAMED_COLUMN_COUNT + AUX_WIDTH;

    for family in column_families().chain(AUX_COLUMN_FAMILIES.iter()) {
        let bit_count = decomposition_bit_count(family.width);

        for column in family.start..family.end() {
            if bit_count == 0 {
                assert_eq!(range_checked_bit_columns(column), None, "{}", family.name);
            } else {
                assert_eq!(
                    range_checked_bit_columns(column),
                    Some(next..next + bit_count),
                    "{}",
                    family.name
                );
                next += bit_count;
            }
        }
    }

    assert_eq!(next, RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(range_checked_bit_columns(NAMED_COLUMN_COUNT + AUX_WIDTH), None);
}

#[test]
fn packed_function_metadata_counts_are_byte_ranged() {
    // The packed ROM word is unpacked linearly. These byte bounds make that
    // decomposition unique: without them, `param += 256; result -= 1` would
    // preserve the authoritative packed value while changing call semantics.
    assert_eq!(
        named_column_family(COL_CALL_PARAM_COUNT).unwrap().width,
        ColumnWidth::Byte
    );
    assert_eq!(
        named_column_family(COL_CALL_RESULT_COUNT).unwrap().width,
        ColumnWidth::Byte
    );
}

/// An out-of-range value in a column no semantic row pins (the global-index
/// word on a row that does not access globals) must be rejected,
/// and the failing row must be one of that column's own range-check rows,
/// proving nothing else constrains the value.
#[test]
fn out_of_range_u32_is_rejected_by_the_column_range_row() {
    // The invariant under attack: the column is declared as a 32-bit value.
    assert_eq!(named_column_family(COL_GLOBAL_INDEX).unwrap().width, ColumnWidth::U32);

    let checked =
        common::checked_main(r#"(module (func (export "main") (result i32) i32.const 20 i32.const 22 i32.add))"#);
    let add_idx = checked
        .trace
        .iter()
        .position(|row| matches!(row.opcode, WasmOpcode::I32Add))
        .expect("i32.add row");
    let mut wit = checked.witnesses[add_idx].clone();

    wit[COL_GLOBAL_INDEX] = F::from_u64(1u64 << 32);
    write_range_check_bits(&mut wit);

    let relation = build_wasm_relation().expect("valid WASM relation");
    let m_in = relation.r1cs().public_input_count();
    let err = check_ccs_rowwise_zero(relation.r1cs().structure(), &wit[..m_in], &wit[m_in..])
        .expect_err("the range-checked CCS must reject an out-of-range global index");

    let detail = err.to_string();
    let row_idx = detail
        .split_once("row ")
        .and_then(|(_, rest)| rest.split_once(':'))
        .and_then(|(row, _)| row.parse::<usize>().ok())
        .unwrap_or_else(|| panic!("could not parse failing row from: {detail}"));
    let tag = relation.r1cs().catalog().rows()[row_idx].tag();
    assert_eq!(
        tag.label(),
        "COL_GLOBAL_INDEX",
        "rejection must come from the column's own range-check row, failed row {row_idx} is tagged {tag:?}"
    );
}

#[test]
fn out_of_range_implied_columns_are_rejected_by_other_rows() {
    use neo_wasm::layout::{
        COL_HOST_IMPORT_SCHEDULE_COUNTS, COL_LOCAL_ADDR_LO, COL_MAX_MEMORY_PAGES_AFTER, COL_OPCODE_CODE,
        COL_STACK_READS, COL_STACK_WRITES,
    };
    let checked =
        common::checked_main(r#"(module (func (export "main") (result i32) i32.const 20 i32.const 22 i32.add))"#);
    for column in [
        COL_OPCODE_CODE,
        COL_STACK_READS,
        COL_STACK_WRITES,
        COL_LOCAL_ADDR_LO,
        COL_MAX_MEMORY_PAGES_AFTER,
        COL_HOST_IMPORT_SCHEDULE_COUNTS,
    ] {
        let family = named_column_family(column).unwrap();
        assert_eq!(family.width, ColumnWidth::Implied(32), "{}", family.name);
        let mut witness = checked.witnesses[0].clone();
        witness[column] = F::from_u64(1u64 << 32);
        common::assert_rejected(&witness, &family.name);
    }
}

#[test]
fn byte_offset_is_bound_when_used() {
    use neo_wasm::layout::COL_LINEAR_MEM_BYTE_OFFSET;
    let checked =
        common::checked_main(r#"(module (memory 1) (func (export "main") (result i32) i32.const 1 i32.load))"#);
    let index = checked
        .trace
        .iter()
        .position(|row| row.opcode == WasmOpcode::I32Load)
        .unwrap();
    let mut witness = checked.witnesses[index].clone();
    witness[COL_LINEAR_MEM_BYTE_OFFSET] = -F::ONE;
    common::assert_rejected(&witness, "an active offset is bound by its byte selectors");
}

#[test]
fn global_values_and_table_index_are_bound_when_used() {
    use neo_wasm::layout::{COL_GLOBAL_VALUE, COL_GLOBAL_VALUE_HI, COL_TABLE_INDEX};
    let checked = common::checked_main(
        r#"(module
          (global $g (mut i64) (i64.const 0))
          (table 1 funcref)
          (func (export "main") (result i64)
            i64.const 7 global.set $g
            i32.const 0 i32.const 0 table.get 0 table.set 0
            i32.const 0 table.get 0 drop
            global.get $g))"#,
    );
    for (column, opcodes) in [
        (COL_GLOBAL_VALUE, [WasmOpcode::GlobalGet, WasmOpcode::GlobalSet]),
        (COL_GLOBAL_VALUE_HI, [WasmOpcode::GlobalGet, WasmOpcode::GlobalSet]),
        (COL_TABLE_INDEX, [WasmOpcode::TableGet, WasmOpcode::TableSet]),
    ] {
        for opcode in opcodes {
            let index = checked
                .trace
                .iter()
                .position(|row| row.opcode == opcode)
                .unwrap();
            let mut witness = checked.witnesses[index].clone();
            witness[column] = -F::ONE;
            common::assert_rejected(&witness, "active advice equals the stack operand");
        }
    }
}

#[test]
fn call_stack_advice_is_bound_when_active_and_repairable_when_inactive() {
    use neo_wasm::layout::{
        COL_CALL_STACK_ADDR, COL_CALL_STACK_CALLER_FBP_VALUE, COL_CALL_STACK_CALLER_SP_BASE_VALUE,
        COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_PUSH_PRESENT,
    };
    let checked = common::checked_main(
        r#"(module
          (func $callee (param i32) (result i32) local.get 0)
          (func (export "main") (result i32) i32.const 7 call $callee))"#,
    );
    let push = checked
        .witnesses
        .iter()
        .position(|w| w[COL_CALL_STACK_PUSH_PRESENT] == F::ONE)
        .unwrap();
    let pop = checked
        .witnesses
        .iter()
        .position(|w| w[COL_CALL_STACK_POP_PRESENT] == F::ONE)
        .unwrap();
    let inactive = checked
        .witnesses
        .iter()
        .position(|w| w[COL_CALL_STACK_PUSH_PRESENT] == F::ZERO && w[COL_CALL_STACK_POP_PRESENT] == F::ZERO)
        .unwrap();
    for column in [
        COL_CALL_STACK_ADDR,
        COL_CALL_STACK_CALLER_FBP_VALUE,
        COL_CALL_STACK_CALLER_SP_BASE_VALUE,
    ] {
        for index in [push, pop] {
            let mut witness = checked.witnesses[index].clone();
            witness[column] = -F::ONE;
            common::assert_rejected(&witness, "active call-stack advice equals a bounded state word");
        }
        let mut forged = checked.witnesses.clone();
        forged[inactive][column] = -F::ONE;
        common::assert_satisfied(&forged[inactive], "inactive call-stack advice has no local effect");
        let layout = neo_wasm::build_wasm_relation_layout();
        let mut preload = neo_wasm::preload_from_program_artifacts(&checked.artifacts);
        let bindings = neo_wasm::host_event_bindings::HostEventBindings::import_free(
            checked.trace[0].state_before.host_events.turn_export_fref,
        );
        neo_wasm::memory_semantics::preload_host_event_tables(&mut preload, &bindings, &checked.artifacts);
        neo_wasm::sanity_check_memory_rows(layout, &forged, &preload).unwrap();
        forged[inactive][column] = F::ZERO;
        common::assert_satisfied(&forged[inactive], "zero repairs the inactive word");
        neo_wasm::sanity_check_memory_rows(layout, &forged, &preload).unwrap();
    }
}

#[test]
fn memory_bounds_reject_a_return_pc_that_the_row_alone_accepts() {
    use neo_wasm::layout::{COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_RETURN_PC_VALUE, COL_EDGE_ADDR, COL_PC_BEFORE};
    let checked = common::checked_main(r#"(module (func $callee) (func (export "main") call $callee))"#);
    let index = checked
        .witnesses
        .iter()
        .position(|w| w[COL_CALL_STACK_PUSH_PRESENT] == F::ONE)
        .unwrap();
    let mut witness = checked.witnesses[index].clone();
    witness[COL_PC_BEFORE] = F::from_u32(u32::MAX);
    witness[COL_EDGE_ADDR] = F::from_u32(u32::MAX);
    witness[COL_CALL_STACK_RETURN_PC_VALUE] = F::from_u64(1 << 32);
    write_range_check_bits(&mut witness);
    common::assert_satisfied(&witness, "pc + 1 needs the active memory-value bound");
    let layout = neo_wasm::build_wasm_relation_layout();
    let preload = neo_wasm::preload_from_program_artifacts(&checked.artifacts);
    let error = neo_wasm::sanity_check_memory_rows(layout, &[witness], &preload).unwrap_err();
    assert!(
        error.contains("call_stack_return_pc") && error.contains("does not fit u32"),
        "{error}"
    );

    let inactive = checked
        .witnesses
        .iter()
        .find(|w| {
            w[COL_CALL_STACK_PUSH_PRESENT] == F::ZERO && w[neo_wasm::layout::COL_CALL_STACK_POP_PRESENT] == F::ZERO
        })
        .unwrap();
    let mut witness = inactive.clone();
    witness[COL_CALL_STACK_RETURN_PC_VALUE] = -F::ONE;
    common::assert_satisfied(&witness, "inactive memory value is not unconditionally bounded");
    witness[COL_CALL_STACK_RETURN_PC_VALUE] = F::ZERO;
    common::assert_satisfied(&witness, "inactive return pc admits a bounded repair");
}
