mod common;

use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::layout::{
    ColumnWidth, COLUMN_SPECS, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_CALL_STACK_RETURN_PC_VALUE,
    NAMED_COLUMN_COUNT,
};
use neo_wasm::range_check::range_checked_bit_columns;
use neo_wasm::{write_range_check_bits, WasmOpcode, WasmVmSpec, RANGE_CHECKED_WITNESS_WIDTH};
use p3_field::PrimeCharacteristicRing;

fn expected_aux_bits() -> usize {
    COLUMN_SPECS
        .iter()
        .map(|spec| match spec.width {
            ColumnWidth::Boolean | ColumnWidth::Field => 0,
            ColumnWidth::Byte => 8,
            ColumnWidth::U32 => 32,
        })
        .sum()
}

#[test]
fn range_checked_width_bookkeeping() {
    assert_eq!(
        RANGE_CHECKED_WITNESS_WIDTH,
        NAMED_COLUMN_COUNT + neo_wasm::ccs::host_event_chain::AUX_WIDTH + expected_aux_bits()
    );

    let vm = WasmVmSpec::default();
    assert_eq!(vm.core_ccs_spec().witness_width, RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(vm.core_ccs_spec().structure.m, RANGE_CHECKED_WITNESS_WIDTH);
}

#[test]
fn range_bit_lookup_exactly_partitions_the_auxiliary_suffix() {
    let mut next = NAMED_COLUMN_COUNT + neo_wasm::ccs::host_event_chain::AUX_WIDTH;

    for spec in COLUMN_SPECS {
        let bit_count = match spec.width {
            ColumnWidth::Boolean | ColumnWidth::Field => 0,
            ColumnWidth::Byte => 8,
            ColumnWidth::U32 => 32,
        };

        if bit_count == 0 {
            assert_eq!(range_checked_bit_columns(spec.index), None, "{}", spec.name);
        } else {
            assert_eq!(
                range_checked_bit_columns(spec.index),
                Some(next..next + bit_count),
                "{}",
                spec.name
            );
            next += bit_count;
        }
    }

    assert_eq!(next, RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(range_checked_bit_columns(NAMED_COLUMN_COUNT), None);
}

#[test]
fn packed_function_metadata_counts_are_byte_ranged() {
    // The packed ROM word is unpacked linearly. These byte bounds make that
    // decomposition unique: without them, `param += 256; result -= 1` would
    // preserve the authoritative packed value while changing call semantics.
    assert_eq!(COLUMN_SPECS[COL_CALL_PARAM_COUNT].width, ColumnWidth::Byte);
    assert_eq!(COLUMN_SPECS[COL_CALL_RESULT_COUNT].width, ColumnWidth::Byte);
}

/// An out-of-range value in a column no semantic row pins (the call-stack
/// return-pc cell on a row that neither pushes nor pops) must be rejected,
/// and the failing row must be one of that column's own range-check rows,
/// proving nothing else constrains the value.
#[test]
fn out_of_range_u32_is_rejected_by_the_column_range_row() {
    // The invariant under attack: the column is declared as a 32-bit value.
    assert_eq!(COLUMN_SPECS[COL_CALL_STACK_RETURN_PC_VALUE].width, ColumnWidth::U32);

    let checked =
        common::checked_main(r#"(module (func (export "main") (result i32) i32.const 20 i32.const 22 i32.add))"#);
    let add_idx = checked
        .trace
        .iter()
        .position(|row| matches!(row.opcode, WasmOpcode::I32Add))
        .expect("i32.add row");
    let mut wit = checked.witnesses[add_idx].clone();

    wit[COL_CALL_STACK_RETURN_PC_VALUE] = F::from_u64(1u64 << 32);
    write_range_check_bits(&mut wit);

    let vm = WasmVmSpec::default();
    let m_in = vm.core_ccs_spec().m_in;
    let err = check_ccs_rowwise_zero(&vm.core_ccs_spec().structure, &wit[..m_in], &wit[m_in..])
        .expect_err("the range-checked CCS must reject an out-of-range call-stack return pc value");

    let detail = err.to_string();
    let row_idx = detail
        .split_once("row ")
        .and_then(|(_, rest)| rest.split_once(':'))
        .and_then(|(row, _)| row.parse::<usize>().ok())
        .unwrap_or_else(|| panic!("could not parse failing row from: {detail}"));
    let tag = &vm.constraint_catalog().row_tags[row_idx];
    assert_eq!(
        tag.label, "COL_CALL_STACK_RETURN_PC_VALUE",
        "rejection must come from the column's own range-check row, failed row {row_idx} is tagged {tag:?}"
    );
}
