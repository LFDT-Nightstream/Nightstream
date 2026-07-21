mod common;

use neo_ccs::check_ccs_rowwise_zero;
use neo_math::F;
use neo_wasm::layout::{ColumnWidth, COLUMN_SPECS, COL_CALL_STACK_RETURN_PC_VALUE, NAMED_COLUMN_COUNT};
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
        NAMED_COLUMN_COUNT + neo_wasm::ccs::poseidon::PERM_GADGET_AUX_WIDTH + expected_aux_bits()
    );

    let vm = WasmVmSpec::default();
    assert_eq!(vm.core_ccs_spec().witness_width, RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(vm.core_ccs_spec().structure.m, RANGE_CHECKED_WITNESS_WIDTH);
}

#[test]
fn range_bit_lookup_exactly_partitions_the_auxiliary_suffix() {
    let mut next = NAMED_COLUMN_COUNT + neo_wasm::ccs::poseidon::PERM_GADGET_AUX_WIDTH;

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

/// The canonical plan must declare the typed widths. This is the only test
/// pinning that: empty widths are legal and self-consistent (they mean
/// "commit every variable as a full 64-bit lane"), so a regression to an
/// undeclared plan would pass every prove/verify test while silently
/// tripling the committed limbs and making the unconditional F' width
/// audit vacuous. Declaring the widths is what makes preprocessing
/// re-derive each one from the range rows and hard-error on any it cannot
/// prove, so this passing also certifies the pass covers every column.
#[test]
fn canonical_preprocessing_audits_declared_widths() {
    let digest = [0u8; 32];
    let batch_size = 2;
    let prep = neo_wasm::preprocess::preprocess_seeded_batched(batch_size, digest).expect("canonical preprocessing");
    assert_eq!(
        prep.plan().app_private_var_widths.len(),
        batch_size * RANGE_CHECKED_WITNESS_WIDTH,
        "the canonical plan must declare (and thus have audited) the typed widths"
    );
}
