//! Range-check pass: enforces each column's declared [`ColumnWidth`] with
//! explicit R1CS rows.
//!
//! This adds cosntraints after the base CCS, by looping over all the columns
//! and deriving the constraints from its declared type.
//!
//! Also provides the function to compute the corresponding witness assignment.

use crate::layout::{ColumnWidth, COLUMN_SPECS, COL_ONE};
use crate::tagged_r1cs_builder::{WasmConstraintScope, WasmConstraintTag, WasmTaggedR1csBuilder};
use crate::witness_layout::{range_bit_region, RANGE_BITS, RANGE_CHECKED_WITNESS_WIDTH};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::ops::Range;

/// Aux bit columns backing one declared byte/u32 column in the extended
/// witness. Boolean and field columns have no separate decomposition.
pub fn range_checked_bit_columns(column: usize) -> Option<Range<usize>> {
    range_bit_region(column).map(|region| region.start..region.end())
}

/// Emit the range-check rows. Each row is tagged with the column's
/// `COL_*` name so constraint provenance dumps itemize the cost per column.
pub(crate) fn push_range_check_rows(b: &mut WasmTaggedR1csBuilder) {
    for spec in COLUMN_SPECS {
        for column in spec.start..spec.end() {
            let tag = WasmConstraintTag {
                label: spec.name,
                scope: WasmConstraintScope::Always,
            };
            match spec.width {
                ColumnWidth::Field => {}
                ColumnWidth::Boolean => {
                    b.with_tag(tag, |b| {
                        b.push_boolean(column);
                    });
                }
                ColumnWidth::Byte | ColumnWidth::U32 => {
                    let region = range_bit_region(column).expect("decomposed column has a range-bit region");
                    b.with_tag(tag, |b| {
                        for bit in region.start..region.end() {
                            b.push_boolean(bit);
                        }
                        b.push_row(
                            (region.start..region.end())
                                .enumerate()
                                .map(|(i, bit)| (bit, F::from_u64(1u64 << i))),
                            [(COL_ONE, F::ONE)],
                            [(column, F::ONE)],
                        );
                    });
                }
            }
        }
    }
}

/// Compute (or refresh) the aux bit columns from the declared columns.
///
/// Accepts a base-width or already-extended witness; the tail is always
/// recomputed, so callers that mutate declared columns can call this again
/// to keep the bits consistent (for tests). A value outside its declared range
/// gets its low bits written as-is, leaving the recomposition row unsatisfiable
/// — the CCS failure then carries the column's name via the row tag.
pub fn write_range_check_bits(witness: &mut Vec<F>) {
    assert!(
        witness.len() == RANGE_BITS.start || witness.len() == RANGE_CHECKED_WITNESS_WIDTH,
        "witness length {} is neither the base width {} nor the range-checked width {}",
        witness.len(),
        RANGE_BITS.start,
        RANGE_CHECKED_WITNESS_WIDTH,
    );
    witness.resize(RANGE_CHECKED_WITNESS_WIDTH, F::ZERO);
    for spec in COLUMN_SPECS {
        for column in spec.start..spec.end() {
            let Some(region) = range_bit_region(column) else {
                continue;
            };
            let value = witness[column].as_canonical_u64();
            for (i, bit) in (region.start..region.end()).enumerate() {
                witness[bit] = F::from_u64((value >> i) & 1);
            }
        }
    }
}
