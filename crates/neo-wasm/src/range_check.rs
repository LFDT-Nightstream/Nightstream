//! Range-check pass: enforces each column's declared [`ColumnWidth`] with
//! explicit R1CS rows.
//!
//! This adds cosntraints after the base CCS, by looping over all the columns
//! and deriving the constraints from its declared type.
//!
//! Also provides the function to compute the corresponding witness assignment.

use crate::ccs::poseidon::PERM_GADGET_AUX_WIDTH;
use crate::layout::{ColumnWidth, COLUMN_SPECS, COL_ONE, NAMED_COLUMN_COUNT};

/// First aux bit column: the bits sit after the named layout and the
/// host-event gadget's internal block (see `ccs::poseidon`).
const BITS_BASE: usize = NAMED_COLUMN_COUNT + PERM_GADGET_AUX_WIDTH;
use crate::tagged_r1cs_builder::{WasmConstraintScope, WasmConstraintTag, WasmTaggedR1csBuilder};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::ops::Range;

fn decomposed_bits(width: ColumnWidth) -> usize {
    match width {
        ColumnWidth::Boolean | ColumnWidth::Field => 0,
        ColumnWidth::Byte => 8,
        ColumnWidth::U32 => 32,
    }
}

/// Witness width of the range-checked wasm CCS: the declared columns, the
/// host-event gadget's internal block, and one aux bit column per
/// decomposed bit.
pub fn range_checked_witness_width() -> usize {
    BITS_BASE
        + COLUMN_SPECS
            .iter()
            .map(|spec| decomposed_bits(spec.width))
            .sum::<usize>()
}

/// Aux bit columns backing one declared byte/u32 column in the extended
/// witness. Boolean and field columns have no separate decomposition.
pub fn range_checked_bit_columns(column: usize) -> Option<Range<usize>> {
    let mut start = NAMED_COLUMN_COUNT;
    for spec in COLUMN_SPECS {
        let bits = decomposed_bits(spec.width);
        if spec.index == column {
            return (bits != 0).then_some(start..start + bits);
        }
        start += bits;
    }
    None
}

/// Emit the range-check rows. Each row is tagged with the column's
/// `COL_*` name so constraint provenance dumps itemize the cost per column.
pub(crate) fn push_range_check_rows(b: &mut WasmTaggedR1csBuilder) {
    let mut aux = BITS_BASE;
    for spec in COLUMN_SPECS {
        let bits = decomposed_bits(spec.width);
        let tag = WasmConstraintTag {
            label: spec.name,
            scope: WasmConstraintScope::Always,
        };
        match spec.width {
            ColumnWidth::Field => {}
            ColumnWidth::Boolean => {
                b.with_tag(tag, |b| {
                    b.push_boolean(spec.index);
                });
            }
            ColumnWidth::Byte | ColumnWidth::U32 => {
                b.with_tag(tag, |b| {
                    for i in 0..bits {
                        b.push_boolean(aux + i);
                    }
                    b.push_row(
                        (0..bits).map(|i| (aux + i, F::from_u64(1u64 << i))),
                        [(COL_ONE, F::ONE)],
                        [(spec.index, F::ONE)],
                    );
                });
                aux += bits;
            }
        }
    }
    debug_assert_eq!(aux, range_checked_witness_width());
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
        witness.len() == BITS_BASE || witness.len() == range_checked_witness_width(),
        "witness length {} is neither the base width {} nor the range-checked width {}",
        witness.len(),
        BITS_BASE,
        range_checked_witness_width(),
    );
    witness.truncate(BITS_BASE);
    witness.reserve(range_checked_witness_width() - BITS_BASE);
    for spec in COLUMN_SPECS {
        let bits = decomposed_bits(spec.width);
        if bits == 0 {
            continue;
        }
        let value = witness[spec.index].as_canonical_u64();
        for i in 0..bits {
            witness.push(F::from_u64((value >> i) & 1));
        }
    }
}
