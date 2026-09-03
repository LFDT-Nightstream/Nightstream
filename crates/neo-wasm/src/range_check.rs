//! Range-check pass: enforces each column's declared
//! [`neo_application::ColumnWidth`] with
//! explicit R1CS rows.
//!
//! This adds constraints after the base CCS, by looping over all the columns
//! and deriving the constraints from its declared type.
//!
//! Also provides the function to compute the corresponding witness assignment.

use crate::tagged_r1cs_builder::{WasmConstraintScope, WasmTaggedR1csBuilder};
use crate::witness_layout::range_check_layout;
use neo_math::F;
use std::ops::Range;

/// Aux bit columns backing one declared byte/u32 column in the extended
/// witness. Boolean and field columns have no separate decomposition.
pub fn range_checked_bit_columns(column: usize) -> Option<Range<usize>> {
    range_check_layout().bit_columns_for(column)
}

/// Emit the range-check rows. Each row is tagged with the column's
/// `COL_*` name so constraint provenance dumps itemize the cost per column.
pub(crate) fn push_range_check_rows(builder: &mut WasmTaggedR1csBuilder<'_>) {
    range_check_layout().push_constraints(builder, WasmConstraintScope::Always);
}

/// Compute (or refresh) the aux bit columns from the declared columns.
///
/// Accepts a base-width or already-extended witness; the tail is always
/// recomputed, so callers that mutate declared columns can call this again
/// to keep the bits consistent (for tests). A value outside its declared range
/// gets its low bits written as-is, leaving the recomposition row unsatisfiable
/// — the CCS failure then carries the column's name via the row tag.
pub fn write_range_check_bits(witness: &mut Vec<F>) {
    range_check_layout()
        .assign_bits(witness)
        .unwrap_or_else(|error| panic!("{error}"));
}
