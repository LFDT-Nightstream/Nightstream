//! Range-check pass for explicit [`neo_application::ColumnWidth`] declarations.
//! `Implied` and `Repairable` bounds emit no rows. Soundness requires their proof
//! obligations to be discharged against the relation without those checks.
//! For WASM, `export_wasm_rocq` exports the obligations and the Rocq audit proves
//! `range_obligations_sound`; this pass itself does not establish them.
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
/// witness. Boolean, field, implied and repairable columns have no separate decomposition.
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
/// to keep the bits consistent (for tests). An explicitly checked value outside
/// its declared range gets its low bits written as-is, leaving recomposition unsatisfied
/// — the CCS failure then carries the column's name via the row tag.
pub fn write_range_check_bits(witness: &mut Vec<F>) {
    range_check_layout()
        .assign_bits(witness)
        .unwrap_or_else(|error| panic!("{error}"));
}
