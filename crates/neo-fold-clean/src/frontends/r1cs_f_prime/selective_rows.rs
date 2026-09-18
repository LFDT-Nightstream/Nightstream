//! Source-row ownership for direct selective lowering.

use super::{trace_error, LowNormR1csError, SparseR1cs, BALANCED_FIELD_WIDTH};

pub(super) fn skipped_selective_rows(arm: &SparseR1cs) -> Result<Vec<bool>, LowNormR1csError> {
    let mut skipped = vec![false; arm.n];
    let mut claim = |range: core::ops::Range<usize>, overlap: &'static str| {
        for row in range {
            if row >= skipped.len() || core::mem::replace(&mut skipped[row], true) {
                return Err(trace_error(overlap));
            }
        }
        Ok(())
    };

    for trace in arm.poseidon2_traces() {
        claim(trace.row_start..trace.row_end, "Poseidon2 traces overlap")?;
    }
    for trace in arm.polynomial_evaluation_traces() {
        claim(trace.row_start..trace.row_end, "selective trace row ranges overlap")?;
    }
    for trace in arm.product_sum_batch_traces() {
        claim(
            trace.row_start..trace.row_end,
            "product-sum trace overlaps another selective trace",
        )?;
    }
    for trace in arm.centered_unit_traces() {
        claim(
            trace.row_start..trace.row_end,
            "centered-unit trace overlaps another selective trace",
        )?;
    }
    for trace in arm.shifted_ternary_canonical_traces() {
        claim(
            trace.digit_rows_start..trace.digit_rows_start + 2 * BALANCED_FIELD_WIDTH,
            "shifted-ternary digit rows overlap another selective trace",
        )?;
        claim(
            trace.transition_rows_start..trace.transition_rows_start + BALANCED_FIELD_WIDTH,
            "shifted-ternary transition rows overlap another selective trace",
        )?;
    }
    Ok(skipped)
}
