//! Matrix expansion for low-norm direct R1CS lowering.

use super::*;

pub(super) fn expand_matrix(
    matrix: &CcsMatrix<F>,
    rows: usize,
    lanes: &[LaneMap],
    out: &mut Vec<(usize, usize, F)>,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n > lanes.len() {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct low-norm R1CS identity matrix exceeds export variable count".into(),
                ));
            }
            for row in 0..(*n).min(rows) {
                push_expanded_term(row, F::ONE, &lanes[row], out);
            }
        }
        CcsMatrix::Csc(csc) => {
            if csc.ncols > lanes.len() || csc.nrows != rows {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct low-norm R1CS matrix shape does not match export shape".into(),
                ));
            }
            for col in 0..csc.ncols {
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    push_expanded_term(csc.row_idx[idx], csc.vals[idx], &lanes[col], out);
                }
            }
        }
    }
    Ok(())
}

fn push_expanded_term(row: usize, coeff: F, lane: &LaneMap, out: &mut Vec<(usize, usize, F)>) {
    let mut bit_coeff = coeff;
    for bit_index in 0..lane.bit_len {
        out.push((row, lane.bits_start_col + bit_index, bit_coeff));
        bit_coeff += bit_coeff;
    }
}
