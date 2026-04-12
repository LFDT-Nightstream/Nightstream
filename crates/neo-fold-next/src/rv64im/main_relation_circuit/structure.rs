//! Owns local CCS-shape helpers for the RV64IM main-relation circuit.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, Mat, RelationError};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

pub fn pad_ccs_structure_to_block_width(structure: &CcsStructure<F>) -> Result<CcsStructure<F>, RelationError> {
    let padded_m = structure.m.div_ceil(D) * D;
    if padded_m == structure.m {
        return Ok(structure.clone());
    }

    let matrices = structure
        .matrices
        .iter()
        .map(|matrix| match matrix {
            CcsMatrix::Identity { n } => {
                let mut dense = Mat::zero(*n, padded_m, F::ZERO);
                for idx in 0..*n {
                    dense[(idx, idx)] = F::ONE;
                }
                CcsMatrix::Csc(CscMat::from_dense_row_major(&dense))
            }
            CcsMatrix::Csc(csc) => {
                let mut col_ptr = csc.col_ptr.clone();
                let last = *col_ptr.last().unwrap_or(&0);
                col_ptr.resize(padded_m + 1, last);
                CcsMatrix::Csc(CscMat {
                    nrows: csc.nrows,
                    ncols: padded_m,
                    col_ptr,
                    row_idx: csc.row_idx.clone(),
                    vals: csc.vals.clone(),
                })
            }
        })
        .collect::<Vec<_>>();

    CcsStructure::new_sparse(matrices, structure.f.clone())
}
