//! Owns low-level circuit helpers for the terminal committed `F'` R2 check.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use neo_ccs::CcsMatrix;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::rv64im::ivc_snark::SpartanF;

pub(super) fn native_to_spartan(value: &F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

pub(super) fn enforce_boolean_allocated<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    label: &str,
) {
    cs.enforce(
        || format!("{label}_boolean"),
        |lc| lc + value.get_variable(),
        |lc| lc + value.get_variable() - CS::one(),
        |lc| lc,
    );
}

pub(super) fn set_z_entry(
    entries: &mut [Option<LinearCombination<SpartanF>>],
    col: usize,
    entry: LinearCombination<SpartanF>,
) -> Result<(), SynthesisError> {
    let slot = entries.get_mut(col).ok_or(SynthesisError::Unsatisfiable)?;
    if slot.is_some() {
        return Err(SynthesisError::Unsatisfiable);
    }
    *slot = Some(entry);
    Ok(())
}

pub(super) fn matrix_row_linear_combinations(
    matrix: &CcsMatrix<F>,
    z_entries: &[LinearCombination<SpartanF>],
) -> Result<Vec<LinearCombination<SpartanF>>, SynthesisError> {
    if matrix.cols() != z_entries.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut rows = vec![LinearCombination::<SpartanF>::zero(); matrix.rows()];
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n > z_entries.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            for row in 0..*n {
                rows[row] = z_entries[row].clone();
            }
        }
        CcsMatrix::Csc(csc) => {
            for col in 0..csc.ncols {
                let start = csc.col_ptr[col];
                let end = csc.col_ptr[col + 1];
                for idx in start..end {
                    let row = csc.row_idx[idx];
                    let coeff = csc.vals[idx];
                    if row >= rows.len() {
                        return Err(SynthesisError::Unsatisfiable);
                    }
                    if coeff != F::ZERO {
                        rows[row] = rows[row].clone() + (native_to_spartan(&coeff), &z_entries[col]);
                    }
                }
            }
        }
    }
    Ok(rows)
}
