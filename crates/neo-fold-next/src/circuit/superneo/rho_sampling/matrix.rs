use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{RotRhoMatrixVar, RotRhoVar};

fn alloc_rot_rhos_from_coeff_values<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    rhos: &[Vec<F>],
    label: &str,
) -> Result<Vec<RotRhoVar>, SynthesisError> {
    let mut out = Vec::with_capacity(rhos.len());
    for (rho_idx, coeff_values) in rhos.iter().enumerate() {
        if coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut coeffs = Vec::with_capacity(D);
        for (row, value) in coeff_values.iter().copied().enumerate() {
            let coeff = alloc_affine(
                cs.namespace(|| format!("{label}_rho_{rho_idx}_coeff_{row}")),
                &[],
                SpartanF::from_canonical_u64(value.as_canonical_u64()),
            )?;
            coeffs.push(coeff);
        }
        out.push(RotRhoVar::from_coeffs(coeffs, coeff_values.clone()));
    }
    Ok(out)
}

pub fn alloc_zero_rot_rhos<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    count: usize,
    label: &str,
) -> Result<Vec<RotRhoVar>, SynthesisError> {
    alloc_rot_rhos_from_coeff_values(
        cs.namespace(|| format!("{label}_zero")),
        &vec![vec![F::ZERO; D]; count],
        label,
    )
}

pub fn materialize_goldilocks_rot_matrices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<Vec<RotRhoMatrixVar>, SynthesisError> {
    let ring = neo_reductions::RotRing::goldilocks();
    let neg_phi = ring
        .phi_coeffs
        .iter()
        .map(|coeff| F::from_i64(-(*coeff as i64)))
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(rhos.len());
    for (rho_idx, rho) in rhos.iter().enumerate() {
        if rho.coeffs.len() != D || rho.coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut entries = vec![rho.coeffs[0].clone(); D * D];
        let mut entry_values = vec![F::ZERO; D * D];

        let mut prev_col = rho.coeffs.clone();
        let mut prev_values = rho.coeff_values.clone();
        for row in 0..D {
            entries[row * D] = prev_col[row].clone();
            entry_values[row * D] = prev_values[row];
        }

        for col in 1..D {
            let tail_var = prev_col[D - 1].clone();
            let tail_value = prev_values[D - 1];
            let mut next_col = Vec::with_capacity(D);
            let mut next_values = Vec::with_capacity(D);

            let top_value = neg_phi[0] * tail_value;
            let top = alloc_affine(
                cs.namespace(|| format!("{label}_rho_{rho_idx}_col_{col}_row_0")),
                &[(
                    tail_var.clone(),
                    SpartanF::from_canonical_u64(neg_phi[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(tail_value.as_canonical_u64()),
                )],
                SpartanF::ZERO,
            )?;
            next_col.push(top.clone());
            next_values.push(top_value);

            for row in 1..D {
                let value = prev_values[row - 1] + neg_phi[row] * tail_value;
                let entry = alloc_affine(
                    cs.namespace(|| format!("{label}_rho_{rho_idx}_col_{col}_row_{row}")),
                    &[
                        (
                            prev_col[row - 1].clone(),
                            SpartanF::ONE,
                            SpartanF::from_canonical_u64(prev_values[row - 1].as_canonical_u64()),
                        ),
                        (
                            tail_var.clone(),
                            SpartanF::from_canonical_u64(neg_phi[row].as_canonical_u64()),
                            SpartanF::from_canonical_u64(tail_value.as_canonical_u64()),
                        ),
                    ],
                    SpartanF::ZERO,
                )?;
                next_col.push(entry.clone());
                next_values.push(value);
            }

            for row in 0..D {
                entries[row * D + col] = next_col[row].clone();
                entry_values[row * D + col] = next_values[row];
            }
            prev_col = next_col;
            prev_values = next_values;
        }

        out.push(RotRhoMatrixVar::from_entries(D, D, entries, entry_values));
    }
    Ok(out)
}

pub fn alloc_rot_rho_matrices_from_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    mats: &[Mat<F>],
    label: &str,
) -> Result<Vec<RotRhoMatrixVar>, SynthesisError> {
    let mut out = Vec::with_capacity(mats.len());
    for (mat_idx, mat) in mats.iter().enumerate() {
        if mat.rows() != D || mat.cols() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
        let mut entries = Vec::with_capacity(D * D);
        let mut entry_values = Vec::with_capacity(D * D);
        for row in 0..D {
            for col in 0..D {
                let value = mat[(row, col)];
                let entry = alloc_affine(
                    cs.namespace(|| format!("{label}_mat_{mat_idx}_{row}_{col}")),
                    &[],
                    SpartanF::from_canonical_u64(value.as_canonical_u64()),
                )?;
                entries.push(entry);
                entry_values.push(value);
            }
        }
        out.push(RotRhoMatrixVar::from_entries(D, D, entries, entry_values));
    }
    Ok(out)
}

pub fn alloc_zero_rot_rho_matrices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    count: usize,
    label: &str,
) -> Result<Vec<RotRhoMatrixVar>, SynthesisError> {
    let zero = Mat::zero(D, D, F::ZERO);
    alloc_rot_rho_matrices_from_native(&mut cs.namespace(|| format!("{label}_zero")), &vec![zero; count], label)
}

fn alloc_affine<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(AllocatedNum<SpartanF>, SpartanF, SpartanF)],
    constant: SpartanF,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let mut value = constant;
    for (_, coeff, term_value) in terms {
        value += *coeff * *term_value;
    }
    let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(value))?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc + (constant, CS::one());
            for (term, coeff, _) in terms {
                rhs = rhs + (*coeff, term.get_variable());
            }
            rhs
        },
    );
    Ok(out)
}
