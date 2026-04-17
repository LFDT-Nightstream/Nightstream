//! Owns public Π_DEC arithmetic checks over CE claims for the RV64IM main-relation circuit.
//!
//! This module owns the pure b-ary homomorphic checks for parent/child CE claims. Rho-driven
//! Π_RLC checks live in `pi_rlc.rs`.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{enforce_k_eq, enforce_k_eq_constant_f_linear_combination, KNumVar};

pub fn enforce_dec_public<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, child) in children.iter().enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.s_col_values != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.c_data.len() != parent.c_data.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &parent.s_col, &child.s_col, &format!("{label}_s_col_{idx}"))?;
    }

    enforce_scalar_power_sum_on_dense_f_slices(
        cs,
        &parent.x,
        &children
            .iter()
            .map(|child| child.x.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_x"),
    )?;

    enforce_scalar_power_sum_on_dense_f_slices(
        cs,
        &parent.c_data,
        &children
            .iter()
            .map(|child| child.c_data.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_c"),
    )?;

    let d_pad = parent
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.y_zcol_values.len());
    for (idx, row) in parent.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_dec_target(
            cs,
            &parent.y_ring[idx],
            children,
            idx,
            d_pad,
            base_b,
            &format!("{label}_y_{idx}"),
        )?;
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_dec_target(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            aux_idx,
            base_b,
            &format!("{label}_aux_{aux_idx}"),
        )?;
    }

    Ok(())
}
pub fn enforce_dec_public_with_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for child in children {
        if child.m_in != parent.m_in
            || child.X.rows() != D
            || child.X.cols() != parent.m_in
            || child.r != parent.r_values
            || child.s_col != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.c.data.len() != parent.c_data_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    enforce_scalar_power_sum_on_dense_constant_f_slices(
        cs,
        &parent.x,
        &parent.x_values,
        &children
            .iter()
            .map(|child| child.X.as_slice().to_vec())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_x"),
    )?;

    enforce_scalar_power_sum_on_dense_constant_f_slices(
        cs,
        &parent.c_data,
        &parent.c_data_values,
        &children
            .iter()
            .map(|child| child.c.data.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_c"),
    )?;

    let d_pad = parent
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.y_zcol_values.len());
    for (idx, row) in parent.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_dec_target_constant_children(
            cs,
            &parent.y_ring[idx],
            children,
            idx,
            d_pad,
            base_b,
            &format!("{label}_y_{idx}"),
        )?;
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_dec_target_constant_children(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            aux_idx,
            base_b,
            &format!("{label}_aux_{aux_idx}"),
        )?;
    }

    Ok(())
}
pub fn enforce_dec_public_non_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_dec_public(cs, parent, children, base_b, label)
}
fn enforce_scalar_power_sum_on_dense_f_slices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    children: &[Vec<AllocatedNum<SpartanF>>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.len() != parent.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    let b = SpartanF::from_canonical_u64(base_b as u64);
    for idx in 0..parent.len() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| {
                let mut acc = lc;
                let mut pow = SpartanF::ONE;
                for child in children {
                    acc = acc + (pow, child[idx].get_variable());
                    pow *= b;
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + parent[idx].get_variable(),
        );
    }
    Ok(())
}
fn enforce_scalar_power_sum_on_dense_constant_f_slices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    parent_values: &[F],
    children: &[Vec<F>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() || parent_values.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if !parent.is_empty() && parent.len() != parent_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.len() != parent_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    let b = F::from_u64(base_b as u64);
    for idx in 0..parent_values.len() {
        let mut pow = F::ONE;
        let mut expected = F::ZERO;
        for child in children {
            expected += pow * child[idx];
            pow *= b;
        }
        if parent.is_empty() {
            if parent_values[idx] != expected {
                return Err(SynthesisError::Unsatisfiable);
            }
        } else {
            cs.enforce(
                || format!("{label}_{idx}"),
                |lc| lc + parent[idx].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (SpartanF::from_canonical_u64(expected.as_canonical_u64()), CS::one()),
            );
        }
    }
    Ok(())
}
fn enforce_y_row_dec_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut scalars = Vec::with_capacity(children.len());
    for _ in children {
        let coeff = pow.as_coeffs();
        if coeff[1] != F::ZERO {
            return Err(SynthesisError::Unsatisfiable);
        }
        scalars.push(SpartanF::from_canonical_u64(coeff[0].as_canonical_u64()));
        pow *= b;
    }
    for (idx, target) in target.iter().enumerate() {
        let mut terms = Vec::new();
        for (child, coeff) in children.iter().zip(scalars.iter()) {
            if *coeff == SpartanF::ZERO {
                continue;
            }
            terms.push((*coeff, child.y_ring[row_idx][idx].c0, child.y_ring[row_idx][idx].c1));
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{idx}"));
    }
    Ok(())
}
fn enforce_aux_dec_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaimVar],
    aux_idx: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut terms = Vec::with_capacity(children.len());
    for child in children {
        let coeff = pow.as_coeffs();
        if coeff[1] != F::ZERO {
            return Err(SynthesisError::Unsatisfiable);
        }
        let coeff = SpartanF::from_canonical_u64(coeff[0].as_canonical_u64());
        if coeff != SpartanF::ZERO {
            terms.push((coeff, child.aux_openings[aux_idx].c0, child.aux_openings[aux_idx].c1));
        }
        pow *= b;
    }
    enforce_k_eq_constant_f_linear_combination(cs, target, &terms, label);
    Ok(())
}
fn enforce_y_row_dec_target_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaim<Commitment, F, K>],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    let b = K::from(F::from_u64(base_b as u64));
    for (idx, target) in target.iter().enumerate() {
        let mut pow = K::ONE;
        let mut expected = K::ZERO;
        for child in children {
            expected += pow * child.y_ring[row_idx][idx];
            pow *= b;
        }
        enforce_k_eq_native(cs, target, expected, &format!("{label}_{idx}"));
    }
    Ok(())
}
fn enforce_aux_dec_target_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaim<Commitment, F, K>],
    aux_idx: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut expected = K::ZERO;
    for child in children {
        expected += pow * child.aux_openings[aux_idx];
        pow *= b;
    }
    enforce_k_eq_native(cs, target, expected, label);
    Ok(())
}
fn enforce_equal_k_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    left: &[KNumVar],
    right: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        enforce_k_eq(cs, l, r, &format!("{label}_{idx}"));
    }
    Ok(())
}
fn enforce_k_eq_native<CS: ConstraintSystem<SpartanF>>(cs: &mut CS, actual: &KNumVar, expected: K, label: &str) {
    let coeffs = expected.as_coeffs();
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| lc + actual.c0,
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()), CS::one()),
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| lc + actual.c1,
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()), CS::one()),
    );
}
