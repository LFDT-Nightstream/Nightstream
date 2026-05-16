use crate::spartan_backend::SpartanF;
use bellpepper_core::ConstraintSystem;
use bellpepper_core::SynthesisError;
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::claim::CircuitCeClaim;
use super::super::k_field::{enforce_k_eq_constant_f_linear_combination, KNumVar};

pub(super) fn enforce_dec_y_ring_from_circuit_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    let d_pad = dec_y_row_width(parent);
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        let target = parent
            .openings
            .y_ring
            .get(idx)
            .ok_or(SynthesisError::Unsatisfiable)?;
        enforce_y_row_dec_target(cs, target, children, idx, d_pad, base_b, &format!("{label}_y_{idx}"))?;
    }
    Ok(())
}

pub(super) fn enforce_dec_y_ring_from_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    let d_pad = dec_y_row_width(parent);
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        let target = parent
            .openings
            .y_ring
            .get(idx)
            .ok_or(SynthesisError::Unsatisfiable)?;
        enforce_y_row_dec_target_constant_children(
            cs,
            target,
            children,
            idx,
            d_pad,
            base_b,
            &format!("{label}_y_{idx}"),
        )?;
    }
    Ok(())
}

pub(super) fn enforce_aux_openings_dec_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.openings.aux_openings.len() != target.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let scalars = scalar_powers_for_k_children(children.len(), base_b)?;
    for (idx, target) in target.iter().enumerate() {
        let mut terms = Vec::new();
        for (child, coeff) in children.iter().zip(scalars.iter()) {
            if *coeff == SpartanF::ZERO {
                continue;
            }
            terms.push((
                *coeff,
                child.openings.aux_openings[idx].c0,
                child.openings.aux_openings[idx].c1,
            ));
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{idx}"));
    }
    Ok(())
}

pub(super) fn enforce_aux_openings_dec_target_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.aux_openings.len() != target.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let b = K::from(F::from_u64(base_b as u64));
    for (idx, target) in target.iter().enumerate() {
        let mut pow = K::ONE;
        let mut expected = K::ZERO;
        for child in children {
            expected += pow * child.aux_openings[idx];
            pow *= b;
        }
        enforce_k_eq_native(cs, target, expected, &format!("{label}_{idx}"));
    }
    Ok(())
}

fn dec_y_row_width(parent: &CircuitCeClaim) -> usize {
    parent
        .openings
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.norm_check.y_zcol_values.len())
}

fn enforce_y_row_dec_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CircuitCeClaim],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        let row = child
            .openings
            .y_ring
            .get(row_idx)
            .ok_or(SynthesisError::Unsatisfiable)?;
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    let scalars = scalar_powers_for_k_children(children.len(), base_b)?;
    for (idx, target) in target.iter().enumerate() {
        let mut terms = Vec::new();
        for (child, coeff) in children.iter().zip(scalars.iter()) {
            if *coeff == SpartanF::ZERO {
                continue;
            }
            terms.push((
                *coeff,
                child.openings.y_ring[row_idx][idx].c0,
                child.openings.y_ring[row_idx][idx].c1,
            ));
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{idx}"));
    }
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
    for child in children {
        let row = child
            .y_ring
            .get(row_idx)
            .ok_or(SynthesisError::Unsatisfiable)?;
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
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

fn scalar_powers_for_k_children(count: usize, base_b: u32) -> Result<Vec<SpartanF>, SynthesisError> {
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut scalars = Vec::with_capacity(count);
    for _ in 0..count {
        let coeff = pow.as_coeffs();
        if coeff[1] != F::ZERO {
            return Err(SynthesisError::Unsatisfiable);
        }
        scalars.push(SpartanF::from_canonical_u64(coeff[0].as_canonical_u64()));
        pow *= b;
    }
    Ok(scalars)
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
