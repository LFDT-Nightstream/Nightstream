use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::claim::CircuitCeClaim;

pub(super) fn enforce_dec_public_input_from_circuit_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_scalar_power_sum_on_dense_f_slices(
        cs,
        &parent.public_input.x,
        &children
            .iter()
            .map(|child| child.public_input.x.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_x"),
    )
}

pub(super) fn enforce_dec_public_input_from_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_scalar_power_sum_on_dense_constant_f_slices(
        cs,
        &parent.public_input.x,
        &parent.public_input.x_values,
        &children
            .iter()
            .map(|child| child.X.as_slice().to_vec())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_x"),
    )
}

pub(super) fn enforce_dec_commitment_from_circuit_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_scalar_power_sum_on_dense_f_slices(
        cs,
        &parent.commitment.data,
        &children
            .iter()
            .map(|child| child.commitment.data.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_c"),
    )
}

pub(super) fn enforce_dec_commitment_from_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_scalar_power_sum_on_dense_constant_f_slices(
        cs,
        &parent.commitment.data,
        &parent.commitment.data_values,
        &children
            .iter()
            .map(|child| child.c.data.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_c"),
    )
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
