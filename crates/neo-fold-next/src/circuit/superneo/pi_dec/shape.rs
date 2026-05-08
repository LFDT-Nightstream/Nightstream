use crate::spartan_backend::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{D, F, K};

use super::super::claim::CircuitCeClaim;
use super::super::k_field::{enforce_k_eq, KNumVar};

pub(super) fn ensure_parent_dec_surface(parent: &CircuitCeClaim) -> Result<(), SynthesisError> {
    if parent.public_input.rows != D
        || parent.public_input.cols != parent.public_input.m_in
        || parent.openings.r.len() != parent.openings.r_values.len()
        || parent.openings.y_ring.len() != parent.openings.y_ring_values.len()
        || parent.norm_check.y_zcol.len() != parent.norm_check.y_zcol_values.len()
        || parent.openings.aux_openings.len() != parent.openings.aux_openings_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub(super) fn ensure_circuit_children_match_parent<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, child) in children.iter().enumerate() {
        if child.public_input.m_in != parent.public_input.m_in
            || child.public_input.rows != D
            || child.public_input.cols != parent.public_input.m_in
            || child.openings.r_values != parent.openings.r_values
            || child.openings.y_ring.len() != parent.openings.y_ring.len()
            || child.norm_check.y_zcol_values.len() != parent.norm_check.y_zcol_values.len()
            || child.openings.aux_openings.len() != parent.openings.aux_openings.len()
            || child.commitment.data.len() != parent.commitment.data.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.openings.r, &child.openings.r, &format!("{label}_r_{idx}"))?;
    }
    Ok(())
}

pub(super) fn ensure_constant_children_match_parent(
    parent: &CircuitCeClaim,
    children: &[CeClaim<Commitment, F, K>],
) -> Result<(), SynthesisError> {
    if children.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.m_in != parent.public_input.m_in
            || child.X.rows() != D
            || child.X.cols() != parent.public_input.m_in
            || child.r != parent.openings.r_values
            || child.y_ring.len() != parent.openings.y_ring.len()
            || child.y_zcol.len() != parent.norm_check.y_zcol_values.len()
            || child.aux_openings.len() != parent.openings.aux_openings.len()
            || child.c.data.len() != parent.commitment.data_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
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
