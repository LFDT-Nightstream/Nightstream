//! Owns public Π_DEC arithmetic checks over circuit CE claims.
//!
//! This module owns the paper-facing b-ary homomorphic recomposition flow:
//! validate parent/child shapes, enforce parent `x`, enforce parent
//! commitment `c`, enforce `{y_j}`, then enforce auxiliary openings. Rho-driven
//! Π_RLC checks live under `pi_rlc`.

mod f_sums;
mod k_sums;
mod shape;

use crate::spartan_backend::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};

use super::claim::CircuitCeClaim;
use f_sums::{
    enforce_dec_commitment_from_circuit_children, enforce_dec_commitment_from_constant_children,
    enforce_dec_public_input_from_circuit_children, enforce_dec_public_input_from_constant_children,
};
use k_sums::{
    enforce_aux_openings_dec_target, enforce_aux_openings_dec_target_constant_children,
    enforce_dec_y_ring_from_circuit_children, enforce_dec_y_ring_from_constant_children,
};
use shape::{ensure_circuit_children_match_parent, ensure_constant_children_match_parent, ensure_parent_dec_surface};

pub fn enforce_dec_public<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    ensure_parent_dec_surface(parent)?;
    ensure_circuit_children_match_parent(cs, parent, children, label)?;
    enforce_parent_dec_from_circuit_children(cs, parent, children, base_b, label)
}

pub fn enforce_dec_public_with_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    ensure_parent_dec_surface(parent)?;
    ensure_constant_children_match_parent(parent, children)?;
    enforce_parent_dec_from_constant_children(cs, parent, children, base_b, label)
}

pub fn enforce_dec_public_non_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_dec_public(cs, parent, children, base_b, label)
}

fn enforce_parent_dec_from_circuit_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_dec_public_input_from_circuit_children(cs, parent, children, base_b, label)?;
    enforce_dec_commitment_from_circuit_children(cs, parent, children, base_b, label)?;
    enforce_dec_y_ring_from_circuit_children(cs, parent, children, base_b, label)?;
    enforce_aux_openings_dec_target(
        cs,
        &parent.openings.aux_openings,
        children,
        base_b,
        &format!("{label}_aux_openings"),
    )
}

fn enforce_parent_dec_from_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_dec_public_input_from_constant_children(cs, parent, children, base_b, label)?;
    enforce_dec_commitment_from_constant_children(cs, parent, children, base_b, label)?;
    enforce_dec_y_ring_from_constant_children(cs, parent, children, base_b, label)?;
    enforce_aux_openings_dec_target_constant_children(
        cs,
        &parent.openings.aux_openings,
        children,
        base_b,
        &format!("{label}_aux_openings"),
    )
}
