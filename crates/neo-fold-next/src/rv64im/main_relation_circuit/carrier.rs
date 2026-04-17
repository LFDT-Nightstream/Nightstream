//! Owns the explicit witness-backed carrier-opening layer for the RV64IM main relation.
//!
//! This module only handles the part of the compiled relation that ties carried CE
//! claims to explicit packed `Z` witnesses. It does not own transcript replay or
//! the verifier-style `Π_CCS -> Π_RLC -> Π_DEC` theorem checks.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ccs::{CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::F;
use neo_params::NeoParams;
use spartan2::provider::goldi::F as SpartanF;

use super::ce_consistency::{
    enforce_ajtai_commitment_consistency, enforce_output_claim_consistency, enforce_paper_dec_child_claim_consistency,
};
use super::claim::{alloc_ce_claim, CeClaimVar};
use super::pi_dec::enforce_dec_public;
use super::pi_rlc::enforce_rlc_public;
use super::witness::{alloc_packed_witness, PackedWitnessVar};
use super::witness_transition::{
    alloc_split_children_from_native, enforce_packed_dec_split, mix_packed_witnesses_with_rho_mats,
};

#[derive(Clone)]
pub struct CarrierOutputs {
    pub claims: Vec<CeClaimVar>,
    pub witnesses: Vec<PackedWitnessVar>,
}

#[derive(Clone)]
pub struct CarrierParent {
    pub claim: CeClaimVar,
    pub mixed_witness: PackedWitnessVar,
}

#[derive(Clone)]
pub struct CarrierChildren {
    pub claims: Vec<CeClaimVar>,
    pub witnesses: Vec<PackedWitnessVar>,
}

pub fn allocate_and_enforce_outputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    fresh_witnesses: &[CcsWitness<F>],
    carried_witnesses: &[PackedWitnessVar],
    claims: &[CeClaim<neo_ajtai::Commitment, F, neo_math::K>],
    delta: SpartanF,
    label: &str,
) -> Result<CarrierOutputs, SynthesisError> {
    let fresh_witnesses = fresh_witnesses
        .iter()
        .enumerate()
        .map(|(idx, witness)| {
            alloc_packed_witness(
                &mut cs.namespace(|| format!("{label}_fresh_witness_{idx}")),
                witness,
                &format!("{label}_fresh_witness_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut output_witnesses = fresh_witnesses;
    output_witnesses.extend(carried_witnesses.iter().cloned());

    let output_claims = claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| {
            alloc_ce_claim(
                &mut cs.namespace(|| format!("{label}_claim_{idx}")),
                claim,
                &format!("{label}_claim_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    if output_claims.len() != output_witnesses.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (claim, witness)) in output_claims
        .iter()
        .zip(output_witnesses.iter())
        .enumerate()
    {
        enforce_output_claim_consistency(
            &mut cs.namespace(|| format!("{label}_ce_{idx}")),
            params,
            base_structure,
            ring_structure,
            witness,
            claim,
            delta,
            &format!("{label}_ce_{idx}"),
        )?;
    }

    Ok(CarrierOutputs {
        claims: output_claims,
        witnesses: output_witnesses,
    })
}

pub fn allocate_and_enforce_parent<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    output_claims: &[CeClaimVar],
    output_witnesses: &[PackedWitnessVar],
    parent: &CeClaim<neo_ajtai::Commitment, F, neo_math::K>,
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<CarrierParent, SynthesisError> {
    let parent_claim = alloc_ce_claim(
        &mut cs.namespace(|| format!("{label}_claim")),
        parent,
        &format!("{label}_claim"),
    )?;
    enforce_rlc_public(
        &mut cs.namespace(|| format!("{label}_public")),
        &parent_claim,
        output_claims,
        rho_mats,
        &format!("{label}_public"),
    )?;
    let mixed_witness = mix_packed_witnesses_with_rho_mats(
        &mut cs.namespace(|| format!("{label}_mix_witness")),
        output_witnesses,
        rho_mats,
        &format!("{label}_mix_witness"),
    )?;
    enforce_ajtai_commitment_consistency(
        &mut cs.namespace(|| format!("{label}_commitment")),
        &mixed_witness,
        &parent_claim,
        &format!("{label}_commitment"),
    )?;
    Ok(CarrierParent {
        claim: parent_claim,
        mixed_witness,
    })
}

pub fn allocate_and_enforce_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    base_structure: &CcsStructure<F>,
    ring_structure: &CcsStructure<F>,
    parent: &CeClaimVar,
    mixed_witness: &PackedWitnessVar,
    claims: &[CeClaim<neo_ajtai::Commitment, F, neo_math::K>],
    z_split: &[Mat<F>],
    _delta: SpartanF,
    label: &str,
) -> Result<CarrierChildren, SynthesisError> {
    let child_claims = claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| {
            alloc_ce_claim(
                &mut cs.namespace(|| format!("{label}_claim_{idx}")),
                claim,
                &format!("{label}_claim_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let child_witnesses = alloc_split_children_from_native(
        &mut cs.namespace(|| format!("{label}_witnesses")),
        z_split,
        &format!("{label}_witnesses"),
    )?;

    enforce_dec_public(
        &mut cs.namespace(|| format!("{label}_public")),
        parent,
        &child_claims,
        params.b,
        &format!("{label}_public"),
    )?;
    enforce_packed_dec_split(
        &mut cs.namespace(|| format!("{label}_split")),
        mixed_witness,
        &child_witnesses,
        params.b,
        &format!("{label}_split"),
    )?;
    for (idx, (claim, witness)) in child_claims.iter().zip(child_witnesses.iter()).enumerate() {
        enforce_paper_dec_child_claim_consistency(
            &mut cs.namespace(|| format!("{label}_ce_{idx}")),
            params,
            base_structure,
            ring_structure,
            witness,
            claim,
            &format!("{label}_ce_{idx}"),
        )?;
    }

    Ok(CarrierChildren {
        claims: child_claims,
        witnesses: child_witnesses,
    })
}
