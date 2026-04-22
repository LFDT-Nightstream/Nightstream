//! Owns recursive fixed-step cover allocation, equality, and digest gadgets.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::alloc_const_field_values;
use super::fingerprint_cs::FingerprintCS;
use crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot;
use crate::rv64im::ivc_snark::{hash_packed_goldilocks_fields, SpartanF};
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_projection_surface, alloc_ce_claim_projection_surface_with_shared_r,
    alloc_ce_claim_with_shared_point, alloc_ce_claim_x_r_surface, alloc_ce_claim_x_r_surface_with_shared_r,
    me_input_projection_digest_poseidon, packed_bytes_field_values, CeClaimVar,
};
use crate::rv64im::main_relation_circuit::transcript::hash_field_linear_combinations_raw;

#[derive(Clone)]
pub(super) struct Rv64imRecursiveCoverClaimVar {
    pub(super) claim: CeClaimVar,
}

#[derive(Clone)]
pub(super) struct Rv64imRecursiveCoverStateVar {
    pub(super) transcript_state: [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH],
    pub(super) transcript_absorbed: AllocatedNum<SpartanF>,
    pub(super) terminal_handle: [AllocatedNum<SpartanF>; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imRecursiveAccumulatorProjectionDigestAuxBreakdown {
    pub after_header: usize,
    pub after_claim_digests: Vec<usize>,
    pub after_outer_hash: usize,
}

fn alloc_recursive_cover_public_state_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &Rv64imChunkFoldTranscriptSnapshot,
    terminal_handle_digest: [u8; 32],
    label: &str,
) -> Result<
    (
        [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH],
        AllocatedNum<SpartanF>,
        [AllocatedNum<SpartanF>; 4],
    ),
    SynthesisError,
> {
    let transcript_state = transcript
        .state
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_transcript_state_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let transcript_absorbed = AllocatedNum::alloc(cs.namespace(|| format!("{label}_transcript_absorbed")), || {
        Ok(SpartanF::from_canonical_u64(transcript.absorbed as u64))
    })?;
    let terminal_handle = super::digest32_as_spartan_fields(terminal_handle_digest)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_terminal_handle_{idx}")), || Ok(value))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    Ok((transcript_state, transcript_absorbed, terminal_handle))
}

pub(super) fn alloc_recursive_cover_state<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    _claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
    transcript: &Rv64imChunkFoldTranscriptSnapshot,
    terminal_handle_digest: [u8; 32],
    label: &str,
) -> Result<Rv64imRecursiveCoverStateVar, SynthesisError> {
    let (transcript_state, transcript_absorbed, terminal_handle) =
        alloc_recursive_cover_public_state_fields(cs, transcript, terminal_handle_digest, label)?;
    Ok(Rv64imRecursiveCoverStateVar {
        transcript_state,
        transcript_absorbed,
        terminal_handle,
    })
}

pub(super) fn alloc_recursive_cover_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
    label: &str,
) -> Result<Vec<Rv64imRecursiveCoverClaimVar>, SynthesisError> {
    let Some((first, rest)) = claims.split_first() else {
        return Ok(Vec::new());
    };
    let mut base_claims = Vec::with_capacity(claims.len());
    let first_var = alloc_ce_claim(&mut cs.namespace(|| format!("{label}_claim_0")), first, "claim_0")?;
    let shared_r = first_var.r.clone();
    let shared_r_values = first_var.r_values.clone();
    let shared_s_col = first_var.s_col.clone();
    let shared_s_col_values = first_var.s_col_values.clone();
    base_claims.push(first_var);
    for (idx, claim) in rest.iter().enumerate() {
        base_claims.push(alloc_ce_claim_with_shared_point(
            &mut cs.namespace(|| format!("{label}_claim_{}", idx + 1)),
            claim,
            &shared_r,
            &shared_r_values,
            &shared_s_col,
            &shared_s_col_values,
            &format!("claim_{}", idx + 1),
        )?);
    }
    base_claims
        .into_iter()
        .map(|claim| Ok::<_, SynthesisError>(Rv64imRecursiveCoverClaimVar { claim }))
        .collect()
}

pub(super) fn alloc_recursive_carried_projection_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
    label: &str,
) -> Result<Vec<Rv64imRecursiveCoverClaimVar>, SynthesisError> {
    let Some((first, rest)) = claims.split_first() else {
        return Ok(Vec::new());
    };
    let mut base_claims = Vec::with_capacity(claims.len());
    let first_var =
        alloc_ce_claim_projection_surface(&mut cs.namespace(|| format!("{label}_claim_0")), first, "claim_0")?;
    let shared_r = first_var.r.clone();
    let shared_r_values = first_var.r_values.clone();
    base_claims.push(first_var);
    for (idx, claim) in rest.iter().enumerate() {
        base_claims.push(alloc_ce_claim_projection_surface_with_shared_r(
            &mut cs.namespace(|| format!("{label}_claim_{}", idx + 1)),
            claim,
            &shared_r,
            &shared_r_values,
            &format!("claim_{}", idx + 1),
        )?);
    }
    base_claims
        .into_iter()
        .map(|claim| Ok::<_, SynthesisError>(Rv64imRecursiveCoverClaimVar { claim }))
        .collect()
}

pub(super) fn carried_projection_claims_have_zero_public_tail(
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
) -> bool {
    claims.iter().all(|claim| {
        claim.c.data.iter().all(|value| *value == neo_math::F::ZERO)
            && claim
                .y_ring
                .iter()
                .all(|row| row.iter().all(|value| *value == neo_math::K::ZERO))
    })
}

pub(super) fn alloc_recursive_carried_x_r_only_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
    label: &str,
) -> Result<Vec<Rv64imRecursiveCoverClaimVar>, SynthesisError> {
    let Some((first, rest)) = claims.split_first() else {
        return Ok(Vec::new());
    };
    let mut base_claims = Vec::with_capacity(claims.len());
    let first_var = alloc_ce_claim_x_r_surface(&mut cs.namespace(|| format!("{label}_claim_0")), first, "claim_0")?;
    let shared_r = first_var.r.clone();
    let shared_r_values = first_var.r_values.clone();
    base_claims.push(first_var);
    for (idx, claim) in rest.iter().enumerate() {
        base_claims.push(alloc_ce_claim_x_r_surface_with_shared_r(
            &mut cs.namespace(|| format!("{label}_claim_{}", idx + 1)),
            claim,
            &shared_r,
            &shared_r_values,
            &format!("claim_{}", idx + 1),
        )?);
    }
    base_claims
        .into_iter()
        .map(|claim| Ok::<_, SynthesisError>(Rv64imRecursiveCoverClaimVar { claim }))
        .collect()
}

pub(crate) fn recursive_accumulator_instance_digest_circuit_from_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[CeClaimVar],
    terminal_handle: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_domain")),
        &packed_bytes_field_values(b"neo.fold.next/rv64im/main_recursion_recursive_accumulator_instance/v2"),
        &format!("{label}_domain"),
    )?;
    preimage.extend(alloc_const_field_values(
        &mut cs.namespace(|| format!("{label}_claim_count")),
        &[SpartanF::from_canonical_u64(claims.len() as u64)],
        &format!("{label}_claim_count"),
    )?);
    preimage.extend(terminal_handle.iter().cloned());
    for (claim_index, claim) in claims.iter().enumerate() {
        let claim_digest = me_input_projection_digest_poseidon(
            &mut cs.namespace(|| format!("{label}_claim_hash_{claim_index}")),
            claim,
            &format!("{label}_claim_hash_{claim_index}"),
        )?;
        preimage.extend(claim_digest.iter().cloned());
    }
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

pub(crate) fn recursive_accumulator_instance_digest_circuit_from_projection_digests<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim_projection_digests: &[[neo_math::F; 4]],
    terminal_handle: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let domain = packed_bytes_field_values(b"neo.fold.next/rv64im/main_recursion_recursive_accumulator_instance/v2");
    let mut field_terms =
        Vec::with_capacity(domain.len() + 1 + terminal_handle.len() + claim_projection_digests.len() * 4);
    let mut field_constants = Vec::with_capacity(field_terms.capacity());
    let mut field_values = Vec::with_capacity(field_terms.capacity());

    for value in domain {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }

    let claim_count = SpartanF::from_canonical_u64(claim_projection_digests.len() as u64);
    field_terms.push(Vec::new());
    field_constants.push(claim_count);
    field_values.push(claim_count);

    for lane in terminal_handle {
        let value = lane.get_value().unwrap_or(SpartanF::ZERO);
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(value);
    }

    for digest in claim_projection_digests {
        for lane in digest {
            let value = SpartanF::from_canonical_u64(lane.as_canonical_u64());
            field_terms.push(Vec::new());
            field_constants.push(value);
            field_values.push(value);
        }
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub(crate) fn debug_measure_recursive_accumulator_instance_digest_circuit_from_projection_digests_aux(
    cs: &mut FingerprintCS,
    claim_projection_digests: &[[neo_math::F; 4]],
    terminal_handle: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<Rv64imRecursiveAccumulatorProjectionDigestAuxBreakdown, SynthesisError> {
    let domain = packed_bytes_field_values(b"neo.fold.next/rv64im/main_recursion_recursive_accumulator_instance/v2");
    let mut field_terms =
        Vec::with_capacity(domain.len() + 1 + terminal_handle.len() + claim_projection_digests.len() * 4);
    let mut field_constants = Vec::with_capacity(field_terms.capacity());
    let mut field_values = Vec::with_capacity(field_terms.capacity());

    for value in domain {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }
    let claim_count = SpartanF::from_canonical_u64(claim_projection_digests.len() as u64);
    field_terms.push(Vec::new());
    field_constants.push(claim_count);
    field_values.push(claim_count);
    for lane in terminal_handle {
        let value = lane.get_value().unwrap_or(SpartanF::ZERO);
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(value);
    }
    let after_header = cs.num_aux();

    let mut after_claim_digests = Vec::with_capacity(claim_projection_digests.len());
    for digest in claim_projection_digests.iter() {
        for lane in digest {
            let value = SpartanF::from_canonical_u64(lane.as_canonical_u64());
            field_terms.push(Vec::new());
            field_constants.push(value);
            field_values.push(value);
        }
        after_claim_digests.push(cs.num_aux());
    }

    let _ = hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )?;
    let after_outer_hash = cs.num_aux();
    Ok(Rv64imRecursiveAccumulatorProjectionDigestAuxBreakdown {
        after_header,
        after_claim_digests,
        after_outer_hash,
    })
}
