//! Owns recursive fixed-step cover allocation, equality, and digest gadgets.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::fingerprint_cs::FingerprintCS;
use crate::rv64im::final_relation::rv64im_recursive_accumulator_instance_digest_from_parts;
use crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot;
use crate::rv64im::ivc_snark::SpartanF;
use crate::rv64im::kernel::rv64im_root_main_lane_context_for_claim_count;
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_dec_surface, alloc_ce_claim_dec_surface_with_shared_r,
    alloc_ce_claim_with_shared_point, alloc_ce_claim_x_r_surface, alloc_ce_claim_x_r_surface_with_shared_r,
    packed_bytes_field_values, CeClaimVar,
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
    pub(super) folded_accumulator_digest: [AllocatedNum<SpartanF>; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rv64imRecursiveAccumulatorProjectionDigestAuxBreakdown {
    pub after_header: usize,
    pub after_claim_digests: Vec<usize>,
    pub after_outer_hash: usize,
}

fn alloc_recursive_cover_public_state_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
    transcript: &Rv64imChunkFoldTranscriptSnapshot,
    terminal_handle_digest: [u8; 32],
    label: &str,
) -> Result<
    (
        [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH],
        AllocatedNum<SpartanF>,
        [AllocatedNum<SpartanF>; 4],
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
    let folded_accumulator_digest =
        rv64im_recursive_accumulator_instance_digest_from_parts(claims, terminal_handle_digest);
    let folded_accumulator_digest = super::digest32_as_spartan_fields(folded_accumulator_digest)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(
                cs.namespace(|| format!("{label}_folded_accumulator_digest_{idx}")),
                || Ok(value),
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    Ok((
        transcript_state,
        transcript_absorbed,
        terminal_handle,
        folded_accumulator_digest,
    ))
}

pub(super) fn alloc_recursive_cover_state<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>],
    transcript: &Rv64imChunkFoldTranscriptSnapshot,
    terminal_handle_digest: [u8; 32],
    label: &str,
) -> Result<Rv64imRecursiveCoverStateVar, SynthesisError> {
    let (transcript_state, transcript_absorbed, terminal_handle, folded_accumulator_digest) =
        alloc_recursive_cover_public_state_fields(cs, claims, transcript, terminal_handle_digest, label)?;
    Ok(Rv64imRecursiveCoverStateVar {
        transcript_state,
        transcript_absorbed,
        terminal_handle,
        folded_accumulator_digest,
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
    let first_var = alloc_ce_claim_dec_surface(&mut cs.namespace(|| format!("{label}_claim_0")), first, "claim_0")?;
    let shared_r = first_var.r.clone();
    let shared_r_values = first_var.r_values.clone();
    base_claims.push(first_var);
    for (idx, claim) in rest.iter().enumerate() {
        base_claims.push(alloc_ce_claim_dec_surface_with_shared_r(
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
    recursive_accumulator_instance_digest_circuit_from_phi_dec_parent_vars(cs, claims, terminal_handle, label)
}

pub(crate) fn debug_measure_recursive_accumulator_instance_digest_circuit_from_claims_aux(
    cs: &mut FingerprintCS,
    claims: &[CeClaimVar],
    terminal_handle: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<Rv64imRecursiveAccumulatorProjectionDigestAuxBreakdown, SynthesisError> {
    let after_header = cs.num_aux();

    let mut after_claim_digests = Vec::with_capacity(claims.len());
    for _claim in claims {
        after_claim_digests.push(cs.num_aux());
    }

    let _ = recursive_accumulator_instance_digest_circuit_from_phi_dec_parent_vars(cs, claims, terminal_handle, label)?;
    let after_outer_hash = cs.num_aux();
    Ok(Rv64imRecursiveAccumulatorProjectionDigestAuxBreakdown {
        after_header,
        after_claim_digests,
        after_outer_hash,
    })
}

fn recursive_accumulator_instance_digest_circuit_from_phi_dec_parent_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[CeClaimVar],
    terminal_handle: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let domain =
        packed_bytes_field_values(b"neo.fold.next/rv64im/main_recursion_recursive_accumulator_phi_dec_parent/v1");
    let parent_field_count = claims
        .first()
        .map(|claim| 1 + claim.c_data.len())
        .unwrap_or(0);
    let mut field_terms = Vec::with_capacity(domain.len() + 1 + terminal_handle.len() + parent_field_count);
    let mut field_constants = Vec::with_capacity(field_terms.capacity());
    let mut field_values = Vec::with_capacity(field_terms.capacity());

    for value in domain {
        field_terms.push(Vec::new());
        field_constants.push(value);
        field_values.push(value);
    }

    let claim_count = SpartanF::from_canonical_u64(claims.len() as u64);
    field_terms.push(Vec::new());
    field_constants.push(claim_count);
    field_values.push(claim_count);

    for lane in terminal_handle {
        let value = lane.get_value().unwrap_or(SpartanF::ZERO);
        field_terms.push(vec![(lane.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_values.push(value);
    }

    if let Some(first_claim) = claims.first() {
        let (params, _, _) =
            rv64im_root_main_lane_context_for_claim_count(claims.len()).map_err(|_| SynthesisError::Unsatisfiable)?;
        let parent_len = first_claim.c_data.len();
        let len = SpartanF::from_canonical_u64(parent_len as u64);
        field_terms.push(Vec::new());
        field_constants.push(len);
        field_values.push(len);

        let base = SpartanF::from_canonical_u64(params.b as u64);
        let mut powers = Vec::with_capacity(claims.len());
        let mut pow = SpartanF::ONE;
        for claim in claims {
            if claim.c_data.len() != parent_len || claim.c_data.len() != claim.c_data_values.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            powers.push(pow);
            pow *= base;
        }

        for lane_idx in 0..parent_len {
            let mut terms = Vec::with_capacity(claims.len());
            let mut value = SpartanF::ZERO;
            for (claim, pow) in claims.iter().zip(powers.iter().copied()) {
                terms.push((claim.c_data[lane_idx].get_variable(), pow));
                value += SpartanF::from_canonical_u64(claim.c_data_values[lane_idx].as_canonical_u64()) * pow;
            }
            field_terms.push(terms);
            field_constants.push(SpartanF::ZERO);
            field_values.push(value);
        }
    } else {
        for claim in claims {
            if !claim.c_data.is_empty() || !claim.c_data_values.is_empty() {
                return Err(SynthesisError::Unsatisfiable);
            }
        }
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}
