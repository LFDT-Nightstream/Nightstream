//! Owns recursive fixed-step cover allocation, equality, and digest gadgets.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_math::KExtensions;
use p3_field::PrimeField64;
use spartan2::{bellpepper::poseidon2::hash_packed_goldilocks_fields, provider::goldi::F as SpartanF};

use super::alloc_const_field_values;
use crate::rv64im::final_relation::Rv64imChunkFoldTranscriptSnapshot;
use crate::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, alloc_ce_claim_with_shared_point, packed_bytes_field_values, CeClaimVar,
};
use crate::rv64im::main_relation_circuit::k_field::{KNum, KNumVar};

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

fn recursive_extend_f_slice_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    dst: &mut Vec<AllocatedNum<SpartanF>>,
    values: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    dst.extend(alloc_const_field_values(
        cs,
        &[SpartanF::from_canonical_u64(values.len() as u64)],
        &format!("{label}_len"),
    )?);
    dst.extend(values.iter().cloned());
    Ok(())
}

fn recursive_extend_k_slice_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    dst: &mut Vec<AllocatedNum<SpartanF>>,
    values: &[KNumVar],
    native_values: &[neo_math::K],
    label: &str,
) -> Result<(), SynthesisError> {
    if values.len() != native_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    dst.extend(alloc_const_field_values(
        cs,
        &[
            SpartanF::from_canonical_u64(values.len() as u64),
            SpartanF::from_canonical_u64(
                native_values
                    .first()
                    .map(|value| value.as_coeffs().len())
                    .unwrap_or(0) as u64,
            ),
        ],
        &format!("{label}_meta"),
    )?);
    for (idx, (value, native_value)) in values.iter().zip(native_values.iter()).enumerate() {
        let native_value = KNum::from_neo_k(*native_value);
        dst.push(AllocatedNum::alloc(
            cs.namespace(|| format!("{label}_{idx}_c0_copy")),
            || Ok(native_value.c0),
        )?);
        let last = dst.last().unwrap().clone();
        cs.enforce(
            || format!("{label}_{idx}_c0_eq"),
            |lc| lc + last.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + value.c0,
        );
        dst.push(AllocatedNum::alloc(
            cs.namespace(|| format!("{label}_{idx}_c1_copy")),
            || Ok(native_value.c1),
        )?);
        let last = dst.last().unwrap().clone();
        cs.enforce(
            || format!("{label}_{idx}_c1_eq"),
            |lc| lc + last.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + value.c1,
        );
    }
    Ok(())
}

pub(super) fn recursive_accumulator_instance_digest_circuit_from_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claims: &[CeClaimVar],
    _terminal_handle: &[AllocatedNum<SpartanF>; 4],
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
    for (claim_index, claim) in claims.iter().enumerate() {
        let mut claim_preimage = alloc_const_field_values(
            &mut cs.namespace(|| format!("{label}_claim_domain_{claim_index}")),
            &packed_bytes_field_values(b"neo/ccs/me_input_digest_poseidon/v2"),
            &format!("{label}_claim_domain_{claim_index}"),
        )?;
        recursive_extend_f_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_c_data_{claim_index}")),
            &mut claim_preimage,
            &claim.c_data,
            &format!("{label}_claim_c_data_{claim_index}"),
        )?;
        recursive_extend_f_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_x_{claim_index}")),
            &mut claim_preimage,
            &claim.x,
            &format!("{label}_claim_x_{claim_index}"),
        )?;
        recursive_extend_k_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_r_{claim_index}")),
            &mut claim_preimage,
            &claim.r,
            &claim.r_values,
            &format!("{label}_claim_r_{claim_index}"),
        )?;
        recursive_extend_k_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_s_col_{claim_index}")),
            &mut claim_preimage,
            &claim.s_col,
            &claim.s_col_values,
            &format!("{label}_claim_s_col_{claim_index}"),
        )?;
        recursive_extend_k_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_y_zcol_{claim_index}")),
            &mut claim_preimage,
            &claim.y_zcol,
            &claim.y_zcol_values,
            &format!("{label}_claim_y_zcol_{claim_index}"),
        )?;
        claim_preimage.extend(alloc_const_field_values(
            &mut cs.namespace(|| format!("{label}_claim_y_ring_len_{claim_index}")),
            &[SpartanF::from_canonical_u64(claim.y_ring.len() as u64)],
            &format!("{label}_claim_y_ring_len_{claim_index}"),
        )?);
        for (row_idx, row) in claim.y_ring.iter().enumerate() {
            recursive_extend_k_slice_as_fields(
                &mut cs.namespace(|| format!("{label}_claim_y_ring_{claim_index}_{row_idx}")),
                &mut claim_preimage,
                row,
                &claim.y_ring_values[row_idx],
                &format!("{label}_claim_y_ring_{claim_index}_{row_idx}"),
            )?;
        }
        recursive_extend_k_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_ct_{claim_index}")),
            &mut claim_preimage,
            &claim.ct,
            &claim.ct_values,
            &format!("{label}_claim_ct_{claim_index}"),
        )?;
        recursive_extend_k_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_aux_{claim_index}")),
            &mut claim_preimage,
            &claim.aux_openings,
            &claim.aux_openings_values,
            &format!("{label}_claim_aux_{claim_index}"),
        )?;
        recursive_extend_f_slice_as_fields(
            &mut cs.namespace(|| format!("{label}_claim_c_step_coords_{claim_index}")),
            &mut claim_preimage,
            &claim.c_step_coords,
            &format!("{label}_claim_c_step_coords_{claim_index}"),
        )?;
        claim_preimage.push(AllocatedNum::alloc(
            cs.namespace(|| format!("{label}_claim_m_in_{claim_index}")),
            || Ok(SpartanF::from_canonical_u64(claim.m_in as u64)),
        )?);
        claim_preimage.push(AllocatedNum::alloc(
            cs.namespace(|| format!("{label}_claim_u_offset_{claim_index}")),
            || Ok(SpartanF::from_canonical_u64(claim.u_offset as u64)),
        )?);
        claim_preimage.push(AllocatedNum::alloc(
            cs.namespace(|| format!("{label}_claim_u_len_{claim_index}")),
            || Ok(SpartanF::from_canonical_u64(claim.u_len as u64)),
        )?);
        claim_preimage.extend(claim.fold_digest_encoding.iter().cloned());
        let claim_digest = hash_packed_goldilocks_fields(
            cs.namespace(|| format!("{label}_claim_hash_{claim_index}")),
            &claim_preimage,
        )?;
        preimage.extend(claim_digest.iter().cloned());
    }
    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}
