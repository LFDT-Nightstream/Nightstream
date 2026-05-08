//! Shared in-circuit SuperNeo NIFS replay entry points.
//!
//! This is the narrow reuse surface direct CCS needs from the existing
//! recursive-step implementation. The underlying RV32IM module still owns VM
//! recursive-step semantics; the exports here are the relation-neutral
//! `Π_CCS -> Π_RLC -> Π_DEC` circuit helpers.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::CcsStructure;
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::engines::utils::Dims;
use p3_goldilocks::Goldilocks;

use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanF};
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;

pub(crate) use crate::rv32im::main_relation_spartan::{digest32_as_spartan_fields, enforce_digest_eq};
pub(crate) use crate::rv32im::main_relation_trace::{
    build_rv32im_main_circuit_chunk_replay_surface as build_superneo_chunk_replay_surface,
    build_rv32im_main_circuit_pi_ccs_replay_surface as build_superneo_pi_ccs_replay_surface,
    Rv32imMainCircuitChunkCover as SuperNeoChunkCover,
    Rv32imMainCircuitChunkReplaySurface as SuperNeoChunkReplaySurface,
    Rv32imMainCircuitHandoff as SuperNeoChunkHandoff, Rv32imMainCircuitPublicInputLayout as SuperNeoPublicInputLayout,
};

pub(crate) type SuperNeoClaimBundle = crate::rv32im::main_relation_spartan::Rv32imClaimBundle;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SuperNeoNifsChunkStageBreakdown {
    pub(crate) chunk_meta: usize,
    pub(crate) pi_ccs: usize,
    pub(crate) pi_rlc: usize,
    pub(crate) pi_dec: usize,
    pub(crate) outer_relation_public_io: usize,
    pub(crate) chunk_done: usize,
    pub(crate) total: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SuperNeoNifsChunkFullBreakdown {
    pub(crate) stages: SuperNeoNifsChunkStageBreakdown,
    pub(crate) pi_ccs_details: Vec<crate::rv32im::main_relation_spartan::ConstraintStageCounts>,
}

impl SuperNeoNifsChunkStageBreakdown {
    fn from_checkpoints(
        checkpoints: crate::rv32im::main_relation_spartan::Rv32imMainRelationChunkStageCheckpoints,
        start: usize,
    ) -> Self {
        let rel = |end: usize| end.saturating_sub(start);
        Self {
            chunk_meta: rel(checkpoints.chunk_meta_end),
            pi_ccs: rel(checkpoints.pi_ccs_end).saturating_sub(rel(checkpoints.chunk_meta_end)),
            pi_rlc: rel(checkpoints.pi_rlc_end).saturating_sub(rel(checkpoints.pi_ccs_end)),
            pi_dec: rel(checkpoints.pi_dec_end).saturating_sub(rel(checkpoints.pi_rlc_end)),
            outer_relation_public_io: checkpoints
                .outer_relation_public_io_end
                .saturating_sub(checkpoints.pi_dec_end),
            chunk_done: checkpoints
                .chunk_done_end
                .saturating_sub(checkpoints.outer_relation_public_io_end),
            total: checkpoints.total_constraints().saturating_sub(start),
        }
    }
}

impl SuperNeoNifsChunkFullBreakdown {
    fn from_checkpoints(
        checkpoints: crate::rv32im::main_relation_spartan::Rv32imMainRelationChunkStageCheckpoints,
        start: usize,
    ) -> Self {
        let pi_ccs_details = checkpoints.pi_ccs_stage_counts.clone();
        Self {
            stages: SuperNeoNifsChunkStageBreakdown::from_checkpoints(checkpoints, start),
            pi_ccs_details,
        }
    }
}

pub(crate) fn synthesize_superneo_nifs_chunk<CS: ConstraintSystem<SpartanF>>(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    cs: &mut CS,
    chunk_index: usize,
    cover_chunk: &SuperNeoChunkCover,
    chunk: &SuperNeoChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: SuperNeoClaimBundle,
    me_input_accumulator_handle: Option<(&[AllocatedNum<SpartanF>; 4], [SpartanF; 4])>,
) -> Result<(SuperNeoClaimBundle, [AllocatedNum<SpartanF>; 4]), SynthesisError> {
    crate::rv32im::main_relation_spartan::synthesize_direct_ccs_nifs_chunk_with_accumulator_handle(
        params,
        structure,
        dims,
        mat_digest,
        cs,
        chunk_index,
        cover_chunk,
        chunk,
        transcript,
        carried_claims,
        me_input_accumulator_handle,
    )
}

pub(crate) fn synthesize_superneo_nifs_chunk_with_stage_breakdown(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    cs: &mut ShapeCS<NeoFoldDeciderEngine>,
    chunk_index: usize,
    cover_chunk: &SuperNeoChunkCover,
    chunk: &SuperNeoChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: SuperNeoClaimBundle,
    me_input_accumulator_handle: Option<(&[AllocatedNum<SpartanF>; 4], [SpartanF; 4])>,
) -> Result<
    (
        SuperNeoClaimBundle,
        [AllocatedNum<SpartanF>; 4],
        SuperNeoNifsChunkFullBreakdown,
    ),
    SynthesisError,
> {
    let mut public_cursor = 0usize;
    let start = cs.num_constraints();
    let (checkpoints, next, chunk_digest) =
        crate::rv32im::main_relation_spartan::debug_synthesize_rv32im_main_relation_chunk_with_stage_ranges(
            params,
            structure,
            dims,
            mat_digest,
            &[],
            cs,
            chunk_index,
            cover_chunk,
            chunk,
            &[],
            &mut public_cursor,
            transcript,
            carried_claims,
            None,
            me_input_accumulator_handle,
            crate::rv32im::main_relation_spartan::Rv32imChunkBoundaryPlan::from_boundary_mode(
                crate::rv32im::main_relation_spartan::Rv32imChunkBoundaryMode::Interior,
                chunk.fresh_claims.len(),
                chunk.pi_ccs.ccs_outputs.len(),
            ),
            false,
            false,
        )?;
    Ok((
        next,
        chunk_digest,
        SuperNeoNifsChunkFullBreakdown::from_checkpoints(checkpoints, start),
    ))
}
