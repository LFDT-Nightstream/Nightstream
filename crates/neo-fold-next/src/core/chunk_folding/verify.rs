//! Replay-witness verification for one chunk.
//!
//! This file owns verifier-side replay of `Π_CCS` plus the shared transition into
//! `Π_RLC -> Π_DEC`.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_math::F;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::optimized_engine::OptimizedStructureCache;
use neo_reductions::error::PiCcsError;
use neo_transcript::Poseidon2Transcript;
use std::time::Instant;

use crate::proof::{Carry, ChunkInput, ChunkProvePerf};

use super::pi_ccs::verify_pi_ccs_replay_witness;
use super::prepare::prepare_chunk_ccs_inputs;
use super::result::chunk_relation_result_from_parts;
use super::transition::finish_chunk_transition_core_with_perf;
use super::types::{ChunkRelationResult, ChunkReplayWitness, CommitmentMixers};

pub fn verify_chunk_relation_with_witness<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<ChunkRelationResult, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(verify_chunk_relation_with_witness_and_instance_digest(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        replay_witness,
        log,
        mixers,
        optimized_cache,
        None,
    )?
    .0)
}

pub(crate) fn verify_chunk_relation_with_witness_and_instance_digest<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: Option<[F; 4]>,
) -> Result<(ChunkRelationResult, [u8; 32]), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(verify_chunk_relation_with_witness_and_instance_digest_with_perf(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        replay_witness,
        log,
        mixers,
        optimized_cache,
        public_chunk_instance_digest,
    )?
    .0)
}

pub(crate) fn verify_chunk_relation_with_witness_and_instance_digest_with_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: Option<[F; 4]>,
) -> Result<((ChunkRelationResult, [u8; 32]), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_chunk_relation_with_witness_and_instance_digest_with_perf_inner(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        replay_witness,
        log,
        mixers,
        optimized_cache,
        public_chunk_instance_digest,
        None,
    )
}

pub(crate) fn verify_chunk_relation_with_witness_and_instance_digest_and_me_input_handle_with_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
) -> Result<((ChunkRelationResult, [u8; 32]), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_chunk_relation_with_witness_and_instance_digest_with_perf_inner(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        replay_witness,
        log,
        mixers,
        optimized_cache,
        Some(public_chunk_instance_digest),
        Some(me_input_accumulator_handle),
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_chunk_relation_with_witness_and_instance_digest_with_perf_inner<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    replay_witness: &ChunkReplayWitness,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: Option<[F; 4]>,
    me_input_accumulator_handle: Option<[F; 4]>,
) -> Result<((ChunkRelationResult, [u8; 32]), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, public_chunk_instance_digest)?;
    let ccs = verify_pi_ccs_replay_witness(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        incoming_main,
        replay_witness,
        prepared.public_chunk_digest,
        me_input_accumulator_handle,
        optimized_cache,
    )?;
    let (transition, perf) = finish_chunk_transition_core_with_perf(
        total_started,
        FoldingMode::Optimized,
        tr,
        params,
        s,
        prepared.start_index,
        prepared.fresh_step_count,
        incoming_main,
        log,
        mixers,
        Some(optimized_cache),
        prepared.prepare_inputs_ms,
        &prepared.fresh_witnesses,
        &replay_witness.ccs_outputs,
        ccs.fold_digest,
        ccs.perf,
        ccs.elapsed_ms,
    )?;
    Ok((
        (
            chunk_relation_result_from_parts(
                &replay_witness.ccs_outputs,
                transition.parent,
                transition.children,
                transition.z_split,
            ),
            ccs.fold_digest,
        ),
        perf,
    ))
}
