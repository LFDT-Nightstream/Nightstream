//! Replay-oriented chunk folding flows.
//!
//! This file owns native replay and replay-witness construction for one chunk.

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

use super::pi_ccs::{compute_pi_ccs_replay_witness, replay_pi_ccs_outputs};
use super::prepare::prepare_chunk_ccs_inputs;
use super::result::chunk_replay_witness_and_result_from_parts;
use super::transition::finish_chunk_transition_with_perf;
use super::types::{CcsTransitionState, ChunkRelationResult, ChunkReplayWitness, CommitmentMixers};

pub fn replay_chunk_relation<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<ChunkRelationResult, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(replay_chunk_relation_with_perf(tr, params, s, chunk, incoming_main, log, mixers, optimized_cache)?.0)
}

pub fn replay_chunk_relation_with_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<(ChunkRelationResult, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let (transition, perf) =
        compute_replay_chunk_relation_with_perf(tr, params, s, chunk, incoming_main, log, mixers, optimized_cache)?;
    Ok((transition.into_relation_result()?, perf))
}

pub(crate) fn compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: Option<[F; 4]>,
) -> Result<((ChunkReplayWitness, ChunkRelationResult, [u8; 32]), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf_inner(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        log,
        mixers,
        optimized_cache,
        public_chunk_instance_digest,
        None,
    )
}

pub(crate) fn compute_chunk_replay_witness_and_relation_with_instance_digest_and_me_input_handle_and_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: [F; 4],
    me_input_accumulator_handle: [F; 4],
) -> Result<((ChunkReplayWitness, ChunkRelationResult, [u8; 32]), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf_inner(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        log,
        mixers,
        optimized_cache,
        Some(public_chunk_instance_digest),
        Some(me_input_accumulator_handle),
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf_inner<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
    public_chunk_instance_digest: Option<[F; 4]>,
    me_input_accumulator_handle: Option<[F; 4]>,
) -> Result<((ChunkReplayWitness, ChunkRelationResult, [u8; 32]), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, public_chunk_instance_digest)?;
    let ccs = compute_pi_ccs_replay_witness(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &prepared.fresh_witnesses,
        incoming_main,
        prepared.public_chunk_digest,
        me_input_accumulator_handle,
        log,
        optimized_cache,
    )?;
    let (transition, perf) = finish_chunk_transition_with_perf(
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
        ccs.outputs,
        ccs.fold_digest,
        ccs.perf,
        ccs.elapsed_ms,
    )?;
    let (replay_witness, relation_result) = chunk_replay_witness_and_result_from_parts(transition, ccs.replay_proof);
    Ok(((replay_witness, relation_result, ccs.fold_digest), perf))
}

fn compute_replay_chunk_relation_with_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<(CcsTransitionState, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, None)?;
    let ccs = replay_pi_ccs_outputs(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &prepared.fresh_witnesses,
        incoming_main,
        prepared.public_chunk_digest,
        log,
        optimized_cache,
    )?;
    finish_chunk_transition_with_perf(
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
        ccs.outputs,
        ccs.fold_digest,
        ccs.perf,
        ccs.elapsed_ms,
    )
}
