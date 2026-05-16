//! Diagnostic trace construction for one chunk.
//!
//! This file owns the expanded trace used by recursive/direct callers that need
//! intermediate replay surfaces, not only the final relation result.

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

use crate::proof::{Carry, ChunkInput};

use super::pi_ccs::trace_pi_ccs_replay;
use super::prepare::prepare_chunk_ccs_inputs;
use super::transition::finish_chunk_transition_with_perf;
use super::types::{ChunkReplayTrace, ChunkReplayWitness, CommitmentMixers};

pub(crate) fn trace_chunk_relation_with_witness_and_instance_digest_and_me_input_handle<L, MR, MB>(
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
) -> Result<ChunkReplayTrace, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    trace_chunk_relation_with_witness_and_instance_digest_inner(
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
        Some(me_input_accumulator_handle),
    )
}

#[allow(clippy::too_many_arguments)]
fn trace_chunk_relation_with_witness_and_instance_digest_inner<L, MR, MB>(
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
    me_input_accumulator_handle: Option<[F; 4]>,
) -> Result<ChunkReplayTrace, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, Some(public_chunk_instance_digest))?;
    let ccs = trace_pi_ccs_replay(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &prepared.fresh_witnesses,
        incoming_main,
        replay_witness,
        prepared.public_chunk_digest,
        me_input_accumulator_handle,
        log,
        optimized_cache,
    )?;
    let (transition, _perf) = finish_chunk_transition_with_perf(
        Instant::now(),
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
        ccs.terminal_state.me_outputs.clone(),
        ccs.terminal_state.fold_digest,
        ccs.terminal_state.perf,
        ccs.terminal_state.perf.total_ms,
    )?;
    Ok(ChunkReplayTrace {
        ccs_outputs: transition.ccs_outputs,
        ccs_replay_proof: ccs.replay_proof,
        ccs_post_transcript_state: ccs.post_transcript_state,
        ccs_post_transcript_absorbed: ccs.post_transcript_absorbed,
        terminal_state: ccs.terminal_state,
        parent: transition.parent,
        children: transition.children,
        z_split: transition.z_split,
    })
}
