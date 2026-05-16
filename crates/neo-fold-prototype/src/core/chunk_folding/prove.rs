//! Prover-side chunk folding flows.
//!
//! This file owns native proving for one chunk: prepare fresh CCS inputs, run
//! `Π_CCS`, and finish with `Π_RLC -> Π_DEC`.

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

use crate::proof::{Carry, ChunkInput, ChunkProvePerf, ProverChunkInput};

use super::pi_ccs::prove_pi_ccs;
use super::prepare::{prepare_chunk_ccs_inputs, prepare_prover_chunk_ccs_inputs};
use super::transition::finish_chunk_transition_with_perf;
use super::types::{ChunkComputation, CommitmentMixers};

pub(crate) fn compute_chunk_relation_with_perf<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: Option<&OptimizedStructureCache>,
) -> Result<(ChunkComputation, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, None)?;
    let ccs = prove_pi_ccs(
        mode.clone(),
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
    let (transition, perf) = finish_chunk_transition_with_perf(
        total_started,
        mode,
        tr,
        params,
        s,
        prepared.start_index,
        prepared.fresh_step_count,
        incoming_main,
        log,
        mixers,
        optimized_cache,
        prepared.prepare_inputs_ms,
        &prepared.fresh_witnesses,
        ccs.outputs,
        ccs.fold_digest,
        ccs.perf,
        ccs.elapsed_ms,
    )?;
    Ok((
        ChunkComputation {
            transition,
            ccs_proof: ccs.proof,
        },
        perf,
    ))
}

pub(crate) fn compute_chunk_relation_for_prover_chunk_with_perf<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ProverChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: Option<&OptimizedStructureCache>,
) -> Result<(ChunkComputation, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let prepared = prepare_prover_chunk_ccs_inputs(tr, chunk, incoming_main)?;
    let ccs = prove_pi_ccs(
        mode.clone(),
        tr,
        params,
        s,
        prepared.fresh_claims,
        prepared.fresh_witnesses,
        incoming_main,
        prepared.public_chunk_digest,
        log,
        optimized_cache,
    )?;
    let (transition, perf) = finish_chunk_transition_with_perf(
        total_started,
        mode,
        tr,
        params,
        s,
        prepared.start_index,
        prepared.fresh_step_count,
        incoming_main,
        log,
        mixers,
        optimized_cache,
        prepared.prepare_inputs_ms,
        prepared.fresh_witnesses,
        ccs.outputs,
        ccs.fold_digest,
        ccs.perf,
        ccs.elapsed_ms,
    )?;
    Ok((
        ChunkComputation {
            transition,
            ccs_proof: ccs.proof,
        },
        perf,
    ))
}
