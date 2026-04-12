//! Owns the generic chunk-relation evaluation spine.
//!
//! Ownership:
//! - evaluates the `Π_CCS -> Π_RLC -> Π_DEC` transition for one chunk relation
//! - packages the relation result independently of any VM-specific recursive wrapper
//! - does not own theorem-facing proof export surfaces
//! - does not own session/run orchestration

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit, prove, rlc_with_commit, sample_rot_rhos_n_typed, split_b_matrix_k_with_nonzero_flags,
    FoldingMode, PiCcsProof, RotRing,
};
use neo_reductions::engines::utils::{self, me_digest_poseidon};
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::{
    optimized_prove_with_cache_and_instance_digest_and_perf,
    optimized_replay_outputs_with_cache_and_instance_digest_and_perf,
    optimized_replay_trace_with_cache_and_instance_digest_and_perf,
    optimized_replay_witness_with_cache_and_instance_digest_and_perf, OptimizedStructureCache, PiCcsReplayProofWitness,
    PiCcsReplayTerminalState,
};
use neo_reductions::pi_rlc_dec::OptimizedRlcDec;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::finalize::public_chunk_digest;
use crate::proof::{
    Carry, ChunkInput, ChunkProof, ChunkProvePerf, ChunkResult, PiDecArtifact, PiRlcArtifact, ProverChunkInput,
    PublicChunk,
};

const CHUNK_META_RAW_TAG: u64 = 14;
const STEP_INDEX_RAW_TAG: u64 = 15;

#[derive(Clone, Copy)]
pub struct CommitmentMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

#[derive(Clone, Debug)]
pub struct ChunkRelationArtifacts {
    pub relation_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct ChunkRelationResult {
    pub next_main: Carry,
    pub artifacts: ChunkRelationArtifacts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkReplayWitness {
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub ccs_replay_proof: PiCcsReplayProofWitness,
}

pub(crate) struct ChunkReplayTrace {
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub ccs_replay_proof: PiCcsReplayProofWitness,
    pub terminal_state: PiCcsReplayTerminalState,
    pub parent: CeClaim<Commitment, F, K>,
    pub children: Vec<CeClaim<Commitment, F, K>>,
    pub z_split: Vec<Mat<F>>,
}

struct ChunkPreparedInputs {
    start_index: usize,
    fresh_step_count: usize,
    fresh_claims: Vec<CcsClaim<Commitment, F>>,
    fresh_witnesses: Vec<CcsWitness<F>>,
    public_chunk_digest: [F; 4],
    prepare_inputs_ms: f64,
}

struct BorrowedChunkPreparedInputs<'a> {
    start_index: usize,
    fresh_step_count: usize,
    fresh_claims: &'a [CcsClaim<Commitment, F>],
    fresh_witnesses: &'a [CcsWitness<F>],
    public_chunk_digest: [F; 4],
    prepare_inputs_ms: f64,
}

struct CcsTransitionState {
    ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    parent: CeClaim<Commitment, F, K>,
    children: Vec<CeClaim<Commitment, F, K>>,
    z_split: Vec<Mat<F>>,
}

pub(crate) struct ChunkComputation {
    transition: CcsTransitionState,
    ccs_proof: neo_reductions::api::PiCcsProof,
}

impl CcsTransitionState {
    fn into_relation_result(self) -> Result<ChunkRelationResult, PiCcsError> {
        Ok(chunk_relation_result_from_transition(self))
    }
}

impl ChunkComputation {
    pub(crate) fn into_chunk_result(self, chunk: &ChunkInput) -> ChunkResult {
        chunk_result_from_transition(self.transition, chunk.public(), self.ccs_proof)
    }

    pub(crate) fn into_chunk_result_with_public_chunk(self, public_chunk: PublicChunk) -> ChunkResult {
        chunk_result_from_transition(self.transition, public_chunk, self.ccs_proof)
    }
}

fn chunk_result_from_transition(
    transition: CcsTransitionState,
    public_chunk: PublicChunk,
    ccs_proof: neo_reductions::api::PiCcsProof,
) -> ChunkResult {
    let CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
        ..
    } = transition;
    let relation_digest = chunk_relation_digest(&ccs_outputs, &parent, &children);
    ChunkResult {
        proof: ChunkProof {
            chunk: public_chunk,
            relation_digest,
            ccs_outputs,
            ccs_proof,
            rlc: PiRlcArtifact { parent },
            dec: PiDecArtifact {
                children: children.clone(),
            },
        },
        next_main: Carry {
            claims: children,
            witnesses: z_split,
        },
    }
}

fn chunk_fresh_witness_mats<'a>(fresh_witnesses: &'a [CcsWitness<F>]) -> impl Iterator<Item = Mat<F>> + 'a {
    fresh_witnesses.iter().map(|witness| witness.Z.clone())
}

fn chunk_relation_result_from_transition(transition: CcsTransitionState) -> ChunkRelationResult {
    let CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
    } = transition;
    let relation_digest = chunk_relation_digest(&ccs_outputs, &parent, &children);
    ChunkRelationResult {
        next_main: Carry {
            claims: children,
            witnesses: z_split,
        },
        artifacts: ChunkRelationArtifacts { relation_digest },
    }
}

fn chunk_replay_witness_and_result_from_parts(
    transition: CcsTransitionState,
    ccs_replay_proof: PiCcsReplayProofWitness,
) -> (ChunkReplayWitness, ChunkRelationResult) {
    let CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
    } = transition;
    let relation_digest = chunk_relation_digest(&ccs_outputs, &parent, &children);
    (
        ChunkReplayWitness {
            ccs_outputs,
            ccs_replay_proof,
        },
        ChunkRelationResult {
            next_main: Carry {
                claims: children,
                witnesses: z_split,
            },
            artifacts: ChunkRelationArtifacts { relation_digest },
        },
    )
}

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
    verify_chunk_relation_with_witness_and_instance_digest(
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
    )
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
) -> Result<ChunkRelationResult, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, public_chunk_instance_digest)?;
    let ccs_proof = replay_witness.ccs_replay_proof.to_pi_ccs_proof();
    let (ok, _perf) = neo_reductions::optimized_engine::optimized_verify_with_cache_and_instance_digest_and_perf(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &incoming_main.claims,
        &replay_witness.ccs_outputs,
        &ccs_proof,
        optimized_cache,
        prepared.public_chunk_digest,
    )?;
    if !ok {
        return Err(PiCcsError::ProtocolError(
            "optimized replay witness does not verify against chunk relation".into(),
        ));
    }
    let expected_fold_digest = replay_witness.ccs_replay_proof.header_digest;
    let fold_digest = tr.digest32();
    if fold_digest != expected_fold_digest {
        return Err(PiCcsError::ProtocolError(
            "optimized replay witness header digest does not match transcript replay".into(),
        ));
    }
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
        replay_witness.ccs_outputs.clone(),
        fold_digest,
        neo_reductions::optimized_engine::PiCcsProvePerf::default(),
        0.0,
    )?;
    transition.into_relation_result()
}

pub(crate) fn trace_chunk_relation_with_witness_and_instance_digest<L, MR, MB>(
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
) -> Result<ChunkReplayTrace, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, Some(public_chunk_instance_digest))?;
    let (terminal_state, derived_replay_proof) = optimized_replay_trace_with_cache_and_instance_digest_and_perf(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &prepared.fresh_witnesses,
        &incoming_main.claims,
        &incoming_main.witnesses,
        prepared.public_chunk_digest,
        log,
        optimized_cache,
    )?;
    if terminal_state.me_outputs != replay_witness.ccs_outputs {
        return Err(PiCcsError::ProtocolError(
            "optimized replay outputs do not match the carried chunk replay witness outputs".into(),
        ));
    }
    if derived_replay_proof != replay_witness.ccs_replay_proof {
        return Err(PiCcsError::ProtocolError(
            "optimized replay proof rounds do not match the carried chunk replay witness".into(),
        ));
    }
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
        terminal_state.me_outputs.clone(),
        terminal_state.fold_digest,
        terminal_state.perf,
        terminal_state.perf.total_ms,
    )?;
    Ok(ChunkReplayTrace {
        ccs_outputs: transition.ccs_outputs,
        ccs_replay_proof: replay_witness.ccs_replay_proof.clone(),
        terminal_state,
        parent: transition.parent,
        children: transition.children,
        z_split: transition.z_split,
    })
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

pub(crate) fn compute_chunk_replay_witness_and_relation_with_perf<L, MR, MB>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: &OptimizedStructureCache,
) -> Result<((ChunkReplayWitness, ChunkRelationResult), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf(
        tr,
        params,
        s,
        chunk,
        incoming_main,
        log,
        mixers,
        optimized_cache,
        None,
    )
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
) -> Result<((ChunkReplayWitness, ChunkRelationResult), ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let prepared = prepare_chunk_ccs_inputs(tr, chunk, incoming_main, public_chunk_instance_digest)?;
    let ccs_started = Instant::now();
    let replay = optimized_replay_witness_with_cache_and_instance_digest_and_perf(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &prepared.fresh_witnesses,
        &incoming_main.claims,
        &incoming_main.witnesses,
        prepared.public_chunk_digest,
        log,
        optimized_cache,
    )?;
    let ccs_ms = ccs_started.elapsed().as_secs_f64() * 1_000.0;
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
        replay.me_outputs,
        replay.replay_proof.header_digest,
        replay.perf,
        ccs_ms,
    )?;
    Ok((
        chunk_replay_witness_and_result_from_parts(transition, replay.replay_proof),
        perf,
    ))
}

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
    let ccs_started = Instant::now();
    let (ccs_outputs, ccs_proof, ccs_perf) = if matches!(mode, FoldingMode::Optimized) {
        let cache = optimized_cache.ok_or_else(|| {
            PiCcsError::InvalidInput("missing optimized structure cache for optimized chunk relation".into())
        })?;
        optimized_prove_with_cache_and_instance_digest_and_perf(
            tr,
            params,
            s,
            &prepared.fresh_claims,
            &prepared.fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            prepared.public_chunk_digest,
            log,
            cache,
        )?
    } else {
        let (ccs_outputs, ccs_proof) = prove(
            mode.clone(),
            tr,
            params,
            s,
            &prepared.fresh_claims,
            &prepared.fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            log,
        )?;
        (
            ccs_outputs,
            ccs_proof,
            neo_reductions::optimized_engine::PiCcsProvePerf::default(),
        )
    };
    let ccs_ms = ccs_started.elapsed().as_secs_f64() * 1_000.0;
    let fold_digest = fold_digest_from_proof(&ccs_proof)?;
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
        ccs_outputs,
        fold_digest,
        ccs_perf,
        ccs_ms,
    )?;
    Ok((ChunkComputation { transition, ccs_proof }, perf))
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
    let ccs_started = Instant::now();
    let (ccs_outputs, ccs_proof, ccs_perf) = if matches!(mode, FoldingMode::Optimized) {
        let cache = optimized_cache.ok_or_else(|| {
            PiCcsError::InvalidInput("missing optimized structure cache for optimized chunk relation".into())
        })?;
        optimized_prove_with_cache_and_instance_digest_and_perf(
            tr,
            params,
            s,
            prepared.fresh_claims,
            prepared.fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            prepared.public_chunk_digest,
            log,
            cache,
        )?
    } else {
        let (ccs_outputs, ccs_proof) = prove(
            mode.clone(),
            tr,
            params,
            s,
            prepared.fresh_claims,
            prepared.fresh_witnesses,
            &incoming_main.claims,
            &incoming_main.witnesses,
            log,
        )?;
        (
            ccs_outputs,
            ccs_proof,
            neo_reductions::optimized_engine::PiCcsProvePerf::default(),
        )
    };
    let ccs_ms = ccs_started.elapsed().as_secs_f64() * 1_000.0;
    let fold_digest = fold_digest_from_proof(&ccs_proof)?;
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
        ccs_outputs,
        fold_digest,
        ccs_perf,
        ccs_ms,
    )?;
    Ok((ChunkComputation { transition, ccs_proof }, perf))
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
    let ccs_started = Instant::now();
    let replay = optimized_replay_outputs_with_cache_and_instance_digest_and_perf(
        tr,
        params,
        s,
        &prepared.fresh_claims,
        &prepared.fresh_witnesses,
        &incoming_main.claims,
        &incoming_main.witnesses,
        prepared.public_chunk_digest,
        log,
        optimized_cache,
    )?;
    let ccs_ms = ccs_started.elapsed().as_secs_f64() * 1_000.0;
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
        replay.me_outputs,
        replay.fold_digest,
        replay.perf,
        ccs_ms,
    )
}

fn prepare_chunk_ccs_inputs(
    tr: &mut Poseidon2Transcript,
    chunk: &ChunkInput,
    incoming_main: &Carry,
    public_chunk_instance_digest: Option<[F; 4]>,
) -> Result<ChunkPreparedInputs, PiCcsError> {
    validate_main_carry("replay_chunk_relation", incoming_main)?;
    validate_chunk_input(chunk)?;
    append_chunk_transcript(tr, chunk);

    let prepare_inputs_started = Instant::now();
    let fresh_claims = chunk
        .steps
        .iter()
        .map(|step| step.mcs.clone())
        .collect::<Vec<_>>();
    let fresh_witnesses = chunk
        .steps
        .iter()
        .map(|step| step.witness.clone())
        .collect::<Vec<_>>();
    let public_chunk_digest = public_chunk_instance_digest.unwrap_or_else(|| public_chunk_digest(&chunk.public()));
    Ok(ChunkPreparedInputs {
        start_index: chunk.start_index,
        fresh_step_count: chunk.steps.len(),
        fresh_claims,
        fresh_witnesses,
        public_chunk_digest,
        prepare_inputs_ms: prepare_inputs_started.elapsed().as_secs_f64() * 1_000.0,
    })
}

fn prepare_prover_chunk_ccs_inputs<'a>(
    tr: &mut Poseidon2Transcript,
    chunk: &'a ProverChunkInput,
    incoming_main: &Carry,
) -> Result<BorrowedChunkPreparedInputs<'a>, PiCcsError> {
    validate_main_carry("replay_chunk_relation", incoming_main)?;
    validate_public_chunk_input(&chunk.public_chunk)?;
    append_public_chunk_transcript(tr, &chunk.public_chunk);

    let prepare_inputs_started = Instant::now();
    Ok(BorrowedChunkPreparedInputs {
        start_index: chunk.start_index(),
        fresh_step_count: chunk.fresh_step_count(),
        fresh_claims: &chunk.fresh_claims,
        fresh_witnesses: &chunk.fresh_witnesses,
        public_chunk_digest: public_chunk_digest(&chunk.public_chunk),
        prepare_inputs_ms: prepare_inputs_started.elapsed().as_secs_f64() * 1_000.0,
    })
}

#[allow(clippy::too_many_arguments)]
fn finish_chunk_transition_with_perf<L, MR, MB>(
    total_started: Instant,
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunk_start_index: usize,
    fresh_step_count: usize,
    incoming_main: &Carry,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    optimized_cache: Option<&OptimizedStructureCache>,
    prepare_inputs_ms: f64,
    fresh_witnesses: &[CcsWitness<F>],
    ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    fold_digest: [u8; 32],
    ccs_perf: neo_reductions::optimized_engine::PiCcsProvePerf,
    ccs_ms: f64,
) -> Result<(CcsTransitionState, ChunkProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    validate_ccs_outputs(
        chunk_start_index,
        fresh_step_count,
        incoming_main,
        &ccs_outputs,
        fold_digest,
    )?;

    let dims_started = Instant::now();
    let dims = utils::build_dims_and_policy(params, s)?;
    let dims_ms = dims_started.elapsed().as_secs_f64() * 1_000.0;
    let rlc_rhos = sample_rlc_rhos(tr, params, ccs_outputs.len())?;

    let rlc_prepare_started = Instant::now();
    let mut rlc_inputs_wit = Vec::with_capacity(fresh_step_count + incoming_main.witnesses.len());
    rlc_inputs_wit.extend(chunk_fresh_witness_mats(fresh_witnesses));
    rlc_inputs_wit.extend(incoming_main.witnesses.iter().cloned());
    let rlc_prepare_ms = rlc_prepare_started.elapsed().as_secs_f64() * 1_000.0;

    let rlc_started = Instant::now();
    let (parent, z_mix) = rlc_with_commit(
        mode.clone(),
        s,
        params,
        &rlc_rhos,
        &ccs_outputs,
        &rlc_inputs_wit,
        dims.ell_d,
        mixers.mix_rhos_commits,
    )?;
    let rlc_ms = rlc_started.elapsed().as_secs_f64() * 1_000.0;

    let k_dec = params.k_rho as usize;
    let dec_split_started = Instant::now();
    let (z_split, digit_nonzero) = split_b_matrix_k_with_nonzero_flags(&z_mix, k_dec, params.b)?;
    let dec_split_ms = dec_split_started.elapsed().as_secs_f64() * 1_000.0;
    let dec_commit_started = Instant::now();
    let child_commitments = commit_split_children(log, &z_split, &digit_nonzero)?;
    let dec_commit_ms = dec_commit_started.elapsed().as_secs_f64() * 1_000.0;
    let dec_started = Instant::now();
    let (children, ok_y, ok_x, ok_c) = if matches!(mode, FoldingMode::Optimized) {
        let cache = optimized_cache
            .ok_or_else(|| PiCcsError::InvalidInput("missing optimized structure cache for optimized DEC".into()))?;
        OptimizedRlcDec::dec_children_with_commit_cached(
            s,
            params,
            &parent,
            &z_split,
            dims.ell_d,
            &child_commitments,
            mixers.combine_b_pows,
            Some(cache.sparse()),
        )
    } else {
        dec_children_with_commit(
            mode,
            s,
            params,
            &parent,
            &z_split,
            dims.ell_d,
            &child_commitments,
            mixers.combine_b_pows,
        )
    };
    let dec_ms = dec_started.elapsed().as_secs_f64() * 1_000.0;
    if !(ok_y && ok_x && ok_c) {
        return Err(PiCcsError::ProtocolError(format!(
            "Π_DEC public checks failed for chunk starting at {}: y={}, X={}, c={}",
            chunk_start_index, ok_y, ok_x, ok_c
        )));
    }

    let ccs_output_count = ccs_outputs.len();
    let dec_children = children.len();
    let transition = CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
    };
    let perf = ChunkProvePerf {
        start_index: chunk_start_index,
        fresh_steps: fresh_step_count,
        incoming_main_claims: incoming_main.claims.len(),
        ccs_outputs: ccs_output_count,
        dec_children,
        prepare_inputs_ms,
        ccs_bind_ms: ccs_perf.bind_ms,
        ccs_sample_challenges_ms: ccs_perf.sample_challenges_ms,
        ccs_fe_sumcheck_ms: ccs_perf.fe_sumcheck_ms,
        ccs_nc_sumcheck_ms: ccs_perf.nc_sumcheck_ms,
        ccs_output_materialize_ms: ccs_perf.output_materialize_ms,
        ccs_ms,
        dims_ms,
        rlc_prepare_ms,
        rlc_ms,
        dec_split_ms,
        dec_commit_ms,
        dec_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
    };
    Ok((transition, perf))
}

fn append_chunk_transcript(tr: &mut Poseidon2Transcript, chunk: &ChunkInput) {
    append_public_chunk_transcript(tr, &chunk.public());
}

fn append_public_chunk_transcript(tr: &mut Poseidon2Transcript, chunk: &PublicChunk) {
    if chunk.steps.len() == 1 {
        tr.append_fields_raw(&[F::from_u64(STEP_INDEX_RAW_TAG), F::from_u64(chunk.start_index as u64)]);
        return;
    }

    tr.append_fields_raw(&[
        F::from_u64(CHUNK_META_RAW_TAG),
        F::from_u64(chunk.start_index as u64),
        F::from_u64(chunk.steps.len() as u64),
    ]);
}

fn validate_main_carry(context: &str, carry: &Carry) -> Result<(), PiCcsError> {
    if carry.claims.len() != carry.witnesses.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "{context}: |claims|={} != |witnesses|={}",
            carry.claims.len(),
            carry.witnesses.len()
        )));
    }
    Ok(())
}

fn validate_ccs_outputs(
    chunk_start_index: usize,
    fresh_step_count: usize,
    incoming_main: &Carry,
    ccs_outputs: &[CeClaim<Commitment, F, K>],
    fold_digest: [u8; 32],
) -> Result<(), PiCcsError> {
    let expected = fresh_step_count
        .checked_add(incoming_main.claims.len())
        .ok_or_else(|| PiCcsError::InvalidInput("Π_CCS output count overflow".into()))?;
    if ccs_outputs.len() != expected {
        return Err(PiCcsError::ProtocolError(format!(
            "Π_CCS returned {} outputs for chunk starting at {}, expected {}",
            ccs_outputs.len(),
            chunk_start_index,
            expected
        )));
    }
    for (idx, out) in ccs_outputs.iter().enumerate() {
        if out.fold_digest != fold_digest {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS output[{idx}] fold_digest mismatch for chunk starting at {}",
                chunk_start_index
            )));
        }
    }
    Ok(())
}

fn fold_digest_from_proof(ccs_proof: &PiCcsProof) -> Result<[u8; 32], PiCcsError> {
    ccs_proof
        .header_digest
        .as_slice()
        .try_into()
        .map_err(|_| PiCcsError::ProtocolError("Π_CCS header digest must be 32 bytes".into()))
}

fn validate_chunk_input(chunk: &ChunkInput) -> Result<(), PiCcsError> {
    validate_public_chunk_input(&chunk.public())
}

fn validate_public_chunk_input(chunk: &PublicChunk) -> Result<(), PiCcsError> {
    if chunk.steps.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "chunk relation evaluation requires at least one fresh step".into(),
        ));
    }
    Ok(())
}

fn sample_rlc_rhos(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    input_count: usize,
) -> Result<Vec<neo_reductions::api::RotRho>, PiCcsError> {
    let ring = RotRing::goldilocks();
    sample_rot_rhos_n_typed(tr, params, &ring, input_count)
}

fn commit_split_children<L>(log: &L, z_split: &[Mat<F>], digit_nonzero: &[bool]) -> Result<Vec<Commitment>, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
{
    if z_split.len() != digit_nonzero.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC split mismatch: |Z_split|={} != |digit_nonzero|={}",
            z_split.len(),
            digit_nonzero.len()
        )));
    }
    if z_split.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "DEC requires at least one child witness".into(),
        ));
    }

    let zero = log.commit(&Mat::zero(z_split[0].rows(), z_split[0].cols(), F::ZERO));
    let mut child_commitments = vec![zero.clone(); z_split.len()];
    let nonzero_idx: Vec<usize> = digit_nonzero
        .iter()
        .enumerate()
        .filter_map(|(idx, &nz)| nz.then_some(idx))
        .collect();
    if nonzero_idx.is_empty() {
        return Ok(child_commitments);
    }

    let mats: Vec<&Mat<F>> = nonzero_idx.iter().map(|&idx| &z_split[idx]).collect();
    let commits = log.commit_many(&mats);
    if commits.len() != mats.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "DEC commit_many returned {} commitments for {} matrices",
            commits.len(),
            mats.len()
        )));
    }
    for (pos, &idx) in nonzero_idx.iter().enumerate() {
        child_commitments[idx] = commits[pos].clone();
    }
    Ok(child_commitments)
}

pub(crate) fn chunk_relation_digest(
    ccs_outputs: &[neo_ccs::CeClaim<Commitment, F, K>],
    parent: &neo_ccs::CeClaim<Commitment, F, K>,
    children: &[neo_ccs::CeClaim<Commitment, F, K>],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chunk_relation_digest");
    tr.append_u64s(
        b"neo.fold.next/chunk_relation_digest/counts",
        &[ccs_outputs.len() as u64, children.len() as u64],
    );
    for digest in claim_digests(ccs_outputs) {
        tr.append_fields(b"neo.fold.next/chunk_relation_digest/ccs_output", &digest);
    }
    tr.append_fields(
        b"neo.fold.next/chunk_relation_digest/rlc_parent",
        &me_digest_poseidon(parent),
    );
    for claim in children {
        tr.append_fields(
            b"neo.fold.next/chunk_relation_digest/dec_child",
            &me_digest_poseidon(claim),
        );
    }
    tr.digest32()
}

pub(crate) fn claim_digests(claims: &[CeClaim<Commitment, F, K>]) -> Vec<[F; 4]> {
    #[cfg(not(target_arch = "wasm32"))]
    let allow_parallel = rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none();
    #[cfg(target_arch = "wasm32")]
    let _allow_parallel = false;

    #[cfg(not(target_arch = "wasm32"))]
    if allow_parallel && claims.len() >= 8 {
        return claims.par_iter().map(me_digest_poseidon).collect();
    }

    let mut digests = Vec::with_capacity(claims.len());
    let mut scratch = Vec::<F>::with_capacity(2048);
    for claim in claims {
        digests.push(utils::me_digest_poseidon_into(&mut scratch, claim));
    }
    digests
}
