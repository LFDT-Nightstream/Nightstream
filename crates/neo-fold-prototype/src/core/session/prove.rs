use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use std::time::Instant;

use crate::proof::{
    partition_step_inputs, Carry, ChunkInput, FoldSchedule, ProverChunkInput, RunProof, RunProvePerf, StepInput,
};
use crate::prover::{CommitmentMixers, ShardProver};

use super::cache::maybe_build_optimized_cache;
use super::layout::validate_chunk_layout;

pub fn prove_chunks<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<RunProof, PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(prove_chunks_with_perf(mode, schedule, params, s, chunks, log, mixers)?.0)
}

pub fn prove_chunks_with_cache<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<RunProof, PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(prove_chunks_with_perf_and_cache(mode, schedule, params, s, chunks, log, mixers, provided_cache)?.0)
}

pub fn prove_chunks_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    prove_chunks_with_perf_and_cache(mode, schedule, params, s, chunks, log, mixers, None)
}

pub(crate) fn prove_chunks_from_slice_with_perf_and_cache<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: &[ChunkInput],
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    prove_chunk_sequence_with_perf(mode, schedule, params, s, chunks, log, mixers, provided_cache)
}

pub fn prove_chunks_with_perf_and_cache<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let chunks = chunks.into_iter().collect::<Vec<_>>();
    prove_chunk_sequence_with_perf(mode, schedule, params, s, &chunks, log, mixers, provided_cache)
}

pub fn prove_run<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<RunProof, PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let steps_vec: Vec<StepInput> = steps.into_iter().collect();
    let chunks = partition_step_inputs(schedule, steps_vec)?;
    prove_chunks(mode, schedule, params, s, chunks, log, mixers)
}

pub fn prove_run_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let steps_vec: Vec<StepInput> = steps.into_iter().collect();
    let chunks = partition_step_inputs(schedule, steps_vec)?;
    prove_chunks_with_perf(mode, schedule, params, s, chunks, log, mixers)
}

pub(super) fn prove_prepared_chunks_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: impl IntoIterator<Item = ProverChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let (session, perf, _) =
        prove_prepared_chunks_with_final_carry_perf(mode, schedule, params, s, chunks, log, mixers)?;
    Ok((session, perf))
}

pub(super) fn prove_prepared_chunks_with_final_carry_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: impl IntoIterator<Item = ProverChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf, Carry), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = Carry::default();
    let mut session = RunProof {
        fold_schedule: schedule,
        ..RunProof::default()
    };
    let built_cache = maybe_build_optimized_cache(&mode, s, None)?;
    let optimized_cache = built_cache.as_ref();
    let mut perf = RunProvePerf::default();

    for chunk in chunks {
        let (proved, chunk_perf) = ShardProver::prove_prepared_chunk_with_perf(
            mode.clone(),
            &mut tr,
            params,
            s,
            &chunk,
            &main_carry,
            log,
            mixers,
            optimized_cache,
        )?;
        main_carry = proved.next_main;
        session.chunks.push(proved.proof);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
        perf.chunks.push(chunk_perf);
    }

    validate_chunk_layout(
        schedule,
        &session
            .chunks
            .iter()
            .map(|chunk| chunk.chunk.clone())
            .collect::<Vec<_>>(),
    )?;
    session.final_main_claims = main_carry.claims.clone();
    perf.total_ms = perf.chunks.iter().map(|chunk| chunk.total_ms).sum();
    Ok((session, perf, main_carry))
}

fn prove_chunk_sequence_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<neo_math::F>,
    chunks: &[ChunkInput],
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<neo_math::F, Commitment> + Sync,
    MR: Fn(&[Mat<neo_math::F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    schedule.validate()?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = Carry::default();
    let mut session = RunProof {
        fold_schedule: schedule,
        ..RunProof::default()
    };
    let mut perf = RunProvePerf::default();
    let built_cache = maybe_build_optimized_cache(&mode, s, provided_cache)?;
    let optimized_cache = provided_cache.or(built_cache.as_ref());

    for chunk in chunks {
        let (proved, chunk_perf) = ShardProver::prove_chunk_with_perf(
            mode.clone(),
            &mut tr,
            params,
            s,
            chunk,
            &main_carry,
            log,
            mixers,
            optimized_cache,
        )?;
        main_carry = proved.next_main;
        session.chunks.push(proved.proof);
        perf.chunks.push(chunk_perf);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
    }

    validate_chunk_layout(
        schedule,
        &session
            .chunks
            .iter()
            .map(|chunk| chunk.chunk.clone())
            .collect::<Vec<_>>(),
    )?;
    session.final_main_claims = main_carry.claims;
    perf.total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    Ok((session, perf))
}
