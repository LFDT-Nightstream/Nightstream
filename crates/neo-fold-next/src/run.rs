//! Owns the run-level SuperNeo driver.
//!
//! This layer threads the main carry and transcript across prepared steps.

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use std::time::Instant;

use crate::finalize::{
    package_session_proof, verify_finalized_session, verify_finalized_session_with_detailed_perf_and_cache,
    verify_finalized_session_with_perf, verify_finalized_session_with_perf_and_cache, PackagedVerifyPerf,
};
use crate::proof::{
    partition_prover_step_inputs, partition_public_steps, partition_step_inputs, Carry, ChunkInput, FoldSchedule,
    PackagedProof, ProverChunkInput, PublicChunk, PublicStep, RunProof, RunProvePerf, RunVerifyPerf, StepInput,
};
use crate::prover::{CommitmentMixers, ShardProver};
use crate::verifier::ShardVerifier;

fn maybe_build_optimized_cache(
    mode: &FoldingMode,
    s: &CcsStructure<F>,
    provided: Option<&OptimizedStructureCache>,
) -> Result<Option<OptimizedStructureCache>, PiCcsError> {
    if matches!(mode, FoldingMode::Optimized) && provided.is_none() {
        Ok(Some(OptimizedStructureCache::build(s)?))
    } else {
        Ok(None)
    }
}

pub fn prove_chunks<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<RunProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(prove_chunks_with_perf(mode, schedule, params, s, chunks, log, mixers)?.0)
}

pub fn prove_chunks_with_cache<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<RunProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = Carry::default();
    let mut session = RunProof {
        fold_schedule: schedule,
        ..RunProof::default()
    };
    let built_cache = maybe_build_optimized_cache(&mode, s, provided_cache)?;
    let optimized_cache = provided_cache.or(built_cache.as_ref());

    for chunk in chunks {
        let proved = ShardProver::prove_chunk(
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
    Ok(session)
}

pub fn prove_chunks_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    prove_chunks_with_perf_and_cache(mode, schedule, params, s, chunks, log, mixers, None)
}

pub(crate) fn prove_chunks_from_slice_with_perf_and_cache<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: &[ChunkInput],
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
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

fn prove_prepared_chunks_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: impl IntoIterator<Item = ProverChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
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
    session.final_main_claims = main_carry.claims;
    perf.total_ms = perf.chunks.iter().map(|chunk| chunk.total_ms).sum();
    Ok((session, perf))
}

pub fn prove_chunks_with_perf_and_cache<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: impl IntoIterator<Item = ChunkInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
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
            &chunk,
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

pub fn verify_chunks<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: &[PublicChunk],
    proof: &RunProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(verify_chunks_with_perf(mode, params, s, chunks, proof, mixers)?.0)
}

pub fn verify_chunks_with_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: &[PublicChunk],
    proof: &RunProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry: &[CeClaim<Commitment, F, K>] = &[];
    let built_cache = maybe_build_optimized_cache(&mode, s, provided_cache)?;
    let optimized_cache = provided_cache.or(built_cache.as_ref());

    validate_chunk_layout(proof.fold_schedule, chunks)?;

    for (idx, chunk_proof) in proof.chunks.iter().enumerate() {
        let chunk = chunks
            .get(idx)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing public chunk {idx} during verification")))?;
        let next_main = ShardVerifier::verify_chunk(
            mode.clone(),
            &mut tr,
            params,
            s,
            chunk,
            &main_carry,
            chunk_proof,
            mixers,
            optimized_cache,
        )?;
        main_carry = next_main;
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
    }
    if chunks.len() != proof.chunks.len() {
        return Err(PiCcsError::InvalidInput(
            "public chunk list is longer than proof chunk list".into(),
        ));
    }
    if main_carry != proof.final_main_claims.as_slice() {
        return Err(PiCcsError::ProtocolError(
            "final carried main claims do not match proof footer".into(),
        ));
    }
    Ok(proof.final_main_claims.clone())
}

pub fn verify_chunks_with_perf<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: &[PublicChunk],
    proof: &RunProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_chunks_with_perf_and_cache(mode, params, s, chunks, proof, mixers, None)
}

pub fn verify_chunks_with_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    chunks: &[PublicChunk],
    proof: &RunProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = Instant::now();
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry: &[CeClaim<Commitment, F, K>] = &[];
    let mut perf = RunVerifyPerf::default();
    let built_cache = maybe_build_optimized_cache(&mode, s, provided_cache)?;
    let optimized_cache = provided_cache.or(built_cache.as_ref());

    validate_chunk_layout(proof.fold_schedule, chunks)?;

    for (idx, chunk_proof) in proof.chunks.iter().enumerate() {
        let chunk = chunks
            .get(idx)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing public chunk {idx} during verification")))?;
        let (next_main, chunk_perf) = ShardVerifier::verify_chunk_with_perf(
            mode.clone(),
            &mut tr,
            params,
            s,
            chunk,
            &main_carry,
            chunk_proof,
            mixers,
            optimized_cache,
        )?;
        main_carry = next_main;
        perf.chunks.push(chunk_perf);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
    }
    if chunks.len() != proof.chunks.len() {
        return Err(PiCcsError::InvalidInput(
            "public chunk list is longer than proof chunk list".into(),
        ));
    }
    if main_carry != proof.final_main_claims.as_slice() {
        return Err(PiCcsError::ProtocolError(
            "final carried main claims do not match proof footer".into(),
        ));
    }
    perf.total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    Ok((proof.final_main_claims.clone(), perf))
}

pub fn prove_run<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<RunProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
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
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(RunProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let steps_vec: Vec<StepInput> = steps.into_iter().collect();
    let chunks = partition_step_inputs(schedule, steps_vec)?;
    prove_chunks_with_perf(mode, schedule, params, s, chunks, log, mixers)
}

pub fn verify_run<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: &[PublicStep],
    proof: &RunProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let chunks = partition_public_steps(proof.fold_schedule, steps.to_vec())?;
    verify_chunks(mode, params, s, &chunks, proof, mixers)
}

pub fn verify_run_with_perf<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: &[PublicStep],
    proof: &RunProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let chunks = partition_public_steps(proof.fold_schedule, steps.to_vec())?;
    verify_chunks_with_perf(mode, params, s, &chunks, proof, mixers)
}

pub fn prove_and_package<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<PackagedProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(prove_and_package_with_perf(mode, schedule, params, s, steps, log, mixers)?.0)
}

pub fn prove_and_package_with_perf<L, MR, MB>(
    mode: FoldingMode,
    schedule: FoldSchedule,
    params: &NeoParams,
    s: &CcsStructure<F>,
    steps: impl IntoIterator<Item = StepInput>,
    log: &L,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(PackagedProof, RunProvePerf), PiCcsError>
where
    L: SModuleHomomorphism<F, Commitment> + Sync,
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let steps_vec: Vec<StepInput> = steps.into_iter().collect();
    let input_chunks = partition_prover_step_inputs(schedule, steps_vec)?;
    let public_chunks = input_chunks
        .iter()
        .map(|chunk| chunk.public_chunk.clone())
        .collect::<Vec<_>>();
    let (session, perf) = prove_prepared_chunks_with_perf(mode, schedule, params, s, input_chunks, log, mixers)?;
    let packaged = package_session_proof(public_chunks, session)?;
    Ok((packaged, perf))
}

pub fn verify_packaged<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session(mode, params, s, proof, mixers)
}

pub fn verify_packaged_with_perf<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_with_perf(mode, params, s, proof, mixers)
}

pub fn verify_packaged_with_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_with_perf_and_cache(mode, params, s, proof, mixers, provided_cache)
}

pub(crate) fn verify_packaged_with_detailed_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    proof: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, PackagedVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_with_detailed_perf_and_cache(mode, params, s, proof, mixers, provided_cache)
}

fn validate_chunk_layout(schedule: FoldSchedule, chunks: &[PublicChunk]) -> Result<(), PiCcsError> {
    schedule.validate()?;
    let mut next_start = 0usize;
    let row_cap = match schedule {
        FoldSchedule::WholeTrace => None,
        FoldSchedule::RowsPerChunk(rows) => Some(rows),
    };

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.steps.is_empty() {
            return Err(PiCcsError::InvalidInput(format!("chunk[{idx}] is empty")));
        }
        if chunk.start_index != next_start {
            return Err(PiCcsError::InvalidInput(format!(
                "chunk[{idx}] starts at {}, expected {}",
                chunk.start_index, next_start
            )));
        }
        if let Some(rows_per_chunk) = row_cap {
            if chunk.steps.len() > rows_per_chunk {
                return Err(PiCcsError::InvalidInput(format!(
                    "chunk[{idx}] has {} steps, exceeds RowsPerChunk({rows_per_chunk})",
                    chunk.steps.len()
                )));
            }
            if idx + 1 != chunks.len() && chunk.steps.len() != rows_per_chunk {
                return Err(PiCcsError::InvalidInput(format!(
                    "chunk[{idx}] has {} steps, expected exactly {} before the final chunk",
                    chunk.steps.len(),
                    rows_per_chunk
                )));
            }
        }
        next_start += chunk.steps.len();
    }

    if matches!(schedule, FoldSchedule::WholeTrace) && chunks.len() > 1 {
        return Err(PiCcsError::InvalidInput(
            "WholeTrace schedule must carry exactly one chunk".into(),
        ));
    }
    if schedule.chunk_count(next_start)? != chunks.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "chunk count {} does not match {:?} for {} steps",
            chunks.len(),
            schedule,
            next_start
        )));
    }
    Ok(())
}
