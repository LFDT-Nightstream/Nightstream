use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use std::time::Instant;

use crate::proof::{partition_public_steps, PublicChunk, PublicStep, RunProof, RunVerifyPerf};
use crate::prover::CommitmentMixers;
use crate::verifier::ShardVerifier;

use super::cache::maybe_build_optimized_cache;
use super::layout::validate_chunk_layout;

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
    Ok(verify_chunks_with_perf_and_cache(mode, params, s, chunks, proof, mixers, provided_cache)?.0)
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
