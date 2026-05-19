//! Replay verification for finalized proof packages.
//!
//! This file owns the verifier flow: recompute public digests, validate schedule
//! binding, replay the session verifier, and compare verified final claims.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::proof::{PackagedProof, RunVerifyPerf};
use crate::prover::CommitmentMixers;
use crate::session::{verify_chunks, verify_chunks_with_perf_and_cache};

use super::digest::{
    digest_final_proof_from_chunk_digests, digest_public_statement_from_digests, final_main_claim_digests,
    public_chunk_digests, validate_packaged_public_digest_limb_encoding,
};
use super::package::{validate_chunk_schedule, validate_session_chunk_relation_digests};

#[derive(Clone, Debug, Default)]
pub(crate) struct PackagedVerifyPerf {
    pub statement_digest_ms: f64,
    pub chunk_digests_ms: f64,
    pub final_main_claim_digests_ms: f64,
    pub statement_hash_ms: f64,
    pub schedule_checks_ms: f64,
    pub proof_digest_ms: f64,
    pub final_claim_match_ms: f64,
    pub session: RunVerifyPerf,
    pub total_ms: f64,
}

pub fn verify_finalized_session<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    Ok(verify_finalized_session_inner(mode, params, s, packaged, mixers, false)?.0)
}

pub fn verify_finalized_session_with_perf<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let (verified, perf) = verify_finalized_session_inner(mode, params, s, packaged, mixers, true)?;
    Ok((verified, perf.expect("verify perf requested")))
}

pub fn verify_finalized_session_with_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, RunVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let (verified, perf) =
        verify_finalized_session_inner_with_cache(mode, params, s, packaged, mixers, true, provided_cache)?;
    Ok((verified, perf.expect("verify perf requested")))
}

pub(crate) fn verify_finalized_session_with_detailed_perf_and_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, PackagedVerifyPerf), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    let total_started = std::time::Instant::now();
    validate_packaged_public_digest_limb_encoding(packaged)?;

    let statement_digest_started = std::time::Instant::now();
    let chunk_digests_started = std::time::Instant::now();
    let chunk_digests = public_chunk_digests(&packaged.statement.chunks);
    let chunk_digests_ms = chunk_digests_started.elapsed().as_secs_f64() * 1_000.0;
    let final_main_claim_digests_started = std::time::Instant::now();
    let final_claim_digests = final_main_claim_digests(&packaged.statement.final_main_claims);
    let final_main_claim_digests_ms = final_main_claim_digests_started.elapsed().as_secs_f64() * 1_000.0;
    let statement_hash_started = std::time::Instant::now();
    let expected_statement_digest =
        digest_public_statement_from_digests(packaged.statement.fold_schedule, &chunk_digests, &final_claim_digests);
    if packaged.statement.digest != expected_statement_digest {
        return Err(PiCcsError::ProtocolError("final statement digest mismatch".into()));
    }
    let statement_hash_ms = statement_hash_started.elapsed().as_secs_f64() * 1_000.0;
    let statement_digest_ms = statement_digest_started.elapsed().as_secs_f64() * 1_000.0;

    let schedule_checks_started = std::time::Instant::now();
    let public_step_count = packaged
        .statement
        .chunks
        .iter()
        .map(|chunk| chunk.steps.len())
        .sum();
    if packaged.statement.chunk_count as usize != packaged.statement.chunks.len() {
        return Err(PiCcsError::ProtocolError(
            "final statement chunk_count does not match chunk list".into(),
        ));
    }
    validate_chunk_schedule(
        packaged.statement.fold_schedule,
        packaged.statement.chunks.len(),
        public_step_count,
    )?;
    if packaged.proof.session.fold_schedule != packaged.statement.fold_schedule {
        return Err(PiCcsError::ProtocolError(
            "final proof schedule does not match public statement schedule".into(),
        ));
    }
    let schedule_checks_ms = schedule_checks_started.elapsed().as_secs_f64() * 1_000.0;

    let proof_digest_started = std::time::Instant::now();
    validate_session_chunk_relation_digests(&packaged.proof.session)?;
    let expected_digest =
        digest_final_proof_from_chunk_digests(&packaged.statement.digest, &packaged.proof.session, &chunk_digests);
    if packaged.proof.proof_digest != expected_digest {
        return Err(PiCcsError::ProtocolError("final proof digest mismatch".into()));
    }
    let proof_digest_ms = proof_digest_started.elapsed().as_secs_f64() * 1_000.0;

    let (verified, session) = verify_chunks_with_perf_and_cache(
        mode,
        params,
        s,
        &packaged.statement.chunks,
        &packaged.proof.session,
        mixers,
        provided_cache,
    )?;

    let final_claim_match_started = std::time::Instant::now();
    if verified != packaged.statement.final_main_claims {
        return Err(PiCcsError::ProtocolError(
            "final public statement claims do not match verified output".into(),
        ));
    }
    let final_claim_match_ms = final_claim_match_started.elapsed().as_secs_f64() * 1_000.0;

    Ok((
        verified,
        PackagedVerifyPerf {
            statement_digest_ms,
            chunk_digests_ms,
            final_main_claim_digests_ms,
            statement_hash_ms,
            schedule_checks_ms,
            proof_digest_ms,
            final_claim_match_ms,
            session,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

fn verify_finalized_session_inner<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    with_perf: bool,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, Option<RunVerifyPerf>), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session_inner_with_cache(mode, params, s, packaged, mixers, with_perf, None)
}

fn verify_finalized_session_inner_with_cache<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
    with_perf: bool,
    provided_cache: Option<&OptimizedStructureCache>,
) -> Result<(Vec<CeClaim<Commitment, F, K>>, Option<RunVerifyPerf>), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    validate_packaged_public_digest_limb_encoding(packaged)?;

    let chunk_digests = public_chunk_digests(&packaged.statement.chunks);
    let final_claim_digests = final_main_claim_digests(&packaged.statement.final_main_claims);
    let expected_statement_digest =
        digest_public_statement_from_digests(packaged.statement.fold_schedule, &chunk_digests, &final_claim_digests);
    if packaged.statement.digest != expected_statement_digest {
        return Err(PiCcsError::ProtocolError("final statement digest mismatch".into()));
    }
    let public_step_count = packaged
        .statement
        .chunks
        .iter()
        .map(|chunk| chunk.steps.len())
        .sum();
    if packaged.statement.chunk_count as usize != packaged.statement.chunks.len() {
        return Err(PiCcsError::ProtocolError(
            "final statement chunk_count does not match chunk list".into(),
        ));
    }
    validate_chunk_schedule(
        packaged.statement.fold_schedule,
        packaged.statement.chunks.len(),
        public_step_count,
    )?;
    if packaged.proof.session.fold_schedule != packaged.statement.fold_schedule {
        return Err(PiCcsError::ProtocolError(
            "final proof schedule does not match public statement schedule".into(),
        ));
    }

    validate_session_chunk_relation_digests(&packaged.proof.session)?;
    let expected_digest =
        digest_final_proof_from_chunk_digests(&packaged.statement.digest, &packaged.proof.session, &chunk_digests);
    if packaged.proof.proof_digest != expected_digest {
        return Err(PiCcsError::ProtocolError("final proof digest mismatch".into()));
    }

    let (verified, perf) = if with_perf {
        let (verified, perf) = verify_chunks_with_perf_and_cache(
            mode,
            params,
            s,
            &packaged.statement.chunks,
            &packaged.proof.session,
            mixers,
            provided_cache,
        )?;
        (verified, Some(perf))
    } else {
        let verified = verify_chunks(
            mode,
            params,
            s,
            &packaged.statement.chunks,
            &packaged.proof.session,
            mixers,
        )?;
        (verified, None)
    };
    if verified != packaged.statement.final_main_claims {
        return Err(PiCcsError::ProtocolError(
            "final public statement claims do not match verified output".into(),
        ));
    }
    Ok((verified, perf))
}

pub fn verify_packaged_proof<MR, MB>(
    mode: FoldingMode,
    params: &NeoParams,
    s: &CcsStructure<F>,
    packaged: &PackagedProof,
    mixers: CommitmentMixers<MR, MB>,
) -> Result<Vec<CeClaim<Commitment, F, K>>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    verify_finalized_session(mode, params, s, packaged, mixers)
}
