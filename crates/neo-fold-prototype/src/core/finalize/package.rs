//! Build finalized proof packages from an already verified session spine.
//!
//! This file owns package assembly and cheap structural checks. It does not run
//! the verifier; see `verify.rs` for verifier replay.

use neo_reductions::error::PiCcsError;

use crate::chunk_folding::chunk_relation_digest;
use crate::proof::{FinalProof, FoldSchedule, PackagedProof, PublicChunk, PublicStatement, RunProof};

use super::digest::{
    digest_final_proof_from_chunk_digests, digest_public_statement_from_digests, final_main_claim_digests,
    public_chunk_digests,
};

fn validate_public_chunks_against_session(chunks: &[PublicChunk], session: &RunProof) -> Result<(), PiCcsError> {
    if chunks.len() != session.chunks.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "finalizer chunk mismatch: public chunks={}, session chunks={}",
            chunks.len(),
            session.chunks.len()
        )));
    }
    for (chunk_idx, (chunk, proved)) in chunks.iter().zip(session.chunks.iter()).enumerate() {
        if chunk.start_index != proved.chunk.start_index {
            return Err(PiCcsError::InvalidInput(format!(
                "finalizer chunk[{chunk_idx}] start mismatch: {} != {}",
                chunk.start_index, proved.chunk.start_index
            )));
        }
        if chunk.steps.len() != proved.chunk.steps.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "finalizer chunk[{chunk_idx}] length mismatch: {} != {}",
                chunk.steps.len(),
                proved.chunk.steps.len()
            )));
        }
        for (step_idx, (step, proved_step)) in chunk
            .steps
            .iter()
            .zip(proved.chunk.steps.iter())
            .enumerate()
        {
            if proved_step.label != step.label
                || proved_step.mcs.m_in != step.mcs.m_in
                || proved_step.mcs.x != step.mcs.x
                || proved_step.mcs.c != step.mcs.c
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "finalizer chunk[{chunk_idx}] step[{step_idx}] public/proof mismatch for '{}'",
                    step.label
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_chunk_schedule(
    schedule: FoldSchedule,
    chunk_count: usize,
    public_step_count: usize,
) -> Result<(), PiCcsError> {
    let expected = schedule.chunk_count(public_step_count)?;
    if expected != chunk_count {
        return Err(PiCcsError::InvalidInput(format!(
            "chunk count {} does not match {:?} for {} public steps",
            chunk_count, schedule, public_step_count
        )));
    }
    Ok(())
}

pub(super) fn validate_session_chunk_relation_digests(session: &RunProof) -> Result<(), PiCcsError> {
    for (idx, chunk) in session.chunks.iter().enumerate() {
        let expected = chunk_relation_digest(&chunk.ccs_outputs, &chunk.rlc.parent, &chunk.dec.children);
        if chunk.relation_digest != expected {
            return Err(PiCcsError::ProtocolError(format!(
                "final proof chunk[{idx}] relation digest does not match authoritative relation fields"
            )));
        }
    }
    Ok(())
}

pub fn package_session_proof(chunks: Vec<PublicChunk>, session: RunProof) -> Result<PackagedProof, PiCcsError> {
    validate_public_chunks_against_session(&chunks, &session)?;
    let public_step_count = chunks.iter().map(|chunk| chunk.steps.len()).sum();
    validate_chunk_schedule(session.fold_schedule, chunks.len(), public_step_count)?;
    validate_session_chunk_relation_digests(&session)?;

    let chunk_digests = public_chunk_digests(&chunks);
    let final_claim_digests = final_main_claim_digests(&session.final_main_claims);
    let statement_digest =
        digest_public_statement_from_digests(session.fold_schedule, &chunk_digests, &final_claim_digests);
    let proof_digest = digest_final_proof_from_chunk_digests(&statement_digest, &session, &chunk_digests);

    Ok(PackagedProof {
        statement: PublicStatement {
            fold_schedule: session.fold_schedule,
            chunk_count: chunks.len() as u64,
            chunks,
            final_main_claims: session.final_main_claims.clone(),
            digest: statement_digest,
        },
        proof: FinalProof { session, proof_digest },
    })
}

pub fn package_proof(chunks: Vec<PublicChunk>, session: RunProof) -> Result<PackagedProof, PiCcsError> {
    package_session_proof(chunks, session)
}
