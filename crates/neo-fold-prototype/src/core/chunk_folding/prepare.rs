use neo_math::F;
use neo_reductions::error::PiCcsError;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use std::time::Instant;

use super::types::{BorrowedChunkPreparedInputs, ChunkPreparedInputs};
use crate::finalize::public_chunk_digest;
use crate::proof::{Carry, ChunkInput, ProverChunkInput, PublicChunk};

const CHUNK_META_RAW_TAG: u64 = 14;

pub(super) fn prepare_chunk_ccs_inputs(
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

pub(super) fn prepare_prover_chunk_ccs_inputs<'a>(
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

fn append_chunk_transcript(tr: &mut Poseidon2Transcript, chunk: &ChunkInput) {
    append_public_chunk_transcript(tr, &chunk.public());
}

fn append_public_chunk_transcript(tr: &mut Poseidon2Transcript, chunk: &PublicChunk) {
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
