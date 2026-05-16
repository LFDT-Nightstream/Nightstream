//! Poseidon2 digest encoding for finalized public statements and proof packages.
//!
//! This file owns byte/field packing and digest construction only. It does not
//! validate chunk schedules and it does not run the folding verifier.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CeClaim};
use neo_math::{F, K};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::chunk_folding::chunk_relation_digest;
use crate::proof::{ChunkProof, FoldSchedule, PackagedProof, PublicChunk, PublicStep, RunProof};

#[inline]
pub(super) fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

#[inline]
pub(super) fn packed_bytes_field_len(bytes_len: usize) -> usize {
    const BYTES_PER_LIMB: usize = 7;
    1 + bytes_len.div_ceil(BYTES_PER_LIMB)
}

pub(crate) const FIXED_SHAPE_DIGEST_FIELD_LEN: usize = 4;

pub(crate) fn digest32_as_fields(digest: [u8; 32]) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
    [
        F::from_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        F::from_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        F::from_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        F::from_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ]
}

pub(crate) fn digest_fields_as_digest32(fields: [F; FIXED_SHAPE_DIGEST_FIELD_LEN]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (index, field) in fields.into_iter().enumerate() {
        out[index * 8..(index + 1) * 8].copy_from_slice(&field.as_canonical_u64().to_le_bytes());
    }
    out
}

fn validate_digest32_canonical_field_limb_bytes(digest: [u8; 32], context: &str) -> Result<(), PiCcsError> {
    for (limb_index, chunk) in digest.chunks_exact(8).enumerate() {
        let raw = u64::from_le_bytes(chunk.try_into().expect("digest32 limb"));
        if F::from_u64(raw).as_canonical_u64() != raw {
            return Err(PiCcsError::InvalidInput(format!(
                "{context} limb {limb_index} is not a canonical Goldilocks field element"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_packaged_public_digest_limb_encoding(packaged: &PackagedProof) -> Result<(), PiCcsError> {
    validate_digest32_canonical_field_limb_bytes(packaged.statement.digest, "final statement digest")?;
    validate_digest32_canonical_field_limb_bytes(packaged.proof.proof_digest, "final proof digest")?;
    for (chunk_index, chunk) in packaged.proof.session.chunks.iter().enumerate() {
        validate_digest32_canonical_field_limb_bytes(
            chunk.relation_digest,
            &format!("final proof chunk[{chunk_index}] relation digest"),
        )?;
    }
    Ok(())
}

#[inline]
pub(super) fn poseidon_digest_fields(input: &[F]) -> [F; 4] {
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(input)
}

fn ccs_claim_digest_fields_into(claim: &CcsClaim<Commitment, F>, scratch: &mut Vec<F>) -> [F; 4] {
    scratch.clear();
    scratch.reserve(256);
    extend_packed_bytes_as_fields(scratch, b"neo.fold.next/finalize/ccs_claim_digest/v1");
    scratch.push(F::from_u64(claim.c.d as u64));
    scratch.push(F::from_u64(claim.c.kappa as u64));
    scratch.push(F::from_u64(claim.c.data.len() as u64));
    scratch.extend_from_slice(&claim.c.data);
    scratch.push(F::from_u64(claim.x.len() as u64));
    scratch.extend_from_slice(&claim.x);
    scratch.push(F::from_u64(claim.m_in as u64));
    poseidon_digest_fields(scratch)
}

fn public_step_digest_fields_into(step: &PublicStep, claim_scratch: &mut Vec<F>, step_scratch: &mut Vec<F>) -> [F; 4] {
    step_scratch.clear();
    step_scratch.reserve(96);
    extend_packed_bytes_as_fields(step_scratch, b"neo.fold.next/finalize/public_step_digest/v1");
    extend_packed_bytes_as_fields(step_scratch, step.label.as_bytes());
    step_scratch.extend_from_slice(&ccs_claim_digest_fields_into(&step.mcs, claim_scratch));
    poseidon_digest_fields(step_scratch)
}

fn append_fold_schedule_meta(tr: &mut Poseidon2Transcript, label: &'static [u8], schedule: FoldSchedule) {
    tr.append_u64s(label, &schedule.meta_words());
}

fn public_chunk_digest_fields_into(
    chunk: &PublicChunk,
    claim_scratch: &mut Vec<F>,
    step_scratch: &mut Vec<F>,
    chunk_scratch: &mut Vec<F>,
) -> [F; 4] {
    chunk_scratch.clear();
    chunk_scratch.reserve(32 + (chunk.steps.len() * 4));
    extend_packed_bytes_as_fields(chunk_scratch, b"neo.fold.next/finalize/public_chunk_digest/v1");
    chunk_scratch.push(F::from_u64(chunk.start_index as u64));
    chunk_scratch.push(F::from_u64(chunk.steps.len() as u64));
    for step in &chunk.steps {
        chunk_scratch.extend_from_slice(&public_step_digest_fields_into(step, claim_scratch, step_scratch));
    }
    poseidon_digest_fields(chunk_scratch)
}

fn chunk_proof_compact_digest_fields(chunk: &ChunkProof, public_chunk_digest: [F; 4]) -> [F; 4] {
    let relation_digest = digest32_as_fields(chunk_relation_digest(
        &chunk.ccs_outputs,
        &chunk.rlc.parent,
        &chunk.dec.children,
    ));
    let mut digest_input = Vec::<F>::with_capacity(128 + (chunk.chunk.steps.len() * 4));
    extend_packed_bytes_as_fields(
        &mut digest_input,
        b"neo.fold.next/finalize/chunk_proof_compact_digest/v2",
    );
    digest_input.extend_from_slice(&public_chunk_digest);
    digest_input.push(F::from_u64(chunk.ccs_outputs.len() as u64));
    digest_input.push(F::from_u64(chunk.dec.children.len() as u64));
    extend_packed_bytes_as_fields(&mut digest_input, &chunk.ccs_proof.header_digest);
    digest_input.extend_from_slice(&relation_digest);
    poseidon_digest_fields(&digest_input)
}

pub(super) fn public_chunk_digests(chunks: &[PublicChunk]) -> Vec<[F; 4]> {
    let mut digests = Vec::with_capacity(chunks.len());
    let mut claim_scratch = Vec::<F>::with_capacity(256);
    let mut step_scratch = Vec::<F>::with_capacity(96);
    let mut chunk_scratch = Vec::<F>::new();
    for chunk in chunks {
        digests.push(public_chunk_digest_fields_into(
            chunk,
            &mut claim_scratch,
            &mut step_scratch,
            &mut chunk_scratch,
        ));
    }
    digests
}

pub(crate) fn public_chunk_digest(chunk: &PublicChunk) -> [F; 4] {
    let mut claim_scratch = Vec::<F>::with_capacity(256);
    let mut step_scratch = Vec::<F>::with_capacity(96);
    let mut chunk_scratch = Vec::<F>::new();
    public_chunk_digest_fields_into(chunk, &mut claim_scratch, &mut step_scratch, &mut chunk_scratch)
}

pub(crate) fn final_main_claim_digests(final_main_claims: &[CeClaim<Commitment, F, K>]) -> Vec<[F; 4]> {
    let mut digests = Vec::with_capacity(final_main_claims.len());
    let mut scratch = Vec::<F>::with_capacity(2048);
    for claim in final_main_claims {
        digests.push(me_digest_poseidon_into(&mut scratch, claim));
    }
    digests
}

pub(crate) fn digest_public_statement_from_digests(
    schedule: FoldSchedule,
    chunk_digests: &[[F; 4]],
    final_main_claim_digests: &[[F; 4]],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/final_statement");
    tr.append_message(b"neo.fold.next/final_statement/version", b"v2");
    append_fold_schedule_meta(&mut tr, b"neo.fold.next/final_statement/fold_schedule", schedule);
    tr.append_u64s(
        b"neo.fold.next/final_statement/header",
        &[chunk_digests.len() as u64, final_main_claim_digests.len() as u64],
    );
    tr.append_fields_iter(
        b"neo.fold.next/final_statement/chunk_digest",
        chunk_digests.len() * 4,
        chunk_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
    tr.append_fields_iter(
        b"neo.fold.next/final_statement/final_main_claim_digest",
        final_main_claim_digests.len() * 4,
        final_main_claim_digests
            .iter()
            .flat_map(|digest| digest.iter().copied()),
    );
    tr.digest32()
}

pub(super) fn digest_final_proof_from_chunk_digests(
    statement_digest: &[u8; 32],
    session: &RunProof,
    public_chunk_digests: &[[F; 4]],
) -> [u8; 32] {
    digest_final_proof_from_chunk_digests_with(
        statement_digest,
        session,
        public_chunk_digests,
        chunk_proof_compact_digest_fields,
    )
}

pub(super) fn digest_final_proof_from_chunk_digests_with(
    statement_digest: &[u8; 32],
    session: &RunProof,
    public_chunk_digests: &[[F; 4]],
    compact_digest: fn(&ChunkProof, [F; 4]) -> [F; 4],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/final_proof");
    tr.append_message(b"neo.fold.next/final_proof/version", b"v4");
    tr.append_message(b"neo.fold.next/final_proof/statement_digest", statement_digest);
    append_fold_schedule_meta(
        &mut tr,
        b"neo.fold.next/final_proof/fold_schedule",
        session.fold_schedule,
    );
    tr.append_u64s(
        b"neo.fold.next/final_proof/header",
        &[session.chunks.len() as u64, session.final_main_claims.len() as u64],
    );
    tr.append_fields_iter(
        b"neo.fold.next/final_proof/chunk_digest",
        session.chunks.len() * 4,
        session
            .chunks
            .iter()
            .zip(public_chunk_digests.iter())
            .flat_map(|(chunk, public_chunk_digest)| compact_digest(chunk, *public_chunk_digest)),
    );
    tr.digest32()
}
