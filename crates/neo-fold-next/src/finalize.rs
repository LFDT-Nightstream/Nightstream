//! Owns the packaged proof boundary for the active main-lane path.
//!
//! Ownership:
//! - packages the verified session spine into one final proof/public statement pair
//! - binds the package with Poseidon2 digests
//! - does not redefine `Π_CCS -> Π_RLC -> Π_DEC`

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::chunk_relation::chunk_relation_digest;
use crate::proof::{
    ChunkProof, FinalProof, FoldSchedule, PackagedProof, PublicChunk, PublicStatement, PublicStep, RunProof,
    RunVerifyPerf,
};
use crate::prover::CommitmentMixers;
use crate::run::{verify_chunks, verify_chunks_with_perf_and_cache};

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

#[inline]
fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

#[inline]
fn packed_bytes_field_len(bytes_len: usize) -> usize {
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

fn validate_packaged_public_digest_limb_encoding(packaged: &PackagedProof) -> Result<(), PiCcsError> {
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedShapeChunkSummary {
    pub start_index: u64,
    pub public_step_count: u64,
    pub public_chunk_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
}

impl FixedShapeChunkSummary {
    pub fn from_public_chunk(
        chunk: &PublicChunk,
        public_chunk_digest: [u8; 32],
        chunk_relation_digest: [u8; 32],
    ) -> Self {
        Self {
            start_index: chunk.start_index as u64,
            public_step_count: chunk.steps.len() as u64,
            public_chunk_digest,
            chunk_relation_digest,
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/fixed_shape_chunk_summary");
        tr.append_u64s(
            b"neo.fold.next/fixed_shape_chunk_summary/meta",
            &[self.start_index, self.public_step_count],
        );
        tr.append_message(
            b"neo.fold.next/fixed_shape_chunk_summary/public_chunk_digest",
            &self.public_chunk_digest,
        );
        tr.append_message(
            b"neo.fold.next/fixed_shape_chunk_summary/chunk_relation_digest",
            &self.chunk_relation_digest,
        );
        tr.digest32()
    }

    pub fn packed_fields(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(Self::packed_field_len());
        out.push(F::from_u64(self.start_index));
        out.push(F::from_u64(self.public_step_count));
        extend_packed_bytes_as_fields(&mut out, &self.public_chunk_digest);
        extend_packed_bytes_as_fields(&mut out, &self.chunk_relation_digest);
        out
    }

    pub fn packed_digest_field_len() -> usize {
        packed_bytes_field_len(32)
    }

    pub fn public_chunk_digest_field_offset() -> usize {
        2
    }

    pub fn chunk_relation_digest_field_offset() -> usize {
        Self::public_chunk_digest_field_offset() + Self::packed_digest_field_len()
    }

    pub fn packed_field_len() -> usize {
        2 + 2 * packed_bytes_field_len(32)
    }
}

pub(crate) fn fixed_shape_chunk_coverage_terminal_index(
    chunk_summaries: &[FixedShapeChunkSummary],
) -> Result<u64, String> {
    let mut next_start = 0u64;
    for (chunk_index, summary) in chunk_summaries.iter().enumerate() {
        if summary.public_step_count == 0 {
            return Err(format!(
                "chunk summary {} has zero public_step_count; fixed-shape chunks must be non-empty",
                chunk_index
            ));
        }
        if summary.start_index != next_start {
            return Err(format!(
                "chunk summary {} start index {} does not match expected contiguous start {}",
                chunk_index, summary.start_index, next_start
            ));
        }
        next_start = next_start
            .checked_add(summary.public_step_count)
            .ok_or_else(|| {
                format!(
                    "chunk summary {} public coverage overflows u64 when extending {} by {} steps",
                    chunk_index, next_start, summary.public_step_count
                )
            })?;
    }
    Ok(next_start)
}

pub(crate) fn validate_fixed_shape_chunk_coverage(
    semantic_step_count: u64,
    chunk_summaries: &[FixedShapeChunkSummary],
) -> Result<(), String> {
    let terminal_index = fixed_shape_chunk_coverage_terminal_index(chunk_summaries)?;
    if terminal_index != semantic_step_count {
        return Err(format!(
            "chunk summary coverage terminal index {} does not match semantic step count {}",
            terminal_index, semantic_step_count
        ));
    }
    Ok(())
}

pub(crate) fn validate_fixed_shape_chunk_layout(
    schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: &[FixedShapeChunkSummary],
) -> Result<(), String> {
    schedule.validate().map_err(|err| err.to_string())?;
    validate_fixed_shape_chunk_coverage(semantic_step_count, chunk_summaries)?;

    let expected_chunk_count = match schedule {
        FoldSchedule::WholeTrace => u64::from(semantic_step_count != 0),
        FoldSchedule::RowsPerChunk(rows) => {
            let rows = rows as u64;
            if semantic_step_count == 0 {
                0
            } else {
                semantic_step_count.div_ceil(rows)
            }
        }
    };
    if chunk_summaries.len() as u64 != expected_chunk_count {
        return Err(format!(
            "chunk summary count {} does not match {:?} for {} semantic steps",
            chunk_summaries.len(),
            schedule,
            semantic_step_count
        ));
    }

    match schedule {
        FoldSchedule::WholeTrace => {
            if chunk_summaries.len() > 1 {
                return Err("WholeTrace fixed-shape schedule must carry at most one chunk summary".into());
            }
        }
        FoldSchedule::RowsPerChunk(rows) => {
            let rows = rows as u64;
            for (idx, summary) in chunk_summaries.iter().enumerate() {
                if summary.public_step_count > rows {
                    return Err(format!(
                        "chunk summary {} has {} steps, exceeds RowsPerChunk({rows})",
                        idx, summary.public_step_count
                    ));
                }
                if idx + 1 != chunk_summaries.len() && summary.public_step_count != rows {
                    return Err(format!(
                        "chunk summary {} has {} steps, expected exactly {} before the final chunk",
                        idx, summary.public_step_count, rows
                    ));
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn fixed_shape_recursive_step_handle_fields(
    previous_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    chunk_index: u64,
    chunk_start_index: u64,
    public_step_count: u64,
    chunk_relation_digest: [u8; 32],
) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
    let mut preimage = Vec::with_capacity(FIXED_SHAPE_DIGEST_FIELD_LEN + 3 + FIXED_SHAPE_DIGEST_FIELD_LEN);
    preimage.extend(previous_handle_digest);
    preimage.push(F::from_u64(chunk_index));
    preimage.push(F::from_u64(chunk_start_index));
    preimage.push(F::from_u64(public_step_count));
    preimage.extend(digest32_as_fields(chunk_relation_digest));
    poseidon_digest_fields(&preimage)
}

pub(crate) fn fixed_shape_recursive_step_handle(
    previous_handle: [u8; 32],
    chunk_index: usize,
    chunk_start_index: usize,
    public_step_count: usize,
    chunk_relation_digest: [u8; 32],
) -> [u8; 32] {
    digest_fields_as_digest32(fixed_shape_recursive_step_handle_fields(
        digest32_as_fields(previous_handle),
        chunk_index as u64,
        chunk_start_index as u64,
        public_step_count as u64,
        chunk_relation_digest,
    ))
}

pub(crate) fn fixed_shape_terminal_handle_digest_fields(
    initial_handle: [u8; 32],
    chunk_summaries: &[FixedShapeChunkSummary],
) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
    let mut handle = digest32_as_fields(initial_handle);
    for (chunk_index, summary) in chunk_summaries.iter().enumerate() {
        handle = fixed_shape_recursive_step_handle_fields(
            handle,
            chunk_index as u64,
            summary.start_index,
            summary.public_step_count,
            summary.chunk_relation_digest,
        );
    }
    handle
}

#[inline]
fn poseidon_digest_fields(input: &[F]) -> [F; 4] {
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(input)
}

pub(crate) fn fixed_shape_recursive_seed(domain: &[u8]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(packed_bytes_field_len(domain.len()));
    extend_packed_bytes_as_fields(&mut preimage, domain);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
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

fn public_chunk_digests(chunks: &[PublicChunk]) -> Vec<[F; 4]> {
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

fn digest_final_proof_from_chunk_digests(
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

fn digest_final_proof_from_chunk_digests_with(
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

pub(crate) fn digest_fixed_shape_final_proof(
    relation_digest: &[u8; 32],
    chunk_count: u64,
    chunk_summaries: &[FixedShapeChunkSummary],
    base_component_digests: &[[u8; 32]],
    chunk_transition_digests: &[[u8; 32]],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/fixed_shape_final_proof");
    tr.append_message(b"neo.fold.next/fixed_shape_final_proof/version", b"v2");
    tr.append_message(
        b"neo.fold.next/fixed_shape_final_proof/relation_digest",
        relation_digest,
    );
    tr.append_u64s(b"neo.fold.next/fixed_shape_final_proof/chunk_count", &[chunk_count]);
    for summary in chunk_summaries {
        tr.append_message(
            b"neo.fold.next/fixed_shape_final_proof/chunk_summary",
            &summary.digest(),
        );
    }
    tr.append_u64s(
        b"neo.fold.next/fixed_shape_final_proof/base_component_count",
        &[base_component_digests.len() as u64],
    );
    tr.append_u64s(
        b"neo.fold.next/fixed_shape_final_proof/chunk_transition_count",
        &[chunk_transition_digests.len() as u64],
    );
    for digest in base_component_digests {
        tr.append_message(b"neo.fold.next/fixed_shape_final_proof/base_component_digest", digest);
    }
    for digest in chunk_transition_digests {
        tr.append_message(b"neo.fold.next/fixed_shape_final_proof/chunk_transition_digest", digest);
    }
    tr.digest32()
}

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

fn validate_chunk_schedule(
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

fn validate_session_chunk_relation_digests(session: &RunProof) -> Result<(), PiCcsError> {
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
    let final_main_claim_digests = final_main_claim_digests(&session.final_main_claims);
    let statement_digest =
        digest_public_statement_from_digests(session.fold_schedule, &chunk_digests, &final_main_claim_digests);
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
    let final_main_claim_digests = final_main_claim_digests(&packaged.statement.final_main_claims);
    let final_main_claim_digests_ms = final_main_claim_digests_started.elapsed().as_secs_f64() * 1_000.0;
    let statement_hash_started = std::time::Instant::now();
    let expected_statement_digest = digest_public_statement_from_digests(
        packaged.statement.fold_schedule,
        &chunk_digests,
        &final_main_claim_digests,
    );
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
    let final_main_claim_digests = final_main_claim_digests(&packaged.statement.final_main_claims);
    let expected_statement_digest = digest_public_statement_from_digests(
        packaged.statement.fold_schedule,
        &chunk_digests,
        &final_main_claim_digests,
    );
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
