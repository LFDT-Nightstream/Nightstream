//! Fixed-shape public handles used by the recursive/public boundary.
//!
//! This file owns the compact chunk-summary shape and the handle chain derived
//! from those summaries. It does not verify the underlying session.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::proof::{FoldSchedule, PublicChunk};

use super::digest::{
    digest32_as_fields, digest_fields_as_digest32, extend_packed_bytes_as_fields, packed_bytes_field_len,
    poseidon_digest_fields, FIXED_SHAPE_DIGEST_FIELD_LEN,
};

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

pub(crate) fn fixed_shape_recursive_seed(domain: &[u8]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(packed_bytes_field_len(domain.len()));
    extend_packed_bytes_as_fields(&mut preimage, domain);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
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
