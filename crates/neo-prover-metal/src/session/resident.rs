//! Shared resident witness masks and sumcheck transcript buffers.
//!
//! FE and NC own their phase-specific plans and command encoding in child modules.

use std::mem::{size_of, size_of_val};
use std::sync::atomic::Ordering;

use neo_math::{KExtensions, D, K};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{Buffer, MetalSession};
use crate::{KWords, MetalError};

mod fe;
mod nc;

pub(crate) use fe::{MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalFeTableInput};
pub(crate) use nc::{
    MetalNcDigitInput, MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan, MetalNcSumcheckTrace,
};

/// A signed-unit witness batch stored as positive/negative ring masks.
///
/// Layout is witness-major: each ring block contributes `[positive, negative]`.
/// Bit `r` represents row `r` in the 54-row block; both bits clear means zero.
#[derive(Clone)]
pub(crate) struct MetalWitnessMasks {
    words: Buffer,
    pub(super) witness_count: usize,
    blocks: usize,
    active_rows: usize,
    // Zero witnesses may be omitted from kernel work, but remain part of the
    // logical batch and are restored at decoding boundaries.
    pub(super) active_witnesses: Vec<u32>,
}

impl MetalWitnessMasks {
    pub(super) fn from_buffer(
        words: Buffer,
        witness_count: usize,
        blocks: usize,
        active_rows: usize,
    ) -> Result<Self, MetalError> {
        let expected_bytes = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2))
            .and_then(|values| values.checked_mul(size_of::<u64>()))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        let scalar_columns = blocks
            .checked_mul(D)
            .ok_or(MetalError::Shape("witness mask column count overflow"))?;
        if witness_count == 0
            || blocks == 0
            || active_rows == 0
            || active_rows > scalar_columns
            || words.length() as usize != expected_bytes
        {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        Ok(Self {
            words,
            witness_count,
            blocks,
            active_rows,
            active_witnesses: (0..witness_count as u32).collect(),
        })
    }

    pub(crate) fn matches(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count == witness_count && self.blocks == blocks
    }

    pub(super) fn contains(&self, witness: usize, blocks: usize) -> bool {
        witness < self.witness_count && self.blocks == blocks
    }

    pub(crate) fn matches_nc(&self, witness_count: usize, blocks: usize, active_rows: usize) -> bool {
        self.matches(witness_count, blocks) && self.active_rows == active_rows
    }

    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }
}

/// Device-produced round data returned to the canonical sumcheck engine.
///
/// This is an execution result, not verifier authority; the protocol layer
/// owns the initial snapshot, absorb order, and subsequent proof assembly.
pub(crate) struct MetalSumcheckTrace {
    pub coeffs: Vec<Vec<KWords>>,
    pub challenges: Vec<KWords>,
    pub transcript_state: [u64; 8],
    pub transcript_absorbed: usize,
}

impl MetalSession {
    /// Validates and uploads one signed-mask batch while deriving the compact
    /// active-witness index once for every downstream kernel.
    pub(crate) fn prepare_witness_masks(
        &self,
        words: &[u64],
        witness_count: usize,
        blocks: usize,
        active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        let expected_words = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        if words.len() != expected_words {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        let active_witnesses = words
            .chunks_exact(2 * blocks)
            .enumerate()
            .filter(|(_, masks)| masks.iter().any(|&mask| mask != 0))
            .map(|(witness, _)| witness as u32)
            .collect();
        MetalWitnessMasks::from_buffer(self.buffer_from_slice(words)?, witness_count, blocks, active_rows)?
            .with_active_witnesses(active_witnesses)
    }

    pub(super) fn write_shared<T: Copy>(&self, buffer: &Buffer, values: &[T]) -> Result<(), MetalError> {
        let bytes = size_of_val(values);
        if bytes > buffer.length() as usize {
            return Err(MetalError::Shape("resident Metal metadata buffer is too small"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes,
            );
        }
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn write_k_table_at(&self, buffer: &Buffer, byte_offset: usize, values: &[K]) -> Result<(), MetalError> {
        let bytes = values
            .len()
            .checked_mul(2 * size_of::<u64>())
            .ok_or(MetalError::Shape("resident K table byte size overflow"))?;
        if byte_offset
            .checked_add(bytes)
            .is_none_or(|end| end > buffer.length() as usize)
        {
            return Err(MetalError::Shape("resident K table destination is too small"));
        }
        let destination = unsafe {
            buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(byte_offset)
                .cast::<u64>()
        };
        for (index, value) in values.iter().enumerate() {
            let (real, imaginary) = value.to_limbs_u64();
            unsafe {
                destination.add(2 * index).write(real);
                destination.add(2 * index + 1).write(imaginary);
            }
        }
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Appends transcript absorb-and-challenge derivation to the producer's
    /// command so the next round can consume the challenge without a host wait.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_transcript_challenge(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        transcript_state: &Buffer,
        fields: &Buffer,
        fields_offset: usize,
        challenge: &Buffer,
        challenge_offset: usize,
        transcript_shape: &Buffer,
    ) -> Result<(), MetalError> {
        // Transcript storage is eight Poseidon words followed by the rate
        // cursor. Coefficients and challenges use interleaved K words.
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.transcript_absorb_challenge2);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(transcript_state), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(fields), fields_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(challenge), challenge_offset, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.poseidon2_constants), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(transcript_shape), 0, 4);
        }
        self.dispatch(&encoder, &self.transcript_absorb_challenge2, 1);
        encoder.endEncoding();
        Ok(())
    }

    /// Performs the single post-trace readback and decodes round coefficients,
    /// challenges, and the continuation transcript state.
    pub(super) fn read_sumcheck_trace(
        &self,
        coefficient_buffer: &Buffer,
        challenge_buffer: &Buffer,
        transcript_buffer: &Buffer,
        coefficient_count: usize,
        rounds: usize,
    ) -> Result<MetalSumcheckTrace, MetalError> {
        let coefficient_words = self.read_buffer::<u64>(coefficient_buffer, rounds * coefficient_count * 2);
        let coeffs = coefficient_words
            .chunks_exact(coefficient_count * 2)
            .map(|round| {
                round
                    .chunks_exact(2)
                    .map(|words| KWords::new(words[0], words[1]))
                    .collect()
            })
            .collect();
        let challenges = self
            .read_buffer::<u64>(challenge_buffer, rounds * 2)
            .chunks_exact(2)
            .map(|words| KWords::new(words[0], words[1]))
            .collect();
        let transcript = self.read_buffer::<u64>(transcript_buffer, 9);
        Ok(MetalSumcheckTrace {
            coeffs,
            challenges,
            transcript_state: transcript[..8]
                .try_into()
                .map_err(|_| MetalError::Shape("resident transcript state has invalid width"))?,
            transcript_absorbed: transcript[8] as usize,
        })
    }
}
