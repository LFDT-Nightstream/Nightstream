//! Device storage for compact signed-digit witness masks.

use std::mem::size_of;

use neo_math::D;
use objc2_metal::MTLBuffer;

use super::{Buffer, MetalSession};
use crate::MetalError;

#[derive(Clone)]
pub(crate) struct MetalWitnessMasks {
    words: Buffer,
    witness_count: usize,
    blocks: usize,
    magnitudes: usize,
}

impl MetalWitnessMasks {
    fn new(
        words: Buffer,
        witness_count: usize,
        blocks: usize,
        magnitudes: usize,
        active_rows: usize,
    ) -> Result<Self, MetalError> {
        let expected_bytes = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2 * magnitudes))
            .and_then(|values| values.checked_mul(size_of::<u64>()))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        let scalar_columns = blocks
            .checked_mul(D)
            .ok_or(MetalError::Shape("witness mask column count overflow"))?;
        if witness_count == 0
            || blocks == 0
            || magnitudes == 0
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
            magnitudes,
        })
    }

    pub(super) fn matches(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count == witness_count && self.blocks == blocks && self.magnitudes == 1
    }

    pub(super) fn matches_joint(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count == witness_count && self.blocks == blocks
    }

    pub(super) fn magnitudes(&self) -> usize {
        self.magnitudes
    }

    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }
}

impl MetalSession {
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
        MetalWitnessMasks::new(self.buffer_from_slice(words)?, witness_count, blocks, 1, active_rows)
    }

    pub(crate) fn prepare_witness_digit_masks(
        &self,
        words: &[u64],
        witness_count: usize,
        blocks: usize,
        magnitudes: usize,
        active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        let expected_words = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2 * magnitudes))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        if words.len() != expected_words {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        MetalWitnessMasks::new(
            self.buffer_from_slice(words)?,
            witness_count,
            blocks,
            magnitudes,
            active_rows,
        )
    }
}
