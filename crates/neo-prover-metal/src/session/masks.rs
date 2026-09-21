//! Device storage for compact signed-digit witness masks.

use std::{mem::size_of, sync::atomic::Ordering};

use neo_math::D;
use neo_reductions::superneo_eval::SuperneoZBlocks;
use objc2_metal::MTLBuffer;

use super::{Buffer, MetalSession};
use crate::MetalError;

#[derive(Clone)]
pub(crate) struct MetalWitnessMasks {
    words: Buffer,
    witness_count: usize,
    stored_witnesses: usize,
    blocks: usize,
    magnitudes: usize,
    active_witnesses: Vec<u32>,
}

impl MetalWitnessMasks {
    fn new(
        words: Buffer,
        source: &[u64],
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
            || size_of_val(source) != expected_bytes
        {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        let active_witnesses = source
            .chunks_exact(blocks * 2 * magnitudes)
            .enumerate()
            .filter(|(_, words)| words.iter().any(|&word| word != 0))
            .map(|(index, _)| u32::try_from(index).map_err(|_| MetalError::Shape("witness index exceeds u32")))
            .collect::<Result<Vec<_>, _>>()?;
        let bytes_per_witness = expected_bytes / witness_count;
        let stored_witnesses = if active_witnesses.is_empty() && words.length() == size_of::<u64>() {
            0
        } else {
            if !words.length().is_multiple_of(bytes_per_witness) {
                return Err(MetalError::Shape("witness masks do not contain whole sources"));
            }
            words.length() / bytes_per_witness
        };
        if stored_witnesses > witness_count
            || active_witnesses
                .last()
                .is_some_and(|&source| source as usize >= stored_witnesses)
        {
            return Err(MetalError::Shape("witness mask suffix omits a nonzero source"));
        }
        Ok(Self {
            words,
            witness_count,
            stored_witnesses,
            blocks,
            magnitudes,
            active_witnesses,
        })
    }

    #[cfg(feature = "legacy-adapter")]
    pub(super) fn matches(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count == witness_count && self.blocks == blocks && self.magnitudes == 1
    }

    pub(super) fn matches_joint(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count == witness_count && self.blocks == blocks
    }

    pub(super) fn magnitudes(&self) -> usize {
        self.magnitudes
    }

    pub(super) fn blocks(&self) -> usize {
        self.blocks
    }

    pub(super) fn stored_witnesses(&self) -> usize {
        self.stored_witnesses
    }

    pub(super) fn active_witnesses(&self) -> &[u32] {
        &self.active_witnesses
    }

    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }
}

impl MetalSession {
    pub(super) fn prepare_joint_witness_masks(
        &self,
        witnesses: &[SuperneoZBlocks],
        base: u32,
        active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        if !(2..=4).contains(&base) {
            return Err(MetalError::Shape("witness mask radix is unsupported"));
        }
        let blocks = witnesses
            .first()
            .ok_or(MetalError::Shape("witness masks need sources"))?
            .block_len();
        let magnitudes = (base - 1) as usize;
        let columns = blocks
            .checked_mul(D)
            .ok_or(MetalError::Shape("witness mask column count overflow"))?;
        if blocks == 0
            || active_rows == 0
            || active_rows > columns
            || witnesses
                .iter()
                .any(|witness| witness.block_len() != blocks || !witness.imag_all_zero())
        {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        let active_witnesses = witnesses
            .iter()
            .enumerate()
            .filter(|(_, witness)| !witness.is_zero())
            .map(|(index, _)| u32::try_from(index).map_err(|_| MetalError::Shape("witness index exceeds u32")))
            .collect::<Result<Vec<_>, _>>()?;
        let stored_witnesses = active_witnesses.last().map_or(0, |&last| last as usize + 1);
        let words_per_source = blocks
            .checked_mul(2 * magnitudes)
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        let word_count = stored_witnesses
            .checked_mul(words_per_source)
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?
            .max(1);
        let bytes = word_count
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        let words = self.buffer(bytes)?;
        // This new shared buffer has no device readers. Initialize it before
        // borrowing its words, then pack directly into the final device storage.
        let destination = unsafe {
            let pointer = words.contents().as_ptr().cast::<u64>();
            pointer.write_bytes(0, word_count);
            std::slice::from_raw_parts_mut(pointer, word_count)
        };
        for (source, witness) in witnesses[..stored_witnesses].iter().enumerate() {
            if witness.is_zero() {
                continue;
            }
            let target = &mut destination[source * words_per_source..(source + 1) * words_per_source];
            if let Some((positive, negative)) = witness.signed_unit_masks() {
                for (block, masks) in target.chunks_exact_mut(2 * magnitudes).enumerate() {
                    masks[0] = positive[block];
                    masks[1] = negative[block];
                }
            } else {
                // General digit input uses one source's temporary packing.
                let packed = witness
                    .signed_digit_masks(base)
                    .ok_or(MetalError::Shape("witness is outside the configured radix alphabet"))?;
                target.copy_from_slice(&packed);
            }
        }
        self.activity
            .uploaded_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(MetalWitnessMasks {
            words,
            witness_count: witnesses.len(),
            stored_witnesses,
            blocks,
            magnitudes,
            active_witnesses,
        })
    }

    #[cfg(feature = "legacy-adapter")]
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
        MetalWitnessMasks::new(
            self.buffer_from_slice(words)?,
            words,
            witness_count,
            blocks,
            1,
            active_rows,
        )
    }

    pub(crate) fn prepare_witness_digit_masks(
        &self,
        words: &[u64],
        witness_count: usize,
        blocks: usize,
        magnitudes: usize,
        active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        if witness_count == 0 || blocks == 0 || magnitudes == 0 {
            return Err(MetalError::Shape("witness masks need nonzero dimensions"));
        }
        let expected_words = witness_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2 * magnitudes))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        if words.len() != expected_words {
            return Err(MetalError::Shape("witness masks have inconsistent dimensions"));
        }
        let words_per_source = expected_words / witness_count;
        let stored = words
            .chunks_exact(words_per_source)
            .rposition(|source| source.iter().any(|&word| word != 0))
            .map_or(0, |last| (last + 1) * words_per_source);
        // Preserve logical source indices. Only an all-zero suffix is absent.
        // An empty source has one bindable word, which no kernel reads.
        let stored_words = if stored == 0 { &[0u64][..] } else { &words[..stored] };
        MetalWitnessMasks::new(
            self.buffer_from_slice(stored_words)?,
            words,
            witness_count,
            blocks,
            magnitudes,
            active_rows,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/witness_mask_storage.rs"]
mod tests;
