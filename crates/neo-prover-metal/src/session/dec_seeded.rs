//! Structure-static seeded-Phi81 ownership shared by Pi_CCS ring forms.

use std::mem::size_of;

use neo_math::D;
use neo_reductions::superneo_eval::SuperneoEvalCache;
use objc2_metal::MTLBuffer;

use super::{Buffer, MetalFeOraclePlan, MetalSession};
use crate::MetalError;

const SEEDED_OUTPUT_HEADER_WORDS: usize = 9;

pub(super) struct DeviceSeededFormPlan {
    pub(super) output_headers: Buffer,
    pub(super) word_starts: Buffer,
    pub(super) rotations: Buffer,
    pub(super) active_indices: Buffer,
    pub(super) group_segment_offsets: Buffer,
    pub(super) segments: Buffer,
    pub(super) group_count: usize,
}

impl MetalSession {
    pub(super) fn prepare_seeded_dec_forms(
        &self,
        cache: &SuperneoEvalCache,
        oracle: &MetalFeOraclePlan,
        active_blocks: &[u32],
        matrix_blocks: usize,
    ) -> Result<Option<DeviceSeededFormPlan>, MetalError> {
        if !oracle.matches(cache) {
            return Err(MetalError::Shape(
                "Pi_DEC seeded form plan does not match the oracle plan",
            ));
        }
        let matrices = cache.matrix_caches();
        let has_seeded = matrices
            .iter()
            .any(|matrix| matrix.has_compact_seeded_phi81_blocks());
        if !has_seeded {
            return Ok(None);
        }
        if matrices
            .iter()
            .flat_map(|matrix| matrix.compact_seeded_phi81_blocks())
            .any(|block| block.has_superneo_transformed_columns())
        {
            return Ok(None);
        }
        let Some(device) = oracle.seeded_rows() else {
            return Ok(None);
        };

        let mut by_active = vec![Vec::<[u32; 2]>::new(); active_blocks.len()];
        let mut output_index = 0usize;
        let mut word_base = 0usize;
        let mut rotation_words = 0usize;
        for (matrix, matrix_cache) in matrices.iter().enumerate() {
            for block in matrix_cache.compact_seeded_phi81_blocks() {
                for _ in 0..block.kappa() {
                    let output = u32::try_from(output_index)
                        .map_err(|_| MetalError::Shape("Pi_DEC seeded output count exceeds u32"))?;
                    for (word, &start) in block.word_starts().iter().enumerate() {
                        let global_word = u32::try_from(word_base + word)
                            .map_err(|_| MetalError::Shape("Pi_DEC seeded word count exceeds u32"))?;
                        let end = start
                            .checked_add(block.word_width())
                            .ok_or(MetalError::Shape("Pi_DEC seeded word range overflow"))?;
                        for column_block in start / D..end.div_ceil(D) {
                            let encoded = u32::try_from(matrix * matrix_blocks + column_block)
                                .map_err(|_| MetalError::Shape("Pi_DEC seeded block index exceeds u32"))?;
                            let active = active_blocks
                                .binary_search(&encoded)
                                .map_err(|_| MetalError::Shape("Pi_DEC seeded block is absent from active forms"))?;
                            by_active[active].push([output, global_word]);
                        }
                    }
                    output_index += 1;
                    rotation_words = rotation_words
                        .checked_add(block.message_cols() * D)
                        .ok_or(MetalError::Shape("Pi_DEC seeded rotation size overflow"))?;
                }
                word_base = word_base
                    .checked_add(block.word_starts().len())
                    .ok_or(MetalError::Shape("Pi_DEC seeded word count overflow"))?;
            }
        }
        if output_index != device.output_count
            || device.output_headers.length() as usize != output_index * SEEDED_OUTPUT_HEADER_WORDS * size_of::<u64>()
            || device.word_starts.length() as usize != word_base * size_of::<u32>()
            || device.rotations.length() as usize != rotation_words * size_of::<u64>()
        {
            return Err(MetalError::Shape("Pi_DEC seeded form metadata is inconsistent"));
        }

        let mut active_indices = Vec::new();
        let mut offsets = Vec::new();
        let mut segments = Vec::new();
        offsets.push(0u32);
        for (active_index, active) in by_active.into_iter().enumerate() {
            if active.is_empty() {
                continue;
            }
            active_indices.push(
                u32::try_from(active_index).map_err(|_| MetalError::Shape("Pi_DEC seeded active index exceeds u32"))?,
            );
            segments.extend(active.into_iter().flatten());
            offsets.push(
                u32::try_from(segments.len() / 2)
                    .map_err(|_| MetalError::Shape("Pi_DEC seeded segment count exceeds u32"))?,
            );
        }
        let group_count = active_indices.len();
        let segment_count = segments.len() / 2;
        if group_count == 0 || segment_count == 0 {
            return Err(MetalError::Shape("Pi_DEC seeded form plan has no segments"));
        }
        Ok(Some(DeviceSeededFormPlan {
            output_headers: device.output_headers.clone(),
            word_starts: device.word_starts.clone(),
            rotations: device.rotations.clone(),
            active_indices: self.buffer_from_slice(&active_indices)?,
            group_segment_offsets: self.buffer_from_slice(&offsets)?,
            segments: self.buffer_from_slice(&segments)?,
            group_count,
        }))
    }
}
