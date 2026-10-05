//! Fixed production-key commitments. Validate on the host, generate key rows
//! and sum signed ring products on the device. No complete key is stored.

use std::borrow::Borrow;

use neo_ajtai::{
    nightstream_fprime_setup::{
        element_input, signed_unit_prefix_blocks, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS, SETUP_ID,
    },
    Commitment,
};
use neo_ccs::Mat;
use neo_math::{D, F};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice};
use p3_field::PrimeCharacteristicRing;

use super::MetalSession;
use crate::MetalError;

/// Columns that one accumulation threadgroup walks before it writes a partial.
/// Measured on an M5 Max with 16 dense witnesses at production width: 128,
/// 512 and 2,048 columns took 5.01 s, 4.91 s and 5.12 s.
const COLUMNS_PER_GROUP: usize = 512;

impl MetalSession {
    pub(crate) fn commit_production_prefixes<W: Borrow<Mat<F>>>(
        &self,
        witnesses: &[W],
    ) -> Result<Vec<Commitment>, MetalError> {
        // This is the CPU commitment's validator, including complete carrier tails.
        // Validate the entire batch before allocating or dispatching device work.
        let blocks = witnesses
            .iter()
            .map(|witness| signed_unit_prefix_blocks(witness.borrow()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = vec![Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize); witnesses.len()];
        let active = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, blocks)| (!blocks.is_empty()).then_some(index))
            .collect::<Vec<_>>();
        if active.is_empty() {
            return Ok(output);
        }
        let width = witnesses
            .iter()
            .map(|witness| witness.borrow().cols())
            .max()
            .unwrap();
        let mut occupied = vec![false; width];
        for blocks in &blocks {
            for block in blocks {
                occupied[block.index() as usize] = true;
            }
        }
        let columns = occupied
            .into_iter()
            .enumerate()
            .filter_map(|(index, nonzero)| nonzero.then_some(index as u32))
            .collect::<Vec<_>>();
        let count = columns.len();
        let mut positions = vec![0usize; width];
        for (position, &column) in columns.iter().enumerate() {
            positions[column as usize] = position;
        }
        let mask_count = count
            .checked_mul(active.len())
            .ok_or(MetalError::Shape("production commitment mask dimensions overflow"))?;
        let mut masks = vec![[0u64; 2]; mask_count];
        for (local, &index) in active.iter().enumerate() {
            for block in &blocks[index] {
                masks[local * count + positions[block.index() as usize]] = [block.positive(), block.negative()];
            }
        }
        drop((blocks, positions));
        let device_masks = self.buffer_from_slice(&masks)?;
        let device_columns = self.buffer_from_slice(&columns)?;
        drop((masks, columns));
        // SHAKE128 input bytes 0..69 (setup ID and seed) as nine little-endian
        // lanes; the kernel adds each row and column.
        let fixed = SETUP_ID.len() + PRODUCTION_SEED.len();
        let input = element_input(&PRODUCTION_SEED, 0, 0);
        let prefix: [u64; 9] = std::array::from_fn(|lane| {
            u64::from_le_bytes(std::array::from_fn(|byte| {
                let position = 8 * lane + byte;
                if position < fixed {
                    input[position]
                } else {
                    0
                }
            }))
        });
        let prefix = self.buffer_from_slice(&prefix)?;

        // A threadgroup serves as many witnesses as fit, 64 lanes each.
        let key_row = &self.production_key_row;
        let accumulate = &self.production_ajtai_accumulate;
        let witnesses_per_group = active
            .len()
            .min(accumulate.maxTotalThreadsPerThreadgroup() / 64);
        let scratch_words = witnesses_per_group * (2 * D - 1);
        if witnesses_per_group == 0
            || scratch_words * size_of::<u64>() + accumulate.staticThreadgroupMemoryLength()
                > self.device.maxThreadgroupMemoryLength()
        {
            return Err(MetalError::Shape("device cannot hold production commitment scratch"));
        }
        let groups = count.div_ceil(COLUMNS_PER_GROUP);
        let witness_blocks = active.len().div_ceil(witnesses_per_group);
        let slab_words = count
            .checked_mul(D)
            .ok_or(MetalError::Shape("production commitment key row overflows"))?;
        let partial_words = active
            .len()
            .checked_mul(groups)
            .and_then(|n| n.checked_mul(D))
            .ok_or(MetalError::Shape("production commitment partial dimensions overflow"))?;
        let rows = PRODUCTION_VERIFIER_ROWS as usize;
        let output_words = rows * active.len() * D;
        let slab = self.buffer(slab_words * size_of::<u64>())?;
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(output_words * size_of::<u64>())?;
        let shape_words: Vec<u64> = (0..PRODUCTION_VERIFIER_ROWS)
            .flat_map(|row| {
                [
                    count as u64,
                    active.len() as u64,
                    groups as u64,
                    COLUMNS_PER_GROUP as u64,
                    row,
                    witnesses_per_group as u64,
                ]
            })
            .collect();
        let shapes = self.buffer_from_slice(&shape_words)?;

        // All rows in one command buffer: the slab and the partials are reused
        // row after row, and buffer hazard tracking orders the encoders.
        let command = self.command_buffer("nightstream.production_commitment")?;
        for row in 0..rows {
            let shape_offset = row * 6 * size_of::<u64>();
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_key_row")));
            encoder.setComputePipelineState(key_row);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&prefix), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&device_columns), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shapes), shape_offset, 2);
                encoder.setBuffer_offset_atIndex(Some(&slab), 0, 3);
            }
            self.dispatch(&encoder, key_row, count);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_ajtai_accumulate")));
            encoder.setComputePipelineState(accumulate);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&slab), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&device_masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shapes), shape_offset, 2);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
                encoder.setThreadgroupMemoryLength_atIndex(scratch_words * size_of::<u64>(), 0);
            }
            self.dispatch_threadgroups(&encoder, accumulate, groups * witness_blocks, 64 * witnesses_per_group);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_ajtai_sum_groups")));
            encoder.setComputePipelineState(&self.production_ajtai_sum_groups);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&shapes), shape_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(&sums), row * active.len() * D * size_of::<u64>(), 2);
            }
            self.dispatch(&encoder, &self.production_ajtai_sum_groups, active.len() * D);
            encoder.endEncoding();
        }
        self.finish(&command)?;
        let words = self.read_buffer::<u64>(&sums, output_words);
        for (row, row_words) in words.chunks_exact(active.len() * D).enumerate() {
            for (&index, coefficients) in active.iter().zip(row_words.chunks_exact(D)) {
                for (target, &value) in output[index].col_mut(row).iter_mut().zip(coefficients) {
                    *target = F::from_u64(value);
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/production_commitment.rs"]
mod tests;
