//! Fixed production-key commitments. Validate on the host, generate key tiles
//! and sum signed ring products on the device. No complete key is stored.

use neo_ajtai::{
    nightstream_fprime_setup::{signed_unit_prefix_blocks, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS},
    Commitment,
};
use neo_ccs::Mat;
use neo_math::{D, F};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice};
use p3_field::PrimeCharacteristicRing;

use super::MetalSession;
use crate::MetalError;

impl MetalSession {
    pub(crate) fn commit_production_prefixes(&self, witnesses: &[Mat<F>]) -> Result<Vec<Commitment>, MetalError> {
        // This is the CPU commitment's validator, including complete carrier tails.
        // Validate the entire batch before allocating or dispatching device work.
        let blocks = witnesses
            .iter()
            .map(signed_unit_prefix_blocks)
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
        let width = witnesses.iter().map(Mat::cols).max().unwrap();
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
        let seed = self.buffer_from_slice(&PRODUCTION_SEED)?;

        // One SIMD-width tile of key columns per group. This follows the device
        // execution width, not a circuit-specific or memory-size constant.
        let pipeline = &self.production_ajtai_partials;
        let tile = pipeline.threadExecutionWidth();
        let threads = (2 * D - 1).div_ceil(tile) * tile;
        let key_bytes = tile * D * size_of::<u64>();
        if threads > pipeline.maxTotalThreadsPerThreadgroup()
            || key_bytes + pipeline.staticThreadgroupMemoryLength() > self.device.maxThreadgroupMemoryLength()
        {
            return Err(MetalError::Shape("device cannot hold a production commitment key tile"));
        }
        let chunks = count.div_ceil(tile);
        let first_words = active
            .len()
            .checked_mul(chunks)
            .and_then(|n| n.checked_mul(D))
            .ok_or(MetalError::Shape("production commitment partial dimensions overflow"))?;
        let first = self.buffer(first_words * size_of::<u64>())?;
        let second = self.buffer(active.len() * chunks.div_ceil(2) * D * size_of::<u64>())?;
        let mut shape_words = Vec::new();
        let mut reduction_shapes = Vec::new();
        for row in 0..PRODUCTION_VERIFIER_ROWS {
            shape_words.extend([
                count as u64,
                active.len() as u64,
                chunks as u64,
                tile as u64,
                row,
                threads as u64,
            ]);
        }
        let mut current = chunks;
        while current > 1 {
            reduction_shapes.extend([active.len() as u64, current as u64]);
            current = current.div_ceil(2);
        }
        let shapes = self.buffer_from_slice(&shape_words)?;
        let reduction_shapes = self.buffer_from_slice(if reduction_shapes.is_empty() {
            &[0, 0]
        } else {
            &reduction_shapes
        })?;
        for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
            let command = self.command_buffer("nightstream.production_commitment")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("production_ajtai_partials")));
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seed), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&device_columns), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&device_masks), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&shapes), row * 6 * size_of::<u64>(), 3);
                encoder.setBuffer_offset_atIndex(Some(&first), 0, 4);
                encoder.setThreadgroupMemoryLength_atIndex(key_bytes, 0);
            }
            self.dispatch_threadgroups(&encoder, pipeline, chunks, threads);
            encoder.endEncoding();
            let mut current = chunks;
            let mut round = 0;
            while current > 1 {
                let next = current.div_ceil(2);
                let (input, destination) = if round % 2 == 0 {
                    (&first, &second)
                } else {
                    (&second, &first)
                };
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setLabel(Some(&NSString::from_str("ajtai_reduce_columns")));
                encoder.setComputePipelineState(&self.ajtai_reduce_columns);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(destination), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&reduction_shapes), round * 2 * size_of::<u64>(), 2);
                }
                self.dispatch(&encoder, &self.ajtai_reduce_columns, active.len() * next * D);
                encoder.endEncoding();
                current = next;
                round += 1;
            }
            self.finish(&command)?;
            let result = if round % 2 == 0 { &first } else { &second };
            for (&index, coefficients) in active.iter().zip(
                self.read_buffer::<u64>(result, active.len() * D)
                    .chunks_exact(D),
            ) {
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
