//! Batched Ajtai commitments over device-resident signed-unit masks.

use std::mem::size_of;
use std::time::Duration;

use neo_fold_clean::paper::relations::LaneRanges;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{command_gpu_duration, MetalAjtaiLowNormPlan, MetalSession, MetalWitnessMasks};
use crate::MetalError;

const RING_DEGREE: usize = 54;
const PRODUCT_COEFFICIENTS: usize = 2 * RING_DEGREE - 1;
const CHUNK_COLUMNS: usize = 512;

impl MetalSession {
    pub(crate) fn ajtai_low_norm_many_from_masks(
        &self,
        plan: &MetalAjtaiLowNormPlan,
        masks: &MetalWitnessMasks,
        count: usize,
    ) -> Result<(Vec<u64>, Duration), MetalError> {
        if count == 0 || !masks.matches(count, plan.cols) {
            return Err(MetalError::Shape(
                "batched Ajtai masks do not match the commitment plan",
            ));
        }
        let chunks = plan.cols.div_ceil(CHUNK_COLUMNS);
        let partial_words = checked_product(&[count, plan.rows, chunks, PRODUCT_COEFFICIENTS])?;
        let sum_words = checked_product(&[count, plan.rows, PRODUCT_COEFFICIENTS])?;
        let output_words = checked_product(&[count, plan.rows, RING_DEGREE])?;
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(sum_words * size_of::<u64>())?;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let active = self.buffer_from_slice(
            &(0..count)
                .map(|index| u32::try_from(index).map_err(|_| MetalError::Shape("batched Ajtai count exceeds u32")))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let shape = self.buffer_from_slice(&[
            (plan.cols * RING_DEGREE) as u64,
            count as u64,
            plan.rows as u64,
            plan.cols as u64,
            chunks as u64,
        ])?;

        let command = self.command_buffer("nightstream.ajtai.low_norm_many")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_ring_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(masks.words()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&active), 0, 4);
        }
        self.dispatch(&encoder, &self.dec_ring_partials, partial_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_ring_sum_chunks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_sum_chunks, sum_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_reduce_phi81, output_words);
        encoder.endEncoding();
        self.finish(&command)?;
        Ok((
            self.read_buffer::<u64>(&output, output_words),
            command_gpu_duration(&command),
        ))
    }

    pub(crate) fn ajtai_lane_commitments_from_masks(
        &self,
        ops_plan: &MetalAjtaiLowNormPlan,
        mem_plan: &MetalAjtaiLowNormPlan,
        masks: &MetalWitnessMasks,
        count: usize,
        full_cols: usize,
        ranges: &LaneRanges,
    ) -> Result<(Vec<u64>, Duration), MetalError> {
        if count == 0
            || !masks.matches(count, full_cols)
            || ops_plan.rows != mem_plan.rows
            || ops_plan.cols != ranges.ops.len()
            || mem_plan.cols != ranges.is.len()
            || mem_plan.cols != ranges.fs.len()
            || ranges.ops.end > full_cols
            || ranges.is.end > full_cols
            || ranges.fs.end > full_cols
        {
            return Err(MetalError::Shape(
                "Nebula lane ranges do not match the resident witness masks",
            ));
        }

        let ops_chunks = ops_plan.cols.div_ceil(CHUNK_COLUMNS);
        let mem_chunks = mem_plan.cols.div_ceil(CHUNK_COLUMNS);
        let total_chunks = ops_chunks
            .checked_add(
                mem_chunks
                    .checked_mul(2)
                    .ok_or(MetalError::Shape("Nebula lane dimensions overflow"))?,
            )
            .ok_or(MetalError::Shape("Nebula lane dimensions overflow"))?;
        let partial_words = checked_product(&[count, ops_plan.rows, total_chunks, PRODUCT_COEFFICIENTS])?;
        let sum_words = checked_product(&[count, 3, ops_plan.rows, PRODUCT_COEFFICIENTS])?;
        let output_words = checked_product(&[count, 3, ops_plan.rows, RING_DEGREE])?;
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(sum_words * size_of::<u64>())?;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let shape = self.buffer_from_slice(&[
            full_cols as u64,
            count as u64,
            ops_plan.rows as u64,
            ops_plan.cols as u64,
            mem_plan.cols as u64,
            ops_chunks as u64,
            mem_chunks as u64,
            ranges.ops.start as u64,
            ranges.is.start as u64,
            ranges.fs.start as u64,
        ])?;

        let command = self.command_buffer("nightstream.ajtai.nebula_lanes")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.ajtai_lane_ring_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&ops_plan.matrix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&mem_plan.matrix), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(masks.words()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 4);
        }
        self.dispatch(&encoder, &self.ajtai_lane_ring_partials, partial_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.ajtai_lane_ring_sum_chunks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
        }
        self.dispatch(&encoder, &self.ajtai_lane_ring_sum_chunks, sum_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.ajtai_lane_ring_reduce_phi81);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        self.dispatch(&encoder, &self.ajtai_lane_ring_reduce_phi81, output_words);
        encoder.endEncoding();
        self.finish(&command)?;
        Ok((
            self.read_buffer::<u64>(&output, output_words),
            command_gpu_duration(&command),
        ))
    }
}

fn checked_product(factors: &[usize]) -> Result<usize, MetalError> {
    factors
        .iter()
        .try_fold(1usize, |value, &factor| value.checked_mul(factor))
        .ok_or(MetalError::Shape("batched Ajtai dimensions overflow"))
}
