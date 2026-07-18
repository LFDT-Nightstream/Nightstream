//! Compact public Pi_DEC projections from resident child masks.

use std::mem::size_of;
use std::time::Duration;

use neo_math::{KExtensions, K};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{command_gpu_duration, Buffer, MetalSession};
use crate::MetalError;

const RING_DEGREE: usize = 54;
const CHUNK_COLUMNS: usize = 512;
const CHILDREN_PER_THREAD: usize = 7;

#[derive(Clone, Copy)]
pub(crate) struct MetalDecPublicProjection<'a> {
    pub active_rows: usize,
    pub s_col: &'a [K],
}

/// Independently queued child projection, completed after the main DEC command.
pub(super) struct PendingDecYzcol {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    output: Buffer,
    output_count: usize,
}

impl PendingDecYzcol {
    pub(super) fn finish_after(
        self,
        session: &MetalSession,
        predecessor: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<(Vec<u64>, Duration), MetalError> {
        // Submit both queues before either wait so the projection can overlap
        // the split/form/commit tail whenever its parent input is already ready.
        session.submit(predecessor);
        session.submit(&self.command);
        session.wait(predecessor)?;
        session.wait(&self.command)?;
        Ok((
            session.read_buffer::<u64>(&self.output, 2 * self.output_count),
            command_gpu_duration(&self.command),
        ))
    }
}

impl MetalSession {
    pub(super) fn enqueue_dec_child_y_zcol(
        &self,
        parent: &ProtocolObject<dyn MTLBuffer>,
        child_count: usize,
        cols: usize,
        projection: MetalDecPublicProjection<'_>,
    ) -> Result<PendingDecYzcol, MetalError> {
        let chi_len = 1usize
            .checked_shl(projection.s_col.len() as u32)
            .ok_or(MetalError::Shape("Pi_DEC column point dimensions overflow"))?;
        if projection.s_col.is_empty()
            || projection.active_rows == 0
            || projection.active_rows > chi_len
            || projection.active_rows > cols * RING_DEGREE
        {
            return Err(MetalError::Shape("Pi_DEC column projection dimensions are invalid"));
        }
        let challenge_words = projection
            .s_col
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let stages = (0..projection.s_col.len() as u64).collect::<Vec<_>>();
        let chunks = cols.div_ceil(CHUNK_COLUMNS);
        let partial_count = checked_product(&[child_count, RING_DEGREE, chunks])?;
        let partial_dispatch = checked_product(&[child_count.div_ceil(CHILDREN_PER_THREAD), RING_DEGREE, chunks])?;
        let output_count = checked_product(&[child_count, RING_DEGREE])?;
        let challenges = self.buffer_from_slice(&challenge_words)?;
        let stages = self.buffer_from_slice(&stages)?;
        let chi = self.buffer(checked_product(&[chi_len, 2, size_of::<u64>()])?)?;
        let shape = self.buffer_from_slice(&[
            projection.active_rows as u64,
            cols as u64,
            child_count as u64,
            chunks as u64,
        ])?;
        let partials = self.buffer(partial_count * 2 * size_of::<u64>())?;
        let output = self.buffer(output_count * 2 * size_of::<u64>())?;
        // This reads the immutable Pi_RLC parent, not the newly split children,
        // so it has no dependency on the main Pi_DEC command buffer.
        let command = self.independent_command_buffer("nightstream.pi_dec.y_zcol")?;
        self.encode_tensor_point_k(&command, &challenges, &stages, &chi, projection.s_col.len())?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_y_zcol_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(parent), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&chi), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_y_zcol_partials, partial_dispatch);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_y_zcol_reduce);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_y_zcol_reduce, output_count);
        encoder.endEncoding();
        Ok(PendingDecYzcol {
            command,
            output,
            output_count,
        })
    }
}

fn checked_product(factors: &[usize]) -> Result<usize, MetalError> {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .ok_or(MetalError::Shape("Pi_DEC column projection dimensions overflow"))
}
