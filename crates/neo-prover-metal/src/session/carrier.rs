//! Resident witness ownership across Pi_RLC and Pi_DEC.

use std::mem::size_of;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{Buffer, MetalSession};
use crate::MetalError;

const RING_DEGREE: usize = 54;

pub(crate) struct MetalResidentWitness {
    pub(super) words: Buffer,
    pub(super) cols: usize,
}

impl MetalResidentWitness {
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }
}

pub(crate) struct MetalResidentChildren {
    pub(super) words: Buffer,
    pub(super) child_count: usize,
    pub(super) cols: usize,
}

impl MetalResidentChildren {
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.child_count, self.cols)
    }
}

impl MetalSession {
    pub(crate) fn retain_running_children(&self, children: MetalResidentChildren) -> u64 {
        let id = self.next_resident_id.get();
        self.next_resident_id.set(id.wrapping_add(1).max(1));
        self.resident_running.replace(Some((id, children)));
        id
    }

    pub(crate) fn resident_running_shape(&self, id: u64) -> Option<(usize, usize)> {
        let resident = self.resident_running.borrow();
        let (resident_id, children) = resident.as_ref()?;
        (*resident_id == id).then(|| children.shape())
    }

    pub(crate) fn mix_rlc_witnesses_resident(
        &self,
        rhos: &[i8],
        witnesses: &[u64],
        input_count: usize,
        cols: usize,
    ) -> Result<MetalResidentWitness, MetalError> {
        let expected_rhos = input_count
            .checked_mul(RING_DEGREE * RING_DEGREE)
            .ok_or(MetalError::Shape("RLC rho dimensions overflow"))?;
        let expected_witnesses = input_count
            .checked_mul(RING_DEGREE)
            .and_then(|values| values.checked_mul(cols))
            .ok_or(MetalError::Shape("RLC witness dimensions overflow"))?;
        if input_count == 0 || cols == 0 || rhos.len() != expected_rhos || witnesses.len() != expected_witnesses {
            return Err(MetalError::Shape("RLC witness-mix inputs have inconsistent dimensions"));
        }

        let rhos = self.buffer_from_slice(rhos)?;
        let witnesses = self.buffer_from_slice(witnesses)?;
        let shape = self.buffer_from_slice(&[input_count as u64, cols as u64])?;
        let words = self.buffer(RING_DEGREE * cols * size_of::<u64>())?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.rlc_witness_mix);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&rhos), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&witnesses), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&words), 0, 3);
        }
        self.dispatch(&encoder, &self.rlc_witness_mix, RING_DEGREE * cols);
        encoder.endEncoding();
        self.finish(&command)?;
        Ok(MetalResidentWitness { words, cols })
    }

    pub(crate) fn mix_rlc_witnesses_with_resident_id(
        &self,
        rhos: &[i8],
        fresh_witnesses: &[u64],
        fresh_count: usize,
        input_count: usize,
        cols: usize,
        resident_id: u64,
    ) -> Result<MetalResidentWitness, MetalError> {
        let resident = self.resident_running.borrow();
        let Some((stored_id, resident_tail)) = resident.as_ref() else {
            return Err(MetalError::Shape("RLC resident witness is no longer available"));
        };
        if *stored_id != resident_id {
            return Err(MetalError::Shape("RLC resident witness generation is stale"));
        }
        let expected_rhos = input_count
            .checked_mul(RING_DEGREE * RING_DEGREE)
            .ok_or(MetalError::Shape("RLC rho dimensions overflow"))?;
        let expected_fresh = fresh_count
            .checked_mul(RING_DEGREE)
            .and_then(|values| values.checked_mul(cols))
            .ok_or(MetalError::Shape("RLC fresh-witness dimensions overflow"))?;
        if fresh_count == 0
            || fresh_count >= input_count
            || cols == 0
            || rhos.len() != expected_rhos
            || fresh_witnesses.len() != expected_fresh
            || resident_tail.child_count != input_count - fresh_count
            || resident_tail.cols != cols
        {
            return Err(MetalError::Shape(
                "RLC resident-tail inputs have inconsistent dimensions",
            ));
        }

        let rhos = self.buffer_from_slice(rhos)?;
        let fresh_witnesses = self.buffer_from_slice(fresh_witnesses)?;
        let shape = self.buffer_from_slice(&[input_count as u64, fresh_count as u64, cols as u64])?;
        let words = self.buffer(RING_DEGREE * cols * size_of::<u64>())?;
        let command = self.command_buffer()?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.rlc_witness_mix_resident_tail);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&rhos), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&fresh_witnesses), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&resident_tail.words), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&words), 0, 4);
        }
        self.dispatch(&encoder, &self.rlc_witness_mix_resident_tail, RING_DEGREE * cols);
        encoder.endEncoding();
        self.finish(&command)?;
        Ok(MetalResidentWitness { words, cols })
    }
}
