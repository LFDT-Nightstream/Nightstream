//! Resident witness ownership across Pi_RLC, Pi_DEC, and the next fold.
//!
//! Generation ids prevent stale carriers from addressing a newer buffer set;
//! immutable snapshots provide the explicit CPU-materialization path.

use std::mem::size_of;

use neo_ccs::Mat;
use neo_math::{D, F};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::{Buffer, MetalNcSumcheckPlan, MetalSession, MetalWitnessMasks};
use crate::MetalError;

const RING_DEGREE: usize = 54;

impl MetalWitnessMasks {
    pub(super) fn with_active_witnesses(mut self, active_witnesses: Vec<u32>) -> Result<Self, MetalError> {
        if active_witnesses
            .iter()
            .any(|&witness| witness as usize >= self.witness_count)
        {
            return Err(MetalError::Shape(
                "active witness index exceeds the resident mask batch",
            ));
        }
        self.active_witnesses = active_witnesses;
        Ok(self)
    }

    pub(super) fn active_witnesses(&self) -> &[u32] {
        &self.active_witnesses
    }
}

/// One dense ring witness whose producing Pi_RLC command may still be running.
pub(crate) struct MetalResidentWitness {
    pub(super) words: Buffer,
    pub(super) cols: usize,
    pending_mix: Option<PendingRlcMix>,
}

struct PendingRlcMix {
    // Retaining inputs until completion is required even though the Rust call
    // that encoded the command has already returned.
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _inputs: [Buffer; 3],
    recycled_children: Option<MetalResidentChildren>,
}

impl MetalResidentWitness {
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    /// Waits at the first true consumer of a deferred Pi_RLC mix and returns any
    /// child allocation whose lifetime was transferred into that command.
    pub(super) fn finish_pending_mix(
        &mut self,
        session: &MetalSession,
    ) -> Result<Option<MetalResidentChildren>, MetalError> {
        if let Some(pending) = self.pending_mix.take() {
            session.wait(&pending.command)?;
            return Ok(pending.recycled_children);
        }
        Ok(None)
    }
}

/// Pi_DEC child witnesses in both dense and signed-mask device layouts.
pub(crate) struct MetalResidentChildren {
    pub(super) words: Buffer,
    pub(super) masks: Buffer,
    pub(super) child_count: usize,
    pub(super) cols: usize,
    pub(super) active_witnesses: Vec<u32>,
}

/// Immutable signed-mask view used only when a running carrier leaves Metal.
pub(crate) struct MetalResidentWitnessSnapshot {
    masks: Buffer,
    child_count: usize,
    cols: usize,
}

// SAFETY: the retained MTLBuffer is immutable after the split command has
// completed. Metal resources support cross-thread retention and reads; this
// snapshot only reads `contents()` when crossing the explicit CPU boundary.
unsafe impl Send for MetalResidentWitnessSnapshot {}
unsafe impl Sync for MetalResidentWitnessSnapshot {}

impl MetalResidentWitnessSnapshot {
    /// Materializes compact signed masks into ordinary matrices only for a
    /// carrier that must leave the Metal execution path.
    pub(crate) fn materialize(&self) -> Result<Vec<Mat<F>>, &'static str> {
        let words = super::read_buffer::<u64>(&self.masks, self.child_count * self.cols * 2);
        let mut witnesses = Vec::with_capacity(self.child_count);
        for masks in words.chunks_exact(2 * self.cols) {
            let mut positive = Vec::with_capacity(self.cols);
            let mut negative = Vec::with_capacity(self.cols);
            for pair in masks.chunks_exact(2) {
                positive.push(pair[0]);
                negative.push(pair[1]);
            }
            witnesses.push(Mat::compact_signed_unit_from_column_masks(
                D, self.cols, &positive, &negative,
            )?);
        }
        Ok(witnesses)
    }
}

impl MetalResidentChildren {
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.child_count, self.cols)
    }

    pub(crate) fn snapshot(&self) -> MetalResidentWitnessSnapshot {
        MetalResidentWitnessSnapshot {
            masks: self.masks.clone(),
            child_count: self.child_count,
            cols: self.cols,
        }
    }
}

impl MetalSession {
    pub(crate) fn resident_child_masks(
        &self,
        children: &MetalResidentChildren,
        active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        MetalWitnessMasks::from_buffer(children.masks.clone(), children.child_count, children.cols, active_rows)?
            .with_active_witnesses(children.active_witnesses.clone())
    }

    pub(crate) fn materialize_resident_child_masks(&self, children: &MetalResidentChildren) -> Vec<u64> {
        self.read_buffer(&children.masks, children.child_count * children.cols * 2)
    }

    pub(crate) fn resident_child_mask_prefix(
        &self,
        children: &MetalResidentChildren,
        prefix_cols: usize,
    ) -> Result<Vec<u64>, MetalError> {
        if prefix_cols > children.cols {
            return Err(MetalError::Shape("Pi_DEC public mask prefix exceeds the child width"));
        }
        let words_per_child = 2 * prefix_cols;
        let mut words = Vec::with_capacity(children.child_count * words_per_child);
        for child in 0..children.child_count {
            words.extend(self.read_buffer_range::<u64>(&children.masks, 2 * child * children.cols, words_per_child));
        }
        Ok(words)
    }

    pub(crate) fn retain_running_children(&self, children: MetalResidentChildren) -> u64 {
        // The session retains exactly one generation. Replacing it invalidates
        // all older ids without making those ids part of the proof protocol.
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

    /// Uploads fresh host masks, appends the retained running masks on-device,
    /// and preserves logical active indices across the concatenated batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_witness_masks_with_resident_id(
        &self,
        fresh_words: &[u64],
        fresh_count: usize,
        input_count: usize,
        blocks: usize,
        active_rows: usize,
        resident_id: u64,
    ) -> Result<MetalWitnessMasks, MetalError> {
        let expected_fresh_words = fresh_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("fresh witness mask dimensions overflow"))?;
        let total_words = input_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2))
            .ok_or(MetalError::Shape("witness mask dimensions overflow"))?;
        let total_bytes = total_words
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::Shape("witness mask byte size overflow"))?;
        let tail_count = input_count
            .checked_sub(fresh_count)
            .filter(|&count| count > 0)
            .ok_or(MetalError::Shape("resident witness mask tail is empty"))?;
        if fresh_words.len() != expected_fresh_words {
            return Err(MetalError::Shape("fresh witness masks have inconsistent dimensions"));
        }
        {
            let resident = self.resident_running.borrow();
            let Some((stored_id, children)) = resident.as_ref() else {
                return Err(MetalError::Shape("resident witness masks are unavailable"));
            };
            let expected_tail_bytes = tail_count
                .checked_mul(blocks)
                .and_then(|values| values.checked_mul(2 * size_of::<u64>()))
                .ok_or(MetalError::Shape("resident witness mask byte size overflow"))?;
            if *stored_id != resident_id
                || children.child_count != tail_count
                || children.cols != blocks
                || children.masks.length() as usize != expected_tail_bytes
            {
                return Err(MetalError::Shape(
                    "resident witness masks do not match the requested tail",
                ));
            }
        }

        let masks = self.buffer(total_bytes)?;
        self.write_shared(&masks, fresh_words)?;
        let command = self.command_buffer("nightstream.pi_ccs.compose_witness_masks")?;
        {
            let resident = self.resident_running.borrow();
            let (_, children) = resident
                .as_ref()
                .expect("resident witness masks validated before command encoding");
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.copy_k_words);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&children.masks), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), fresh_words.len() * size_of::<u64>(), 1);
            }
            self.dispatch(&encoder, &self.copy_k_words, tail_count * blocks);
            encoder.endEncoding();
        }
        self.submit(&command);
        let mut active_witnesses = fresh_words
            .chunks_exact(2 * blocks)
            .enumerate()
            .filter(|(_, masks)| masks.iter().any(|&mask| mask != 0))
            .map(|(witness, _)| witness as u32)
            .collect::<Vec<_>>();
        {
            let resident = self.resident_running.borrow();
            let (_, children) = resident
                .as_ref()
                .expect("resident witness masks validated before command encoding");
            active_witnesses.extend(
                children
                    .active_witnesses
                    .iter()
                    .map(|&witness| witness + fresh_count as u32),
            );
        }
        MetalWitnessMasks::from_buffer(masks, input_count, blocks, active_rows)?.with_active_witnesses(active_witnesses)
    }

    /// Concatenates device-resident fresh and running masks without exposing
    /// either source to the CPU.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compose_witness_masks_from_device(
        &self,
        fresh: &MetalWitnessMasks,
        fresh_count: usize,
        input_count: usize,
        blocks: usize,
        active_rows: usize,
        resident_id: Option<u64>,
    ) -> Result<MetalWitnessMasks, MetalError> {
        if !fresh.matches_nc(fresh_count, blocks, active_rows) || fresh_count > input_count {
            return Err(MetalError::Shape("resident fresh masks have inconsistent dimensions"));
        }
        let tail_count = input_count - fresh_count;
        if tail_count == 0 {
            return Ok(fresh.clone());
        }
        let resident_id = resident_id.ok_or(MetalError::Shape("running witness masks are not resident"))?;
        let total_bytes = input_count
            .checked_mul(blocks)
            .and_then(|values| values.checked_mul(2 * size_of::<u64>()))
            .ok_or(MetalError::Shape("witness mask byte size overflow"))?;
        {
            let resident = self.resident_running.borrow();
            let Some((stored_id, children)) = resident.as_ref() else {
                return Err(MetalError::Shape("resident witness masks are unavailable"));
            };
            let expected_tail_bytes = tail_count
                .checked_mul(blocks)
                .and_then(|values| values.checked_mul(2 * size_of::<u64>()))
                .ok_or(MetalError::Shape("resident witness mask byte size overflow"))?;
            if *stored_id != resident_id
                || children.child_count != tail_count
                || children.cols != blocks
                || children.masks.length() as usize != expected_tail_bytes
            {
                return Err(MetalError::Shape(
                    "resident witness masks do not match the requested tail",
                ));
            }
        }

        let masks = self.buffer(total_bytes)?;
        let command = self.command_buffer("nightstream.pi_ccs.compose_resident_witness_masks")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.copy_k_words);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(fresh.words()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
        }
        self.dispatch(&encoder, &self.copy_k_words, fresh_count * blocks);
        encoder.endEncoding();
        {
            let resident = self.resident_running.borrow();
            let (_, children) = resident
                .as_ref()
                .expect("resident witness masks validated before command encoding");
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.copy_k_words);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&children.masks), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), fresh_count * blocks * 2 * size_of::<u64>(), 1);
            }
            self.dispatch(&encoder, &self.copy_k_words, tail_count * blocks);
            encoder.endEncoding();
        }
        self.submit(&command);
        let mut active_witnesses = fresh.active_witnesses().to_vec();
        {
            let resident = self.resident_running.borrow();
            let (_, children) = resident
                .as_ref()
                .expect("resident witness masks validated before command encoding");
            active_witnesses.extend(
                children
                    .active_witnesses
                    .iter()
                    .map(|&witness| witness + fresh_count as u32),
            );
        }
        MetalWitnessMasks::from_buffer(masks, input_count, blocks, active_rows)?.with_active_witnesses(active_witnesses)
    }

    pub(crate) fn enqueue_rlc_witness_mix_from_signed_masks(
        &self,
        rhos: &[i8],
        plan: &MetalNcSumcheckPlan,
        input_count: usize,
        cols: usize,
    ) -> Result<Option<MetalResidentWitness>, MetalError> {
        let expected_rhos = input_count
            .checked_mul(RING_DEGREE * RING_DEGREE)
            .ok_or(MetalError::Shape("RLC rho dimensions overflow"))?;
        if input_count == 0 || cols == 0 || rhos.len() != expected_rhos {
            return Err(MetalError::Shape("RLC signed-mask inputs have inconsistent dimensions"));
        }
        let Some(masks) = plan.signed_mask_buffer(input_count, cols) else {
            return Ok(None);
        };

        let rhos = self.buffer_from_slice(rhos)?;
        let shape = self.buffer_from_slice(&[input_count as u64, cols as u64])?;
        let words = self.buffer(RING_DEGREE * cols * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.pi_rlc.mix_signed_masks")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.rlc_witness_mix_signed_masks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&rhos), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&words), 0, 3);
        }
        self.dispatch(&encoder, &self.rlc_witness_mix_signed_masks, RING_DEGREE * cols);
        encoder.endEncoding();
        self.submit(&command);
        Ok(Some(MetalResidentWitness {
            words,
            cols,
            pending_mix: Some(PendingRlcMix {
                command,
                _inputs: [rhos, masks, shape],
                recycled_children: None,
            }),
        }))
    }

    /// Mixes fresh signed masks with a retained dense tail and moves ownership
    /// of the old generation into the pending command's lifetime.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_rlc_witness_mix_from_signed_masks_with_resident_id(
        &self,
        rhos: &[i8],
        plan: &MetalNcSumcheckPlan,
        fresh_count: usize,
        input_count: usize,
        cols: usize,
        resident_id: u64,
    ) -> Result<Option<MetalResidentWitness>, MetalError> {
        let expected_rhos = input_count
            .checked_mul(RING_DEGREE * RING_DEGREE)
            .ok_or(MetalError::Shape("RLC rho dimensions overflow"))?;
        let Some(masks) = plan.signed_mask_buffer(input_count, cols) else {
            return Ok(None);
        };
        {
            let resident = self.resident_running.borrow();
            let Some((stored_id, resident_tail)) = resident.as_ref() else {
                return Err(MetalError::Shape("RLC resident witness is no longer available"));
            };
            if *stored_id != resident_id {
                return Err(MetalError::Shape("RLC resident witness generation is stale"));
            }
            if fresh_count == 0
                || fresh_count >= input_count
                || cols == 0
                || rhos.len() != expected_rhos
                || resident_tail.child_count != input_count - fresh_count
                || resident_tail.cols != cols
            {
                return Err(MetalError::Shape(
                    "RLC resident signed-mask inputs have inconsistent dimensions",
                ));
            }
        }

        let rhos = self.buffer_from_slice(rhos)?;
        let shape = self.buffer_from_slice(&[input_count as u64, fresh_count as u64, cols as u64])?;
        let words = self.buffer(RING_DEGREE * cols * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.pi_rlc.mix_signed_masks_resident")?;
        {
            let resident = self.resident_running.borrow();
            let (_, resident_tail) = resident
                .as_ref()
                .expect("resident tail validated before command encoding");
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.rlc_witness_mix_signed_masks_resident_tail);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&rhos), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&resident_tail.words), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&words), 0, 4);
            }
            self.dispatch(
                &encoder,
                &self.rlc_witness_mix_signed_masks_resident_tail,
                RING_DEGREE * cols,
            );
            encoder.endEncoding();
        }
        // Transfer the old generation into the pending command's lifetime.
        // Once Pi_RLC completes, Pi_DEC can recycle its dense child storage.
        let (_, recycled_children) = self
            .resident_running
            .borrow_mut()
            .take()
            .expect("resident tail validated before ownership transfer");
        self.submit(&command);
        Ok(Some(MetalResidentWitness {
            words,
            cols,
            pending_mix: Some(PendingRlcMix {
                command,
                _inputs: [rhos, masks, shape],
                recycled_children: Some(recycled_children),
            }),
        }))
    }

    pub(crate) fn enqueue_rlc_witness_mix(
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
        let command = self.command_buffer("nightstream.pi_rlc.mix")?;
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
        self.submit(&command);
        Ok(MetalResidentWitness {
            words,
            cols,
            pending_mix: Some(PendingRlcMix {
                command,
                _inputs: [rhos, witnesses, shape],
                recycled_children: None,
            }),
        })
    }

    pub(crate) fn enqueue_rlc_witness_mix_with_resident_id(
        &self,
        rhos: &[i8],
        fresh_witnesses: &[u64],
        fresh_count: usize,
        input_count: usize,
        cols: usize,
        resident_id: u64,
    ) -> Result<MetalResidentWitness, MetalError> {
        let expected_rhos = input_count
            .checked_mul(RING_DEGREE * RING_DEGREE)
            .ok_or(MetalError::Shape("RLC rho dimensions overflow"))?;
        let expected_fresh = fresh_count
            .checked_mul(RING_DEGREE)
            .and_then(|values| values.checked_mul(cols))
            .ok_or(MetalError::Shape("RLC fresh-witness dimensions overflow"))?;
        {
            let resident = self.resident_running.borrow();
            let Some((stored_id, resident_tail)) = resident.as_ref() else {
                return Err(MetalError::Shape("RLC resident witness is no longer available"));
            };
            if *stored_id != resident_id {
                return Err(MetalError::Shape("RLC resident witness generation is stale"));
            }
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
        }

        let rhos = self.buffer_from_slice(rhos)?;
        let fresh_witnesses = self.buffer_from_slice(fresh_witnesses)?;
        let shape = self.buffer_from_slice(&[input_count as u64, fresh_count as u64, cols as u64])?;
        let words = self.buffer(RING_DEGREE * cols * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.pi_rlc.mix_resident")?;
        {
            let resident = self.resident_running.borrow();
            let (_, resident_tail) = resident
                .as_ref()
                .expect("resident tail validated before command encoding");
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
        }
        // The running generation cannot remain addressable after its buffer is
        // consumed by this mix, so ownership moves into the pending command.
        let (_, recycled_children) = self
            .resident_running
            .borrow_mut()
            .take()
            .expect("resident tail validated before ownership transfer");
        self.submit(&command);
        Ok(MetalResidentWitness {
            words,
            cols,
            pending_mix: Some(PendingRlcMix {
                command,
                _inputs: [rhos, fresh_witnesses, shape],
                recycled_children: Some(recycled_children),
            }),
        })
    }
}
