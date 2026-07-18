//! Base-2 DEC split, validation, child projection, commitment, and residency.
//!
//! Children remain device-owned; only compact claim material, commitments, and
//! validation status cross to the host during the normal fold path.

use std::mem::size_of;
use std::time::Duration;

use neo_math::{KExtensions, D, K};
use neo_reductions::superneo_eval::SuperneoEvalCache;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::forms::{DeviceFormBuild, MetalAjtaiRingForms, MetalDecFormPlan};
use super::{checked_product, CHUNK_COLUMNS, PRODUCT_COEFFICIENTS};
use crate::session::carrier::{MetalResidentChildren, MetalResidentWitness};
use crate::session::{Buffer, MetalAjtaiLowNormPlan, MetalDecPublicProjection, MetalSession};
use crate::MetalError;

/// Minimal host outputs plus ownership of the complete resident child batch.
pub(crate) struct MetalDecMaterial {
    pub child_nonzero: Vec<bool>,
    pub y_words: Vec<u64>,
    pub y_zcol_words: Vec<u64>,
    pub y_zcol_gpu: Duration,
    pub commitment_words: Vec<u64>,
    pub resident_children: MetalResidentChildren,
}

impl MetalSession {
    pub(crate) fn split_dec_base2_with_prebuilt_ring_forms(
        &self,
        parent: &mut MetalResidentWitness,
        child_count: usize,
        plan: &MetalDecFormPlan,
        forms: &MetalAjtaiRingForms,
        row_challenges: &[K],
        n_eff: usize,
        public_projection: Option<MetalDecPublicProjection<'_>>,
        commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        let point_matches = if forms.row_challenge_words.is_empty() {
            let chi_words = neo_ccs::utils::tensor_point_parallel::<K>(row_challenges)
                .into_iter()
                .flat_map(|value| {
                    let (real, imaginary) = value.to_limbs_u64();
                    [real, imaginary]
                })
                .collect::<Vec<_>>();
            forms.matches(plan, &chi_words, n_eff)
        } else {
            forms.matches_row_challenges(plan, row_challenges, n_eff)
        };
        if parent.cols != plan.blocks || !point_matches {
            return Err(MetalError::Shape(
                "Pi_DEC prebuilt ring forms do not match the folded row point",
            ));
        }
        self.split_dec_base2_with_form_buffer(
            parent,
            child_count,
            forms.form_rows,
            &forms.words,
            Some(plan),
            None,
            public_projection,
            commitment_plan,
        )
    }

    pub(crate) fn split_dec_base2_with_ring_forms(
        &self,
        parent: &mut MetalResidentWitness,
        child_count: usize,
        form_rows: usize,
        form_words: &[u64],
        public_projection: Option<MetalDecPublicProjection<'_>>,
        commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        let entries = checked_product(&[D, parent.cols], "Pi_DEC parent dimensions overflow")?;
        let expected_forms = checked_product(&[form_rows, entries], "Pi_DEC form dimensions overflow")?;
        if form_words.len() != expected_forms {
            return Err(MetalError::Shape("Pi_DEC ring forms do not match the resident witness"));
        }
        let forms = self.buffer_from_slice(form_words)?;
        self.split_dec_base2_with_form_buffer(
            parent,
            child_count,
            form_rows,
            &forms,
            None,
            None,
            public_projection,
            commitment_plan,
        )
    }

    pub(crate) fn split_dec_base2_with_ring_form_plan(
        &self,
        parent: &mut MetalResidentWitness,
        child_count: usize,
        plan: &MetalDecFormPlan,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        public_projection: Option<MetalDecPublicProjection<'_>>,
        commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        if chi_r.is_empty() || n_eff > chi_r.len() || plan.blocks != parent.cols || !plan.matches(cache) {
            return Err(MetalError::Shape(
                "Pi_DEC device form inputs have inconsistent dimensions",
            ));
        }
        let chi_words = chi_r
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let form_rows = 2 * plan.matrix_count;
        let form_words = checked_product(
            &[plan.active_block_count, 2, D],
            "Pi_DEC device form dimensions overflow",
        )?;
        let forms = self.buffer(form_words * size_of::<u64>())?;
        let chi = self.buffer_from_slice(&chi_words)?;
        let form_shape = self.buffer_from_slice(&[
            plan.matrix_count as u64,
            plan.blocks as u64,
            n_eff as u64,
            chi_r.len() as u64,
        ])?;
        let seeded = self.prepare_seeded_form_patch(plan, cache, chi_r, n_eff)?;
        self.split_dec_base2_with_form_buffer(
            parent,
            child_count,
            form_rows,
            &forms,
            Some(plan),
            Some(DeviceFormBuild {
                plan,
                chi,
                shape: form_shape,
                seeded,
            }),
            public_projection,
            commitment_plan,
        )
    }

    /// Encodes form construction, digit split, recomposition checks, projection,
    /// and child commitments as one dependency chain before minimal readback.
    fn split_dec_base2_with_form_buffer(
        &self,
        parent: &mut MetalResidentWitness,
        child_count: usize,
        form_rows: usize,
        forms: &Buffer,
        sparse_plan: Option<&MetalDecFormPlan>,
        form_build: Option<DeviceFormBuild<'_>>,
        public_projection: Option<MetalDecPublicProjection<'_>>,
        commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        if child_count == 0 || form_rows == 0 || parent.cols == 0 {
            return Err(MetalError::Shape("Pi_DEC resident dimensions are invalid"));
        }
        if commitment_plan.cols != parent.cols || commitment_plan.rows == 0 {
            return Err(MetalError::Shape(
                "Pi_DEC commitment plan does not match the resident witness",
            ));
        }
        // This is the first consumer of Pi_RLC output, so it is also the single
        // synchronization point for the pending mix and its recyclable inputs.
        let recycled_children = parent.finish_pending_mix(self)?;
        let entries = checked_product(&[D, parent.cols], "Pi_DEC parent dimensions overflow")?;
        let child_words = checked_product(&[child_count, entries], "Pi_DEC child dimensions overflow")?;

        let chunks = parent.cols.div_ceil(CHUNK_COLUMNS);
        let mask_words = checked_product(&[child_count, parent.cols, 2], "Pi_DEC mask dimensions overflow")?;
        let y_words = checked_product(&[child_count, form_rows, D], "Pi_DEC y dimensions overflow")?;
        let commitment_groups = commitment_plan.rows;
        let commitment_words = checked_product(
            &[child_count, commitment_groups, D],
            "Pi_DEC commitment dimensions overflow",
        )?;

        let children = match recycled_children {
            Some(children) if children.child_count == child_count && children.cols == parent.cols => children.words,
            Some(children) => {
                self.recycle_dec_children(children);
                self.take_recycled_dec_children(child_count, parent.cols, child_words * size_of::<u64>())?
            }
            None => self.take_recycled_dec_children(child_count, parent.cols, child_words * size_of::<u64>())?,
        };
        let masks = self.buffer(mask_words * size_of::<u64>())?;
        let split_status = self.buffer_from_slice(&[0u32])?;
        let child_nonzero = self.buffer_from_slice(&vec![0u32; child_count])?;
        let split_shape = self.buffer_from_slice(&[
            entries as u64,
            child_count as u64,
            form_rows as u64,
            parent.cols as u64,
            chunks as u64,
        ])?;
        // Command-buffer order is the dependency graph: build forms, split,
        // validate, pack masks, project, and commit before any host readback.
        let command = self.command_buffer("nightstream.pi_dec.split_project_commit")?;

        let seeded_scratch = if let Some(build) = form_build {
            self.encode_dec_form_build(
                &command,
                build.plan,
                &build.chi,
                &build.shape,
                forms,
                build.seeded.as_ref(),
            )?
        } else {
            None
        };

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.split_base2")));
        encoder.setComputePipelineState(&self.dec_split_base2);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&parent.words), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&split_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&children), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_split_base2, entries);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.validate_split")));
        encoder.setComputePipelineState(&self.dec_validate_split);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&parent.words), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&children), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&split_shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&split_status), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_validate_split, entries);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.binary_masks")));
        encoder.setComputePipelineState(&self.dec_binary_masks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&children), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&split_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&child_nonzero), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_binary_masks, child_count * parent.cols);
        encoder.endEncoding();

        // Project every fixed child before the first host readback. Inactive
        // children have zero masks, so their outputs remain zero without a
        // CPU compaction pass or a second command-buffer submission.
        let groups = checked_product(&[child_count, form_rows], "Pi_DEC output dimensions overflow")?;
        let y_chunks = sparse_plan.map_or(chunks, |plan| plan.active_chunk_count);
        let partial_words = if sparse_plan.is_some() {
            checked_product(
                &[child_count, y_chunks, 2, PRODUCT_COEFFICIENTS],
                "Pi_DEC partial dimensions overflow",
            )?
        } else {
            checked_product(
                &[groups, y_chunks, PRODUCT_COEFFICIENTS],
                "Pi_DEC partial dimensions overflow",
            )?
        };
        let sum_words = checked_product(&[groups, PRODUCT_COEFFICIENTS], "Pi_DEC sum dimensions overflow")?;
        let commitment_partial_words = checked_product(
            &[child_count, commitment_groups, chunks, PRODUCT_COEFFICIENTS],
            "Pi_DEC commitment partial dimensions overflow",
        )?;
        let commitment_sum_words = checked_product(
            &[child_count, commitment_groups, PRODUCT_COEFFICIENTS],
            "Pi_DEC commitment sum dimensions overflow",
        )?;

        let all_children = (0..child_count)
            .map(|child| child as u32)
            .collect::<Vec<_>>();
        let all_children = self.buffer_from_slice(&all_children)?;
        let partials = self.take_recycled_buffer(&self.recycled_dec_partials, partial_words * size_of::<u64>())?;
        let sums = self.buffer(sum_words * size_of::<u64>())?;
        let y = self.buffer(y_words * size_of::<u64>())?;
        let commitment_partials = self.buffer(commitment_partial_words * size_of::<u64>())?;
        let commitment_sums = self.buffer(commitment_sum_words * size_of::<u64>())?;
        let commitments = self.buffer(commitment_words * size_of::<u64>())?;
        let shape = self.buffer_from_slice(&[
            entries as u64,
            child_count as u64,
            form_rows as u64,
            parent.cols as u64,
            y_chunks as u64,
        ])?;
        let commitment_shape = self.buffer_from_slice(&[
            entries as u64,
            child_count as u64,
            commitment_groups as u64,
            parent.cols as u64,
            chunks as u64,
        ])?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.y_partials")));
        if let Some(plan) = sparse_plan {
            encoder.setComputePipelineState(&self.dec_sparse_ring_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(forms), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&all_children), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_bases), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_matrices), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&plan.matrix_active_offsets), 0, 8);
            }
            self.dispatch(&encoder, &self.dec_sparse_ring_partials, partial_words);
        } else {
            encoder.setComputePipelineState(&self.dec_ring_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(forms), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&all_children), 0, 4);
            }
            self.dispatch(&encoder, &self.dec_ring_partials, partial_words);
        }
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.y_sum_chunks")));
        if let Some(plan) = sparse_plan {
            encoder.setComputePipelineState(&self.dec_sparse_ring_sum_chunks);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&plan.matrix_chunk_offsets), 0, 3);
            }
            self.dispatch(&encoder, &self.dec_sparse_ring_sum_chunks, sum_words);
        } else {
            encoder.setComputePipelineState(&self.dec_ring_sum_chunks);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
            }
            self.dispatch(&encoder, &self.dec_ring_sum_chunks, sum_words);
        }
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.y_reduce_phi81")));
        encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&y), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_reduce_phi81, y_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.commit_partials")));
        encoder.setComputePipelineState(&self.dec_ring_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&commitment_plan.matrix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&commitment_shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&commitment_partials), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&all_children), 0, 4);
        }
        self.dispatch(&encoder, &self.dec_ring_partials, commitment_partial_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.commit_sum_chunks")));
        encoder.setComputePipelineState(&self.dec_ring_sum_chunks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&commitment_partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&commitment_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&commitment_sums), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_sum_chunks, commitment_sum_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_dec.commit_reduce_phi81")));
        encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&commitment_sums), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&commitment_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&commitments), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_reduce_phi81, commitment_words);
        encoder.endEncoding();
        let pending_y_zcol = public_projection
            .map(|projection| self.enqueue_dec_child_y_zcol(&parent.words, child_count, parent.cols, projection))
            .transpose()?;
        let (y_zcol_words, y_zcol_gpu) = if let Some(pending) = pending_y_zcol {
            pending.finish_after(self, &command)?
        } else {
            self.finish(&command)?;
            (Vec::new(), Duration::ZERO)
        };
        if let Some(scratch) = seeded_scratch {
            Self::recycle_largest_buffer(&self.recycled_seeded_forms, scratch);
        }

        // Reject if any digit is outside {-1, 0, 1} or the base-2 children fail
        // to recompose the parent. This accelerator check does not replace the
        // canonical proof verifier.
        if self.read_buffer::<u32>(&split_status, 1)[0] != 0 {
            return Err(MetalError::Shape(
                "Metal Pi_DEC digits are out of range or do not recompose",
            ));
        }
        let child_nonzero_words = self.read_buffer::<u32>(&child_nonzero, child_count);
        let active_witnesses = child_nonzero_words
            .iter()
            .enumerate()
            .filter(|(_, &value)| value != 0)
            .map(|(child, _)| child as u32)
            .collect();
        let child_nonzero = child_nonzero_words
            .into_iter()
            .map(|value| value != 0)
            .collect();
        Self::recycle_largest_buffer(&self.recycled_dec_partials, partials);
        Ok(MetalDecMaterial {
            child_nonzero,
            y_words: self.read_buffer::<u64>(&y, y_words),
            y_zcol_words,
            y_zcol_gpu,
            commitment_words: self.read_buffer::<u64>(&commitments, commitment_words),
            resident_children: MetalResidentChildren {
                words: children,
                masks,
                child_count,
                cols: parent.cols,
                active_witnesses,
            },
        })
    }

    pub(crate) fn recycle_dec_children(&self, children: MetalResidentChildren) {
        let bytes = children.words.length();
        let mut slot = self.recycled_dec_children.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|cached| cached.words.length() <= bytes)
        {
            *slot = Some(children);
        }
    }

    fn take_recycled_dec_children(&self, child_count: usize, cols: usize, bytes: usize) -> Result<Buffer, MetalError> {
        let recycled = {
            let mut slot = self.recycled_dec_children.borrow_mut();
            slot.as_ref()
                .is_some_and(|children| {
                    children.child_count == child_count
                        && children.cols == cols
                        && children.words.length() as usize == bytes
                })
                .then(|| {
                    slot.take()
                        .expect("matching recycled Pi_DEC children exist above")
                        .words
                })
        };
        recycled.map_or_else(|| self.buffer(bytes), Ok)
    }
}
