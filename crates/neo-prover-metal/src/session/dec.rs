//! Metal ownership of the base-2 Pi_DEC split and child opening evaluation.

use std::mem::size_of;

use neo_reductions::superneo_eval::SuperneoEvalCache;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::PrimeField64;

use super::carrier::{MetalResidentChildren, MetalResidentWitness};
use super::{Buffer, MetalAjtaiLowNormPlan, MetalSession};
use crate::MetalError;

const RING_DEGREE: usize = 54;
const PRODUCT_COEFFICIENTS: usize = 2 * RING_DEGREE - 1;
const CHUNK_COLUMNS: usize = 64;

pub(crate) struct MetalDecMaterial {
    pub child_mask_words: Vec<u64>,
    pub child_nonzero: Vec<bool>,
    pub y_words: Vec<u64>,
    pub commitment_words: Vec<u64>,
    pub resident_children: MetalResidentChildren,
}

pub(crate) struct MetalDecFormPlan {
    matrix_block_offsets: Buffer,
    entry_rows: Buffer,
    entry_bars: Buffer,
    matrix_count: usize,
    blocks: usize,
    cache_identity: usize,
    matrix_digest: [u64; 4],
}

impl MetalDecFormPlan {
    pub(crate) fn matches(&self, cache: &SuperneoEvalCache) -> bool {
        self.cache_identity == cache as *const SuperneoEvalCache as usize
            && self.matrix_digest == cache.mat_digest().map(|value| value.as_canonical_u64())
            && self.matrix_count == cache.matrix_caches().len()
    }
}

struct DeviceFormBuild<'a> {
    plan: &'a MetalDecFormPlan,
    chi: &'a Buffer,
    shape: &'a Buffer,
}

impl MetalSession {
    pub(crate) fn prepare_dec_ring_forms(&self, cache: &SuperneoEvalCache) -> Result<MetalDecFormPlan, MetalError> {
        let matrices = cache.matrix_caches();
        let Some(first) = matrices.first() else {
            return Err(MetalError::Shape("Pi_DEC form plan requires CCS matrices"));
        };
        let (_, scalar_columns, _, _) = first.bar_shape();
        let blocks = scalar_columns.div_ceil(RING_DEGREE);
        if blocks == 0 {
            return Err(MetalError::Shape("Pi_DEC form plan has zero columns"));
        }

        let mut matrix_block_offsets = Vec::with_capacity(matrices.len() * (blocks + 1));
        let mut entry_rows = Vec::new();
        let mut entry_bars = Vec::new();
        let mut entry_base = 0usize;
        for matrix in matrices {
            let (rows, columns, row_offsets, entry_count) = matrix.bar_shape();
            if columns != scalar_columns || row_offsets.len() != rows + 1 {
                return Err(MetalError::Shape("Pi_DEC matrices have inconsistent shapes"));
            }
            let mut block_offsets = vec![0usize; blocks + 1];
            for entry in 0..entry_count {
                let (block, _) = matrix.bar_entry(entry);
                if block >= blocks {
                    return Err(MetalError::Shape("Pi_DEC matrix block is out of range"));
                }
                block_offsets[block + 1] += 1;
            }
            for block in 0..blocks {
                block_offsets[block + 1] += block_offsets[block];
            }
            matrix_block_offsets.extend(
                block_offsets
                    .iter()
                    .map(|&offset| (entry_base + offset) as u64),
            );

            let mut next = block_offsets[..blocks].to_vec();
            let mut matrix_rows = vec![0u64; entry_count];
            let mut matrix_bars = vec![0u64; entry_count * RING_DEGREE];
            for row in 0..rows {
                for entry in row_offsets[row]..row_offsets[row + 1] {
                    let (block, bar) = matrix.bar_entry(entry);
                    let slot = next[block];
                    next[block] += 1;
                    matrix_rows[slot] = row as u64;
                    for (coefficient, value) in bar.0.iter().enumerate() {
                        matrix_bars[slot * RING_DEGREE + coefficient] = value.as_canonical_u64();
                    }
                }
            }
            entry_rows.extend(matrix_rows);
            entry_bars.extend(matrix_bars);
            entry_base = entry_base
                .checked_add(entry_count)
                .ok_or(MetalError::Shape("Pi_DEC form entry count overflow"))?;
        }

        Ok(MetalDecFormPlan {
            matrix_block_offsets: self.buffer_from_slice(&matrix_block_offsets)?,
            entry_rows: self.buffer_from_slice(&entry_rows)?,
            entry_bars: self.buffer_from_slice(&entry_bars)?,
            matrix_count: matrices.len(),
            blocks,
            cache_identity: cache as *const SuperneoEvalCache as usize,
            matrix_digest: cache.mat_digest().map(|value| value.as_canonical_u64()),
        })
    }

    pub(crate) fn split_dec_base2_with_ring_forms(
        &self,
        parent: &MetalResidentWitness,
        child_count: usize,
        form_rows: usize,
        form_words: &[u64],
        commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        let entries = checked_product(&[RING_DEGREE, parent.cols], "Pi_DEC parent dimensions overflow")?;
        let expected_forms = checked_product(&[form_rows, entries], "Pi_DEC form dimensions overflow")?;
        if form_words.len() != expected_forms {
            return Err(MetalError::Shape("Pi_DEC ring forms do not match the resident witness"));
        }
        let forms = self.buffer_from_slice(form_words)?;
        self.split_dec_base2_with_form_buffer(parent, child_count, form_rows, &forms, None, commitment_plan)
    }

    pub(crate) fn split_dec_base2_with_ring_form_plan(
        &self,
        parent: &MetalResidentWitness,
        child_count: usize,
        plan: &MetalDecFormPlan,
        chi_words: &[u64],
        n_eff: usize,
        commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        if chi_words.is_empty() || chi_words.len() % 2 != 0 || n_eff > chi_words.len() / 2 || plan.blocks != parent.cols
        {
            return Err(MetalError::Shape(
                "Pi_DEC device form inputs have inconsistent dimensions",
            ));
        }
        let form_rows = 2 * plan.matrix_count;
        let form_words = checked_product(
            &[form_rows, parent.cols, RING_DEGREE],
            "Pi_DEC device form dimensions overflow",
        )?;
        let forms = self.buffer(form_words * size_of::<u64>())?;
        let chi = self.buffer_from_slice(chi_words)?;
        let form_shape = self.buffer_from_slice(&[
            plan.matrix_count as u64,
            plan.blocks as u64,
            n_eff as u64,
            (chi_words.len() / 2) as u64,
        ])?;
        self.split_dec_base2_with_form_buffer(
            parent,
            child_count,
            form_rows,
            &forms,
            Some(DeviceFormBuild {
                plan,
                chi: &chi,
                shape: &form_shape,
            }),
            commitment_plan,
        )
    }

    fn split_dec_base2_with_form_buffer(
        &self,
        parent: &MetalResidentWitness,
        child_count: usize,
        form_rows: usize,
        forms: &Buffer,
        form_build: Option<DeviceFormBuild<'_>>,
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
        let entries = checked_product(&[RING_DEGREE, parent.cols], "Pi_DEC parent dimensions overflow")?;
        let child_words = checked_product(&[child_count, entries], "Pi_DEC child dimensions overflow")?;

        let chunks = parent.cols.div_ceil(CHUNK_COLUMNS);
        let mask_words = checked_product(&[child_count, parent.cols, 2], "Pi_DEC mask dimensions overflow")?;
        let y_words = checked_product(&[child_count, form_rows, RING_DEGREE], "Pi_DEC y dimensions overflow")?;
        let commitment_groups = commitment_plan.rows;
        let commitment_words = checked_product(
            &[child_count, commitment_groups, RING_DEGREE],
            "Pi_DEC commitment dimensions overflow",
        )?;

        let children = self.buffer(child_words * size_of::<u64>())?;
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
        let command = self.command_buffer()?;

        if let Some(build) = form_build {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_build_ring_forms);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&build.plan.matrix_block_offsets), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&build.plan.entry_rows), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&build.plan.entry_bars), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(build.chi), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(build.shape), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(forms), 0, 5);
            }
            self.dispatch(&encoder, &self.dec_build_ring_forms, form_rows * entries);
            encoder.endEncoding();
        }

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.dec_split_base2);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&parent.words), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&split_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&children), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_split_base2, entries);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
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
        encoder.setComputePipelineState(&self.dec_binary_masks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&children), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&split_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&child_nonzero), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_binary_masks, child_count * parent.cols);
        encoder.endEncoding();

        self.finish(&command)?;
        if self.read_buffer::<u32>(&split_status, 1)[0] != 0 {
            return Err(MetalError::Shape(
                "Metal Pi_DEC digits are out of range or do not recompose",
            ));
        }
        let child_nonzero_words = self.read_buffer::<u32>(&child_nonzero, child_count);
        let active_children = child_nonzero_words
            .iter()
            .enumerate()
            .filter_map(|(child, &nonzero)| (nonzero != 0).then_some(child as u32))
            .collect::<Vec<_>>();
        let active_count = active_children.len();
        let (y_output_words, commitment_output_words) = if active_count == 0 {
            (vec![0; y_words], vec![0; commitment_words])
        } else {
            let groups = checked_product(&[active_count, form_rows], "Pi_DEC active output dimensions overflow")?;
            let partial_words = checked_product(
                &[groups, chunks, PRODUCT_COEFFICIENTS],
                "Pi_DEC active partial dimensions overflow",
            )?;
            let sum_words = checked_product(&[groups, PRODUCT_COEFFICIENTS], "Pi_DEC active sum dimensions overflow")?;
            let active_y_words = checked_product(&[groups, RING_DEGREE], "Pi_DEC active y dimensions overflow")?;
            let commitment_partial_words = checked_product(
                &[active_count, commitment_groups, chunks, PRODUCT_COEFFICIENTS],
                "Pi_DEC active commitment partial dimensions overflow",
            )?;
            let commitment_sum_words = checked_product(
                &[active_count, commitment_groups, PRODUCT_COEFFICIENTS],
                "Pi_DEC active commitment sum dimensions overflow",
            )?;
            let active_commitment_words = checked_product(
                &[active_count, commitment_groups, RING_DEGREE],
                "Pi_DEC active commitment dimensions overflow",
            )?;

            let active_children_buffer = self.buffer_from_slice(&active_children)?;
            let partials = self.buffer(partial_words * size_of::<u64>())?;
            let sums = self.buffer(sum_words * size_of::<u64>())?;
            let y = self.buffer(active_y_words * size_of::<u64>())?;
            let commitment_partials = self.buffer(commitment_partial_words * size_of::<u64>())?;
            let commitment_sums = self.buffer(commitment_sum_words * size_of::<u64>())?;
            let commitments = self.buffer(active_commitment_words * size_of::<u64>())?;
            let shape = self.buffer_from_slice(&[
                entries as u64,
                active_count as u64,
                form_rows as u64,
                parent.cols as u64,
                chunks as u64,
            ])?;
            let commitment_shape = self.buffer_from_slice(&[
                entries as u64,
                active_count as u64,
                commitment_groups as u64,
                parent.cols as u64,
                chunks as u64,
            ])?;
            let projection_command = self.command_buffer()?;

            let encoder = projection_command
                .computeCommandEncoder()
                .ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(forms), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&active_children_buffer), 0, 4);
            }
            self.dispatch(&encoder, &self.dec_ring_partials, partial_words);
            encoder.endEncoding();

            let encoder = projection_command
                .computeCommandEncoder()
                .ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_sum_chunks);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
            }
            self.dispatch(&encoder, &self.dec_ring_sum_chunks, sum_words);
            encoder.endEncoding();

            let encoder = projection_command
                .computeCommandEncoder()
                .ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&y), 0, 2);
            }
            self.dispatch(&encoder, &self.dec_ring_reduce_phi81, groups);
            encoder.endEncoding();

            let encoder = projection_command
                .computeCommandEncoder()
                .ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&commitment_plan.matrix), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&commitment_shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&commitment_partials), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&active_children_buffer), 0, 4);
            }
            self.dispatch(&encoder, &self.dec_ring_partials, commitment_partial_words);
            encoder.endEncoding();

            let encoder = projection_command
                .computeCommandEncoder()
                .ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_sum_chunks);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&commitment_partials), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&commitment_shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&commitment_sums), 0, 2);
            }
            self.dispatch(&encoder, &self.dec_ring_sum_chunks, commitment_sum_words);
            encoder.endEncoding();

            let encoder = projection_command
                .computeCommandEncoder()
                .ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&commitment_sums), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&commitment_shape), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&commitments), 0, 2);
            }
            self.dispatch(&encoder, &self.dec_ring_reduce_phi81, active_count * commitment_groups);
            encoder.endEncoding();
            self.finish(&projection_command)?;

            let active_y = self.read_buffer::<u64>(&y, active_y_words);
            let active_commitments = self.read_buffer::<u64>(&commitments, active_commitment_words);
            let mut full_y = vec![0; y_words];
            let mut full_commitments = vec![0; commitment_words];
            for (active, &child) in active_children.iter().enumerate() {
                let child = child as usize;
                let y_per_child = form_rows * RING_DEGREE;
                let commitment_per_child = commitment_groups * RING_DEGREE;
                full_y[child * y_per_child..(child + 1) * y_per_child]
                    .copy_from_slice(&active_y[active * y_per_child..(active + 1) * y_per_child]);
                full_commitments[child * commitment_per_child..(child + 1) * commitment_per_child].copy_from_slice(
                    &active_commitments[active * commitment_per_child..(active + 1) * commitment_per_child],
                );
            }
            (full_y, full_commitments)
        };
        Ok(MetalDecMaterial {
            child_mask_words: self.read_buffer::<u64>(&masks, mask_words),
            child_nonzero: child_nonzero_words
                .into_iter()
                .map(|value| value != 0)
                .collect(),
            y_words: y_output_words,
            commitment_words: commitment_output_words,
            resident_children: MetalResidentChildren {
                words: children,
                child_count,
                cols: parent.cols,
            },
        })
    }
}

fn checked_product(factors: &[usize], message: &'static str) -> Result<usize, MetalError> {
    factors
        .iter()
        .try_fold(1usize, |value, &factor| value.checked_mul(factor))
        .ok_or(MetalError::Shape(message))
}
