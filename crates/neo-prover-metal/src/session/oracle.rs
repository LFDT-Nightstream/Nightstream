//! Device-owned preparation of the carried Pi_CCS Eval table.

use std::collections::BTreeMap;
use std::mem::size_of;
use std::time::{Duration, Instant};

use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{
    weighted_projection_basis_forms, SuperneoEvalCache, SuperneoMatrixCache, SuperneoZBlocks,
};
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::PrimeField64;
use rayon::prelude::*;

use super::{Buffer, MetalSession, MetalWitnessMasks};
use crate::MetalError;

const SEEDED_CHUNK_COLUMNS: usize = 128;
const RING_PRODUCT_COEFFICIENTS: usize = 2 * D - 1;
const SEEDED_OUTPUT_HEADER_WORDS: usize = 9;
const SEEDED_WORK_HEADER_WORDS: usize = 3;

pub(super) struct DeviceSeededRows {
    pub(super) output_headers: Buffer,
    work_headers: Buffer,
    pub(super) word_starts: Buffer,
    pub(super) rotations: Buffer,
    eval_group_headers: Buffer,
    eval_group_outputs: Buffer,
    pub(super) output_count: usize,
    work_count: usize,
    eval_group_count: usize,
}

struct SeededOutputMeta {
    row_start: usize,
}

pub(crate) struct MetalFeOraclePlan {
    matrix_row_offsets: Buffer,
    matrix_entry_bases: Buffer,
    matrix_identity: Buffer,
    entry_columns: Buffer,
    entry_coefficients: Buffer,
    matrix_count: usize,
    rows: usize,
    blocks: usize,
    has_seeded: bool,
    seeded: Option<DeviceSeededRows>,
    explicit_coefficients: usize,
    explicit_row_list_histogram: [usize; 8],
    max_explicit_row_entries: usize,
    cache_identity: usize,
    matrix_digest: [u64; 4],
}

pub(crate) struct MetalDeferredMcsRowTables {
    words: Buffer,
    mcs_idx: usize,
    n_pad: usize,
    table_count: usize,
    seeded_build: Duration,
    seeded_patch_entries: usize,
    seeded_patch_bytes: usize,
}

pub(crate) struct MetalDeferredEvalTable {
    words: Buffer,
    n_pad: usize,
}

impl MetalFeOraclePlan {
    pub(crate) fn matches(&self, cache: &SuperneoEvalCache) -> bool {
        self.cache_identity == cache as *const SuperneoEvalCache as usize
            && self.matrix_digest == cache.mat_digest().map(|value| value.as_canonical_u64())
            && self.matrix_count == cache.matrix_caches().len()
    }

    pub(super) fn seeded_rows(&self) -> Option<&DeviceSeededRows> {
        self.seeded.as_ref()
    }

    pub(crate) fn supports_resident_eval(&self) -> bool {
        !self.has_seeded || self.seeded.is_some()
    }

    pub(crate) fn explicit_coefficients(&self) -> usize {
        self.explicit_coefficients
    }

    pub(crate) fn explicit_row_list_histogram(&self) -> [usize; 8] {
        self.explicit_row_list_histogram
    }

    pub(crate) fn max_explicit_row_entries(&self) -> usize {
        self.max_explicit_row_entries
    }
}

impl MetalDeferredMcsRowTables {
    pub(crate) fn matches(&self, mcs_idx: usize, n_pad: usize, table_count: usize) -> bool {
        self.mcs_idx == mcs_idx
            && self.n_pad == n_pad
            && self.table_count == table_count
            && self.words.length() as usize == table_count * n_pad * size_of::<u64>()
    }

    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }

    pub(super) fn n_pad(&self) -> usize {
        self.n_pad
    }

    pub(super) fn table_count(&self) -> usize {
        self.table_count
    }

    pub(crate) fn seeded_build(&self) -> Duration {
        self.seeded_build
    }

    pub(crate) fn seeded_patch_entries(&self) -> usize {
        self.seeded_patch_entries
    }

    pub(crate) fn seeded_patch_bytes(&self) -> usize {
        self.seeded_patch_bytes
    }
}

impl MetalDeferredEvalTable {
    pub(crate) fn matches(&self, n_pad: usize) -> bool {
        self.n_pad == n_pad && self.words.length() as usize == 2 * n_pad * size_of::<u64>()
    }

    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }
}

impl MetalSession {
    pub(crate) fn prepare_fe_oracle(&self, cache: &SuperneoEvalCache) -> Result<MetalFeOraclePlan, MetalError> {
        let matrices = cache.matrix_caches();
        let Some(first) = matrices.first() else {
            return Err(MetalError::Shape("Pi_CCS oracle plan requires CCS matrices"));
        };
        let (rows, scalar_columns, _, _) = first.bar_shape();
        let blocks = scalar_columns.div_ceil(D);
        if rows == 0 || blocks == 0 {
            return Err(MetalError::Shape("Pi_CCS oracle matrices have an empty shape"));
        }

        let total_coefficients = matrices
            .iter()
            .map(|matrix| matrix.compact_explicit_coefficient_count())
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or(MetalError::Shape("Pi_CCS explicit coefficient count overflow"))?;
        let row_offset_count = matrices
            .len()
            .checked_mul(rows + 1)
            .ok_or(MetalError::Shape("Pi_CCS row offset count overflow"))?;
        let entry_counts = matrices
            .iter()
            .map(SuperneoMatrixCache::compact_explicit_coefficient_count)
            .collect::<Vec<_>>();
        let mut matrix_entry_bases = Vec::with_capacity(matrices.len());
        let mut entry_base = 0usize;
        for &count in &entry_counts {
            matrix_entry_bases.push(entry_base as u64);
            entry_base = entry_base
                .checked_add(count)
                .ok_or(MetalError::Shape("Pi_CCS explicit coefficient count overflow"))?;
        }
        if entry_base != total_coefficients {
            return Err(MetalError::Shape("Pi_CCS compact row coefficient count changed"));
        }
        let matrix_identity = matrices
            .iter()
            .map(|matrix| u32::from(matrix.compact_explicit_shape().2))
            .collect::<Vec<_>>();
        let has_seeded = matrices
            .iter()
            .any(SuperneoMatrixCache::has_compact_seeded_phi81_blocks);

        let matrix_row_offsets = self.buffer(row_offset_count * size_of::<u32>())?;
        let entry_columns = self.buffer(total_coefficients.max(1) * size_of::<u32>())?;
        let entry_coefficients = self.buffer(total_coefficients.max(1) * size_of::<u64>())?;
        let row_offsets = unsafe {
            std::slice::from_raw_parts_mut(matrix_row_offsets.contents().as_ptr().cast::<u32>(), row_offset_count)
        };
        let columns = unsafe {
            std::slice::from_raw_parts_mut(
                entry_columns.contents().as_ptr().cast::<u32>(),
                total_coefficients.max(1),
            )
        };
        let coefficients = unsafe {
            std::slice::from_raw_parts_mut(
                entry_coefficients.contents().as_ptr().cast::<u64>(),
                total_coefficients.max(1),
            )
        };
        let mut row_tail = row_offsets;
        let mut column_tail = &mut columns[..total_coefficients];
        let mut coefficient_tail = &mut coefficients[..total_coefficients];
        let mut matrix_parts = Vec::with_capacity(matrices.len());
        for ((matrix, &count), &identity) in matrices.iter().zip(&entry_counts).zip(&matrix_identity) {
            let (matrix_offsets, remaining_offsets) = row_tail.split_at_mut(rows + 1);
            let (matrix_columns, remaining_columns) = column_tail.split_at_mut(count);
            let (matrix_coefficients, remaining_coefficients) = coefficient_tail.split_at_mut(count);
            matrix_parts.push((matrix, identity, matrix_offsets, matrix_columns, matrix_coefficients));
            row_tail = remaining_offsets;
            column_tail = remaining_columns;
            coefficient_tail = remaining_coefficients;
        }
        let row_profiles = matrix_parts
            .into_par_iter()
            .map(|(matrix, identity, offsets, columns, coefficients)| {
                let (matrix_rows, matrix_columns, actual_identity) = matrix.compact_explicit_shape();
                if matrix_rows != rows || matrix_columns != scalar_columns || u32::from(actual_identity) != identity {
                    return Err(MetalError::Shape("Pi_CCS oracle matrices have inconsistent shapes"));
                }
                let mut histogram = [0usize; 8];
                let mut max_entries = 0usize;
                let mut local_offset = 0u32;
                offsets[0] = 0;
                let mut invalid = false;
                for row in 0..rows {
                    let row_start = local_offset;
                    matrix.for_each_compact_explicit_row_coefficient(row, |block, local, coefficient| {
                        let column = block as usize * D + local as usize;
                        let destination = local_offset as usize;
                        if column >= scalar_columns || destination >= columns.len() {
                            invalid = true;
                            return;
                        }
                        columns[destination] = column as u32;
                        coefficients[destination] = coefficient.as_canonical_u64();
                        let Some(next) = local_offset.checked_add(1) else {
                            invalid = true;
                            return;
                        };
                        local_offset = next;
                    });
                    if invalid {
                        return Err(MetalError::Shape("Pi_CCS compact row entry is out of range"));
                    }
                    let list_len = (local_offset - row_start) as usize;
                    histogram[row_list_histogram_bin(list_len)] += 1;
                    max_entries = max_entries.max(list_len);
                    offsets[row + 1] = local_offset;
                }
                if local_offset as usize != columns.len() {
                    return Err(MetalError::Shape("Pi_CCS compact row coefficient count changed"));
                }
                Ok((histogram, max_entries))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut explicit_row_list_histogram = [0usize; 8];
        let mut max_explicit_row_entries = 0usize;
        for (histogram, max_entries) in row_profiles {
            for (total, count) in explicit_row_list_histogram.iter_mut().zip(histogram) {
                *total += count;
            }
            max_explicit_row_entries = max_explicit_row_entries.max(max_entries);
        }
        self.record_host_write(row_offset_count * size_of::<u32>());
        self.record_host_write(total_coefficients * (size_of::<u32>() + size_of::<u64>()));
        let seeded = self.prepare_device_seeded_rows(matrices, rows)?;

        Ok(MetalFeOraclePlan {
            matrix_row_offsets,
            matrix_entry_bases: self.buffer_from_slice(&matrix_entry_bases)?,
            matrix_identity: self.buffer_from_slice(&matrix_identity)?,
            entry_columns,
            entry_coefficients,
            matrix_count: matrices.len(),
            rows,
            blocks,
            has_seeded,
            seeded,
            explicit_coefficients: total_coefficients,
            explicit_row_list_histogram,
            max_explicit_row_entries,
            cache_identity: cache as *const SuperneoEvalCache as usize,
            matrix_digest: cache.mat_digest().map(|value| value.as_canonical_u64()),
        })
    }

    fn prepare_device_seeded_rows(
        &self,
        matrices: &[neo_reductions::superneo_eval::SuperneoMatrixCache],
        rows: usize,
    ) -> Result<Option<DeviceSeededRows>, MetalError> {
        let blocks = matrices
            .iter()
            .flat_map(|matrix| matrix.compact_seeded_phi81_blocks())
            .collect::<Vec<_>>();
        if blocks.is_empty() {
            return Ok(None);
        }
        if blocks
            .iter()
            .any(|block| block.has_superneo_transformed_columns())
        {
            return Ok(None);
        }

        let mut output_headers = Vec::<u64>::new();
        let mut output_meta = Vec::<SeededOutputMeta>::new();
        let mut work_headers = Vec::<u64>::new();
        let mut word_starts = Vec::<u32>::new();
        let mut rotations = Vec::<u64>::new();
        for (matrix, cache) in matrices.iter().enumerate() {
            for block in cache.compact_seeded_phi81_blocks() {
                if block.row_end() > rows {
                    return Err(MetalError::Shape("Pi_CCS seeded row range exceeds the matrix"));
                }
                let word_base = word_starts.len();
                word_starts.extend(
                    block
                        .word_starts()
                        .iter()
                        .map(|&column| {
                            u32::try_from(column).map_err(|_| MetalError::Shape("Pi_CCS seeded column exceeds u32"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
                for output in 0..block.kappa() {
                    let output_index = output_meta.len();
                    let row_start = block.row_start() + output * D;
                    let rotation_base = rotations.len();
                    let mut next_message_col = 0usize;
                    block.for_each_original_output_rotation::<F, _>(output, |message_col, rotation| {
                        debug_assert_eq!(message_col, next_message_col);
                        next_message_col += 1;
                        rotations.extend(rotation.iter().map(PrimeField64::as_canonical_u64));
                    });
                    if next_message_col != block.message_cols()
                        || rotations.len() - rotation_base != block.message_cols() * D
                    {
                        return Err(MetalError::Shape("Pi_CCS seeded rotation stream is inconsistent"));
                    }
                    let work_base = work_headers.len() / SEEDED_WORK_HEADER_WORDS;
                    for start in (0..block.message_cols()).step_by(SEEDED_CHUNK_COLUMNS) {
                        work_headers.extend_from_slice(&[
                            output_index as u64,
                            start as u64,
                            block.message_cols().min(start + SEEDED_CHUNK_COLUMNS) as u64,
                        ]);
                    }
                    let work_count = work_headers.len() / SEEDED_WORK_HEADER_WORDS - work_base;
                    output_headers.extend_from_slice(&[
                        matrix as u64,
                        row_start as u64,
                        block.message_cols() as u64,
                        block.word_width() as u64,
                        block.word_starts().len() as u64,
                        word_base as u64,
                        rotation_base as u64,
                        work_base as u64,
                        work_count as u64,
                    ]);
                    output_meta.push(SeededOutputMeta { row_start });
                }
            }
        }

        let mut eval_groups = BTreeMap::<usize, Vec<u32>>::new();
        for (output, meta) in output_meta.iter().enumerate() {
            let output =
                u32::try_from(output).map_err(|_| MetalError::Shape("Pi_CCS seeded output count exceeds u32"))?;
            eval_groups.entry(meta.row_start).or_default().push(output);
        }
        let mut eval_group_headers = Vec::<u64>::new();
        let mut eval_group_outputs = Vec::<u32>::new();
        for (row_start, outputs) in eval_groups {
            let output_base = eval_group_outputs.len();
            eval_group_outputs.extend_from_slice(&outputs);
            eval_group_headers.extend_from_slice(&[row_start as u64, output_base as u64, outputs.len() as u64]);
        }
        let work_count = work_headers.len() / SEEDED_WORK_HEADER_WORDS;
        let eval_group_count = eval_group_headers.len() / 3;
        if output_headers.len() != output_meta.len() * SEEDED_OUTPUT_HEADER_WORDS
            || work_count == 0
            || eval_group_count == 0
        {
            return Err(MetalError::Shape("Pi_CCS seeded device plan is incomplete"));
        }
        Ok(Some(DeviceSeededRows {
            output_headers: self.buffer_from_slice(&output_headers)?,
            work_headers: self.buffer_from_slice(&work_headers)?,
            word_starts: self.buffer_from_slice(&word_starts)?,
            rotations: self.buffer_from_slice(&rotations)?,
            eval_group_headers: self.buffer_from_slice(&eval_group_headers)?,
            eval_group_outputs: self.buffer_from_slice(&eval_group_outputs)?,
            output_count: output_meta.len(),
            work_count,
            eval_group_count,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_mcs_row_tables(
        &self,
        plan: &MetalFeOraclePlan,
        cache: &SuperneoEvalCache,
        mcs_idx: usize,
        matrix_indices: &[usize],
        z_blocks: &SuperneoZBlocks,
        witness_masks: Option<&MetalWitnessMasks>,
        n_eff: usize,
        n_pad: usize,
    ) -> Result<MetalDeferredMcsRowTables, MetalError> {
        if !plan.matches(cache) {
            return Err(MetalError::Shape("Pi_CCS MCS row-table plan is stale"));
        }
        if !z_blocks.imag_all_zero() {
            return Err(MetalError::Shape("Pi_CCS MCS row-table witness is not base-field"));
        }
        if matrix_indices.is_empty() {
            return Err(MetalError::Shape("Pi_CCS MCS row-table matrix set is empty"));
        }
        if matrix_indices
            .iter()
            .any(|&index| index >= plan.matrix_count)
        {
            return Err(MetalError::Shape("Pi_CCS MCS row-table matrix index is out of range"));
        }
        if n_eff == 0 || n_eff > plan.rows {
            return Err(MetalError::Shape("Pi_CCS MCS row-table active row count is invalid"));
        }
        if n_pad < n_eff || !n_pad.is_power_of_two() {
            return Err(MetalError::Shape("Pi_CCS MCS row-table padded row count is invalid"));
        }
        let resident_masks = witness_masks.filter(|masks| masks.contains(mcs_idx, plan.blocks));
        let (z, z_kind, z_index) = if let Some((positive, negative)) = z_blocks.signed_unit_masks() {
            if positive.len() != plan.blocks || negative.len() != plan.blocks {
                return Err(MetalError::Shape("Pi_CCS signed witness plane has the wrong width"));
            }
            match resident_masks {
                Some(masks) => (masks.words().clone(), 2u64, mcs_idx as u64),
                None => {
                    let mut words = Vec::with_capacity(2 * plan.blocks);
                    words.extend_from_slice(positive);
                    words.extend_from_slice(negative);
                    (self.buffer_from_slice(&words)?, 1u64, 0u64)
                }
            }
        } else {
            let words = z_blocks.re_plane_words();
            if words.len() != plan.blocks * D {
                return Err(MetalError::Shape("Pi_CCS witness plane has the wrong width"));
            }
            (self.buffer_from_slice(&words)?, 0u64, 0u64)
        };
        let matrix_words = matrix_indices
            .iter()
            .map(|&index| index as u32)
            .collect::<Vec<_>>();
        let shape = self.buffer_from_slice(&[
            plan.matrix_count as u64,
            plan.rows as u64,
            plan.blocks as u64,
            n_eff as u64,
            n_pad as u64,
            matrix_indices.len() as u64,
            z_kind,
            z_index,
        ])?;
        let selected = self.buffer_from_slice(&matrix_words)?;
        let output_words = matrix_indices
            .len()
            .checked_mul(n_pad)
            .ok_or(MetalError::Shape("Pi_CCS MCS row table size overflow"))?;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.pi_ccs.mcs_rows.explicit")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.fe_build_mcs_row_tables);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_row_offsets), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_entry_bases), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_identity), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_columns), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&selected), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&z), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 8);
        }
        self.dispatch(&encoder, &self.fe_build_mcs_row_tables, output_words);
        encoder.endEncoding();
        self.submit(&command);

        let seeded_started = Instant::now();
        let mut patch_indices = Vec::new();
        let mut patch_values = Vec::new();
        for (table, &matrix_index) in matrix_indices.iter().enumerate() {
            for (row, value) in
                cache.matrix_caches()[matrix_index].seeded_row_dot_patches_base_with_blocks(z_blocks, n_eff)
            {
                patch_indices.push((table * n_pad + row) as u64);
                patch_values.push(value.as_canonical_u64());
            }
        }
        let seeded_build = seeded_started.elapsed();
        let seeded_patch_entries = patch_indices.len();
        let seeded_patch_bytes = patch_indices
            .len()
            .checked_mul(2 * size_of::<u64>())
            .ok_or(MetalError::Shape("Pi_CCS seeded patch size overflow"))?;
        if !patch_indices.is_empty() {
            let indices = self.buffer_from_slice(&patch_indices)?;
            let values = self.buffer_from_slice(&patch_values)?;
            let command = self.command_buffer("nightstream.pi_ccs.mcs_rows.seeded")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.fe_add_sparse_base_rows);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&indices), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&values), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
            }
            self.dispatch(&encoder, &self.fe_add_sparse_base_rows, patch_indices.len());
            encoder.endEncoding();
            self.submit(&command);
        }

        Ok(MetalDeferredMcsRowTables {
            words: output,
            mcs_idx,
            n_pad,
            table_count: matrix_indices.len(),
            seeded_build,
            seeded_patch_entries,
            seeded_patch_bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_carried_eval_table(
        &self,
        plan: &MetalFeOraclePlan,
        resident_id: u64,
        carried_coeffs: &[K],
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Result<MetalDeferredEvalTable, MetalError> {
        if !plan.supports_resident_eval()
            || carried_coeffs.is_empty()
            || mat_coeffs.len() != plan.matrix_count
            || n_eff == 0
            || n_eff > plan.rows
            || n_pad < n_eff
            || !n_pad.is_power_of_two()
        {
            return Err(MetalError::Shape("Pi_CCS carried Eval dimensions are invalid"));
        }
        {
            let resident = self.resident_running.borrow();
            let Some((stored_id, children)) = resident.as_ref() else {
                return Err(MetalError::Shape("Pi_CCS resident witnesses are unavailable"));
            };
            if *stored_id != resident_id || children.child_count != carried_coeffs.len() || children.cols != plan.blocks
            {
                return Err(MetalError::Shape(
                    "Pi_CCS resident witnesses do not match the oracle plan",
                ));
            }
        }

        let coeff_words = carried_coeffs
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let mat_coeff_words = mat_coeffs
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let (basis_re, basis_im) = weighted_projection_basis_forms(weights);
        let basis_re_words = basis_re
            .iter()
            .flat_map(|form| form.0.iter().map(PrimeField64::as_canonical_u64))
            .collect::<Vec<_>>();
        let basis_im_words = basis_im
            .iter()
            .flat_map(|form| form.0.iter().map(PrimeField64::as_canonical_u64))
            .collect::<Vec<_>>();

        let plane_len = plan
            .blocks
            .checked_mul(D)
            .ok_or(MetalError::Shape("Pi_CCS carried plane dimensions overflow"))?;
        let coeffs = self.buffer_from_slice(&coeff_words)?;
        let mat_coeffs = self.buffer_from_slice(&mat_coeff_words)?;
        let basis_re = self.buffer_from_slice(&basis_re_words)?;
        let basis_im = self.buffer_from_slice(&basis_im_words)?;
        let shape = self.buffer_from_slice(&[
            carried_coeffs.len() as u64,
            plan.blocks as u64,
            plan.matrix_count as u64,
            plan.rows as u64,
            n_eff as u64,
            n_pad as u64,
        ])?;
        let z_re = self.buffer(plane_len * size_of::<u64>())?;
        let z_im = self.buffer(plane_len * size_of::<u64>())?;
        let qk = self.buffer(2 * plane_len * size_of::<u64>())?;
        let output = self.buffer(2 * n_pad * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.pi_ccs.carried_eval")?;

        {
            let resident = self.resident_running.borrow();
            let (_, children) = resident
                .as_ref()
                .expect("resident Pi_CCS witnesses validated before command encoding");
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.carried_eval.combine")));
            encoder.setComputePipelineState(&self.fe_carried_plane_lin_comb);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&children.words), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&coeffs), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&z_re), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&z_im), 0, 4);
            }
            self.dispatch(&encoder, &self.fe_carried_plane_lin_comb, plane_len);
            encoder.endEncoding();
        }

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.carried_eval.basis")));
        encoder.setComputePipelineState(&self.fe_weighted_basis_dots);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&basis_re), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&basis_im), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&z_re), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&z_im), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&qk), 0, 5);
        }
        self.dispatch(&encoder, &self.fe_weighted_basis_dots, plane_len);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.carried_eval.rows")));
        encoder.setComputePipelineState(&self.fe_weighted_row_table);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_row_offsets), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_entry_bases), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_identity), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_columns), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&qk), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&mat_coeffs), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 8);
        }
        self.dispatch(&encoder, &self.fe_weighted_row_table, n_pad);
        encoder.endEncoding();

        let seeded_partials = if let Some(seeded) = &plan.seeded {
            let partial_values = seeded
                .work_count
                .checked_mul(RING_PRODUCT_COEFFICIENTS)
                .ok_or(MetalError::Shape("Pi_CCS seeded Eval partial size overflow"))?;
            let partials = self.buffer(2 * partial_values * size_of::<u64>())?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str(
                "nightstream.pi_ccs.carried_eval.seeded_partials",
            )));
            encoder.setComputePipelineState(&self.fe_seeded_k_partials);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seeded.output_headers), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&seeded.work_headers), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&seeded.word_starts), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&seeded.rotations), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&qk), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&mat_coeffs), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 7);
            }
            self.dispatch(&encoder, &self.fe_seeded_k_partials, partial_values);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str(
                "nightstream.pi_ccs.carried_eval.seeded_reduce",
            )));
            encoder.setComputePipelineState(&self.fe_seeded_k_reduce);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seeded.output_headers), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&seeded.eval_group_headers), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&seeded.eval_group_outputs), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&output), 0, 5);
            }
            self.dispatch(&encoder, &self.fe_seeded_k_reduce, seeded.eval_group_count * D);
            encoder.endEncoding();
            Some(partials)
        } else {
            None
        };

        self.submit(&command);
        drop(seeded_partials);

        Ok(MetalDeferredEvalTable { words: output, n_pad })
    }
}

fn row_list_histogram_bin(entries: usize) -> usize {
    match entries {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        32..=63 => 6,
        _ => 7,
    }
}
