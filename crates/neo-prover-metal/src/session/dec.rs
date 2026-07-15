//! Metal ownership of the base-2 Pi_DEC split and child opening evaluation.

use std::mem::size_of;
use std::time::{Duration, Instant};

use neo_math::{KExtensions, F, K};
use neo_reductions::superneo_eval::SuperneoEvalCache;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::PrimeField64;
use rayon::prelude::*;

use super::carrier::{MetalResidentChildren, MetalResidentWitness};
use super::dec_seeded::DeviceSeededFormPlan;
use super::{
    command_gpu_duration, Buffer, MetalAjtaiLowNormPlan, MetalDecPublicProjection, MetalFeOraclePlan, MetalSession,
    MetalWitnessMasks,
};
use crate::{MetalAjtaiYProfile, MetalError};

const RING_DEGREE: usize = 54;
const PRODUCT_COEFFICIENTS: usize = 2 * RING_DEGREE - 1;
const CHUNK_COLUMNS: usize = 512;
const PARALLEL_FORM_LIST_THRESHOLD: usize = 128;
const FORM_REDUCTION_THREADS: usize = 256;

pub(crate) struct MetalDecMaterial {
    pub child_nonzero: Vec<bool>,
    pub y_words: Vec<u64>,
    pub y_zcol_words: Vec<u64>,
    pub y_zcol_gpu: Duration,
    pub commitment_words: Vec<u64>,
    pub resident_children: MetalResidentChildren,
}

pub(crate) struct MetalDecFormPlan {
    active_local_offsets: Buffer,
    active_entry_bases: Buffer,
    active_blocks: Buffer,
    active_chunk_bases: Buffer,
    active_chunk_matrices: Buffer,
    matrix_active_offsets: Buffer,
    matrix_chunk_offsets: Buffer,
    matrix_identity: Buffer,
    entry_rows: Buffer,
    entry_coefficients: Buffer,
    parallel_form_lists: Buffer,
    seeded_forms: Option<DeviceSeededFormPlan>,
    active_blocks_host: Vec<u32>,
    matrix_active_offsets_host: Vec<u32>,
    active_block_count: usize,
    active_chunk_count: usize,
    explicit_coefficient_count: usize,
    signed_unit_coefficient_count: usize,
    explicit_form_list_histogram: [usize; 8],
    max_explicit_form_list_entries: usize,
    parallel_form_list_count: usize,
    parallel_form_entry_count: usize,
    matrix_count: usize,
    blocks: usize,
    cache_identity: usize,
    matrix_digest: [u64; 4],
}

pub(crate) struct MetalAjtaiRingForms {
    words: Buffer,
    form_rows: usize,
    blocks: usize,
    cache_identity: usize,
    matrix_digest: [u64; 4],
    chi_words: Vec<u64>,
    row_challenge_words: Vec<u64>,
    n_eff: usize,
}

impl MetalDecFormPlan {
    pub(crate) fn matches(&self, cache: &SuperneoEvalCache) -> bool {
        self.cache_identity == cache as *const SuperneoEvalCache as usize
            && self.matrix_digest == cache.mat_digest().map(|value| value.as_canonical_u64())
            && self.matrix_count == cache.matrix_caches().len()
    }
}

impl MetalAjtaiRingForms {
    fn matches(&self, plan: &MetalDecFormPlan, chi_words: &[u64], n_eff: usize) -> bool {
        self.blocks == plan.blocks
            && self.cache_identity == plan.cache_identity
            && self.matrix_digest == plan.matrix_digest
            && self.form_rows == 2 * plan.matrix_count
            && self.n_eff == n_eff
            && self.chi_words == chi_words
    }

    fn matches_row_challenges(&self, plan: &MetalDecFormPlan, row_challenges: &[K], n_eff: usize) -> bool {
        self.blocks == plan.blocks
            && self.cache_identity == plan.cache_identity
            && self.matrix_digest == plan.matrix_digest
            && self.form_rows == 2 * plan.matrix_count
            && self.n_eff == n_eff
            && self.row_challenge_words.len() == 2 * row_challenges.len()
            && self
                .row_challenge_words
                .chunks_exact(2)
                .zip(row_challenges)
                .all(|(words, challenge)| {
                    let (real, imaginary) = challenge.to_limbs_u64();
                    words == [real, imaginary]
                })
    }
}

struct DeviceSeededFormPatch {
    bases: Buffer,
    coefficients: Buffer,
    entries: usize,
}

struct DeviceFormBuild<'a> {
    plan: &'a MetalDecFormPlan,
    chi: Buffer,
    shape: Buffer,
    seeded: Option<DeviceSeededFormPatch>,
}

impl MetalSession {
    pub(crate) fn prepare_dec_ring_forms(
        &self,
        cache: &SuperneoEvalCache,
        oracle: &MetalFeOraclePlan,
    ) -> Result<MetalDecFormPlan, MetalError> {
        let matrices = cache.matrix_caches();
        let Some(first) = matrices.first() else {
            return Err(MetalError::Shape("Pi_DEC form plan requires CCS matrices"));
        };
        let (scalar_rows, scalar_columns, _) = first.compact_explicit_shape();
        let blocks = scalar_columns.div_ceil(RING_DEGREE);
        if blocks == 0 {
            return Err(MetalError::Shape("Pi_DEC form plan has zero columns"));
        }
        let local_slots = checked_product(&[blocks, RING_DEGREE], "Pi_DEC compact form slot count overflow")?;
        let matrix_stride = local_slots
            .checked_add(1)
            .ok_or(MetalError::Shape("Pi_DEC compact form slot count overflow"))?;
        let total_coefficients = matrices
            .iter()
            .map(|matrix| matrix.compact_explicit_coefficient_count())
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or(MetalError::Shape("Pi_DEC compact coefficient count overflow"))?;

        struct MatrixLayout {
            offsets: Vec<u32>,
            active: Vec<bool>,
            identity: bool,
            entry_count: usize,
        }

        let layouts = matrices
            .par_iter()
            .map(|matrix| {
                let (rows, columns, identity) = matrix.compact_explicit_shape();
                if rows != scalar_rows || columns != scalar_columns || rows == 0 || u32::try_from(rows - 1).is_err() {
                    return Err(MetalError::Shape("Pi_DEC matrices have inconsistent shapes"));
                }
                let mut active = vec![false; blocks];
                for block in matrix.compact_seeded_column_blocks() {
                    if block >= blocks {
                        return Err(MetalError::Shape("Pi_DEC seeded form block is out of range"));
                    }
                    active[block] = true;
                }
                if identity {
                    let identity_blocks = rows.div_ceil(RING_DEGREE);
                    if identity_blocks > blocks {
                        return Err(MetalError::Shape("Pi_DEC identity matrix exceeds its columns"));
                    }
                    active[..identity_blocks].fill(true);
                    return Ok(MatrixLayout {
                        offsets: Vec::new(),
                        active,
                        identity,
                        entry_count: 0,
                    });
                }

                let entry_count = matrix.compact_explicit_coefficient_count();
                if u32::try_from(entry_count).is_err() {
                    return Err(MetalError::Shape("Pi_DEC compact matrix coefficient count exceeds u32"));
                }
                let mut offsets = vec![0u32; matrix_stride];
                let mut invalid = false;
                for row in 0..rows {
                    matrix.for_each_compact_explicit_row_coefficient(row, |block, local, _| {
                        let block = block as usize;
                        let local = local as usize;
                        if block >= blocks || local >= RING_DEGREE {
                            invalid = true;
                            return;
                        }
                        active[block] = true;
                        let slot = block * RING_DEGREE + local + 1;
                        let Some(next) = offsets[slot].checked_add(1) else {
                            invalid = true;
                            return;
                        };
                        offsets[slot] = next;
                    });
                }
                if invalid {
                    return Err(MetalError::Shape(
                        "Pi_DEC compact matrix coefficient dimensions are invalid",
                    ));
                }
                for slot in 0..local_slots {
                    offsets[slot + 1] = offsets[slot + 1]
                        .checked_add(offsets[slot])
                        .ok_or(MetalError::Shape("Pi_DEC compact matrix offset overflow"))?;
                }
                if offsets[local_slots] as usize != entry_count {
                    return Err(MetalError::Shape("Pi_DEC compact matrix coefficient count changed"));
                }
                Ok(MatrixLayout {
                    offsets,
                    active,
                    identity,
                    entry_count,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut matrix_entry_bases = Vec::with_capacity(layouts.len());
        let mut entry_base = 0usize;
        for layout in &layouts {
            matrix_entry_bases.push(entry_base);
            entry_base = entry_base
                .checked_add(layout.entry_count)
                .ok_or(MetalError::Shape("Pi_DEC compact coefficient count overflow"))?;
        }
        if entry_base != total_coefficients {
            return Err(MetalError::Shape("Pi_DEC compact form plan is incomplete"));
        }

        let mut active_local_offsets = Vec::new();
        let mut active_entry_bases = Vec::new();
        let mut active_blocks_host = Vec::new();
        let mut matrix_active_offsets_host = vec![0u32];
        let mut matrix_identity = Vec::with_capacity(matrices.len());
        let mut explicit_form_list_histogram = [0usize; 8];
        let mut max_explicit_form_list_entries = 0usize;
        let mut parallel_form_lists = Vec::new();
        let mut parallel_form_entry_count = 0usize;
        for (matrix_index, layout) in layouts.iter().enumerate() {
            matrix_identity.push(u32::from(layout.identity));
            if layout.identity {
                for (block, _) in layout
                    .active
                    .iter()
                    .enumerate()
                    .filter(|(_, active)| **active)
                {
                    active_blocks_host.push(
                        u32::try_from(matrix_index * blocks + block)
                            .map_err(|_| MetalError::Shape("Pi_DEC active block index exceeds u32"))?,
                    );
                    active_entry_bases.push(0);
                    active_local_offsets.resize(active_local_offsets.len() + RING_DEGREE + 1, 0);
                }
                matrix_active_offsets_host.push(
                    u32::try_from(active_blocks_host.len())
                        .map_err(|_| MetalError::Shape("Pi_DEC active block count exceeds u32"))?,
                );
                continue;
            }
            let offsets = &layout.offsets;
            let entry_base = matrix_entry_bases[matrix_index];
            for (block, _) in layout
                .active
                .iter()
                .enumerate()
                .filter(|(_, active)| **active)
            {
                let active_index = active_blocks_host.len();
                active_blocks_host.push(
                    u32::try_from(matrix_index * blocks + block)
                        .map_err(|_| MetalError::Shape("Pi_DEC active block index exceeds u32"))?,
                );
                let block_start = offsets[block * RING_DEGREE];
                active_entry_bases.push(
                    u64::try_from(entry_base + block_start as usize)
                        .map_err(|_| MetalError::Shape("Pi_DEC compact entry base exceeds u64"))?,
                );
                for local in 0..RING_DEGREE {
                    let entries =
                        (offsets[block * RING_DEGREE + local + 1] - offsets[block * RING_DEGREE + local]) as usize;
                    max_explicit_form_list_entries = max_explicit_form_list_entries.max(entries);
                    let bucket = match entries {
                        0 => 0,
                        1 => 1,
                        2..=3 => 2,
                        4..=7 => 3,
                        8..=15 => 4,
                        16..=31 => 5,
                        32..=63 => 6,
                        _ => 7,
                    };
                    explicit_form_list_histogram[bucket] += 1;
                    if entries >= PARALLEL_FORM_LIST_THRESHOLD {
                        parallel_form_lists.push(
                            u32::try_from(active_index * RING_DEGREE + local)
                                .map_err(|_| MetalError::Shape("Pi_DEC parallel form list index exceeds u32"))?,
                        );
                        parallel_form_entry_count = parallel_form_entry_count
                            .checked_add(entries)
                            .ok_or(MetalError::Shape("Pi_DEC parallel form entry count overflow"))?;
                    }
                }
                active_local_offsets.extend(
                    offsets[block * RING_DEGREE..=block * RING_DEGREE + RING_DEGREE]
                        .iter()
                        .map(|&offset| offset - block_start),
                );
            }
            matrix_active_offsets_host.push(
                u32::try_from(active_blocks_host.len())
                    .map_err(|_| MetalError::Shape("Pi_DEC active block count exceeds u32"))?,
            );
        }

        let entry_rows = self.buffer(total_coefficients.max(1) * size_of::<u32>())?;
        let entry_coefficients = self.buffer(total_coefficients.max(1) * size_of::<u64>())?;
        let rows = unsafe {
            std::slice::from_raw_parts_mut(entry_rows.contents().as_ptr().cast::<u32>(), total_coefficients.max(1))
        };
        let coefficients = unsafe {
            std::slice::from_raw_parts_mut(
                entry_coefficients.contents().as_ptr().cast::<u64>(),
                total_coefficients.max(1),
            )
        };
        let mut row_tail = &mut rows[..total_coefficients];
        let mut coefficient_tail = &mut coefficients[..total_coefficients];
        let mut fill_parts = Vec::with_capacity(matrices.len());
        for (matrix, layout) in matrices.iter().zip(layouts) {
            let (matrix_rows, remaining_rows) = row_tail.split_at_mut(layout.entry_count);
            let (matrix_coefficients, remaining_coefficients) = coefficient_tail.split_at_mut(layout.entry_count);
            fill_parts.push((matrix, layout, matrix_rows, matrix_coefficients));
            row_tail = remaining_rows;
            coefficient_tail = remaining_coefficients;
        }
        let signed_unit_coefficient_count = fill_parts
            .into_par_iter()
            .map(|(matrix, mut layout, entry_rows, entry_coefficients)| {
                if layout.identity {
                    return Ok(0usize);
                }
                let mut invalid = false;
                let mut filled = 0usize;
                let mut signed = 0usize;
                for row in 0..scalar_rows {
                    matrix.for_each_compact_explicit_row_coefficient(row, |block, local, coefficient| {
                        let slot = block as usize * RING_DEGREE + local as usize;
                        if slot >= local_slots {
                            invalid = true;
                            return;
                        }
                        let destination = layout.offsets[slot] as usize;
                        if destination >= entry_rows.len() {
                            invalid = true;
                            return;
                        }
                        layout.offsets[slot] += 1;
                        entry_rows[destination] = row as u32;
                        let coefficient = coefficient.as_canonical_u64();
                        entry_coefficients[destination] = coefficient;
                        signed += usize::from(coefficient == 1 || coefficient == F::ORDER_U64 - 1);
                        filled += 1;
                    });
                }
                if invalid || filled != entry_rows.len() {
                    return Err(MetalError::Shape("Pi_DEC compact matrix fill is inconsistent"));
                }
                Ok(signed)
            })
            .sum::<Result<usize, _>>()?;
        self.record_host_write(total_coefficients * (size_of::<u32>() + size_of::<u64>()));
        let parallel_form_list_count = parallel_form_lists.len();
        if parallel_form_lists.is_empty() {
            parallel_form_lists.push(0);
        }
        let active_block_count = active_blocks_host.len();
        if active_block_count == 0 {
            return Err(MetalError::Shape("Pi_DEC form plan has no active blocks"));
        }
        let mut active_chunk_bases = Vec::new();
        let mut active_chunk_matrices = Vec::new();
        let mut matrix_chunk_offsets = vec![0u32];
        for matrix in 0..matrices.len() {
            let start = matrix_active_offsets_host[matrix] as usize;
            let end = matrix_active_offsets_host[matrix + 1] as usize;
            for base in (start..end).step_by(CHUNK_COLUMNS) {
                active_chunk_bases
                    .push(u32::try_from(base).map_err(|_| MetalError::Shape("Pi_DEC active chunk base exceeds u32"))?);
                active_chunk_matrices
                    .push(u32::try_from(matrix).map_err(|_| MetalError::Shape("Pi_DEC matrix index exceeds u32"))?);
            }
            matrix_chunk_offsets.push(
                u32::try_from(active_chunk_bases.len())
                    .map_err(|_| MetalError::Shape("Pi_DEC active chunk count exceeds u32"))?,
            );
        }
        let active_chunk_count = active_chunk_bases.len();
        if active_chunk_count == 0 {
            return Err(MetalError::Shape("Pi_DEC form plan has no active chunks"));
        }

        let seeded_forms = self.prepare_seeded_dec_forms(cache, oracle, &active_blocks_host, blocks)?;
        let active_blocks = self.buffer_from_slice(&active_blocks_host)?;
        let matrix_active_offsets = self.buffer_from_slice(&matrix_active_offsets_host)?;
        Ok(MetalDecFormPlan {
            active_local_offsets: self.buffer_from_slice(&active_local_offsets)?,
            active_entry_bases: self.buffer_from_slice(&active_entry_bases)?,
            active_blocks,
            active_chunk_bases: self.buffer_from_slice(&active_chunk_bases)?,
            active_chunk_matrices: self.buffer_from_slice(&active_chunk_matrices)?,
            matrix_active_offsets,
            matrix_chunk_offsets: self.buffer_from_slice(&matrix_chunk_offsets)?,
            matrix_identity: self.buffer_from_slice(&matrix_identity)?,
            entry_rows,
            entry_coefficients,
            parallel_form_lists: self.buffer_from_slice(&parallel_form_lists)?,
            seeded_forms,
            active_blocks_host,
            matrix_active_offsets_host,
            active_block_count,
            active_chunk_count,
            explicit_coefficient_count: total_coefficients,
            signed_unit_coefficient_count,
            explicit_form_list_histogram,
            max_explicit_form_list_entries,
            parallel_form_list_count,
            parallel_form_entry_count,
            matrix_count: matrices.len(),
            blocks,
            cache_identity: cache as *const SuperneoEvalCache as usize,
            matrix_digest: cache.mat_digest().map(|value| value.as_canonical_u64()),
        })
    }

    fn prepare_seeded_form_patch(
        &self,
        plan: &MetalDecFormPlan,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
    ) -> Result<Option<DeviceSeededFormPatch>, MetalError> {
        if plan.seeded_forms.is_some() {
            return Ok(None);
        }
        let forms = cache.build_seeded_ring_linear_forms(chi_r, n_eff);
        self.prepare_seeded_form_patch_from_forms(plan, &forms)
    }

    fn prepare_seeded_form_patch_from_row_challenges(
        &self,
        plan: &MetalDecFormPlan,
        cache: &SuperneoEvalCache,
        row_challenges: &[K],
        n_eff: usize,
    ) -> Result<Option<DeviceSeededFormPatch>, MetalError> {
        if plan.seeded_forms.is_some() {
            return Ok(None);
        }
        let forms = cache.build_seeded_ring_linear_forms_from_row_challenges(row_challenges, n_eff);
        self.prepare_seeded_form_patch_from_forms(plan, &forms)
    }

    fn prepare_seeded_form_patch_from_forms(
        &self,
        plan: &MetalDecFormPlan,
        forms: &[neo_reductions::superneo_eval::SuperneoRingLinearForm],
    ) -> Result<Option<DeviceSeededFormPatch>, MetalError> {
        let mut bases = Vec::new();
        let mut coefficients = Vec::new();
        for (matrix, form) in forms.iter().enumerate() {
            for (block, real, imaginary) in form.to_sparse_block_coeffs() {
                if matrix >= plan.matrix_count || block >= plan.blocks {
                    return Err(MetalError::Shape("Pi_DEC seeded form block is out of range"));
                }
                let start = plan.matrix_active_offsets_host[matrix] as usize;
                let end = plan.matrix_active_offsets_host[matrix + 1] as usize;
                let encoded = u32::try_from(matrix * plan.blocks + block)
                    .map_err(|_| MetalError::Shape("Pi_DEC seeded form block index exceeds u32"))?;
                let relative = plan.active_blocks_host[start..end]
                    .binary_search(&encoded)
                    .map_err(|_| MetalError::Shape("Pi_DEC seeded form block is absent from the compact plan"))?;
                let base = checked_product(&[start + relative, 2, RING_DEGREE], "Pi_DEC seeded form base overflow")?;
                bases.push(base as u64);
                coefficients.extend(
                    real.iter()
                        .chain(&imaginary)
                        .map(PrimeField64::as_canonical_u64),
                );
            }
        }
        if bases.is_empty() {
            return Ok(None);
        }
        Ok(Some(DeviceSeededFormPatch {
            bases: self.buffer_from_slice(&bases)?,
            coefficients: self.buffer_from_slice(&coefficients)?,
            entries: bases.len(),
        }))
    }

    pub(crate) fn eval_ajtai_y_from_signed_masks(
        &self,
        plan: &MetalDecFormPlan,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        mask_words: &[u64],
        resident_masks: Option<&MetalWitnessMasks>,
        witness_count: usize,
    ) -> Result<(Vec<u64>, MetalAjtaiRingForms, MetalAjtaiYProfile), MetalError> {
        if witness_count == 0 || chi_r.is_empty() || n_eff > chi_r.len() || !plan.matches(cache) {
            return Err(MetalError::Shape("Pi_CCS Y_eval dimensions are invalid"));
        }
        let chi_words = chi_r
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let seeded_started = Instant::now();
        let seeded = self.prepare_seeded_form_patch(plan, cache, chi_r, n_eff)?;
        let seeded_build = seeded_started.elapsed();
        let seeded_patch_entries = seeded.as_ref().map_or(0, |patch| patch.entries);
        let seeded_patch_bytes =
            seeded_patch_entries.saturating_mul(size_of::<u64>() + 2 * RING_DEGREE * size_of::<u64>());
        let expected_masks = checked_product(
            &[witness_count, plan.blocks, 2],
            "Pi_CCS Y_eval mask dimensions overflow",
        )?;
        if resident_masks.is_some_and(|masks| !masks.matches(witness_count, plan.blocks))
            || resident_masks.is_none() && mask_words.len() != expected_masks
        {
            return Err(MetalError::Shape("Pi_CCS Y_eval masks do not match the witness shape"));
        }

        let form_rows = 2 * plan.matrix_count;
        let form_words = checked_product(
            &[plan.active_block_count, 2, RING_DEGREE],
            "Pi_CCS Y_eval form dimensions overflow",
        )?;
        let groups = checked_product(&[witness_count, form_rows], "Pi_CCS Y_eval output dimensions overflow")?;
        let partial_words = checked_product(
            &[witness_count, plan.active_chunk_count, 2, PRODUCT_COEFFICIENTS],
            "Pi_CCS Y_eval partial dimensions overflow",
        )?;
        let sum_words = checked_product(&[groups, PRODUCT_COEFFICIENTS], "Pi_CCS Y_eval sum dimensions overflow")?;
        let output_words = checked_product(&[groups, RING_DEGREE], "Pi_CCS Y_eval output dimensions overflow")?;

        let form_bytes = form_words * size_of::<u64>();
        let forms = self.take_recycled_buffer(&self.recycled_ajtai_forms, form_bytes)?;
        let chi = self.buffer_from_slice(&chi_words)?;
        let form_shape = self.buffer_from_slice(&[
            plan.matrix_count as u64,
            plan.blocks as u64,
            n_eff as u64,
            chi_r.len() as u64,
        ])?;
        let masks = match resident_masks {
            Some(masks) => masks.words().clone(),
            None => self.buffer_from_slice(mask_words)?,
        };
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(sum_words * size_of::<u64>())?;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let witnesses = (0..witness_count)
            .map(|witness| {
                u32::try_from(witness).map_err(|_| MetalError::Shape("Pi_CCS Y_eval witness count exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let witnesses = self.buffer_from_slice(&witnesses)?;
        let shape = self.buffer_from_slice(&[
            plan.active_block_count as u64,
            witness_count as u64,
            form_rows as u64,
            plan.blocks as u64,
            plan.active_chunk_count as u64,
        ])?;

        let device_started = Instant::now();
        let command = self.command_buffer("nightstream.pi_ccs.ajtai_y_eval")?;
        let seeded_scratch = self.encode_dec_form_build(&command, plan, &chi, &form_shape, &forms, seeded.as_ref())?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.y_eval.partials")));
        encoder.setComputePipelineState(&self.dec_sparse_ring_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&forms), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&witnesses), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_bases), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_matrices), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_active_offsets), 0, 8);
        }
        self.dispatch(&encoder, &self.dec_sparse_ring_partials, partial_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.y_eval.sum_chunks")));
        encoder.setComputePipelineState(&self.dec_sparse_ring_sum_chunks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_chunk_offsets), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_sparse_ring_sum_chunks, sum_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.y_eval.reduce_phi81")));
        encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_reduce_phi81, output_words);
        encoder.endEncoding();
        self.finish(&command)?;
        if let Some(scratch) = seeded_scratch {
            Self::recycle_largest_buffer(&self.recycled_seeded_forms, scratch);
        }

        Ok((
            self.read_buffer::<u64>(&output, output_words),
            MetalAjtaiRingForms {
                words: forms,
                form_rows,
                blocks: plan.blocks,
                cache_identity: plan.cache_identity,
                matrix_digest: plan.matrix_digest,
                chi_words,
                row_challenge_words: Vec::new(),
                n_eff,
            },
            MetalAjtaiYProfile {
                seeded_build,
                device_eval: device_started.elapsed(),
                tensor_gpu: Duration::ZERO,
                form_gpu: Duration::ZERO,
                tail_gpu: command_gpu_duration(&command),
                seeded_patch_entries,
                seeded_patch_bytes,
                form_blocks: plan.active_block_count,
                form_bytes,
                explicit_coefficients: plan.explicit_coefficient_count,
                signed_unit_coefficients: plan.signed_unit_coefficient_count,
                explicit_form_list_histogram: plan.explicit_form_list_histogram,
                max_explicit_form_list_entries: plan.max_explicit_form_list_entries,
                parallel_form_lists: plan.parallel_form_list_count,
                parallel_form_entries: plan.parallel_form_entry_count,
            },
        ))
    }

    pub(crate) fn eval_ajtai_y_from_signed_masks_and_row_challenges(
        &self,
        plan: &MetalDecFormPlan,
        cache: &SuperneoEvalCache,
        row_challenges: &[K],
        n_eff: usize,
        mask_words: &[u64],
        resident_masks: Option<&MetalWitnessMasks>,
        witness_count: usize,
    ) -> Result<(Vec<u64>, MetalAjtaiRingForms, MetalAjtaiYProfile), MetalError> {
        let challenge_count = u32::try_from(row_challenges.len())
            .map_err(|_| MetalError::Shape("Pi_CCS row challenge count exceeds u32"))?;
        let chi_len = 1usize
            .checked_shl(challenge_count)
            .ok_or(MetalError::Shape("Pi_CCS row challenge dimensions overflow"))?;
        if witness_count == 0
            || row_challenges.is_empty()
            || n_eff > chi_len
            || u32::try_from(chi_len).is_err()
            || !plan.matches(cache)
        {
            return Err(MetalError::Shape(
                "Pi_CCS challenge-backed Y_eval dimensions are invalid",
            ));
        }
        let expected_masks = checked_product(
            &[witness_count, plan.blocks, 2],
            "Pi_CCS Y_eval mask dimensions overflow",
        )?;
        if resident_masks.is_some_and(|masks| !masks.matches(witness_count, plan.blocks))
            || resident_masks.is_none() && mask_words.len() != expected_masks
        {
            return Err(MetalError::Shape("Pi_CCS Y_eval masks do not match the witness shape"));
        }

        let challenge_words = row_challenges
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let stages = (0..challenge_count).map(u64::from).collect::<Vec<_>>();
        let form_rows = 2 * plan.matrix_count;
        let form_words = checked_product(
            &[plan.active_block_count, 2, RING_DEGREE],
            "Pi_CCS Y_eval form dimensions overflow",
        )?;
        let groups = checked_product(&[witness_count, form_rows], "Pi_CCS Y_eval output dimensions overflow")?;
        let partial_words = checked_product(
            &[witness_count, plan.active_chunk_count, 2, PRODUCT_COEFFICIENTS],
            "Pi_CCS Y_eval partial dimensions overflow",
        )?;
        let sum_words = checked_product(&[groups, PRODUCT_COEFFICIENTS], "Pi_CCS Y_eval sum dimensions overflow")?;
        let output_words = checked_product(&[groups, RING_DEGREE], "Pi_CCS Y_eval output dimensions overflow")?;

        let form_bytes = form_words * size_of::<u64>();
        let forms = self.take_recycled_buffer(&self.recycled_ajtai_forms, form_bytes)?;
        let chi = self.buffer(checked_product(
            &[chi_len, 2, size_of::<u64>()],
            "Pi_CCS chi bytes overflow",
        )?)?;
        let challenges = self.buffer_from_slice(&challenge_words)?;
        let stages = self.buffer_from_slice(&stages)?;
        let form_shape = self.buffer_from_slice(&[
            plan.matrix_count as u64,
            plan.blocks as u64,
            n_eff as u64,
            chi_len as u64,
        ])?;
        let masks = match resident_masks {
            Some(masks) => masks.words().clone(),
            None => self.buffer_from_slice(mask_words)?,
        };
        let partials = self.buffer(partial_words * size_of::<u64>())?;
        let sums = self.buffer(sum_words * size_of::<u64>())?;
        let output = self.buffer(output_words * size_of::<u64>())?;
        let witnesses = (0..witness_count)
            .map(|witness| {
                u32::try_from(witness).map_err(|_| MetalError::Shape("Pi_CCS Y_eval witness count exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let witnesses = self.buffer_from_slice(&witnesses)?;
        let shape = self.buffer_from_slice(&[
            plan.active_block_count as u64,
            witness_count as u64,
            form_rows as u64,
            plan.blocks as u64,
            plan.active_chunk_count as u64,
        ])?;

        let device_started = Instant::now();
        let tensor_command = self.command_buffer("nightstream.pi_ccs.tensor_point_from_row_challenges")?;
        self.encode_tensor_point_k(&tensor_command, &challenges, &stages, &chi, challenge_count as usize)?;
        self.submit(&tensor_command);
        let form_command = self.command_buffer("nightstream.pi_ccs.forms_from_row_challenges")?;
        let seeded_scratch = self.encode_dec_form_build(&form_command, plan, &chi, &form_shape, &forms, None)?;
        self.submit(&form_command);

        let seeded_started = Instant::now();
        let seeded = self.prepare_seeded_form_patch_from_row_challenges(plan, cache, row_challenges, n_eff)?;
        let seeded_build = seeded_started.elapsed();
        let seeded_patch_entries = seeded.as_ref().map_or(0, |patch| patch.entries);
        let seeded_patch_bytes =
            seeded_patch_entries.saturating_mul(size_of::<u64>() + 2 * RING_DEGREE * size_of::<u64>());

        let command = self.command_buffer("nightstream.pi_ccs.ajtai_y_eval_from_row_challenges")?;
        self.encode_seeded_form_patch(&command, &forms, seeded.as_ref())?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.y_eval.partials")));
        encoder.setComputePipelineState(&self.dec_sparse_ring_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&forms), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&masks), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&witnesses), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_bases), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_chunk_matrices), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_active_offsets), 0, 8);
        }
        self.dispatch(&encoder, &self.dec_sparse_ring_partials, partial_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.y_eval.sum_chunks")));
        encoder.setComputePipelineState(&self.dec_sparse_ring_sum_chunks);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_chunk_offsets), 0, 3);
        }
        self.dispatch(&encoder, &self.dec_sparse_ring_sum_chunks, sum_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.y_eval.reduce_phi81")));
        encoder.setComputePipelineState(&self.dec_ring_reduce_phi81);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&sums), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_ring_reduce_phi81, output_words);
        encoder.endEncoding();
        self.finish(&command)?;
        if let Some(error) = tensor_command.error() {
            return Err(MetalError::Execution(format!("{error:?}")));
        }
        if let Some(error) = form_command.error() {
            return Err(MetalError::Execution(format!("{error:?}")));
        }
        if let Some(scratch) = seeded_scratch {
            Self::recycle_largest_buffer(&self.recycled_seeded_forms, scratch);
        }
        Ok((
            self.read_buffer::<u64>(&output, output_words),
            MetalAjtaiRingForms {
                words: forms,
                form_rows,
                blocks: plan.blocks,
                cache_identity: plan.cache_identity,
                matrix_digest: plan.matrix_digest,
                chi_words: Vec::new(),
                row_challenge_words: challenge_words,
                n_eff,
            },
            MetalAjtaiYProfile {
                seeded_build,
                device_eval: device_started.elapsed(),
                tensor_gpu: command_gpu_duration(&tensor_command),
                form_gpu: command_gpu_duration(&form_command),
                tail_gpu: command_gpu_duration(&command),
                seeded_patch_entries,
                seeded_patch_bytes,
                form_blocks: plan.active_block_count,
                form_bytes,
                explicit_coefficients: plan.explicit_coefficient_count,
                signed_unit_coefficients: plan.signed_unit_coefficient_count,
                explicit_form_list_histogram: plan.explicit_form_list_histogram,
                max_explicit_form_list_entries: plan.max_explicit_form_list_entries,
                parallel_form_lists: plan.parallel_form_list_count,
                parallel_form_entries: plan.parallel_form_entry_count,
            },
        ))
    }

    pub(super) fn encode_tensor_point_k(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        challenges: &Buffer,
        stages: &Buffer,
        table: &Buffer,
        challenge_count: usize,
    ) -> Result<(), MetalError> {
        for stage in 0..challenge_count {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.tensor_point.expand")));
            encoder.setComputePipelineState(&self.tensor_point_expand_k);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(challenges), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(stages), stage * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(table), 0, 2);
            }
            self.dispatch(&encoder, &self.tensor_point_expand_k, 1usize << stage);
            encoder.endEncoding();
        }
        Ok(())
    }

    fn encode_dec_form_build(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        plan: &MetalDecFormPlan,
        chi: &Buffer,
        shape: &Buffer,
        forms: &Buffer,
        seeded: Option<&DeviceSeededFormPatch>,
    ) -> Result<Option<Buffer>, MetalError> {
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.forms.compact_explicit")));
        encoder.setComputePipelineState(&self.dec_build_ring_forms);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&plan.active_local_offsets), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_entry_bases), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&plan.matrix_identity), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_rows), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(chi), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(shape), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(forms), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 8);
        }
        let form_work_items = checked_product(
            &[plan.active_block_count, 2, RING_DEGREE],
            "Pi_DEC compact form work dimensions overflow",
        )?;
        self.dispatch(&encoder, &self.dec_build_ring_forms, form_work_items);
        encoder.endEncoding();

        if plan.parallel_form_list_count != 0 {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.forms.parallel_explicit")));
            encoder.setComputePipelineState(&self.dec_build_parallel_original_forms);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&plan.active_local_offsets), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&plan.active_entry_bases), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&plan.entry_rows), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&plan.entry_coefficients), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(chi), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(shape), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(forms), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(&plan.parallel_form_lists), 0, 9);
            }
            self.dispatch_threadgroups(
                &encoder,
                &self.dec_build_parallel_original_forms,
                plan.parallel_form_list_count * 2,
                FORM_REDUCTION_THREADS,
            );
            encoder.endEncoding();
        }

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.forms.bar_in_place")));
        encoder.setComputePipelineState(&self.dec_bar_ring_forms_in_place);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(forms), 0, 0);
        }
        self.dispatch(
            &encoder,
            &self.dec_bar_ring_forms_in_place,
            plan.active_block_count * 2 * 14,
        );
        encoder.endEncoding();

        let Some(device_seeded) = &plan.seeded_forms else {
            self.encode_seeded_form_patch(command, forms, seeded)?;
            return Ok(None);
        };
        if seeded.is_some() {
            return Err(MetalError::Shape("Pi_DEC seeded forms have two owners"));
        }
        let seeded_words = checked_product(
            &[device_seeded.group_count, 2, RING_DEGREE],
            "Pi_DEC seeded form dimensions overflow",
        )?;
        let scratch = self.take_recycled_buffer(&self.recycled_seeded_forms, seeded_words * size_of::<u64>())?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.forms.seeded_resident")));
        encoder.setComputePipelineState(&self.dec_build_seeded_ring_forms);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.output_headers), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.word_starts), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.rotations), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.group_segment_offsets), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.segments), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(chi), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(shape), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&scratch), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&plan.active_blocks), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.active_indices), 0, 9);
        }
        self.dispatch(&encoder, &self.dec_build_seeded_ring_forms, seeded_words);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.forms.seeded_bar_add")));
        encoder.setComputePipelineState(&self.dec_add_bar_seeded_ring_forms);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&scratch), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(forms), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&device_seeded.active_indices), 0, 2);
        }
        self.dispatch(&encoder, &self.dec_add_bar_seeded_ring_forms, seeded_words);
        encoder.endEncoding();
        Ok(Some(scratch))
    }

    fn encode_seeded_form_patch(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        forms: &Buffer,
        seeded: Option<&DeviceSeededFormPatch>,
    ) -> Result<(), MetalError> {
        if let Some(seeded) = seeded {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setLabel(Some(&NSString::from_str("nightstream.forms.seeded_patch")));
            encoder.setComputePipelineState(&self.dec_add_sparse_ring_forms);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seeded.bases), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&seeded.coefficients), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(forms), 0, 2);
            }
            self.dispatch(
                &encoder,
                &self.dec_add_sparse_ring_forms,
                seeded.entries * 2 * RING_DEGREE,
            );
            encoder.endEncoding();
        }
        Ok(())
    }

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
        let entries = checked_product(&[RING_DEGREE, parent.cols], "Pi_DEC parent dimensions overflow")?;
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
            &[plan.active_block_count, 2, RING_DEGREE],
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
        let recycled_children = parent.finish_pending_mix(self)?;
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

    pub(crate) fn recycle_ajtai_ring_forms(&self, forms: MetalAjtaiRingForms) {
        Self::recycle_largest_buffer(&self.recycled_ajtai_forms, forms.words);
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

    fn take_recycled_buffer(
        &self,
        slot: &std::cell::RefCell<Option<Buffer>>,
        bytes: usize,
    ) -> Result<Buffer, MetalError> {
        let recycled = {
            let mut slot = slot.borrow_mut();
            slot.as_ref()
                .is_some_and(|buffer| buffer.length() as usize == bytes)
                .then(|| {
                    slot.take()
                        .expect("matching recycled Metal buffer exists above")
                })
        };
        recycled.map_or_else(|| self.buffer(bytes), Ok)
    }

    fn recycle_largest_buffer(slot: &std::cell::RefCell<Option<Buffer>>, buffer: Buffer) {
        let bytes = buffer.length();
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_none_or(|cached| cached.length() <= bytes) {
            *slot = Some(buffer);
        }
    }
}

fn checked_product(factors: &[usize], message: &'static str) -> Result<usize, MetalError> {
    factors
        .iter()
        .try_fold(1usize, |value, &factor| value.checked_mul(factor))
        .ok_or(MetalError::Shape(message))
}
