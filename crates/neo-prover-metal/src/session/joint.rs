//! Metal tables for the canonical one-joint padded-row PiCCS oracle.

use std::collections::BTreeMap;
use std::mem::size_of;

use neo_ccs::Mat;
use neo_math::{KExtensions, Rq, D, F, K};
use neo_reductions::optimized_engine::{PaperJointOracleInput, PaperJointRoundOracle};
use neo_reductions::superneo_eval::{
    weighted_projection_basis_forms, SuperneoCompactRowOffsets, SuperneoEvalCache, SuperneoZBlocks,
};
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{Buffer, MetalSession, MetalWitnessMasks};
use crate::MetalError;

mod opening;
use opening::MetalJointOpeningPlan;

const EQUALITY_CHUNK_BITS: usize = 8;
const EQUALITY_CHUNK_VALUES: usize = 1 << EQUALITY_CHUNK_BITS;
const MAX_COEFFICIENTS: usize = 10;
const SEEDED_CHUNK_COLUMNS: usize = 128;
const RING_PRODUCT_COEFFICIENTS: usize = 2 * D - 1;
const SEEDED_OUTPUT_HEADER_WORDS: usize = 9;
const SEEDED_WORK_HEADER_WORDS: usize = 3;

struct DeviceSeededRows {
    output_headers: Buffer,
    work_headers: Buffer,
    word_starts: Buffer,
    rotations: Buffer,
    eval_group_headers: Buffer,
    eval_group_outputs: Buffer,
    base_group_headers: Buffer,
    base_group_outputs: Buffer,
    work_count: usize,
    eval_group_count: usize,
    base_group_count: usize,
    selective_copy_compatible: bool,
}

struct SeededOutputMeta {
    matrix: usize,
    row_start: usize,
}

struct MetalCompactMatrix {
    row_offsets: Buffer,
    row_offset_width: u64,
    row_blocks: Buffer,
    dense_offsets: Buffer,
    dense_locals: Buffer,
    dense_coefficients: Buffer,
    identity: bool,
}

/// Structure-static compact matrix index for the one-joint oracle.
pub(crate) struct MetalJointMatrixPlan {
    matrices: Vec<MetalCompactMatrix>,
    opening: MetalJointOpeningPlan,
    matrix_count: usize,
    rows: usize,
    blocks: usize,
    seeded: Option<DeviceSeededRows>,
    has_seeded: bool,
    cache_identity: usize,
}

impl MetalJointMatrixPlan {
    pub(crate) fn matches(&self, cache: &SuperneoEvalCache) -> bool {
        self.cache_identity == cache as *const SuperneoEvalCache as usize
            && self.matrix_count == cache.matrix_caches().len()
    }
}

/// Device-resident implementation of the reduction engine's oracle seam.
pub(crate) struct MetalPaperJointOracle<'a> {
    session: &'a MetalSession,
    plan: &'a MetalJointMatrixPlan,
    masks: MetalWitnessMasks,
    application_base: Option<Buffer>,
    application_k: Option<[Buffer; 2]>,
    assignments_k: Option<[Buffer; 2]>,
    common: Option<[Buffer; 2]>,
    common_len: usize,
    assignment_sources: Buffer,
    equality_chunks: Buffer,
    prior_equality_chunks: Buffer,
    equality_chunks_per_round: usize,
    alpha_point: Vec<K>,
    prior_point: Option<Vec<K>>,
    alpha_prefix: K,
    prior_prefix: K,
    weights: Buffer,
    term_headers: Buffer,
    term_variables: Buffer,
    partials: Buffer,
    output: Buffer,
    challenge: Buffer,
    fresh_count: usize,
    matrix_count: usize,
    assignment_count: usize,
    opening_assignment_count: usize,
    blocks: usize,
    assignment_width: usize,
    coefficient_count: usize,
    selective_f_prime: bool,
    rounds: usize,
    round: usize,
    current_len: usize,
    active_len: usize,
    application_len: usize,
    assignment_len: usize,
    k_slot: usize,
    common_slot: usize,
}

impl MetalSession {
    pub(crate) fn eval_joint_dec_openings(
        &self,
        plan: &MetalJointMatrixPlan,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Vec<Vec<[K; D]>>, MetalError> {
        if witnesses.is_empty() || assignment_width > plan.blocks * D {
            return Err(MetalError::Shape("one-joint PiDEC opening shape is invalid"));
        }
        let mut mask_words = Vec::with_capacity(witnesses.len() * plan.blocks * 2);
        for witness in witnesses {
            let blocks = SuperneoZBlocks::from_witness_mat(witness, assignment_width)
                .map_err(|_| MetalError::Shape("one-joint PiDEC witness is not canonical"))?;
            if let Some((positive, negative)) = blocks.signed_unit_masks() {
                if positive.len() != plan.blocks || negative.len() != plan.blocks {
                    return Err(MetalError::Shape(
                        "one-joint PiDEC witness width does not match the matrix plan",
                    ));
                }
                for (&positive, &negative) in positive.iter().zip(negative) {
                    mask_words.extend_from_slice(&[positive, negative]);
                }
            } else if witness
                .virtual_constant_value()
                .is_some_and(|value| *value == F::ZERO)
            {
                mask_words.resize(mask_words.len() + 2 * plan.blocks, 0);
            } else {
                return Err(MetalError::Shape(
                    "one-joint PiDEC openings require signed-unit witnesses",
                ));
            }
        }
        let masks = self.prepare_witness_masks(&mask_words, witnesses.len(), plan.blocks, assignment_width)?;
        self.eval_joint_openings(
            &plan.opening,
            plan.seeded.as_ref(),
            &masks,
            point,
            witnesses.len(),
            assignment_width,
        )?
        .into_iter()
        .map(|matrices| {
            matrices
                .into_iter()
                .map(|coefficients| {
                    coefficients
                        .try_into()
                        .map_err(|_| MetalError::Shape("one-joint PiDEC opening has the wrong ring degree"))
                })
                .collect()
        })
        .collect()
    }

    pub(crate) fn prepare_joint_matrix_plan(
        &self,
        cache: &SuperneoEvalCache,
    ) -> Result<MetalJointMatrixPlan, MetalError> {
        let cache_matrices = cache.matrix_caches();
        let Some(first) = cache_matrices.first() else {
            return Err(MetalError::Shape("one-joint oracle requires application matrices"));
        };
        let (rows, scalar_columns, _) = first.compact_explicit_shape();
        let blocks = scalar_columns.div_ceil(D);
        if rows == 0 || blocks == 0 {
            return Err(MetalError::Shape("one-joint matrices have an empty shape"));
        }

        let has_seeded = cache_matrices
            .iter()
            .any(|matrix| matrix.has_compact_seeded_phi81_blocks());
        let seeded = self.prepare_joint_seeded_rows(cache_matrices, rows)?;
        let matrices = cache_matrices
            .iter()
            .map(|matrix| {
                let (matrix_rows, matrix_columns, identity) = matrix.compact_explicit_shape();
                if matrix_rows != rows || matrix_columns != scalar_columns {
                    return Err(MetalError::Shape("one-joint matrices have inconsistent shapes"));
                }
                let parts = matrix
                    .compact_device_parts()
                    .ok_or(MetalError::Shape("one-joint compact matrix cache is not finished"))?;
                let (row_offsets, row_offset_width) = match parts.row_offsets {
                    SuperneoCompactRowOffsets::Empty => (self.buffer(size_of::<u64>())?, 0),
                    SuperneoCompactRowOffsets::U24(values) => (self.buffer_from_slice(values)?, 3),
                    SuperneoCompactRowOffsets::U32(values) => (self.buffer_from_slice(values)?, 4),
                };
                let row_blocks = if parts.row_blocks.is_empty() {
                    self.buffer(size_of::<u64>())?
                } else {
                    self.buffer_from_slice(parts.row_blocks)?
                };
                let dense_offsets = if parts.dense_offsets.is_empty() {
                    self.buffer(size_of::<u64>())?
                } else {
                    self.buffer_from_slice(parts.dense_offsets)?
                };
                let dense_locals = if parts.dense_locals.is_empty() {
                    self.buffer(size_of::<u64>())?
                } else {
                    self.buffer_from_slice(parts.dense_locals)?
                };
                let dense_coefficients = if parts.dense_coefficients.is_empty() {
                    self.buffer(size_of::<u64>())?
                } else {
                    self.buffer_from_slice(parts.dense_coefficients)?
                };
                Ok(MetalCompactMatrix {
                    row_offsets,
                    row_offset_width,
                    row_blocks,
                    dense_offsets,
                    dense_locals,
                    dense_coefficients,
                    identity,
                })
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        let opening = self.prepare_joint_opening_plan(cache_matrices, scalar_columns, seeded.as_ref())?;

        Ok(MetalJointMatrixPlan {
            matrix_count: matrices.len(),
            matrices,
            opening,
            rows,
            blocks,
            seeded,
            has_seeded,
            cache_identity: cache as *const SuperneoEvalCache as usize,
        })
    }

    fn prepare_joint_seeded_rows(
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
                    return Err(MetalError::Shape("one-joint seeded row range exceeds matrix"));
                }
                let word_base = word_starts.len();
                word_starts.extend(
                    block
                        .word_starts()
                        .iter()
                        .map(|&column| {
                            u32::try_from(column).map_err(|_| MetalError::Shape("one-joint seeded column exceeds u32"))
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
                        return Err(MetalError::Shape("one-joint seeded rotation stream changed"));
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
                    output_meta.push(SeededOutputMeta { matrix, row_start });
                }
            }
        }

        let mut eval_groups = BTreeMap::<usize, Vec<u32>>::new();
        for (output, meta) in output_meta.iter().enumerate() {
            let output =
                u32::try_from(output).map_err(|_| MetalError::Shape("one-joint seeded output count exceeds u32"))?;
            eval_groups.entry(meta.row_start).or_default().push(output);
        }
        let mut eval_group_headers = Vec::<u64>::new();
        let mut eval_group_outputs = Vec::<u32>::new();
        for (row_start, outputs) in eval_groups {
            let output_base = eval_group_outputs.len();
            eval_group_outputs.extend_from_slice(&outputs);
            eval_group_headers.extend_from_slice(&[row_start as u64, output_base as u64, outputs.len() as u64]);
        }
        let mut base_groups = BTreeMap::<(usize, usize), Vec<u32>>::new();
        for (output, meta) in output_meta.iter().enumerate() {
            let output =
                u32::try_from(output).map_err(|_| MetalError::Shape("one-joint seeded output count exceeds u32"))?;
            base_groups
                .entry((meta.matrix, meta.row_start))
                .or_default()
                .push(output);
        }
        let mut base_group_headers = Vec::<u64>::new();
        let mut base_group_outputs = Vec::<u32>::new();
        for ((matrix, row_start), outputs) in base_groups {
            let output_base = base_group_outputs.len();
            base_group_outputs.extend_from_slice(&outputs);
            base_group_headers.extend_from_slice(&[
                matrix as u64,
                row_start as u64,
                output_base as u64,
                outputs.len() as u64,
            ]);
        }
        let work_count = work_headers.len() / SEEDED_WORK_HEADER_WORDS;
        let eval_group_count = eval_group_headers.len() / 3;
        let base_group_count = base_group_headers.len() / 4;
        if output_headers.len() != output_meta.len() * SEEDED_OUTPUT_HEADER_WORDS
            || work_count == 0
            || eval_group_count == 0
            || base_group_count == 0
        {
            return Err(MetalError::Shape("one-joint seeded device plan is incomplete"));
        }
        Ok(Some(DeviceSeededRows {
            output_headers: self.buffer_from_slice(&output_headers)?,
            work_headers: self.buffer_from_slice(&work_headers)?,
            word_starts: self.buffer_from_slice(&word_starts)?,
            rotations: self.buffer_from_slice(&rotations)?,
            eval_group_headers: self.buffer_from_slice(&eval_group_headers)?,
            eval_group_outputs: self.buffer_from_slice(&eval_group_outputs)?,
            base_group_headers: self.buffer_from_slice(&base_group_headers)?,
            base_group_outputs: self.buffer_from_slice(&base_group_outputs)?,
            work_count,
            eval_group_count,
            base_group_count,
            selective_copy_compatible: output_meta.iter().all(|meta| meta.matrix == 2),
        }))
    }

    fn build_joint_application_tables(
        &self,
        plan: &MetalJointMatrixPlan,
        cache: &SuperneoEvalCache,
        masks: &MetalWitnessMasks,
        fresh_count: usize,
        n_eff: usize,
        _n_pad: usize,
        selective_f_prime: bool,
    ) -> Result<Buffer, MetalError> {
        if !plan.matches(cache) || n_eff == 0 || n_eff > plan.rows {
            return Err(MetalError::Shape("one-joint application table shape is invalid"));
        }
        let table_count = fresh_count
            .checked_mul(plan.matrix_count)
            .ok_or(MetalError::Shape("one-joint application table count overflow"))?;
        let output = self.buffer(
            table_count
                .checked_mul(n_eff)
                .and_then(|words| words.checked_mul(size_of::<u64>()))
                .ok_or(MetalError::Shape("one-joint application table size overflow"))?,
        )?;
        let command = self.command_buffer("nightstream.pi_ccs.joint.application")?;
        let mut resources = Vec::<Buffer>::new();
        let assignment_width = plan.blocks * D;
        let dense_assignments = self.buffer(fresh_count * assignment_width * size_of::<u64>())?;
        let dense_shape = self.buffer_from_slice(&[plan.blocks as u64, fresh_count as u64, assignment_width as u64])?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.joint_expand_mask_assignments_f);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(masks.words()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&dense_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&dense_assignments), 0, 2);
        }
        self.dispatch(
            &encoder,
            &self.joint_expand_mask_assignments_f,
            fresh_count * assignment_width,
        );
        encoder.endEncoding();
        for source in 0..fresh_count {
            for (matrix_index, matrix) in plan.matrices.iter().enumerate() {
                let shape = self.buffer_from_slice(&[
                    plan.rows as u64,
                    plan.blocks as u64,
                    n_eff as u64,
                    n_eff as u64,
                    source as u64,
                    (source * plan.matrix_count + matrix_index) as u64,
                    matrix.row_offset_width,
                    u64::from(matrix.identity),
                    assignment_width as u64,
                ])?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.joint_build_application_tables);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&matrix.row_offsets), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.row_blocks), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_offsets), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_locals), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&matrix.dense_coefficients), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&dense_assignments), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(&shape), 0, 6);
                    encoder.setBuffer_offset_atIndex(Some(&output), 0, 7);
                }
                self.dispatch(&encoder, &self.joint_build_application_tables, n_eff);
                encoder.endEncoding();
                resources.push(shape);
            }

            if let Some(seeded) = &plan.seeded {
                if selective_f_prime && seeded.selective_copy_compatible && plan.matrix_count > 4 {
                    let copy_shape = self.buffer_from_slice(&[n_eff as u64, (source * plan.matrix_count) as u64, 4])?;
                    let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                    encoder.setComputePipelineState(&self.joint_copy_seeded_satisfied_rows);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&seeded.base_group_headers), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&copy_shape), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&output), 0, 2);
                    }
                    self.dispatch(
                        &encoder,
                        &self.joint_copy_seeded_satisfied_rows,
                        seeded.base_group_count * D,
                    );
                    encoder.endEncoding();
                    resources.push(copy_shape);
                    continue;
                }
                let partial_values = seeded
                    .work_count
                    .checked_mul(RING_PRODUCT_COEFFICIENTS)
                    .ok_or(MetalError::Shape("one-joint seeded base partial size overflow"))?;
                let partials = self.buffer(partial_values * size_of::<u64>())?;
                let seeded_shape = self.buffer_from_slice(&[
                    plan.blocks as u64,
                    source as u64,
                    n_eff as u64,
                    n_eff as u64,
                    (source * plan.matrix_count) as u64,
                ])?;
                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.joint_seeded_base_partials);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&seeded.output_headers), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.work_headers), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.word_starts), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.rotations), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(masks.words()), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&seeded_shape), 0, 5);
                    encoder.setBuffer_offset_atIndex(Some(&partials), 0, 6);
                }
                self.dispatch(&encoder, &self.joint_seeded_base_partials, partial_values);
                encoder.endEncoding();

                let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
                encoder.setComputePipelineState(&self.joint_seeded_base_reduce);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&seeded.output_headers), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.base_group_headers), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&seeded.base_group_outputs), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&seeded_shape), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&partials), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(&output), 0, 5);
                }
                self.dispatch(&encoder, &self.joint_seeded_base_reduce, seeded.base_group_count * D);
                encoder.endEncoding();
                resources.extend([seeded_shape, partials]);
            }
        }
        self.finish(&command)?;
        drop(dense_shape);
        drop(dense_assignments);
        drop(resources);
        Ok(output)
    }

    fn build_joint_common_tables(
        &self,
        plan: &MetalJointMatrixPlan,
        input: &PaperJointOracleInput<'_>,
        masks: &MetalWitnessMasks,
        has_carried: bool,
    ) -> Result<([Buffer; 2], usize), MetalError> {
        let common_len = input.structure.n.max(input.dims.assignment_width);
        let first_words = has_carried
            .then_some(common_len)
            .unwrap_or(1)
            .checked_mul(2)
            .ok_or(MetalError::Shape("one-joint common table size overflow"))?;
        let second_words = has_carried
            .then_some(common_len.div_ceil(2))
            .unwrap_or(1)
            .checked_mul(2)
            .ok_or(MetalError::Shape("one-joint folded common table size overflow"))?;
        let first = self.buffer(first_words * size_of::<u64>())?;
        let second = self.buffer(second_words * size_of::<u64>())?;

        if has_carried {
            let command = self.command_buffer("nightstream.pi_ccs.joint.common.zero")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.joint_zero_words);
            unsafe { encoder.setBuffer_offset_atIndex(Some(&first), 0, 0) };
            self.dispatch(&encoder, &self.joint_zero_words, first_words);
            encoder.endEncoding();
            self.finish(&command)?;
            self.build_joint_carried_table(plan, input, masks, &first, 0, common_len)?;
        }
        Ok(([first, second], common_len))
    }

    fn encode_joint_tensor_point(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        destination: &Buffer,
        table: usize,
        point: &[K],
        resources: &mut Vec<Buffer>,
    ) -> Result<(), MetalError> {
        let words = point
            .iter()
            .flat_map(|value| {
                let (real, imaginary) = value.to_limbs_u64();
                [real, imaginary]
            })
            .collect::<Vec<_>>();
        let challenges = self.buffer_from_slice(&words)?;
        let stages = self.buffer_from_slice(&(0..point.len() as u64).collect::<Vec<_>>())?;
        let destination_offset = table * (1usize << point.len()) * 2 * size_of::<u64>();
        for stage in 0..point.len() {
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.tensor_point_expand_k);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&challenges), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&stages), stage * size_of::<u64>(), 1);
                encoder.setBuffer_offset_atIndex(Some(destination), destination_offset, 2);
            }
            self.dispatch(&encoder, &self.tensor_point_expand_k, 1usize << stage);
            encoder.endEncoding();
        }
        resources.extend([challenges, stages]);
        Ok(())
    }

    fn build_joint_carried_table(
        &self,
        plan: &MetalJointMatrixPlan,
        input: &PaperJointOracleInput<'_>,
        masks: &MetalWitnessMasks,
        output: &Buffer,
        output_offset: usize,
        table_len: usize,
    ) -> Result<(), MetalError> {
        if plan.has_seeded && plan.seeded.is_none() {
            return Err(MetalError::Shape("one-joint seeded carried table is unsupported"));
        }
        let running_count = input.running_witnesses.len();
        let fresh_count = input.fresh_witnesses.len();
        let matrix_count = plan.matrix_count + 1;
        let gamma = input.challenges.gamma;
        let carried_coefficients = (0..running_count)
            .map(|running| k_power(gamma, running))
            .collect::<Vec<_>>();
        let weights = std::array::from_fn(|coefficient| k_power(gamma, running_count * matrix_count * coefficient));
        let identity_coefficient = k_power(gamma, 2 * fresh_count + running_count);
        let matrix_coefficients = (0..plan.matrix_count)
            .map(|matrix| k_power(gamma, 2 * fresh_count + running_count + running_count * (matrix + 1)))
            .collect::<Vec<_>>();
        let coeffs = self.buffer_from_slice(&k_words(&carried_coefficients))?;
        let mat_coeffs = self.buffer_from_slice(&k_words(&matrix_coefficients))?;
        let identity_coeff = self.buffer_from_slice(&k_words(&[identity_coefficient]))?;
        let (basis_re, basis_im) = weighted_projection_basis_forms(&weights);
        let basis_re = self.buffer_from_slice(&ring_words(&basis_re))?;
        let basis_im = self.buffer_from_slice(&ring_words(&basis_im))?;
        let n_pad = table_len;
        let shape = self.buffer_from_slice(&[
            running_count as u64,
            plan.blocks as u64,
            plan.matrix_count as u64,
            plan.rows as u64,
            input.structure.n as u64,
            n_pad as u64,
        ])?;
        let identity_shape = self.buffer_from_slice(&[input.dims.assignment_width as u64, n_pad as u64])?;
        let plane_len = plan.blocks * D;
        let z_re = self.buffer(plane_len * size_of::<u64>())?;
        let z_im = self.buffer(plane_len * size_of::<u64>())?;
        let qk = self.buffer(2 * plane_len * size_of::<u64>())?;
        let command = self.command_buffer("nightstream.pi_ccs.joint.carried")?;

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setLabel(Some(&NSString::from_str("nightstream.pi_ccs.joint.carried.combine")));
        encoder.setComputePipelineState(&self.fe_carried_mask_lin_comb);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(masks.words()), fresh_count * plan.blocks * 2 * size_of::<u64>(), 0);
            encoder.setBuffer_offset_atIndex(Some(&coeffs), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&z_re), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&z_im), 0, 4);
        }
        self.dispatch(&encoder, &self.fe_carried_mask_lin_comb, plane_len);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
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

        let mut matrix_shapes = Vec::with_capacity(plan.matrix_count);
        for (matrix_index, matrix) in plan.matrices.iter().enumerate() {
            let matrix_shape = self.buffer_from_slice(&[
                plan.rows as u64,
                input.structure.n as u64,
                n_pad as u64,
                matrix.row_offset_width,
                u64::from(matrix.identity),
            ])?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.fe_weighted_row_table);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&matrix.row_offsets), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&matrix.row_blocks), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&matrix.dense_offsets), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&matrix.dense_locals), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&matrix.dense_coefficients), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&qk), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(&mat_coeffs), matrix_index * 2 * size_of::<u64>(), 6);
                encoder.setBuffer_offset_atIndex(Some(&matrix_shape), 0, 7);
                encoder.setBuffer_offset_atIndex(Some(output), output_offset, 8);
            }
            self.dispatch(&encoder, &self.fe_weighted_row_table, n_pad);
            encoder.endEncoding();
            matrix_shapes.push(matrix_shape);
        }

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.joint_add_identity_carried);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&qk), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&identity_coeff), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&identity_shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), output_offset, 3);
        }
        self.dispatch(&encoder, &self.joint_add_identity_carried, input.dims.assignment_width);
        encoder.endEncoding();

        let seeded_partials = if let Some(seeded) = &plan.seeded {
            let partial_values = seeded
                .work_count
                .checked_mul(RING_PRODUCT_COEFFICIENTS)
                .ok_or(MetalError::Shape("one-joint seeded partial size overflow"))?;
            let partials = self.buffer(2 * partial_values * size_of::<u64>())?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.joint_seeded_k_partials);
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
            self.dispatch(&encoder, &self.joint_seeded_k_partials, partial_values);
            encoder.endEncoding();

            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.joint_seeded_k_reduce);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seeded.output_headers), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&seeded.eval_group_headers), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&seeded.eval_group_outputs), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&partials), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(output), output_offset, 5);
            }
            self.dispatch(&encoder, &self.joint_seeded_k_reduce, seeded.eval_group_count * D);
            encoder.endEncoding();
            Some(partials)
        } else {
            None
        };
        self.finish(&command)?;
        drop(matrix_shapes);
        drop(seeded_partials);
        Ok(())
    }
}

impl<'a> MetalPaperJointOracle<'a> {
    pub(crate) fn new(
        session: &'a MetalSession,
        plan: &'a MetalJointMatrixPlan,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Self, neo_reductions::PiCcsError> {
        if !plan.matches(input.cache.superneo())
            || plan.rows != input.structure.n
            || plan.matrix_count != input.structure.t()
            || input.params.b != 2
            || input.dims.degree + 1 > MAX_COEFFICIENTS
            || input.dims.row_count < 2
        {
            return Err(protocol_error(MetalError::Shape(
                "Metal one-joint oracle does not support this shape",
            )));
        }
        let fresh_count = input.fresh_witnesses.len();
        let opening_assignment_count = fresh_count + input.running_witnesses.len();
        if input.prior_point.is_some() != !input.running_witnesses.is_empty() {
            return Err(protocol_error(MetalError::Shape(
                "one-joint prior point and running witnesses disagree",
            )));
        }
        let mut mask_words = Vec::with_capacity(opening_assignment_count * plan.blocks * 2);
        let witness_mats = input
            .fresh_witnesses
            .iter()
            .map(|witness| &witness.Z)
            .chain(input.running_witnesses.iter())
            .collect::<Vec<_>>();
        let source_blocks = witness_mats
            .iter()
            .map(|witness| SuperneoZBlocks::from_witness_mat(witness, input.structure.m))
            .collect::<Result<Vec<_>, _>>()?;
        for (witness, blocks) in witness_mats.iter().zip(&source_blocks) {
            if let Some((positive, negative)) = blocks.signed_unit_masks() {
                if positive.len() != plan.blocks || negative.len() != plan.blocks {
                    return Err(protocol_error(MetalError::Shape(
                        "Metal one-joint witness width does not match the matrix plan",
                    )));
                }
                for (&positive, &negative) in positive.iter().zip(negative) {
                    mask_words.extend_from_slice(&[positive, negative]);
                }
                continue;
            }
            if witness
                .virtual_constant_value()
                .is_some_and(|value| *value == F::ZERO)
            {
                mask_words.resize(mask_words.len() + 2 * plan.blocks, 0);
                continue;
            }
            return Err(protocol_error(MetalError::Shape(
                "Metal one-joint oracle requires signed-unit witnesses",
            )));
        }
        let words_per_source = plan.blocks * 2;
        let assignment_source_indices = mask_words
            .chunks_exact(words_per_source)
            .enumerate()
            .filter_map(|(source, words)| words.iter().any(|&word| word != 0).then_some(source))
            .collect::<Vec<_>>();
        let assignment_count = assignment_source_indices.len();
        let has_carried = assignment_source_indices
            .iter()
            .any(|&source| source >= fresh_count);
        let assignment_source_words = assignment_source_indices
            .iter()
            .map(|&source| source as u64)
            .collect::<Vec<_>>();
        let assignment_sources = session
            .buffer_from_slice(nonempty(&assignment_source_words))
            .map_err(protocol_error)?;
        let masks = session
            .prepare_witness_masks(&mask_words, opening_assignment_count, plan.blocks, input.structure.m)
            .map_err(protocol_error)?;
        let selective_f_prime =
            neo_fold_clean::frontends::r1cs_f_prime::is_canonical_selective_low_norm_polynomial(&input.structure.f);
        let application_base = session
            .build_joint_application_tables(
                plan,
                input.cache.superneo(),
                &masks,
                fresh_count,
                input.structure.n,
                input.dims.row_count,
                selective_f_prime,
            )
            .map_err(protocol_error)?;
        let (common, common_len) = session
            .build_joint_common_tables(plan, &input, &masks, has_carried)
            .map_err(protocol_error)?;

        let application_count = fresh_count * plan.matrix_count;
        let application_len = input.structure.n;
        let assignment_len = input.dims.assignment_width;
        let application_half = application_len.div_ceil(2).max(1);
        let application_quarter = application_half.div_ceil(2).max(1);
        let assignment_half = assignment_len.div_ceil(2).max(1);
        let assignment_quarter = assignment_half.div_ceil(2).max(1);
        let application_k = [
            session
                .buffer(application_count * application_half * 2 * size_of::<u64>())
                .map_err(protocol_error)?,
            session
                .buffer(application_count * application_quarter * 2 * size_of::<u64>())
                .map_err(protocol_error)?,
        ];
        let assignments_k = [
            session
                .buffer(assignment_count * assignment_half * 2 * size_of::<u64>())
                .map_err(protocol_error)?,
            session
                .buffer(assignment_count * assignment_quarter * 2 * size_of::<u64>())
                .map_err(protocol_error)?,
        ];

        let mut weights = Vec::with_capacity(fresh_count + assignment_count);
        weights.extend((0..fresh_count).map(|source| k_power(input.challenges.gamma, source)));
        weights.extend(
            assignment_source_indices
                .iter()
                .map(|&source| k_power(input.challenges.gamma, fresh_count + source)),
        );
        let weights = session
            .buffer_from_slice(&k_words(&weights))
            .map_err(protocol_error)?;
        let (term_headers, term_variables) = joint_term_metadata(input.structure)?;
        let term_headers = session
            .buffer_from_slice(&term_headers)
            .map_err(protocol_error)?;
        let term_variables = session
            .buffer_from_slice(nonempty(&term_variables))
            .map_err(protocol_error)?;
        let coefficient_count = input.dims.degree + 1;
        let active_len = input.structure.n.max(input.dims.assignment_width);
        let groups = active_len.div_ceil(2).div_ceil(64).max(1);
        let partials = session
            .buffer(groups * coefficient_count * 2 * size_of::<u64>())
            .map_err(protocol_error)?;
        let output = session
            .buffer(coefficient_count * 2 * size_of::<u64>())
            .map_err(protocol_error)?;
        let challenge = session
            .buffer(2 * size_of::<u64>())
            .map_err(protocol_error)?;
        let alpha_point = input.challenges.alpha.clone();
        let prior_point = has_carried.then(|| {
            input
                .prior_point
                .expect("running witnesses have a prior point")
                .to_vec()
        });
        let equality_chunks_per_round = input
            .dims
            .variables
            .saturating_sub(1)
            .div_ceil(EQUALITY_CHUNK_BITS)
            .max(1);
        let equality_chunks = session
            .buffer_from_slice(&equality_suffix_chunk_words(&alpha_point, equality_chunks_per_round))
            .map_err(protocol_error)?;
        let prior_equality_chunks = session
            .buffer_from_slice(
                prior_point
                    .as_deref()
                    .map(|point| equality_suffix_chunk_words(point, equality_chunks_per_round))
                    .as_deref()
                    .unwrap_or(&[0]),
            )
            .map_err(protocol_error)?;

        Ok(Self {
            session,
            plan,
            masks,
            application_base: Some(application_base),
            application_k: Some(application_k),
            assignments_k: Some(assignments_k),
            common: Some(common),
            common_len,
            assignment_sources,
            equality_chunks,
            prior_equality_chunks,
            equality_chunks_per_round,
            alpha_point,
            prior_point,
            alpha_prefix: K::ONE,
            prior_prefix: K::ONE,
            weights,
            term_headers,
            term_variables,
            partials,
            output,
            challenge,
            fresh_count,
            matrix_count: plan.matrix_count,
            assignment_count,
            opening_assignment_count,
            blocks: plan.blocks,
            assignment_width: input.dims.assignment_width,
            coefficient_count,
            selective_f_prime,
            rounds: input.dims.variables,
            round: 0,
            current_len: input.dims.row_count,
            active_len,
            application_len,
            assignment_len,
            k_slot: 0,
            common_slot: 0,
        })
    }

    fn round_coefficients(&mut self) -> Result<Vec<K>, MetalError> {
        let base_round = self.round == 0;
        let application = if base_round {
            self.application_base
                .as_ref()
                .ok_or(MetalError::Shape("one-joint base tables were released too early"))?
        } else {
            &self
                .application_k
                .as_ref()
                .expect("one-joint application tables exist during SumCheck")[self.k_slot]
        };
        let assignments = if base_round {
            self.masks.words()
        } else {
            &self
                .assignments_k
                .as_ref()
                .expect("one-joint assignment tables exist during SumCheck")[self.k_slot]
        };
        let (alpha_low, alpha_slope) = equality_round_affine(&self.alpha_point, self.alpha_prefix, self.round);
        let (prior_low, prior_slope) = self
            .prior_point
            .as_deref()
            .map(|point| equality_round_affine(point, self.prior_prefix, self.round))
            .unwrap_or((K::ZERO, K::ZERO));
        let (alpha_low_re, alpha_low_im) = alpha_low.to_limbs_u64();
        let (alpha_slope_re, alpha_slope_im) = alpha_slope.to_limbs_u64();
        let (prior_low_re, prior_low_im) = prior_low.to_limbs_u64();
        let (prior_slope_re, prior_slope_im) = prior_slope.to_limbs_u64();
        let shape = self.session.buffer_from_slice(&[
            self.current_len as u64,
            self.fresh_count as u64,
            self.matrix_count as u64,
            self.assignment_count as u64,
            self.coefficient_count as u64,
            (self.term_headers.length() as usize / (3 * size_of::<u64>())) as u64,
            u64::from(base_round),
            self.blocks as u64,
            self.assignment_width as u64,
            self.active_len as u64,
            self.application_len as u64,
            self.assignment_len as u64,
            u64::from(self.prior_point.is_some()),
            self.common_len as u64,
            self.equality_chunks_per_round as u64,
            self.round as u64,
            alpha_low_re,
            alpha_low_im,
            alpha_slope_re,
            alpha_slope_im,
            prior_low_re,
            prior_low_im,
            prior_slope_re,
            prior_slope_im,
        ])?;
        let groups = self.active_len.div_ceil(2).div_ceil(64).max(1);
        let reduction_shape = self
            .session
            .buffer_from_slice(&[groups as u64, self.coefficient_count as u64])?;
        let command = self
            .session
            .command_buffer("nightstream.pi_ccs.joint.round")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let round_pipeline = if self.selective_f_prime {
            &self.session.joint_selective_round_partials
        } else {
            &self.session.joint_round_partials
        };
        encoder.setComputePipelineState(round_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(application), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(assignments), 0, 1);
            encoder.setBuffer_offset_atIndex(
                Some(
                    &self
                        .common
                        .as_ref()
                        .expect("one-joint common tables exist during SumCheck")[self.common_slot],
                ),
                0,
                2,
            );
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.weights), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.term_headers), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.term_variables), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&self.partials), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&self.equality_chunks), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&self.prior_equality_chunks), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(&self.assignment_sources), 0, 10);
        }
        self.session
            .dispatch_threadgroups(&encoder, round_pipeline, groups, 64);
        encoder.endEncoding();

        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.session.sumcheck_reduce_partials);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.partials), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&reduction_shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.output), 0, 2);
        }
        self.session
            .dispatch(&encoder, &self.session.sumcheck_reduce_partials, self.coefficient_count);
        encoder.endEncoding();
        self.session.finish(&command)?;
        Ok(self
            .session
            .read_buffer::<u64>(&self.output, self.coefficient_count * 2)
            .chunks_exact(2)
            .map(|words| K::from_coeffs([F::from_u64(words[0]), F::from_u64(words[1])]))
            .collect())
    }

    fn fold_tables(&mut self, challenge: K) -> Result<(), MetalError> {
        let (real, imaginary) = challenge.to_limbs_u64();
        self.session
            .write_shared(&self.challenge, &[real, imaginary])?;
        let half = self.current_len / 2;
        let application_next_len = self.application_len.div_ceil(2).max(1);
        let assignment_next_len = self.assignment_len.div_ceil(2).max(1);
        let command = self
            .session
            .command_buffer("nightstream.pi_ccs.joint.fold")?;
        if self.round == 0 {
            let app_shape = self.session.buffer_from_slice(&[
                self.application_len as u64,
                (self.fresh_count * self.matrix_count) as u64,
            ])?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.session.joint_fold_base_tables);
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(
                        self.application_base
                            .as_ref()
                            .expect("one-joint base tables exist during the first fold"),
                    ),
                    0,
                    0,
                );
                encoder.setBuffer_offset_atIndex(Some(&self.challenge), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&app_shape), 0, 2);
                encoder.setBuffer_offset_atIndex(
                    Some(
                        &self
                            .application_k
                            .as_ref()
                            .expect("one-joint application tables exist during SumCheck")[0],
                    ),
                    0,
                    3,
                );
            }
            self.session.dispatch(
                &encoder,
                &self.session.joint_fold_base_tables,
                self.fresh_count * self.matrix_count * application_next_len,
            );
            encoder.endEncoding();

            let assignment_shape = self.session.buffer_from_slice(&[
                self.assignment_len as u64,
                self.assignment_count as u64,
                self.blocks as u64,
                self.assignment_width as u64,
            ])?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            encoder.setComputePipelineState(&self.session.joint_fold_mask_assignments);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(self.masks.words()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.challenge), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&assignment_shape), 0, 2);
                encoder.setBuffer_offset_atIndex(
                    Some(
                        &self
                            .assignments_k
                            .as_ref()
                            .expect("one-joint assignment tables exist during SumCheck")[0],
                    ),
                    0,
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(&self.assignment_sources), 0, 4);
            }
            self.session.dispatch(
                &encoder,
                &self.session.joint_fold_mask_assignments,
                self.assignment_count * assignment_next_len,
            );
            encoder.endEncoding();
            self.k_slot = 0;
        } else {
            let next = self.k_slot ^ 1;
            let application_k = self
                .application_k
                .as_ref()
                .expect("one-joint application tables exist during SumCheck");
            self.encode_compact_k_fold(
                &command,
                &application_k[self.k_slot],
                &application_k[next],
                self.application_len,
                self.fresh_count * self.matrix_count,
            )?;
            let assignments_k = self
                .assignments_k
                .as_ref()
                .expect("one-joint assignment tables exist during SumCheck");
            self.encode_compact_k_fold(
                &command,
                &assignments_k[self.k_slot],
                &assignments_k[next],
                self.assignment_len,
                self.assignment_count,
            )?;
            self.k_slot = next;
        }
        if self.prior_point.is_some() {
            let common_next = self.common_slot ^ 1;
            let common = self
                .common
                .as_ref()
                .expect("one-joint carried tables exist during SumCheck");
            self.encode_compact_k_fold(
                &command,
                &common[self.common_slot],
                &common[common_next],
                self.common_len,
                1,
            )?;
            self.common_slot = common_next;
            self.common_len = self.common_len.div_ceil(2).max(1);
        }
        self.session.finish(&command)?;
        self.alpha_prefix = restrict_equality_prefix(self.alpha_prefix, self.alpha_point[self.round], challenge);
        if let Some(prior_point) = &self.prior_point {
            self.prior_prefix = restrict_equality_prefix(self.prior_prefix, prior_point[self.round], challenge);
        }
        if self.round == 0 {
            self.application_base = None;
        }
        self.current_len = half;
        self.application_len = application_next_len;
        self.assignment_len = assignment_next_len;
        self.active_len = self.application_len.max(self.assignment_len);
        self.round += 1;
        Ok(())
    }

    fn encode_compact_k_fold(
        &self,
        command: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        input: &Buffer,
        output: &Buffer,
        table_len: usize,
        table_count: usize,
    ) -> Result<(), MetalError> {
        let shape = self
            .session
            .buffer_from_slice(&[table_len as u64, table_count as u64])?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        encoder.setComputePipelineState(&self.session.joint_fold_k_tables);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.challenge), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
        }
        self.session.dispatch(
            &encoder,
            &self.session.joint_fold_k_tables,
            table_count * table_len.div_ceil(2),
        );
        encoder.endEncoding();
        Ok(())
    }
}

impl PaperJointRoundOracle for MetalPaperJointOracle<'_> {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, neo_reductions::PiCcsError> {
        let coefficients = self.round_coefficients().map_err(protocol_error)?;
        if self.selective_f_prime {
            if points.len() != coefficients.len()
                || points
                    .iter()
                    .enumerate()
                    .any(|(index, &point)| point != K::from(F::from_u64(index as u64)))
            {
                return Err(protocol_error(MetalError::Shape(
                    "selective one-joint oracle received non-canonical evaluation points",
                )));
            }
            return Ok(coefficients);
        }
        Ok(points
            .iter()
            .map(|&point| {
                coefficients
                    .iter()
                    .rev()
                    .fold(K::ZERO, |value, &coefficient| value * point + coefficient)
            })
            .collect())
    }

    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.coefficient_count - 1
    }

    fn fold(&mut self, challenge: K) -> Result<(), neo_reductions::PiCcsError> {
        self.fold_tables(challenge).map_err(protocol_error)
    }

    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<Vec<Vec<K>>>>, neo_reductions::PiCcsError> {
        self.application_base = None;
        self.application_k = None;
        self.assignments_k = None;
        self.common = None;
        self.session
            .eval_joint_openings(
                &self.plan.opening,
                self.plan.seeded.as_ref(),
                &self.masks,
                point,
                self.opening_assignment_count,
                self.assignment_width,
            )
            .map(Some)
            .map_err(protocol_error)
    }
}

fn joint_term_metadata(
    structure: &neo_ccs::CcsStructure<F>,
) -> Result<(Vec<u64>, Vec<u64>), neo_reductions::PiCcsError> {
    let mut headers = Vec::with_capacity(structure.f.terms().len() * 3);
    let mut variables = Vec::new();
    for term in structure.f.terms() {
        let start = variables.len() / 2;
        for (matrix, &exponent) in term.exps.iter().enumerate() {
            if exponent != 0 {
                variables.extend_from_slice(&[matrix as u64, exponent as u64]);
            }
        }
        headers.extend_from_slice(&[
            term.coeff.as_canonical_u64(),
            start as u64,
            (variables.len() / 2 - start) as u64,
        ]);
    }
    Ok((headers, variables))
}

fn equality_suffix_chunk_words(point: &[K], chunks_per_round: usize) -> Vec<u64> {
    let mut values = Vec::with_capacity(point.len() * chunks_per_round * EQUALITY_CHUNK_VALUES);
    for round in 0..point.len() {
        for chunk in 0..chunks_per_round {
            let start = round + 1 + chunk * EQUALITY_CHUNK_BITS;
            let end = (start + EQUALITY_CHUNK_BITS).min(point.len());
            for index in 0..EQUALITY_CHUNK_VALUES {
                let mut value = K::ONE;
                for (bit, &coordinate) in point[start.min(point.len())..end].iter().enumerate() {
                    value *= if index & (1 << bit) == 0 {
                        K::ONE - coordinate
                    } else {
                        coordinate
                    };
                }
                values.push(value);
            }
        }
    }
    k_words(&values)
}

fn equality_round_affine(point: &[K], prefix: K, round: usize) -> (K, K) {
    let high = prefix * point[round];
    let low = prefix * (K::ONE - point[round]);
    (low, high - low)
}

fn restrict_equality_prefix(prefix: K, point: K, challenge: K) -> K {
    prefix * ((K::ONE - point) * (K::ONE - challenge) + point * challenge)
}

fn k_power(value: K, exponent: usize) -> K {
    (0..exponent).fold(K::ONE, |power, _| power * value)
}

fn k_words(values: &[K]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|value| {
            let (real, imaginary) = value.to_limbs_u64();
            [real, imaginary]
        })
        .collect()
}

fn ring_words(values: &[Rq; D]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|ring| ring.0.iter().map(PrimeField64::as_canonical_u64))
        .collect()
}

fn nonempty(values: &[u64]) -> &[u64] {
    if values.is_empty() {
        &[0]
    } else {
        values
    }
}

fn protocol_error(error: MetalError) -> neo_reductions::PiCcsError {
    neo_reductions::PiCcsError::ProtocolError(format!("Metal one-joint oracle: {error}"))
}
