//! CUDA tables for the canonical one-joint padded-row PiCCS oracle.

use std::mem::size_of;
#[cfg(test)]
use std::sync::Arc;

use neo_ccs::{CcsStructure, Mat, V1_2Evaluations};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::{PaperJointOracleInput, PaperJointRoundOracle};
use neo_reductions::superneo_eval::{weighted_projection_basis_forms, MatrixRows, SuperneoEvalCache, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;
#[cfg(test)]
use p3_field::PrimeField64;

use super::{Buffer, CudaSession, CudaWitnessMasks, Grid};
use crate::CudaError;

mod application;
mod assignments;
mod dense;
mod matrix_window;
mod opening;
mod relation;
mod support;

use crate::oracle_error;
use application::ApplicationTables;
use assignments::AssignmentTables;
use dense::{CudaDenseWitnesses, JointSources};
use support::{
    equality_round_affine, equality_suffix_chunk_words, joint_term_metadata, k_power, k_words, nonempty, nonempty_u32,
    restrict_equality_prefix, ring_words,
};

use crate::kernels::carried::CARRIED_THREADS;
use crate::kernels::joint::SUMCHECK_REDUCTION_THREADS;

#[cfg(test)]
#[path = "../../tests/unit/joint_buffers.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/carried_projection.rs"]
mod carried_tests;

#[cfg(test)]
#[path = "../../tests/unit/application_replay.rs"]
mod application_tests;

const EQUALITY_CHUNK_BITS: usize = 8;
const EQUALITY_CHUNK_VALUES: usize = 1 << EQUALITY_CHUNK_BITS;
const MAX_COEFFICIENTS: usize = 10;
const ZERO_FAMILY_LEN: usize = 1;

struct CudaCompactMatrix {
    row_offsets: Buffer,
    row_offset_width: u64,
    row_blocks: Buffer,
    dense_row_blocks: Buffer,
    dense_offsets: Buffer,
    dense_locals: Buffer,
    dense_coefficients: Buffer,
    geometric_row_offsets: Buffer,
    geometric_row_offset_width: u64,
    geometric_runs: Buffer,
    identity: bool,
}

pub(crate) struct CudaJointMatrixPlan<'a> {
    source: &'a dyn MatrixRows,
    matrix_count: usize,
    rows: usize,
    blocks: usize,
    workspace_bytes: usize,
}

impl CudaJointMatrixPlan<'_> {
    pub(crate) fn matches(&self, source: &dyn MatrixRows) -> bool {
        let shape = source.shape();
        std::ptr::eq(self.source, source)
            && self.rows == shape.rows
            && self.blocks * D == shape.columns
            && self.matrix_count == shape.matrices
    }
}

pub(crate) struct CudaPaperJointOracle<'a> {
    session: &'a CudaSession,
    plan: CudaJointMatrixPlan<'a>,
    masks: CudaWitnessMasks,
    dense: Option<CudaDenseWitnesses>,
    application: Option<ApplicationTables>,
    application_peak_bytes: usize,
    assignments: Option<AssignmentTables>,
    common: Option<Buffer>,
    common_len: usize,
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
    output: Buffer,
    fresh_count: usize,
    matrix_count: usize,
    assignment_count: usize,
    opening_assignment_count: usize,
    blocks: usize,
    assignment_width: usize,
    coefficient_count: usize,
    range_base: u32,
    f_vanishes_at_zero: bool,
    rounds: usize,
    round: usize,
    current_len: usize,
    active_len: usize,
    application_len: usize,
    assignment_len: usize,
}

impl CudaSession {
    pub(crate) fn eval_joint_dec_openings(
        &self,
        plan: &CudaJointMatrixPlan,
        witnesses: &[Mat<F>],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Option<Vec<V1_2Evaluations<K>>>, CudaError> {
        let variables = (usize::BITS
            - plan
                .rows
                .max(plan.blocks * D)
                .saturating_sub(1)
                .leading_zeros()) as usize;
        if witnesses.is_empty() || assignment_width.div_ceil(D) != plan.blocks || point.len() < variables {
            return Err(CudaError::Shape("one-joint PiDEC opening shape is invalid"));
        }
        let source_blocks = witnesses
            .iter()
            .map(|witness| {
                SuperneoZBlocks::from_witness_mat(witness, assignment_width)
                    .map_err(|_| CudaError::Shape("one-joint PiDEC witness is not canonical"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if source_blocks.iter().all(SuperneoZBlocks::is_zero) {
            return Ok(Some(
                witnesses
                    .iter()
                    .map(|_| V1_2Evaluations {
                        eval_k: vec![K::ZERO; D],
                        eval_a: vec![vec![K::ZERO; D]; plan.matrix_count],
                    })
                    .collect(),
            ));
        }
        let signed_unit = source_blocks
            .iter()
            .zip(witnesses)
            .all(|(blocks, witness)| {
                blocks.signed_unit_masks().is_some()
                    || witness
                        .virtual_constant_value()
                        .is_some_and(|value| *value == F::ZERO)
            });
        let base = if signed_unit { 2 } else { 4 };
        let masks = self.prepare_joint_witness_masks(&source_blocks, base, assignment_width)?;
        drop(source_blocks);
        let openings = self.eval_streamed_joint_openings(plan, &masks, point, witnesses.len(), assignment_width)?;
        Ok(Some(openings))
    }

    pub(crate) fn evaluate_terminal_rows(
        &self,
        plan: &CudaJointMatrixPlan,
        structure: &CcsStructure<F>,
        witnesses: &[Mat<F>],
        point: &[K],
        fresh: &Mat<F>,
    ) -> Result<neo_reductions::superneo_eval::TerminalEvaluations, CudaError> {
        let variables = (usize::BITS
            - structure
                .domain_rows()
                .max(plan.rows)
                .max(plan.blocks * D)
                .saturating_sub(1)
                .leading_zeros()) as usize;
        if witnesses.is_empty() || point.len() != variables {
            return Err(CudaError::Shape("terminal running opening shape"));
        }
        let terminal = self.prepare_terminal_row_check(plan, structure, fresh)?;
        let blocks = witnesses
            .iter()
            .map(|witness| {
                SuperneoZBlocks::from_witness_mat(witness, structure.m())
                    .map_err(|_| CudaError::Shape("terminal running witness shape"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let masks = self.prepare_joint_witness_masks(&blocks, 2, structure.m())?;
        drop(blocks);
        self.eval_streamed_rows(plan, &masks, point, witnesses.len(), structure.m(), Some(&terminal))
    }

    pub(crate) fn prepare_joint_matrix_plan<'a>(
        &self,
        source: &'a dyn MatrixRows,
        workspace_bytes: usize,
    ) -> Result<CudaJointMatrixPlan<'a>, CudaError> {
        let shape = source.shape();
        if shape.rows == 0 || shape.columns == 0 || shape.matrices == 0 || !shape.columns.is_multiple_of(D) {
            return Err(CudaError::Shape("one-joint matrix source has an invalid shape"));
        }
        Ok(CudaJointMatrixPlan {
            source,
            matrix_count: shape.matrices,
            rows: shape.rows,
            blocks: shape.columns / D,
            workspace_bytes,
        })
    }

    fn build_joint_common_tables<'a>(
        &self,
        plan: &CudaJointMatrixPlan,
        input: &PaperJointOracleInput<'_>,
        sources: impl Into<JointSources<'a>>,
        has_carried: bool,
    ) -> Result<(Buffer, usize), CudaError> {
        if !has_carried {
            return self.zero_family_table();
        }
        let common_len = input.structure.n().max(input.dims.assignment_width);
        let first_words = common_len
            .checked_mul(2)
            .ok_or(CudaError::Shape("one-joint common table size overflow"))?;
        let first = self.buffer(first_words * size_of::<u64>())?;
        self.build_joint_carried_table(plan, input, sources.into(), &first, common_len)?;
        Ok((first, common_len))
    }

    fn zero_family_table(&self) -> Result<(Buffer, usize), CudaError> {
        Ok((self.buffer_from_slice(&[0u64; 2 * ZERO_FAMILY_LEN])?, ZERO_FAMILY_LEN))
    }

    /// Launch the tensor expansion of `point` into table `table` of
    /// `destination`; `resources` keeps its inputs until the caller
    /// finishes.
    fn encode_joint_tensor_point(
        &self,
        destination: &Buffer,
        table: usize,
        point: &[K],
        resources: &mut Vec<Buffer>,
    ) -> Result<(), CudaError> {
        let challenges = self.buffer_from_slice(&k_words(point))?;
        let table = destination.at::<u64>(table * (1usize << point.len()) * 2 * size_of::<u64>());
        let kernels = &self.kernels.nightstream;
        for stage in 0..point.len() {
            self.launch(Grid::Threads(1usize << stage), |stream, config| {
                kernels.tensor_point_expand_k(stream, config, challenges.ptr(), table, stage as u64)
            })?;
        }
        resources.push(challenges);
        Ok(())
    }

    fn build_joint_carried_table(
        &self,
        plan: &CudaJointMatrixPlan,
        input: &PaperJointOracleInput<'_>,
        sources: JointSources<'_>,
        output: &Buffer,
        table_len: usize,
    ) -> Result<(), CudaError> {
        let masks = sources.masks;
        let width = plan.blocks * D;
        let rows = input.structure.n();
        if input.dims.assignment_width != width
            || rows == 0
            || table_len < width.max(rows)
            || output.length() / size_of::<K>() < table_len
        {
            return Err(CudaError::Shape("carried output cannot hold the complete projection"));
        }
        let running_count = input.running_witnesses.len();
        let fresh_count = input.fresh_witnesses.len();
        let stored_running = match sources.dense {
            Some(dense) => dense.count(),
            None => masks
                .stored_witnesses()
                .saturating_sub(fresh_count)
                .min(running_count),
        };
        if stored_running == 0 {
            return Err(CudaError::Shape("carried projection has no stored running witness"));
        }
        // The running sources' values: their masks after the fresh sources',
        // or their dense values.
        let running_values: *const u64 = match sources.dense {
            Some(dense) => dense.words().ptr(),
            None => masks
                .words()
                .at(fresh_count * plan.blocks * 2 * masks.magnitudes() * size_of::<u64>()),
        };
        let dense_projection = sources.dense.is_some();
        let matrix_count = plan.matrix_count;
        let gamma = input.challenges.gamma;
        let carried_coefficients = (0..running_count)
            .map(|running| k_power(gamma, running))
            .collect::<Vec<_>>();
        let weights = std::array::from_fn(|coefficient| k_power(gamma, running_count * matrix_count * coefficient));
        let matrix_coefficients = (0..plan.matrix_count)
            .map(|matrix| k_power(gamma, running_count * D + running_count * matrix))
            .collect::<Vec<_>>();
        let coeffs = self.buffer_from_slice(&k_words(&carried_coefficients))?;
        let mat_coeffs = self.buffer_from_slice(&k_words(&matrix_coefficients))?;
        let (basis_re, basis_im) = weighted_projection_basis_forms(&weights);
        let basis_re = self.buffer_from_slice(&ring_words(&basis_re))?;
        let basis_im = self.buffer_from_slice(&ring_words(&basis_im))?;
        let pad_weights = std::array::from_fn(|coefficient| k_power(gamma, running_count * coefficient));
        let (pad_re, pad_im) = weighted_projection_basis_forms(&pad_weights);
        let pad_re = self.buffer_from_slice(&ring_words(&pad_re))?;
        let pad_im = self.buffer_from_slice(&ring_words(&pad_im))?;
        let row_sums = self.buffer(
            rows.checked_mul(size_of::<K>())
                .ok_or(CudaError::Shape("carried row sums overflow"))?,
        )?;
        let shape = [
            stored_running as u64,
            plan.blocks as u64,
            plan.matrix_count as u64,
            plan.rows as u64,
            input.structure.n() as u64,
            rows as u64,
            masks.magnitudes() as u64,
            width as u64,
            0,
        ];
        let carried = &self.kernels.carried;
        let project = |shape: [u64; 9], basis_re: &Buffer, basis_im: &Buffer, groups: usize| {
            self.launch(Grid::groups(groups, CARRIED_THREADS), |stream, config| {
                let args = (
                    coeffs.ptr(),
                    shape,
                    basis_re.ptr(),
                    basis_im.ptr(),
                    row_sums.ptr(),
                    output.ptr(),
                );
                if dense_projection {
                    carried.joint_carried_projection_dense(
                        stream,
                        config,
                        running_values,
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                    )
                } else {
                    carried.joint_carried_projection(
                        stream,
                        config,
                        running_values,
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                        args.5,
                    )
                }
            })
        };
        self.zero_words(&row_sums, rows * 2)?;
        project(shape, &basis_re, &basis_im, plan.blocks)?;
        self.finish()?;

        let reserved = plan
            .matrix_count
            .checked_mul(7 * size_of::<u64>())
            .ok_or(CudaError::Shape("carried matrix metadata size overflow"))?;
        let mut next_row = 0;
        while next_row < rows {
            let window = self.load_matrix_window(plan, next_row..rows, reserved, true)?;
            let local_rows = window.rows.end - window.rows.start;
            let kernels = &self.kernels.nightstream;
            for (matrix_index, matrix) in window.matrices.iter().enumerate() {
                let matrix_shape = [
                    local_rows as u64,
                    rows as u64,
                    rows as u64,
                    matrix.row_offset_width,
                    u64::from(matrix.identity),
                    matrix.geometric_row_offset_width,
                    window.rows.start as u64,
                ];
                self.launch(Grid::Threads(local_rows), |stream, config| {
                    kernels.fe_weighted_row_table(
                        stream,
                        config,
                        matrix.row_offsets.ptr(),
                        matrix.row_blocks.ptr(),
                        matrix.dense_offsets.ptr(),
                        matrix.dense_locals.ptr(),
                        matrix.dense_coefficients.ptr(),
                        matrix.geometric_row_offsets.ptr(),
                        matrix.geometric_runs.ptr(),
                        output.ptr(),
                        mat_coeffs.at(matrix_index * 2 * size_of::<u64>()),
                        matrix_shape,
                        row_sums.ptr(),
                        matrix.dense_row_blocks.ptr(),
                    )
                })?;
            }
            self.finish()?;
            next_row = window.rows.end;
            drop(window);
        }

        let final_shape = [
            stored_running as u64,
            plan.blocks as u64,
            plan.matrix_count as u64,
            plan.rows as u64,
            rows as u64,
            rows as u64,
            masks.magnitudes() as u64,
            table_len as u64,
            rows as u64,
        ];
        project(final_shape, &pad_re, &pad_im, table_len.div_ceil(D))?;
        self.finish()?;
        Ok(())
    }
}

impl<'a> CudaPaperJointOracle<'a> {
    pub(crate) fn new(
        session: &'a CudaSession,
        plan: CudaJointMatrixPlan<'a>,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Self, neo_reductions::PiCcsError> {
        Self::with_application_workspace(session, plan, input, None)
    }

    #[cfg(test)]
    pub(crate) fn new_with_application_workspace(
        session: &'a CudaSession,
        plan: CudaJointMatrixPlan<'a>,
        input: PaperJointOracleInput<'a>,
        workspace_bytes: usize,
    ) -> Result<Self, neo_reductions::PiCcsError> {
        Self::with_application_workspace(session, plan, input, Some(workspace_bytes))
    }

    fn with_application_workspace(
        session: &'a CudaSession,
        plan: CudaJointMatrixPlan<'a>,
        input: PaperJointOracleInput<'a>,
        workspace: Option<usize>,
    ) -> Result<Self, neo_reductions::PiCcsError> {
        if !plan.matches(input.rows)
            || plan.rows != input.structure.n()
            || plan.matrix_count != input.structure.t()
            || input.params.b() != 2
            || input.dims.degree + 1 > MAX_COEFFICIENTS
            || input.dims.row_count < 2
        {
            return Err(oracle_error(CudaError::Shape(
                "CUDA one-joint oracle does not support this shape",
            )));
        }
        let fresh_count = input.fresh_witnesses.len();
        let running_count = input.running_witnesses.len();
        let opening_assignment_count = fresh_count + running_count;
        if input.prior_point.is_some() == input.running_witnesses.is_empty() {
            return Err(oracle_error(CudaError::Shape(
                "one-joint prior point and running witnesses disagree",
            )));
        }
        // Option B's light fold checks the fresh sources' norms only; its
        // running sources have no masks and stay dense.
        let dense_running = match input.norm_sources {
            sources if sources == fresh_count + running_count => false,
            sources if sources == fresh_count => true,
            _ => {
                return Err(oracle_error(CudaError::Shape(
                    "norm terms cover neither every source nor the fresh sources",
                )))
            }
        };
        let witness_mats = input
            .fresh_witnesses
            .iter()
            .map(|witness| &witness.Z)
            .chain(input.running_witnesses.iter())
            .collect::<Vec<_>>();
        let mut source_blocks = witness_mats
            .iter()
            .map(|witness| SuperneoZBlocks::from_witness_mat(witness, input.structure.m()))
            .collect::<Result<Vec<_>, _>>()?;
        let dense = dense_running
            .then(|| session.prepare_dense_witnesses(&source_blocks.split_off(fresh_count)))
            .transpose()
            .map_err(oracle_error)?;
        let masks = session
            .prepare_joint_witness_masks(&source_blocks, input.params.b(), input.structure.m())
            .map_err(oracle_error)?;
        let assignment_source_indices = masks
            .active_witnesses()
            .iter()
            .map(|&source| source as usize)
            .collect::<Vec<_>>();
        let assignment_count = assignment_source_indices.len();
        let has_carried = assignment_source_indices
            .iter()
            .any(|&source| source >= fresh_count)
            || dense
                .as_ref()
                .is_some_and(|dense| !dense.active().is_empty());
        drop(source_blocks);
        let sources = JointSources {
            masks: &masks,
            dense: dense.as_ref(),
        };
        let (common, common_len) = session
            .build_joint_common_tables(&plan, &input, sources, has_carried)
            .map_err(oracle_error)?;

        let application_len = input.structure.n();
        let assignment_len = input.dims.assignment_width;
        let assignments = AssignmentTables::new(session, &masks, assignment_len).map_err(oracle_error)?;

        let constraint_exponent = input.running_witnesses.len() * D * (plan.matrix_count + 1);
        let mut weights = Vec::with_capacity(fresh_count + assignment_count);
        weights.extend((0..fresh_count).map(|source| k_power(input.challenges.gamma, constraint_exponent + source)));
        weights.extend(
            assignment_source_indices
                .iter()
                .map(|&source| k_power(input.challenges.gamma, constraint_exponent + fresh_count + source)),
        );
        let weights = session
            .buffer_from_slice(&k_words(&weights))
            .map_err(oracle_error)?;
        let (term_headers, term_variables) = joint_term_metadata(input.structure)?;
        let term_headers = session
            .buffer_from_slice(&term_headers)
            .map_err(oracle_error)?;
        let term_variables = session
            .buffer_from_slice(nonempty(&term_variables))
            .map_err(oracle_error)?;
        let coefficient_count = input.dims.degree + 1;
        let active_len = input.structure.n().max(input.dims.assignment_width);
        let output = session
            .buffer(coefficient_count * 2 * size_of::<u64>())
            .map_err(oracle_error)?;
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
            .map_err(oracle_error)?;
        let prior_equality_chunks = session
            .buffer_from_slice(
                prior_point
                    .as_deref()
                    .map(|point| equality_suffix_chunk_words(point, equality_chunks_per_round))
                    .as_deref()
                    .unwrap_or(&[0]),
            )
            .map_err(oracle_error)?;

        let reserved = ApplicationTables::scratch_bytes(&plan, fresh_count)
            .map_err(oracle_error)?
            .max(
                active_len.div_ceil(2).div_ceil(64).max(1) * coefficient_count * size_of::<K>() + 33 * size_of::<u64>(),
            );
        let application =
            ApplicationTables::new(session, &plan, &masks, fresh_count, workspace, reserved).map_err(oracle_error)?;

        Ok(Self {
            session,
            masks,
            dense,
            application_peak_bytes: application.peak_bytes(),
            application: Some(application),
            assignments: Some(assignments),
            common: Some(common),
            common_len,
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
            output,
            fresh_count,
            matrix_count: plan.matrix_count,
            assignment_count,
            opening_assignment_count,
            blocks: plan.blocks,
            assignment_width: input.dims.assignment_width,
            coefficient_count,
            range_base: input.params.b(),
            f_vanishes_at_zero: input.structure.f().eval(&vec![F::ZERO; plan.matrix_count]) == F::ZERO,
            rounds: input.dims.variables,
            round: 0,
            current_len: input.dims.row_count,
            active_len,
            application_len,
            assignment_len,
            plan,
        })
    }

    #[cfg(test)]
    fn application_workspace_peak_bytes(&self) -> usize {
        self.application
            .as_ref()
            .map_or(self.application_peak_bytes, |tables| {
                tables.peak_bytes().max(self.application_peak_bytes)
            })
    }

    #[cfg(test)]
    fn application_is_resident(&self) -> bool {
        self.application
            .as_ref()
            .is_some_and(|tables| tables.resident().is_some())
    }

    fn round_coefficients(&mut self) -> Result<Vec<K>, CudaError> {
        let mut application = self
            .application
            .take()
            .ok_or(CudaError::Shape("one-joint application tables were released too early"))?;
        let result = (|| {
            let reserved = ApplicationTables::scratch_bytes(&self.plan, self.fresh_count)?.max(
                self.active_len.div_ceil(2).div_ceil(64).max(1) * self.coefficient_count * size_of::<K>()
                    + 33 * size_of::<u64>(),
            );
            application.prepare_round(self.session, &self.plan, &self.masks, reserved)?;
            if let Some(window) = application.resident() {
                return self.round_window(&window.values, 0, window.stride, 0, self.active_len.div_ceil(2));
            }
            let mut coefficients = vec![K::ZERO; self.coefficient_count];
            let pairs = self.application_len.div_ceil(2);
            let window_pairs = application.window_rows()? / 2;
            for pair in (0..pairs).step_by(window_pairs) {
                let count = (pairs - pair).min(window_pairs);
                let window = application.replay_window(self.session, &self.plan, &self.masks, pair * 2)?;
                let partial = self.round_window(&window.values, pair * 2, window.stride, pair, count)?;
                for (sum, value) in coefficients.iter_mut().zip(partial) {
                    *sum += value;
                }
            }
            let remaining = self.active_len.div_ceil(2) - pairs;
            if remaining != 0 {
                let partial = self.round_window(&self.output, pairs * 2, 0, pairs, remaining)?;
                for (sum, value) in coefficients.iter_mut().zip(partial) {
                    *sum += value;
                }
            }
            Ok(coefficients)
        })();
        self.application_peak_bytes = self.application_peak_bytes.max(application.peak_bytes());
        self.application = Some(application);
        result
    }

    fn round_window(
        &self,
        application: &Buffer,
        application_offset: usize,
        application_stride: usize,
        pair_offset: usize,
        pair_count: usize,
    ) -> Result<Vec<K>, CudaError> {
        let base_round = self.round == 0;
        let assignments = self
            .assignments
            .as_ref()
            .expect("assignment prefixes exist during SumCheck");
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
        let shape = [
            self.current_len as u64,
            self.fresh_count as u64,
            self.matrix_count as u64,
            self.assignment_count as u64,
            self.coefficient_count as u64,
            (self.term_headers.length() / (3 * size_of::<u64>())) as u64,
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
            self.range_base as u64,
            u64::from(self.f_vanishes_at_zero),
            assignments.encoding() as u64,
            pair_offset as u64,
            pair_count as u64,
            application_offset as u64,
            application_stride as u64,
        ];
        let groups = pair_count.div_ceil(SUMCHECK_REDUCTION_THREADS).max(1);
        let partials = self
            .session
            .buffer(groups * self.coefficient_count * size_of::<K>())?;
        let common = self
            .common
            .as_ref()
            .expect("one-joint common tables exist during SumCheck");
        let kernels = &self.session.kernels;
        self.session
            .launch(Grid::groups(groups, SUMCHECK_REDUCTION_THREADS), |stream, config| {
                kernels.joint.joint_round_partials(
                    stream,
                    config,
                    application.ptr(),
                    assignments.data(&self.masks).ptr(),
                    common.ptr(),
                    shape,
                    self.weights.ptr(),
                    self.term_headers.ptr(),
                    self.term_variables.ptr(),
                    partials.ptr(),
                    self.equality_chunks.ptr(),
                    self.prior_equality_chunks.ptr(),
                    assignments.sources.ptr(),
                    assignments.values().ptr(),
                )
            })?;
        self.session
            .launch(Grid::Threads(self.coefficient_count), |stream, config| {
                kernels.nightstream.sumcheck_reduce_partials(
                    stream,
                    config,
                    partials.ptr(),
                    self.output.ptr(),
                    groups as u64,
                    self.coefficient_count as u64,
                )
            })?;
        self.session.finish()?;
        Ok(self
            .session
            .read_buffer::<u64>(&self.output, self.coefficient_count * 2)?
            .chunks_exact(2)
            .map(|words| K::from_coeffs([F::from_u64(words[0]), F::from_u64(words[1])]))
            .collect())
    }

    fn fold_tables(&mut self, challenge: K) -> Result<(), CudaError> {
        let half = self.current_len / 2;
        let application_next_len = self.application_len.div_ceil(2).max(1);
        let assignment_next_len = self.assignment_len.div_ceil(2).max(1);
        let common_scratch = if self.prior_point.is_some() {
            self.common_len.div_ceil(2) * size_of::<K>() + 2 * size_of::<u64>()
        } else {
            0
        };
        let assignment_scratch = self
            .assignments
            .as_ref()
            .expect("assignment prefixes exist")
            .fold_allocation_bytes()?;
        let application = self
            .application
            .as_mut()
            .expect("application prefixes exist");
        application.fold(
            self.session,
            challenge,
            common_scratch.max(assignment_scratch) + 4 * size_of::<u64>(),
        )?;
        self.application_peak_bytes = self.application_peak_bytes.max(application.peak_bytes());
        if self.prior_point.is_some() {
            let next_len = self.common_len.div_ceil(2).max(1);
            let output = self.session.buffer(next_len * size_of::<K>())?;
            let common = self
                .common
                .as_ref()
                .expect("one-joint carried table exists during SumCheck");
            self.session
                .fold_tables(common, &output, challenge, self.common_len, 1, false)?;
            self.session.finish()?;
            self.common = Some(output);
            self.common_len = next_len;
        }
        self.assignments
            .as_mut()
            .expect("assignment prefixes exist during SumCheck")
            .fold(self.session, &self.masks, challenge)?;
        self.alpha_prefix = restrict_equality_prefix(self.alpha_prefix, self.alpha_point[self.round], challenge);
        if let Some(prior_point) = &self.prior_point {
            self.prior_prefix = restrict_equality_prefix(self.prior_prefix, prior_point[self.round], challenge);
        }
        self.current_len = half;
        self.application_len = application_next_len;
        self.assignment_len = assignment_next_len;
        self.active_len = self.application_len.max(self.assignment_len);
        self.round += 1;
        Ok(())
    }
}

impl PaperJointRoundOracle for CudaPaperJointOracle<'_> {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, neo_reductions::PiCcsError> {
        let coefficients = self.round_coefficients().map_err(oracle_error)?;
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
        self.fold_tables(challenge).map_err(oracle_error)
    }

    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<V1_2Evaluations<K>>>, neo_reductions::PiCcsError> {
        self.application = None;
        self.assignments = None;
        self.common = None;
        let sources = JointSources {
            masks: &self.masks,
            dense: self.dense.as_ref(),
        };
        self.session
            .eval_streamed_joint_openings(
                &self.plan,
                sources,
                point,
                self.opening_assignment_count,
                self.assignment_width,
            )
            .map(Some)
            .map_err(oracle_error)
    }
}
