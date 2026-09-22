//! The normal CPU PiCCS evaluator over actual witnesses and compact matrices.

use neo_ccs::{CcsStructure, CcsWitness, Mat};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[path = "application.rs"]
mod application;

use super::prefix::{self, Assignment};
use crate::engines::pi_ccs_joint::{gamma_power, range_product, JointDims};
use crate::engines::pi_ccs_joint_protocol::{PaperJointRoundOracle, V1_1OutputOpening};
use crate::engines::pi_ccs_protocol::Challenges;
use crate::superneo_eval::{fill_combined_projection, EqualityWeights, MatrixRows, SuperneoZBlocks};
use crate::PiCcsError;
use application::ApplicationTables;

// The production call uses offset zero. An offset preserves absolute tensor
// indices for a separately stored contiguous input range.
fn norm_coefficients(
    assignments: &[Assignment<'_>],
    gamma: K,
    weights: &EqualityWeights,
    pair_offset: usize,
) -> [K; 4] {
    let mut result = [K::ZERO; 4];
    for (source, table) in assignments.iter().enumerate() {
        let term = |index| {
            let (low, high) = table.pair(index);
            prefix::norm_pair(low, high).map(|value| value * weights.at(pair_offset + index))
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let coefficients = (0..table.len().div_ceil(2))
            .into_par_iter()
            .map(term)
            .reduce(
                || [K::ZERO; 4],
                |left, right| std::array::from_fn(|index| left[index] + right[index]),
            );
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let coefficients = (0..table.len().div_ceil(2))
            .map(term)
            .fold([K::ZERO; 4], |left, right| {
                std::array::from_fn(|index| left[index] + right[index])
            });
        let weight = gamma_power(gamma, source);
        for index in 0..4 {
            result[index] += weight * coefficients[index];
        }
    }
    result
}

pub struct OptimizedPaperJointOracle<'a> {
    structure: &'a CcsStructure<F>,
    source: &'a dyn MatrixRows,
    workspace_bytes: usize,
    base: u32,
    challenges: Challenges,
    dims: JointDims,
    point: Vec<K>,
    fixed_equality: K,
    prior_point: Option<Vec<K>>,
    fixed_prior_equality: K,
    fresh_tables: ApplicationTables,
    assignments: Vec<Assignment<'a>>,
    witness_blocks: Vec<SuperneoZBlocks>,
    evaluation_table: Vec<K>,
    constraint_shift: K,
}

impl<'a> OptimizedPaperJointOracle<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        structure: &'a CcsStructure<F>,
        params: &neo_params::NeoParams,
        fresh: &'a [CcsWitness<F>],
        running: &'a [Mat<F>],
        challenges: Challenges,
        prior_point: Option<&[K]>,
        dims: JointDims,
        source: &'a dyn MatrixRows,
        workspace_bytes: usize,
    ) -> Result<Self, PiCcsError> {
        let shape = source.shape();
        if !challenges.has_expected_dimension(dims.variables)
            || running.is_empty() != prior_point.is_none()
            || prior_point.is_some_and(|point| point.len() != dims.variables)
            || (shape.rows, shape.columns, shape.matrices) != (structure.n, dims.assignment_width, structure.t())
            || structure.f.eval(&vec![F::ZERO; structure.t()]) != F::ZERO
        {
            return Err(PiCcsError::InvalidInput("optimized CPU oracle input shape".into()));
        }
        let witnesses = fresh
            .iter()
            .map(|source| &source.Z)
            .chain(running)
            .collect::<Vec<_>>();
        let witness_blocks = witnesses
            .iter()
            .map(|source| SuperneoZBlocks::from_witness_mat(*source, structure.m))
            .collect::<Result<Vec<_>, _>>()?;
        let assignments = witnesses
            .iter()
            .map(|source| Assignment::new(source, dims.assignment_width))
            .collect::<Vec<_>>();
        let fresh_tables = ApplicationTables::new(source, &witness_blocks[..fresh.len()], workspace_bytes)?;
        let evaluation_table = carried_table(
            source,
            &witness_blocks[fresh.len()..],
            challenges.gamma,
            dims,
            structure.n,
            workspace_bytes,
        )?;
        Ok(Self {
            structure,
            source,
            workspace_bytes,
            base: params.b,
            challenges: challenges.clone(),
            dims,
            point: Vec::with_capacity(dims.variables),
            fixed_equality: K::ONE,
            prior_point: prior_point.map(<[K]>::to_vec),
            fixed_prior_equality: K::ONE,
            fresh_tables,
            assignments,
            witness_blocks,
            evaluation_table,
            constraint_shift: gamma_power(challenges.gamma, running.len() * D * (structure.t() + 1)),
        })
    }

    fn matrix_coefficients(&mut self, points: &[K], weights: &EqualityWeights) -> Result<Vec<K>, PiCcsError> {
        self.fresh_tables.evals_at(
            self.source,
            &self.witness_blocks[..self.fresh_tables.fresh_count()],
            &self.structure.f,
            self.challenges.gamma,
            points,
            weights,
        )
    }

    fn norm_coefficients(&self, weights: &EqualityWeights) -> [K; 4] {
        norm_coefficients(&self.assignments, self.challenges.gamma, weights, 0)
    }

    fn general_norm(&self, weights: &EqualityWeights, point: K) -> K {
        self.assignments
            .iter()
            .enumerate()
            .map(|(source, table)| {
                gamma_power(self.challenges.gamma, source)
                    * (0..table.len().div_ceil(2))
                        .map(|index| {
                            let (low, high) = table.pair(index);
                            weights.at(index) * range_product::<F>(prefix::interpolate(low, high, point), self.base)
                        })
                        .sum::<K>()
            })
            .sum()
    }

    fn carried_coefficients(&self, weights: &EqualityWeights) -> [K; 2] {
        let term = |index| {
            let (low, high) = prefix::pair(&self.evaluation_table, index);
            let weight = weights.at(index);
            [weight * low, weight * (high - low)]
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            (0..self.evaluation_table.len().div_ceil(2))
                .into_par_iter()
                .map(term)
                .reduce(|| [K::ZERO; 2], |left, right| [left[0] + right[0], left[1] + right[1]])
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            (0..self.evaluation_table.len().div_ceil(2))
                .map(term)
                .fold([K::ZERO; 2], |left, right| [left[0] + right[0], left[1] + right[1]])
        }
    }
}

impl PaperJointRoundOracle for OptimizedPaperJointOracle<'_> {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, PiCcsError> {
        let round = self.point.len();
        if round >= self.dims.variables {
            return Err(PiCcsError::InvalidInput("completed optimized CPU oracle".into()));
        }
        let weights = EqualityWeights::new(&self.challenges.alpha[round + 1..]);
        let matrix = self.matrix_coefficients(points, &weights)?;
        let norm = if self.base == 2 {
            self.norm_coefficients(&weights)
        } else {
            [K::ZERO; 4]
        };
        let carried = self
            .prior_point
            .as_ref()
            .map(|prior| self.carried_coefficients(&EqualityWeights::new(&prior[round + 1..])))
            .unwrap_or([K::ZERO; 2]);
        Ok(points
            .iter()
            .zip(matrix)
            .map(|(&point, matrix)| {
                let norm = if self.base == 2 {
                    norm.iter()
                        .rev()
                        .fold(K::ZERO, |value, &coefficient| value * point + coefficient)
                } else {
                    self.general_norm(&weights, point)
                };
                let alpha = self.challenges.alpha[round];
                let equality = (K::ONE - point) * (K::ONE - alpha) + point * alpha;
                let prior_equality = self.prior_point.as_ref().map_or(K::ZERO, |prior| {
                    (K::ONE - point) * (K::ONE - prior[round]) + point * prior[round]
                });
                self.fixed_prior_equality * prior_equality * (carried[0] + point * carried[1])
                    + self.constraint_shift
                        * self.fixed_equality
                        * equality
                        * (matrix + gamma_power(self.challenges.gamma, self.fresh_tables.fresh_count()) * norm)
            })
            .collect())
    }
    fn num_rounds(&self) -> usize {
        self.dims.variables
    }
    fn degree_bound(&self) -> usize {
        self.dims.degree
    }
    fn fold(&mut self, challenge: K) -> Result<(), PiCcsError> {
        let round = self.point.len();
        if round >= self.dims.variables {
            return Err(PiCcsError::InvalidInput("completed optimized CPU oracle".into()));
        }
        for table in &mut self.assignments {
            table.fold(challenge);
        }
        self.fresh_tables.fold(challenge);
        prefix::fold(&mut self.evaluation_table, challenge);
        let alpha = self.challenges.alpha[round];
        self.fixed_equality *= (K::ONE - challenge) * (K::ONE - alpha) + challenge * alpha;
        if let Some(prior) = &self.prior_point {
            self.fixed_prior_equality *= (K::ONE - challenge) * (K::ONE - prior[round]) + challenge * prior[round];
        }
        self.point.push(challenge);
        Ok(())
    }

    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<V1_1OutputOpening>>, PiCcsError> {
        if self.point.len() != self.dims.variables || point != self.point {
            return Err(PiCcsError::InvalidInput(
                "optimized CPU opening point is not the completed point".into(),
            ));
        }
        // The completed SumCheck tables are not inputs to witness openings.
        self.assignments.clear();
        self.fresh_tables.clear();
        let storage = core::mem::take(&mut self.evaluation_table);
        crate::superneo_eval::eval_real_v1_1_openings_from_rows_reusing(
            self.source,
            point,
            &self.witness_blocks,
            self.workspace_bytes,
            storage,
        )
        .map(Some)
    }
}

fn carried_table(
    source: &dyn MatrixRows,
    running: &[SuperneoZBlocks],
    gamma: K,
    dims: JointDims,
    rows: usize,
    workspace_bytes: usize,
) -> Result<Vec<K>, PiCcsError> {
    if running.iter().all(SuperneoZBlocks::is_zero) {
        return Ok(Vec::new());
    }
    let powers = (0..running.len())
        .map(|source| gamma_power(gamma, source))
        .collect::<Vec<_>>();
    let matrix_weights =
        std::array::from_fn(|coefficient| gamma_power(gamma, running.len() * dims.matrix_count * coefficient));
    let matrix_coefficients = (0..dims.matrix_count)
        .map(|matrix| gamma_power(gamma, running.len() * matrix))
        .collect::<Vec<_>>();
    let width = dims.assignment_width;
    let mut result = vec![K::ZERO; width.max(rows)];
    fill_combined_projection(running, &powers, &matrix_weights, &mut result[..width]);
    let mut matrix = vec![K::ZERO; rows];
    crate::superneo_eval::fill_weighted_rows_from_source(
        source,
        &result[..width],
        &matrix_coefficients,
        &mut matrix,
        workspace_bytes,
    )?;
    let pad_weights = std::array::from_fn(|coefficient| gamma_power(gamma, running.len() * coefficient));
    fill_combined_projection(running, &powers, &pad_weights, &mut result[..width]);
    let matrix_shift = gamma_power(gamma, running.len() * D);
    for (output, value) in result.iter_mut().zip(matrix) {
        *output += matrix_shift * value;
    }
    while result.last().is_some_and(|value| *value == K::ZERO) {
        result.pop();
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../../../tests/unit/piccs_first_round_norm.rs"]
mod first_round_norm_tests;

#[cfg(test)]
#[path = "../../../tests/unit/carried_table_storage.rs"]
mod carried_storage_tests;
