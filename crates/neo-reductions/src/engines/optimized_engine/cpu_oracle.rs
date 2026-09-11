//! The normal CPU PiCCS evaluator over actual witnesses and compact matrices.

use neo_ccs::{CcsStructure, CcsWitness, Mat};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::prefix::{self, Assignment};
use crate::engines::pi_ccs_joint::{gamma_power, range_product, JointDims};
use crate::engines::pi_ccs_joint_protocol::{PaperJointRoundOracle, V1_1OutputOpening};
use crate::engines::pi_ccs_protocol::Challenges;
use crate::superneo_eval::{weighted_identity_projection, EqualityWeights, SuperneoEvalCache, SuperneoZBlocks};
use crate::PiCcsError;

pub struct OptimizedPaperJointOracle<'a> {
    structure: &'a CcsStructure<F>,
    cache: &'a SuperneoEvalCache,
    base: u32,
    challenges: Challenges,
    dims: JointDims,
    point: Vec<K>,
    fixed_equality: K,
    prior_point: Option<Vec<K>>,
    fixed_prior_equality: K,
    fresh_tables: Vec<Vec<Vec<K>>>,
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
        cache: &'a super::OptimizedStructureCache,
    ) -> Result<Self, PiCcsError> {
        Self::from_rows(
            structure,
            params,
            fresh,
            running,
            challenges,
            prior_point,
            dims,
            cache.superneo(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_rows(
        structure: &'a CcsStructure<F>,
        params: &neo_params::NeoParams,
        fresh: &'a [CcsWitness<F>],
        running: &'a [Mat<F>],
        challenges: Challenges,
        prior_point: Option<&[K]>,
        dims: JointDims,
        cache: &'a SuperneoEvalCache,
    ) -> Result<Self, PiCcsError> {
        if !challenges.has_expected_dimension(dims.variables)
            || running.is_empty() != prior_point.is_none()
            || prior_point.is_some_and(|point| point.len() != dims.variables)
            || cache.relation_shape() != Some((structure.n, dims.assignment_width, structure.t()))
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
        let mut fresh_tables = Vec::with_capacity(fresh.len());
        for blocks in witness_blocks.iter().take(fresh.len()) {
            let table = |matrix: &crate::superneo_eval::SuperneoMatrixCache| {
                if matrix.compact_device_parts().is_some_and(|parts| {
                    !parts.identity && parts.row_blocks.is_empty() && parts.geometric_runs.is_empty()
                }) && !matrix.has_compact_seeded_phi81_blocks()
                {
                    return Vec::new();
                }
                let mut values = vec![F::ZERO; structure.n];
                matrix.fill_row_dots_base_with_blocks(&mut values, blocks);
                while values.last().is_some_and(|value| *value == F::ZERO) {
                    values.pop();
                }
                values.into_iter().map(K::from).collect()
            };
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let tables = cache.matrix_caches().par_iter().map(table).collect();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let tables = cache.matrix_caches().iter().map(table).collect();
            fresh_tables.push(tables);
        }
        let evaluation_table = carried_table(cache, &assignments[fresh.len()..], challenges.gamma, dims, structure.n);
        Ok(Self {
            structure,
            cache,
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

    fn matrix_coefficients(&self, points: &[K], weights: &EqualityWeights) -> Vec<K> {
        let pairs = self
            .fresh_tables
            .iter()
            .flatten()
            .map(|table| table.len().div_ceil(2))
            .max()
            .unwrap_or(0);
        let evaluate = |mut state: (Vec<K>, Vec<K>), index| {
            let weight = weights.at(index);
            for (output, &point) in state.0.iter_mut().zip(points) {
                for (source, tables) in self.fresh_tables.iter().enumerate() {
                    for (value, table) in state.1.iter_mut().zip(tables) {
                        let (low, high) = prefix::pair(table, index);
                        *value = prefix::interpolate(low, high, point);
                    }
                    // Skip zero factors just as the sparse polynomial does
                    // algebraically; this is important for the active prefix.
                    let polynomial: K = self
                        .structure
                        .f
                        .terms()
                        .iter()
                        .map(|term| {
                            let mut value = K::from(term.coeff);
                            for (&coordinate, &exponent) in state.1.iter().zip(&term.exps) {
                                if exponent == 0 {
                                    continue;
                                }
                                if coordinate == K::ZERO {
                                    return K::ZERO;
                                }
                                for _ in 0..exponent {
                                    value *= coordinate;
                                }
                            }
                            value
                        })
                        .sum();
                    *output += weight * gamma_power(self.challenges.gamma, source) * polynomial;
                }
            }
            state
        };
        let initial = || (vec![K::ZERO; points.len()], vec![K::ZERO; self.structure.t()]);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            (0..pairs)
                .into_par_iter()
                .fold(initial, evaluate)
                .map(|state| state.0)
                .reduce(
                    || vec![K::ZERO; points.len()],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                )
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            (0..pairs).fold(initial(), evaluate).0
        }
    }

    fn norm_coefficients(&self, weights: &EqualityWeights) -> [K; 4] {
        let mut result = [K::ZERO; 4];
        for (source, table) in self.assignments.iter().enumerate() {
            let term = |index| {
                let (low, high) = table.pair(index);
                prefix::norm_pair(low, high).map(|value| value * weights.at(index))
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
            let weight = gamma_power(self.challenges.gamma, source);
            for index in 0..4 {
                result[index] += weight * coefficients[index];
            }
        }
        result
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
        let matrix = self.matrix_coefficients(points, &weights);
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
                        * (matrix + gamma_power(self.challenges.gamma, self.fresh_tables.len()) * norm)
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
        for table in self.fresh_tables.iter_mut().flatten() {
            prefix::fold(table, challenge);
        }
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
        self.cache.eval_real_v1_1_openings(point, &self.witness_blocks).map(Some)
    }
}

fn carried_table(
    cache: &SuperneoEvalCache,
    running: &[Assignment<'_>],
    gamma: K,
    dims: JointDims,
    rows: usize,
) -> Vec<K> {
    if running.iter().all(|source| source.len() == 0) {
        return Vec::new();
    }
    let powers = (0..running.len())
        .map(|source| gamma_power(gamma, source))
        .collect::<Vec<_>>();
    let coefficient = |index| {
        running
            .iter()
            .zip(&powers)
            .map(|(source, &weight)| weight * source.get(index))
            .sum::<K>()
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let combined = (0..dims.assignment_width)
        .into_par_iter()
        .map(coefficient)
        .collect::<Vec<_>>();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let combined = (0..dims.assignment_width)
        .map(coefficient)
        .collect::<Vec<_>>();
    let blocks = SuperneoZBlocks::from_z(&combined);
    drop(combined);
    let matrix_weights =
        std::array::from_fn(|coefficient| gamma_power(gamma, running.len() * dims.matrix_count * coefficient));
    let matrix_coefficients = (0..dims.matrix_count)
        .map(|matrix| gamma_power(gamma, running.len() * matrix))
        .collect::<Vec<_>>();
    let matrix = cache.eval_weighted_row_table(&blocks, &matrix_weights, &matrix_coefficients, rows, rows);
    let pad_weights = std::array::from_fn(|coefficient| gamma_power(gamma, running.len() * coefficient));
    let mut result = weighted_identity_projection(&blocks, &pad_weights);
    result.resize(result.len().max(rows), K::ZERO);
    let matrix_shift = gamma_power(gamma, running.len() * D);
    for (output, value) in result.iter_mut().zip(matrix) {
        *output += matrix_shift * value;
    }
    while result.last().is_some_and(|value| *value == K::ZERO) {
        result.pop();
    }
    result
}
