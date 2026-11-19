use neo_ccs::{CcsStructure, CcsWitness, Mat};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[path = "application.rs"]
mod application;

use super::prefix::{self, Assignment, PackedNorm, PACKED_NORM_ROUNDS};
use crate::engines::pi_ccs_joint::{gamma_power, range_product, JointDims};
use crate::engines::pi_ccs_joint_protocol::{PaperJointRoundOracle, V1_2OutputOpening};
use crate::engines::pi_ccs_protocol::Challenges;
use crate::superneo_eval::{
    fill_geometric_projection, EqualityWeights, MatrixRows, PadProjection, SuperneoZBlocks, LANE_BITS,
};
use crate::PiCcsError;
use application::ApplicationTables;

fn norm_coefficients(assignments: &[Assignment<'_>], gamma: K, weights: &EqualityWeights) -> [K; 4] {
    let mut result = [K::ZERO; 4];
    for (source, table) in assignments.iter().enumerate() {
        let pairs = table.len().div_ceil(2);
        let coefficients = match table.early_pairs() {
            Some(early) => prefix::early_norm_coefficients(&early, pairs, weights),
            None => pair_norm_coefficients(table, pairs, weights),
        };
        let weight = gamma_power(gamma, source);
        for index in 0..4 {
            result[index] += weight * coefficients[index];
        }
    }
    result
}

fn pair_norm_coefficients(table: &Assignment<'_>, pairs: usize, weights: &EqualityWeights) -> [K; 4] {
    let term = |index| {
        let (low, high) = table.pair(index);
        prefix::norm_pair(low, high).map(|value| value * weights.at(index))
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        (0..pairs)
            .into_par_iter()
            .with_min_len(prefix::MIN_JOB_PAIRS)
            .map(term)
            .reduce(
                || [K::ZERO; 4],
                |left, right| std::array::from_fn(|index| left[index] + right[index]),
            )
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        (0..pairs).map(term).fold([K::ZERO; 4], |left, right| {
            std::array::from_fn(|index| left[index] + right[index])
        })
    }
}

struct CarriedPad {
    projection: PadProjection,
    lanes: Vec<K>,
}

impl CarriedPad {
    fn new(witness: &SuperneoZBlocks, high_point: &[K], gamma: K) -> Self {
        let projection = PadProjection::new(gamma);
        let lanes = projection.contract_blocks(witness, high_point).to_vec();
        Self { projection, lanes }
    }
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
    carried_pad: Option<CarriedPad>,
    norm_polynomial: Option<[K; 4]>,
    packed_norm: Option<PackedNorm<'a>>,
    packed_norm_value: Option<K>,
    carried_polynomial: Option<[K; 2]>,
    constraint_shift: K,
}

impl<'a> OptimizedPaperJointOracle<'a> {
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
        norm_sources: usize,
    ) -> Result<Self, PiCcsError> {
        let shape = source.shape();
        if !challenges.has_expected_dimension(dims.variables)
            || running.is_empty() != prior_point.is_none()
            || prior_point.is_some_and(|point| point.len() != dims.variables)
            || norm_sources > fresh.len() + running.len()
            || (shape.rows, shape.columns, shape.matrices) != (structure.n(), dims.assignment_width, structure.t())
            || structure.f().eval(&vec![F::ZERO; structure.t()]) != F::ZERO
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
            .map(|source| SuperneoZBlocks::from_witness_mat(*source, structure.m()))
            .collect::<Result<Vec<_>, _>>()?;
        let assignments = witnesses[..norm_sources]
            .iter()
            .map(|source| Assignment::new(source, dims.assignment_width))
            .collect::<Vec<_>>();
        let fresh_tables = ApplicationTables::new(source, &witness_blocks[..fresh.len()], workspace_bytes)?;
        let running_blocks = &witness_blocks[fresh.len()..];
        let (evaluation_table, carried_pad) = match (running_blocks, prior_point) {
            ([witness], Some(prior))
                if dims.variables > LANE_BITS && structure.n() <= dims.assignment_width && !witness.is_zero() =>
            {
                let matrix =
                    single_carried_matrix(source, witness, challenges.gamma, dims, structure.n(), workspace_bytes)?;
                (
                    matrix,
                    Some(CarriedPad::new(witness, &prior[LANE_BITS..], challenges.gamma)),
                )
            }
            _ => (
                carried_table(
                    source,
                    running_blocks,
                    challenges.gamma,
                    dims,
                    structure.n(),
                    workspace_bytes,
                )?,
                None,
            ),
        };
        let packed_norm = match assignments.as_slice() {
            [assignment]
                if params.b() == 2 && dims.variables >= PACKED_NORM_ROUNDS && assignment.len() >= (1 << 16) =>
            {
                PackedNorm::new(witnesses[0], assignment.len(), &challenges.alpha)
            }
            _ => None,
        };
        Ok(Self {
            structure,
            source,
            workspace_bytes,
            base: params.b(),
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
            carried_pad,
            norm_polynomial: None,
            packed_norm,
            packed_norm_value: None,
            carried_polynomial: None,
            constraint_shift: gamma_power(challenges.gamma, running.len() * D * (structure.t() + 1)),
        })
    }

    fn matrix_coefficients(&mut self, points: &[K], weights: &EqualityWeights) -> Result<Vec<K>, PiCcsError> {
        self.fresh_tables.evals_at(
            self.source,
            &self.witness_blocks[..self.fresh_tables.fresh_count()],
            self.structure.f(),
            self.challenges.gamma,
            points,
            weights,
        )
    }

    fn norm_coefficients(&self, weights: &EqualityWeights) -> [K; 4] {
        norm_coefficients(&self.assignments, self.challenges.gamma, weights)
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

    fn carried_coefficients(&self, round: usize, prior: &[K]) -> [K; 2] {
        let weights = EqualityWeights::new(&prior[round + 1..]);
        let coefficients = |low, high| [low, high - low];
        let matrix = prefix::sum_products(&self.evaluation_table, &weights, coefficients);
        self.carried_pad.as_ref().map_or(matrix, |pad| {
            let weights = EqualityWeights::new(&prior[round + 1..LANE_BITS]);
            prefix::add_coefficients(matrix, prefix::sum_products(&pad.lanes, &weights, coefficients))
        })
    }

    fn bind_norm(&mut self, round: usize, challenge: K) {
        let ordinary = self.norm_polynomial.take();
        if self.packed_norm.is_some() {
            if round + 1 == PACKED_NORM_ROUNDS {
                let point = std::array::from_fn(|i| if i == round { challenge } else { self.point[i] });
                self.packed_norm_value =
                    ordinary.map(|polynomial| crate::sumcheck::poly_eval_k(&polynomial, challenge));
                self.assignments[0] = self
                    .packed_norm
                    .take()
                    .expect("packing before the third bind")
                    .into_assignment(&point);
            }
            return;
        }
        if self.packed_norm_value.take().is_some() && round + 1 < self.dims.variables {
            if let (Some(previous), [assignment]) = (ordinary, self.assignments.as_mut_slice()) {
                if let Assignment::Encoded { codes, values, zero } = assignment {
                    let weights = EqualityWeights::new(&self.challenges.alpha[round + 2..]);
                    let (dense, higher) = prefix::bind_encoded(codes, values, *zero, challenge, &weights);
                    *assignment = Assignment::Folded(dense);
                    self.norm_polynomial = Some(prefix::recover_weighted_constant(
                        crate::sumcheck::poly_eval_k(&previous, challenge),
                        self.challenges.alpha[round + 1],
                        higher,
                    ));
                    return;
                }
            }
        }
        if let Some(previous) = ordinary.filter(|_| {
            round + 1 < self.dims.variables
                && self
                    .assignments
                    .iter()
                    .all(|table| matches!(table, Assignment::Folded(_)))
        }) {
            let weights = EqualityWeights::new(&self.challenges.alpha[round + 2..]);
            let mut higher = [K::ZERO; 3];
            let mut weight = K::ONE;
            for table in &mut self.assignments {
                let Assignment::Folded(values) = table else {
                    unreachable!()
                };
                let contribution =
                    prefix::bind_sum_products(values, challenge, &weights, prefix::norm_higher_coefficients);
                higher = prefix::add_coefficients(higher, contribution.map(|value| value * weight));
                weight *= self.challenges.gamma;
            }
            self.norm_polynomial = Some(prefix::recover_weighted_constant(
                crate::sumcheck::poly_eval_k(&previous, challenge),
                self.challenges.alpha[round + 1],
                higher,
            ));
        } else {
            for table in &mut self.assignments {
                table.fold(challenge);
            }
        }
    }

    fn bind_carried(&mut self, round: usize, challenge: K) {
        let polynomial = self.carried_polynomial.take();
        if let Some(pad) = &mut self.carried_pad {
            prefix::fold(&mut self.evaluation_table, challenge);
            if round + 1 == LANE_BITS {
                let mut point = [K::ZERO; LANE_BITS];
                point[..round].copy_from_slice(&self.point);
                point[round] = challenge;
                let witness = &self.witness_blocks[self.fresh_tables.fresh_count()];
                self.evaluation_table.resize(witness.block_len(), K::ZERO);
                pad.projection
                    .add_contracted_lanes(witness, &point, &mut self.evaluation_table);
                self.carried_pad = None;
            } else {
                prefix::fold(&mut pad.lanes, challenge);
            }
            return;
        }
        if let (Some(previous), Some(prior)) = (polynomial, &self.prior_point) {
            if round + 1 < self.dims.variables {
                let weights = EqualityWeights::new(&prior[round + 2..]);
                let higher =
                    prefix::bind_sum_products(
                        &mut self.evaluation_table,
                        challenge,
                        &weights,
                        |low, high| [high - low],
                    );
                self.carried_polynomial = Some(prefix::recover_weighted_constant(
                    crate::sumcheck::poly_eval_k(&previous, challenge),
                    prior[round + 1],
                    higher,
                ));
                return;
            }
        }
        prefix::fold(&mut self.evaluation_table, challenge);
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
            let polynomial = self.norm_polynomial.unwrap_or_else(|| {
                if let Some(packed) = &self.packed_norm {
                    return packed.coefficients(&self.point, &self.challenges.alpha[round + 1..PACKED_NORM_ROUNDS]);
                }
                if let (Some(previous), [Assignment::Encoded { codes, values, zero }]) =
                    (self.packed_norm_value, self.assignments.as_slice())
                {
                    return prefix::encoded_moments(
                        codes,
                        values,
                        *zero,
                        &weights,
                        previous,
                        self.challenges.alpha[round],
                    );
                }
                self.norm_coefficients(&weights)
            });
            self.norm_polynomial = Some(polynomial);
            polynomial
        } else {
            [K::ZERO; 4]
        };
        let carried = if let Some(prior) = &self.prior_point {
            let polynomial = self
                .carried_polynomial
                .unwrap_or_else(|| self.carried_coefficients(round, prior));
            self.carried_polynomial = Some(polynomial);
            polynomial
        } else {
            [K::ZERO; 2]
        };
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
        self.bind_norm(round, challenge);
        if round + 1 < self.dims.variables {
            let weights = EqualityWeights::new(&self.challenges.alpha[round + 2..]);
            self.fresh_tables
                .fold_next_gate(challenge, self.structure.f(), &weights);
        } else {
            self.fresh_tables.fold(challenge);
        }
        self.bind_carried(round, challenge);
        let alpha = self.challenges.alpha[round];
        self.fixed_equality *= (K::ONE - challenge) * (K::ONE - alpha) + challenge * alpha;
        if let Some(prior) = &self.prior_point {
            self.fixed_prior_equality *= (K::ONE - challenge) * (K::ONE - prior[round]) + challenge * prior[round];
        }
        self.point.push(challenge);
        Ok(())
    }

    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<V1_2OutputOpening>>, PiCcsError> {
        if self.point.len() != self.dims.variables || point != self.point {
            return Err(PiCcsError::InvalidInput(
                "optimized CPU opening point is not the completed point".into(),
            ));
        }
        self.assignments.clear();
        let complete = self.fresh_tables.take_complete_window();
        self.fresh_tables.clear();
        let storage = core::mem::take(&mut self.evaluation_table);
        crate::superneo_eval::eval_real_v1_2_openings_from_rows_reusing(
            self.source,
            point,
            &self.witness_blocks,
            self.workspace_bytes,
            storage,
            complete,
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
    let (mut result, matrix) = carried_matrix(source, running, gamma, dims, rows, workspace_bytes, &powers)?;
    let pad_ratio = gamma_power(gamma, running.len());
    fill_geometric_projection(running, &powers, pad_ratio, &mut result[..dims.assignment_width]);
    for (output, value) in result.iter_mut().zip(matrix) {
        *output += value;
    }
    while result.last().is_some_and(|value| *value == K::ZERO) {
        result.pop();
    }
    Ok(result)
}

fn single_carried_matrix(
    source: &dyn MatrixRows,
    witness: &SuperneoZBlocks,
    gamma: K,
    dims: JointDims,
    rows: usize,
    workspace_bytes: usize,
) -> Result<Vec<K>, PiCcsError> {
    let ratio = gamma_power(gamma, dims.matrix_count);
    let factors = (0..dims.matrix_count)
        .map(|matrix| gamma_power(gamma, matrix))
        .collect::<Vec<_>>();
    if let Some(mut matrix) =
        crate::superneo_eval::geometric_weighted_rows_from_source(source, witness, ratio, &factors, workspace_bytes)?
    {
        let matrix_shift = gamma_power(gamma, D);
        for value in &mut matrix {
            *value *= matrix_shift;
        }
        return Ok(matrix);
    }
    carried_matrix(
        source,
        std::slice::from_ref(witness),
        gamma,
        dims,
        rows,
        workspace_bytes,
        &[K::ONE],
    )
    .map(|(_, matrix)| matrix)
}

fn carried_matrix(
    source: &dyn MatrixRows,
    running: &[SuperneoZBlocks],
    gamma: K,
    dims: JointDims,
    rows: usize,
    workspace_bytes: usize,
    powers: &[K],
) -> Result<(Vec<K>, Vec<K>), PiCcsError> {
    let matrix_ratio = gamma_power(gamma, running.len() * dims.matrix_count);
    let matrix_coefficients = (0..dims.matrix_count)
        .map(|matrix| gamma_power(gamma, running.len() * matrix))
        .collect::<Vec<_>>();
    let width = dims.assignment_width;
    let mut projection = vec![K::ZERO; width.max(rows)];
    fill_geometric_projection(running, powers, matrix_ratio, &mut projection[..width]);
    let mut matrix = vec![K::ZERO; rows];
    crate::superneo_eval::fill_weighted_rows_from_source(
        source,
        &projection[..width],
        &matrix_coefficients,
        &mut matrix,
        workspace_bytes,
    )?;
    let matrix_shift = gamma_power(gamma, running.len() * D);
    for value in &mut matrix {
        *value *= matrix_shift;
    }
    Ok((projection, matrix))
}

#[cfg(test)]
#[path = "../../../tests/unit/piccs_first_round_norm.rs"]
mod first_round_norm_tests;

#[cfg(test)]
#[path = "../../../tests/unit/carried_table_storage.rs"]
mod carried_storage_tests;

#[cfg(test)]
#[path = "../../../tests/unit/factored_oracle.rs"]
mod factored_oracle_tests;
