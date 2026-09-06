//! Completion sums of the fixed PiCCS polynomial and its running term.
//! The table prefixes have implicit zero suffixes. Every fold consumes
//! one low bit, including singleton and odd-length prefixes.

use std::time::Instant;

use neo_ccs::SparsePoly;
use neo_math::{F, K};
use neo_reductions::{
    engines::{pi_ccs_joint::gamma_power, pi_ccs_joint_protocol::PaperJointRoundOracle},
    PiCcsError,
};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

pub const MATRICES: usize = 14;
pub const LIVE_MATRICES: usize = MATRICES - 1;
pub const ROUNDS: usize = 28;

pub(super) struct EqualityWeights {
    low: Vec<K>,
    high: Vec<K>,
    low_bits: usize,
}

impl EqualityWeights {
    fn table(point: &[K]) -> Vec<K> {
        let mut values = vec![K::ONE];
        for &coordinate in point {
            let width = values.len();
            values.resize(width * 2, K::ZERO);
            for index in 0..width {
                let value = values[index];
                values[index] = value * (K::ONE - coordinate);
                values[index + width] = value * coordinate;
            }
        }
        values
    }

    pub(super) fn new(point: &[K]) -> Self {
        // Splitting the remaining coordinates evenly minimizes the two
        // stored tensor factors for this exact domain.
        let low_bits = point.len() / 2;
        Self {
            low: Self::table(&point[..low_bits]),
            high: Self::table(&point[low_bits..]),
            low_bits,
        }
    }

    pub(super) fn at(&self, index: usize) -> K {
        self.low[index & (self.low.len() - 1)] * self.high[index >> self.low_bits]
    }
}

struct PolynomialTerm {
    coefficient: K,
    factors: Vec<(usize, u32)>,
}

fn norm_pair(low: K, high: K) -> [K; 4] {
    let delta = high - low;
    let three = K::from(F::from_u64(3));
    let low_squared = low * low;
    let delta_squared = delta * delta;
    [
        low_squared * low - low,
        delta * (three * low_squared - K::ONE),
        three * low * delta_squared,
        delta_squared * delta,
    ]
}

fn field_norm_coefficients(values: &[K], weights: &EqualityWeights) -> [K; 4] {
    values
        .par_chunks(2)
        .enumerate()
        .map(|(index, pair)| {
            let weight = weights.at(index);
            norm_pair(pair[0], pair.get(1).copied().unwrap_or(K::ZERO)).map(|coefficient| coefficient * weight)
        })
        .reduce(
            || [K::ZERO; 4],
            |left, right| std::array::from_fn(|index| left[index] + right[index]),
        )
}

fn signed(value: u8) -> K {
    match value {
        0 => K::ZERO,
        1 => K::ONE,
        255 => -K::ONE,
        _ => panic!("running opening is not a signed unit"),
    }
}

// Keep initial bounded data in bytes. The first fold halves its length before
// extension-field storage is allocated; sources are folded one at a time.
enum RunningAssignment {
    Signed(Vec<u8>),
    Folded(Vec<K>),
}

impl RunningAssignment {
    fn new(mut values: Vec<u8>) -> Self {
        assert!(values.len() <= 1 << ROUNDS);
        assert!(
            values.iter().all(|value| matches!(value, 0 | 1 | 255)),
            "running opening is not a signed unit"
        );
        let Some(last) = values.iter().rposition(|&value| value != 0) else {
            return Self::Signed(Vec::new());
        };
        values.truncate(last + 1);
        Self::Signed(values)
    }

    fn coefficients(&self, weights: &EqualityWeights) -> [K; 4] {
        match self {
            Self::Folded(values) => field_norm_coefficients(values, weights),
            Self::Signed(values) => values
                .par_chunks(2)
                .enumerate()
                .map(|(index, pair)| {
                    let weight = weights.at(index);
                    norm_pair(signed(pair[0]), signed(pair.get(1).copied().unwrap_or(0)))
                        .map(|coefficient| coefficient * weight)
                })
                .reduce(
                    || [K::ZERO; 4],
                    |left, right| std::array::from_fn(|index| left[index] + right[index]),
                ),
        }
    }

    fn fold(&mut self, challenge: K) {
        let next = match self {
            Self::Signed(values) => values
                .par_chunks(2)
                .map(|pair| {
                    let low = signed(pair[0]);
                    low + challenge * (signed(pair.get(1).copied().unwrap_or(0)) - low)
                })
                .collect(),
            Self::Folded(values) => values
                .par_chunks(2)
                .map(|pair| pair[0] + challenge * (pair.get(1).copied().unwrap_or(K::ZERO) - pair[0]))
                .collect(),
        };
        *self = Self::Folded(next);
    }

    fn terminal(&self) -> K {
        match self {
            Self::Signed(_) => panic!("running assignment has not been folded"),
            Self::Folded(values) => {
                assert!(values.len() <= 1);
                values.first().copied().unwrap_or(K::ZERO)
            }
        }
    }
}

/// The fresh-source polynomial plus a combined multilinear running term.
/// Prefix construction and opening validity are separate responsibilities.
pub struct Oracle {
    values: Vec<K>,
    images: Vec<[K; LIVE_MATRICES]>,
    polynomial: Vec<PolynomialTerm>,
    alpha: Vec<K>,
    gamma: K,
    constraint_shift: K,
    fixed_equality: K,
    running: Vec<K>,
    prior_point: Vec<K>,
    fixed_prior_equality: K,
    norm_weight: K,
    running_assignments: [RunningAssignment; 16],
    round: usize,
}

impl Oracle {
    pub fn new(
        values: Vec<K>,
        images: Vec<[K; LIVE_MATRICES]>,
        polynomial: &SparsePoly<F>,
        alpha: Vec<K>,
        gamma: K,
    ) -> Self {
        assert_eq!(alpha.len(), ROUNDS);
        assert!(values.len() <= 1 << ROUNDS && images.len() <= 1 << ROUNDS);
        assert_eq!(polynomial.arity(), MATRICES);
        assert_eq!(polynomial.max_degree(), 8);
        assert_eq!(
            polynomial.eval(&[F::ZERO; MATRICES]),
            F::ZERO,
            "zero padding polynomial"
        );
        Self {
            values,
            images,
            polynomial: polynomial
                .terms()
                .iter()
                .map(|term| PolynomialTerm {
                    coefficient: K::from(term.coeff),
                    factors: term
                        .exps
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, exponent)| *exponent != 0)
                        .collect(),
                })
                .collect(),
            alpha,
            gamma,
            constraint_shift: gamma_power(gamma, 16 * 54 * (MATRICES + 1)),
            fixed_equality: K::ONE,
            running: Vec::new(),
            prior_point: vec![K::ZERO; ROUNDS],
            fixed_prior_equality: K::ONE,
            norm_weight: K::ONE,
            running_assignments: std::array::from_fn(|_| RunningAssignment::Signed(Vec::new())),
            round: 0,
        }
    }

    /// `running` is the combined Eval_K + gamma^(kd) Eval_A prefix before
    /// multiplication by eq(X, prior_point). For signed copies of the fresh
    /// opening, `norm_weight` includes the fresh source and all running signs.
    pub fn with_running(mut self, running: Vec<K>, prior_point: Vec<K>, norm_weight: K) -> Self {
        assert_eq!(prior_point.len(), ROUNDS);
        assert!(!running.is_empty() && running.len() <= 1 << ROUNDS);
        self.running = running;
        self.prior_point = prior_point;
        self.norm_weight = norm_weight;
        self
    }

    /// The sixteen native openings are indexed separately. Only their linear
    /// evaluation terms are combined in `running`; each cubic norm is retained.
    pub fn with_distinct_running(self, running: Vec<K>, prior_point: Vec<K>, sources: [Vec<u8>; 16]) -> Self {
        assert_eq!(self.round, 0);
        let mut result = self.with_running(running, prior_point, K::ONE);
        result.running_assignments = sources.map(RunningAssignment::new);
        result
    }

    fn running_coefficients(&self) -> [K; 2] {
        if self.running.is_empty() {
            return [K::ZERO; 2];
        }
        let weights = EqualityWeights::new(&self.prior_point[self.round + 1..]);
        self.running
            .par_chunks(2)
            .enumerate()
            .map(|(index, pair)| {
                let weight = weights.at(index);
                let low = pair[0];
                let delta = pair.get(1).copied().unwrap_or(K::ZERO) - low;
                [weight * low, weight * delta]
            })
            .reduce(|| [K::ZERO; 2], |left, right| [left[0] + right[0], left[1] + right[1]])
    }

    fn polynomial_value(&self, values: &[K; MATRICES]) -> K {
        self.polynomial
            .iter()
            .map(|term| {
                let mut value = term.coefficient;
                for &(matrix, exponent) in &term.factors {
                    if values[matrix] == K::ZERO {
                        return K::ZERO;
                    }
                    for _ in 0..exponent {
                        value *= values[matrix];
                    }
                }
                value
            })
            .sum()
    }

    fn norm_coefficients(&self, weights: &EqualityWeights) -> [K; 4] {
        let mut result = field_norm_coefficients(&self.values, weights).map(|value| value * self.norm_weight);
        let mut power = self.gamma;
        for source in &self.running_assignments {
            let coefficients = source.coefficients(weights);
            for index in 0..4 {
                result[index] += power * coefficients[index];
            }
            power *= self.gamma;
        }
        result
    }

    pub fn scalar_outputs(&self) -> (K, [K; MATRICES]) {
        assert_eq!(self.round, ROUNDS);
        assert_eq!(self.values.len(), 1);
        assert_eq!(self.images.len(), 1);
        (
            self.values[0],
            std::array::from_fn(|matrix| {
                if matrix < LIVE_MATRICES {
                    self.images[0][matrix]
                } else {
                    K::ZERO
                }
            }),
        )
    }

    pub fn terminal(&self) -> K {
        let (value, images) = self.scalar_outputs();
        let mut norm = self.norm_weight * (value * value * value - value);
        let mut power = self.gamma;
        for source in &self.running_assignments {
            let value = source.terminal();
            norm += power * (value * value * value - value);
            power *= self.gamma;
        }
        self.fixed_prior_equality * self.running.first().copied().unwrap_or(K::ZERO)
            + self.constraint_shift * self.fixed_equality * (self.polynomial_value(&images) + self.gamma * norm)
    }
}

impl PaperJointRoundOracle for Oracle {
    fn num_rounds(&self) -> usize {
        ROUNDS
    }

    fn degree_bound(&self) -> usize {
        9
    }

    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, PiCcsError> {
        if self.round >= ROUNDS {
            return Err(PiCcsError::InvalidInput("completed fixture oracle".into()));
        }
        let started = Instant::now();
        let running = self.running_coefficients();
        let weights = EqualityWeights::new(&self.alpha[self.round + 1..]);
        let norm = self.norm_coefficients(&weights);
        let norm_time = started.elapsed();
        let matrix_values = self
            .images
            .par_chunks(2)
            .enumerate()
            .fold(
                || vec![K::ZERO; points.len()],
                |mut sums, (index, pair)| {
                    let low = pair[0];
                    let high = pair.get(1).copied().unwrap_or([K::ZERO; LIVE_MATRICES]);
                    let delta: [K; LIVE_MATRICES] = std::array::from_fn(|matrix| high[matrix] - low[matrix]);
                    let weight = weights.at(index);
                    for (&point, sum) in points.iter().zip(&mut sums) {
                        let values = std::array::from_fn(|matrix| {
                            if matrix < LIVE_MATRICES {
                                low[matrix] + point * delta[matrix]
                            } else {
                                K::ZERO
                            }
                        });
                        *sum += weight * self.polynomial_value(&values);
                    }
                    sums
                },
            )
            .reduce(
                || vec![K::ZERO; points.len()],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            );
        let result = points
            .iter()
            .zip(matrix_values)
            .map(|(&point, matrix)| {
                let norm = norm
                    .iter()
                    .rev()
                    .fold(K::ZERO, |value, coefficient| value * point + *coefficient);
                let equality = (K::ONE - point) * (K::ONE - self.alpha[self.round]) + point * self.alpha[self.round];
                let prior_equality =
                    (K::ONE - point) * (K::ONE - self.prior_point[self.round]) + point * self.prior_point[self.round];
                self.fixed_prior_equality * prior_equality * (running[0] + point * running[1])
                    + self.constraint_shift * self.fixed_equality * equality * (matrix + self.gamma * norm)
            })
            .collect();
        println!(
            "honest round {}: norm={norm_time:?} total={:?}",
            self.round,
            started.elapsed()
        );
        Ok(result)
    }

    fn fold(&mut self, challenge: K) -> Result<(), PiCcsError> {
        for source in &mut self.running_assignments {
            source.fold(challenge);
        }
        self.running = self
            .running
            .par_chunks(2)
            .map(|pair| pair[0] + challenge * (pair.get(1).copied().unwrap_or(K::ZERO) - pair[0]))
            .collect();
        self.fixed_prior_equality *=
            (K::ONE - challenge) * (K::ONE - self.prior_point[self.round]) + challenge * self.prior_point[self.round];
        self.values = self
            .values
            .par_chunks(2)
            .map(|pair| pair[0] + challenge * (pair.get(1).copied().unwrap_or(K::ZERO) - pair[0]))
            .collect();
        self.images = self
            .images
            .par_chunks(2)
            .map(|pair| {
                let low = pair[0];
                let high = pair.get(1).copied().unwrap_or([K::ZERO; LIVE_MATRICES]);
                std::array::from_fn(|matrix| low[matrix] + challenge * (high[matrix] - low[matrix]))
            })
            .collect();
        self.fixed_equality *=
            (K::ONE - challenge) * (K::ONE - self.alpha[self.round]) + challenge * self.alpha[self.round];
        self.round += 1;
        Ok(())
    }
}
