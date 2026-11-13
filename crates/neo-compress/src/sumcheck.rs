//! The sum-check of SPEC.md section 6 over the cells `x` of `W`:
//!
//! `Σ_x W·(G − E·A) + E·B + S·A`,
//!
//! where `G` is the public weights, `A = Σ_k ψ_k I_k` and `B = Σ_k ψ_k τ_k I_k`
//! combine the committed inverses `I_k`, `E` is the weight of the inverse
//! check and `S` the weight of the lookup sum. Each table is a prefix of its
//! evaluations, then zeros.

use neo_math::K;
use neo_reductions::sumcheck::{interpolate_from_evals, poly_eval_k};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::transcript::{absorb_extension, challenge};
use crate::Error;

/// The coefficients of one round's polynomial, of degree 3.
pub type Round = [K; 4];

const PAIRS_PER_TASK: usize = 1 << 12;

pub(crate) struct Tables {
    pub(crate) witness: Vec<K>,
    pub(crate) weights: Vec<K>,
    pub(crate) check: Vec<K>,
    pub(crate) sum: Vec<K>,
    pub(crate) inverse: Vec<K>,
    pub(crate) scaled_inverse: Vec<K>,
}

/// The values at 0 and 1 of the table at pair `pair`.
fn pair(table: &[K], pair: usize) -> (K, K) {
    let at = |index: usize| table.get(index).copied().unwrap_or(K::ZERO);
    (at(2 * pair), at(2 * pair + 1))
}

/// The table after its lowest variable takes the value `challenge`.
fn bind(table: &[K], challenge: K) -> Vec<K> {
    table
        .par_chunks(2)
        .with_min_len(PAIRS_PER_TASK)
        .map(|pair| {
            let high = pair.get(1).copied().unwrap_or(K::ZERO);
            pair[0] + (high - pair[0]) * challenge
        })
        .collect()
}

impl Tables {
    fn all(&self) -> [&Vec<K>; 6] {
        [
            &self.witness,
            &self.weights,
            &self.check,
            &self.sum,
            &self.inverse,
            &self.scaled_inverse,
        ]
    }

    /// The round polynomial at 0, 2 and 3.
    fn round(&self) -> [K; 3] {
        let pairs = self
            .all()
            .map(|table| table.len().div_ceil(2))
            .into_iter()
            .max()
            .unwrap_or(0);
        let lookup_pairs = self.check.len().div_ceil(2);
        (0..pairs.div_ceil(PAIRS_PER_TASK))
            .into_par_iter()
            .map(|task| {
                let mut sums = [K::ZERO; 3];
                for index in task * PAIRS_PER_TASK..((task + 1) * PAIRS_PER_TASK).min(pairs) {
                    let lines = if index < lookup_pairs {
                        self.all().map(|table| pair(table, index))
                    } else {
                        let zero = (K::ZERO, K::ZERO);
                        [
                            pair(&self.witness, index),
                            pair(&self.weights, index),
                            zero,
                            zero,
                            zero,
                            zero,
                        ]
                    };
                    for (sum, point) in sums.iter_mut().zip([0u8, 2, 3]) {
                        let [witness, weights, check, lookup_sum, inverse, scaled] = lines.map(|(low, high)| {
                            let step = high - low;
                            match point {
                                0 => low,
                                2 => high + step,
                                _ => high + step.double(),
                            }
                        });
                        *sum += witness * (weights - check * inverse) + check * scaled + lookup_sum * inverse;
                    }
                }
                sums
            })
            .reduce(
                || [K::ZERO; 3],
                |left, right| std::array::from_fn(|point| left[point] + right[point]),
            )
    }

    fn bind(&mut self, challenge: K) {
        for table in [
            &mut self.witness,
            &mut self.weights,
            &mut self.check,
            &mut self.sum,
            &mut self.inverse,
            &mut self.scaled_inverse,
        ] {
            *table = bind(table, challenge);
        }
    }
}

fn points() -> [K; 4] {
    [0u64, 1, 2, 3].map(K::from_u64)
}

/// Prove that the sum is `claim`, in `rounds` rounds. Returns the rounds, the
/// point and `W` at the point.
pub(crate) fn prove(
    transcript: &mut Poseidon2Transcript,
    mut tables: Tables,
    mut claim: K,
    rounds: usize,
) -> (Vec<Round>, Vec<K>, K) {
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let [zero, two, three] = tables.round();
        let coefficients: Round = interpolate_from_evals(&points(), &[zero, claim - zero, two, three])
            .try_into()
            .expect("four coefficients");
        absorb_extension(transcript, &coefficients);
        let challenge = challenge(transcript);
        claim = poly_eval_k(&coefficients, challenge);
        tables.bind(challenge);
        messages.push(coefficients);
        point.push(challenge);
    }
    let witness = tables.witness.first().copied().unwrap_or(K::ZERO);
    (messages, point, witness)
}

/// Check the rounds against `claim`. Returns the point and the value that
/// the summand must have at the point.
pub(crate) fn verify(
    transcript: &mut Poseidon2Transcript,
    rounds: &[Round],
    mut claim: K,
) -> Result<(Vec<K>, K), Error> {
    let mut point = Vec::with_capacity(rounds.len());
    for coefficients in rounds {
        if coefficients[0] + coefficients.iter().copied().sum::<K>() != claim {
            return Err(Error::Rejected("a sum-check round differs from its claim"));
        }
        absorb_extension(transcript, coefficients);
        let challenge = challenge(transcript);
        claim = poly_eval_k(coefficients, challenge);
        point.push(challenge);
    }
    Ok((point, claim))
}
