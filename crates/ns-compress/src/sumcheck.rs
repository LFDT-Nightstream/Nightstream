//! The sum-checks of SPEC.md section 6.

use neo_math::K;
use neo_reductions::sumcheck::poly_eval_k;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::transcript::{absorb_extension, challenge};
use crate::Error;

pub type Round = [K; 4];
pub type KeyRound = [K; 3];

const PAIRS_PER_TASK: usize = 1 << 12;

pub(crate) struct Tables {
    pub(crate) witness: Vec<K>,
    pub(crate) weights: Vec<K>,
    pub(crate) check: Vec<K>,
    pub(crate) sum: Vec<K>,
    pub(crate) inverse: Vec<K>,
    pub(crate) scaled_inverse: Vec<K>,
}

pub(crate) struct ZeroPrefix<'a> {
    pub(crate) beta: &'a [K],
    pub(crate) xi: K,
    pub(crate) lambda: K,
    pub(crate) inverse: K,
    pub(crate) scaled: K,
}

#[derive(Clone, Copy)]
struct Line {
    zero: K,
    two: K,
    slope: K,
}

fn line(table: &[K], pair: usize) -> Line {
    let at = |index: usize| table.get(index).copied().unwrap_or(K::ZERO);
    let (zero, one) = (at(2 * pair), at(2 * pair + 1));
    let slope = one - zero;
    Line {
        zero,
        two: one + slope,
        slope,
    }
}

fn bind_in_place(table: &mut Vec<K>, challenge: K) {
    let line = |low: K, high: K| low + (high - low) * challenge;
    let length = table.len();
    if length == 0 {
        return;
    }
    table[0] = line(table[0], table.get(1).copied().unwrap_or(K::ZERO));
    let mut first = 1;
    while 2 * first < length {
        let (written, read) = table.split_at_mut(2 * first);
        let read = &read[..read.len().min(2 * first)];
        written[first..]
            .par_iter_mut()
            .with_min_len(PAIRS_PER_TASK)
            .zip(read.par_chunks(2))
            .for_each(|(cell, pair)| *cell = line(pair[0], pair.get(1).copied().unwrap_or(K::ZERO)));
        first *= 2;
    }
    table.truncate(length.div_ceil(2));
}

impl Tables {
    fn round(&self) -> [K; 3] {
        let pairs = self.witness.len().max(self.weights.len()).div_ceil(2);
        let lookup_pairs = self.check.len().div_ceil(2);
        (0..pairs.div_ceil(PAIRS_PER_TASK))
            .into_par_iter()
            .map(|task| {
                let mut sums = [K::ZERO; 3];
                for pair in task * PAIRS_PER_TASK..((task + 1) * PAIRS_PER_TASK).min(pairs) {
                    let (witness, weights) = (line(&self.witness, pair), line(&self.weights, pair));
                    if pair >= lookup_pairs {
                        sums[0] += witness.zero * weights.zero;
                        sums[1] += witness.two * weights.two;
                        continue;
                    }
                    let (check, sum) = (line(&self.check, pair), line(&self.sum, pair));
                    let (inverse, scaled) = (line(&self.inverse, pair), line(&self.scaled_inverse, pair));
                    sums[0] += witness.zero * (weights.zero - check.zero * inverse.zero)
                        + check.zero * scaled.zero
                        + sum.zero * inverse.zero;
                    sums[1] += witness.two * (weights.two - check.two * inverse.two)
                        + check.two * scaled.two
                        + sum.two * inverse.two;
                    sums[2] -= witness.slope * check.slope * inverse.slope;
                }
                sums
            })
            .reduce(
                || [K::ZERO; 3],
                |left, right| std::array::from_fn(|point| left[point] + right[point]),
            )
    }

    fn bind(&mut self, challenge: K) {
        [
            &mut self.witness,
            &mut self.weights,
            &mut self.check,
            &mut self.sum,
            &mut self.inverse,
            &mut self.scaled_inverse,
        ]
        .into_par_iter()
        .for_each(|table| bind_in_place(table, challenge));
    }
}

fn coefficients(zero: K, one: K, two: K, cubic: K) -> Round {
    let sum = one - zero - cubic;
    let quadratic = (two - zero - cubic.double().double().double() - sum.double()).halve();
    [zero, sum - quadratic, quadratic, cubic]
}

fn prove_round(transcript: &mut Poseidon2Transcript, [zero, two, cubic]: [K; 3], claim: &mut K) -> (Round, K) {
    let coefficients = coefficients(zero, *claim - zero, two, cubic);
    absorb_extension(transcript, &coefficients);
    let challenge = challenge(transcript);
    *claim = poly_eval_k(&coefficients, challenge);
    (coefficients, challenge)
}

pub(crate) fn prove(
    transcript: &mut Poseidon2Transcript,
    mut tables: Tables,
    mut claim: K,
    rounds: usize,
) -> (Vec<Round>, Vec<K>, K) {
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let (coefficients, challenge) = prove_round(transcript, tables.round(), &mut claim);
        tables.bind(challenge);
        messages.push(coefficients);
        point.push(challenge);
    }
    let witness = tables.witness.first().copied().unwrap_or(K::ZERO);
    (messages, point, witness)
}

pub(crate) fn prove_with_zero_prefix(
    transcript: &mut Poseidon2Transcript,
    mut tables: Tables,
    mut claim: K,
    rounds: usize,
    lookup: ZeroPrefix<'_>,
    weight_at: impl FnOnce(&[K]) -> Result<K, Error>,
) -> Result<(Vec<Round>, Vec<K>, K), Error> {
    let vars = lookup.beta.len();
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut check = lookup.xi;
    for (bit, &beta) in lookup.beta.iter().enumerate() {
        let mut sums = tables.round();
        let sum = lookup.lambda * lookup.inverse * K::from_u64(1 << (vars - bit - 1));
        let scaled = check * lookup.scaled;
        sums[0] += scaled * (K::ONE - beta) + sum;
        sums[1] += scaled * (beta.double() - (K::ONE - beta)) + sum;
        let (coefficients, challenge) = prove_round(transcript, sums, &mut claim);
        check *= (K::ONE - beta) * (K::ONE - challenge) + beta * challenge;
        tables.bind(challenge);
        messages.push(coefficients);
        point.push(challenge);
    }
    tables.witness.insert(0, K::ZERO);
    tables.weights.insert(0, weight_at(&point)?);
    tables.check.push(check);
    tables.sum.push(lookup.lambda);
    tables.inverse.push(lookup.inverse);
    tables.scaled_inverse.push(lookup.scaled);
    let (later, remaining, witness) = prove(transcript, tables, claim, rounds - vars);
    messages.extend(later);
    point.extend(remaining);
    Ok((messages, point, witness))
}

pub(crate) fn prove_product(
    transcript: &mut Poseidon2Transcript,
    mut values: Vec<K>,
    mut weights: Vec<K>,
    mut claim: K,
    rounds: usize,
) -> (Vec<KeyRound>, Vec<K>, K) {
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let pairs = values.len().min(weights.len()).div_ceil(2);
        let [zero, two] = (0..pairs)
            .into_par_iter()
            .with_min_len(PAIRS_PER_TASK)
            .map(|pair| {
                let (value, weight) = (line(&values, pair), line(&weights, pair));
                [value.zero * weight.zero, value.two * weight.two]
            })
            .reduce(|| [K::ZERO; 2], |left, right| [left[0] + right[0], left[1] + right[1]]);
        let sum = claim - zero.double();
        let quadratic = (two - zero - sum.double()).halve();
        let coefficients = [zero, sum - quadratic, quadratic];
        absorb_extension(transcript, &coefficients);
        let challenge = challenge(transcript);
        claim = poly_eval_k(&coefficients, challenge);
        rayon::join(
            || bind_in_place(&mut values, challenge),
            || bind_in_place(&mut weights, challenge),
        );
        messages.push(coefficients);
        point.push(challenge);
    }
    (messages, point, values.first().copied().unwrap_or(K::ZERO))
}

pub(crate) fn verify<const COEFFICIENTS: usize>(
    transcript: &mut Poseidon2Transcript,
    rounds: &[[K; COEFFICIENTS]],
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
