//! The sum-checks of SPEC.md section 6.

pub(crate) mod lookup;
use std::ops::Range;

use neo_math::{ProductSum, D, F, K};
use neo_reductions::sumcheck::poly_eval_k;
use neo_transcript::Poseidon2Transcript;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::layout::LANE_VARS;
use crate::transcript::{absorb_extension, challenge};
use crate::Error;

pub type Round = [K; 7];
pub type KeyRound = [K; 3];

const PAIRS_PER_TASK: usize = 1 << 12;

trait Cell: PrimeCharacteristicRing + Copy {
    fn times(self, value: K) -> K;
    fn lift(self) -> K;
    fn add_to(self, sum: ProductSum, value: K) -> ProductSum;
}

impl Cell for F {
    fn times(self, value: K) -> K {
        value * self
    }

    fn lift(self) -> K {
        K::from(self)
    }

    fn add_to(self, sum: ProductSum, value: K) -> ProductSum {
        sum.add_scaled(self, value)
    }
}

impl Cell for K {
    fn times(self, value: K) -> K {
        value * self
    }

    fn lift(self) -> K {
        self
    }

    fn add_to(self, sum: ProductSum, value: K) -> ProductSum {
        sum.add(self, value)
    }
}

#[derive(Clone, Copy)]
struct Line<T> {
    zero: T,
    two: T,
    slope: T,
}

impl<T: Cell> Line<T> {
    fn new(zero: T, one: T) -> Self {
        let slope = one - zero;
        Self {
            zero,
            two: one + slope,
            slope,
        }
    }

    fn at(self, point: K) -> K {
        self.slope.times(point) + self.zero.lift()
    }
}

fn line<T: Cell>(table: &[T], pair: usize) -> Line<T> {
    let at = |index: usize| table.get(index).copied().unwrap_or(T::ZERO);
    Line::new(at(2 * pair), at(2 * pair + 1))
}

fn add(left: [K; 3], right: [K; 3]) -> [K; 3] {
    std::array::from_fn(|point| left[point] + right[point])
}

fn equality(left: K, right: K) -> K {
    (K::ONE - left) * (K::ONE - right) + left * right
}

fn bind_in_place(table: &mut Vec<K>, challenge: K) {
    let line = |low: K, high: K| low + (high - low) * challenge;
    let length = table.len();
    let mut first = length.div_ceil(2).min(PAIRS_PER_TASK);
    for cell in 0..first {
        table[cell] = line(table[2 * cell], table.get(2 * cell + 1).copied().unwrap_or(K::ZERO));
    }
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

fn sum_tasks(pairs: Range<usize>, sum: impl Fn(Range<usize>) -> [K; 3] + Sync) -> [K; 3] {
    (0..pairs.len().div_ceil(PAIRS_PER_TASK))
        .into_par_iter()
        .map(|task| {
            let first = pairs.start + task * PAIRS_PER_TASK;
            sum(first..(first + PAIRS_PER_TASK).min(pairs.end))
        })
        .reduce(|| [K::ZERO; 3], add)
}

fn plain<T: Cell>(witness: &[T], weights: &[K], pairs: Range<usize>) -> [K; 3] {
    let [zero, two] = pairs.fold([ProductSum::default(); 2], |[zero, two], pair| {
        let (witness, weights) = (line(witness, pair), line(weights, pair));
        [
            witness.zero.add_to(zero, weights.zero),
            witness.two.add_to(two, weights.two),
        ]
    });
    [zero.value(), two.value(), K::ZERO]
}

enum WitnessCells {
    Base(Vec<F>),
    Bound(Vec<K>),
}

impl WitnessCells {
    fn samples(&self, weights: &[K]) -> [K; 3] {
        fn samples<T: Cell + Sync>(cells: &[T], weights: &[K]) -> [K; 3] {
            sum_tasks(0..cells.len().min(weights.len()).div_ceil(2), |pairs| {
                plain(cells, weights, pairs)
            })
        }
        match self {
            Self::Base(cells) => samples(cells, weights),
            Self::Bound(cells) => samples(cells, weights),
        }
    }

    fn bound(&mut self) -> &mut Vec<K> {
        match self {
            Self::Bound(cells) => cells,
            Self::Base(_) => unreachable!("a lookup slice has lane variables"),
        }
    }

    fn bind(&mut self, challenge: K) {
        match self {
            Self::Base(cells) => {
                if cells.len() % 2 == 1 {
                    cells.push(F::ZERO);
                }
                cells
                    .par_chunks_mut(2)
                    .with_min_len(PAIRS_PER_TASK)
                    .for_each(|pair| {
                        let value = Line::new(pair[0], pair[1]).at(challenge);
                        pair.copy_from_slice(value.as_basis_coefficients_slice());
                    });
                *self = Self::Bound(K::reconstitute_from_base(std::mem::take(cells)));
            }
            Self::Bound(cells) => bind_in_place(cells, challenge),
        }
    }
}

fn lane_sums(values: &[K], lanes: &[K], blocks: &[K]) -> [K; 3] {
    let lines: Vec<Line<K>> = (0..lanes.len() / 2).map(|pair| line(lanes, pair)).collect();
    let [zero, two] = values
        .par_chunks(lanes.len())
        .zip(blocks)
        .with_min_len((2 * PAIRS_PER_TASK / lanes.len()).max(1))
        .fold(
            || [ProductSum::default(); 2],
            |[zero, two], (cells, &block)| {
                let [cells_zero, cells_two] =
                    lines
                        .iter()
                        .enumerate()
                        .fold([ProductSum::default(); 2], |[zero, two], (pair, lane)| {
                            let cell = line(cells, pair);
                            [zero.add(cell.zero, lane.zero), two.add(cell.two, lane.two)]
                        });
                [zero.add(block, cells_zero.value()), two.add(block, cells_two.value())]
            },
        )
        .map(|sums| sums.map(ProductSum::value))
        .reduce(|| [K::ZERO; 2], |left, right| [left[0] + right[0], left[1] + right[1]]);
    [zero, two, K::ZERO]
}

/// SPEC.md Algorithm 9, step 10.
fn lane_marginal(values: &[K], blocks: &[K]) -> [K; D] {
    const BLOCKS_PER_TASK: usize = 128;
    values
        .par_chunks(D * BLOCKS_PER_TASK)
        .zip(blocks.par_chunks(BLOCKS_PER_TASK))
        .map(|(values, blocks)| {
            let mut lanes = [ProductSum::default(); D];
            for (row, &weight) in values.chunks(D).zip(blocks) {
                for (sum, &cell) in lanes.iter_mut().zip(row) {
                    *sum = sum.add(cell, weight);
                }
            }
            lanes.map(ProductSum::value)
        })
        .reduce(
            || [K::ZERO; D],
            |left, right| std::array::from_fn(|lane| left[lane] + right[lane]),
        )
}

fn key_dot(values: &[K], weights: &[K]) -> K {
    values
        .iter()
        .zip(weights)
        .fold(ProductSum::default(), |sum, (&value, &weight)| sum.add(value, weight))
        .value()
}

pub(crate) fn prove_product(
    transcript: &mut Poseidon2Transcript,
    values: Vec<K>,
    lanes: &[K],
    blocks: &[K],
    claim: K,
    rounds: usize,
    complete: impl FnOnce(&[K], &mut Vec<K>),
) -> (Vec<KeyRound>, Vec<K>, K) {
    if lanes.len() != D || rounds < LANE_VARS || !values.len().is_multiple_of(D) {
        return prove_product_direct(transcript, values, lanes, blocks, claim, rounds);
    }
    let marginal = lane_marginal(&values, blocks);
    let prefix_rounds = LANE_VARS;
    let (mut messages, mut point, _) =
        prove_product_direct(transcript, marginal.to_vec(), lanes, &[K::ONE], claim, prefix_rounds);
    let claim = poly_eval_k(&messages[prefix_rounds - 1], point[prefix_rounds - 1]);
    let prefix_width = 1 << prefix_rounds;
    let weights = crate::weights::equality(&point, prefix_width);
    let mut folded = values
        .par_chunks(prefix_width)
        .with_min_len(128)
        .map(|row| key_dot(row, &weights))
        .collect();
    let remaining_lanes: Vec<K> = lanes
        .chunks(prefix_width)
        .map(|row| key_dot(row, &weights))
        .collect();
    drop(values);
    complete(&weights, &mut folded);
    let (tail, remaining, value) = prove_product_direct(
        transcript,
        folded,
        &remaining_lanes,
        blocks,
        claim,
        rounds - prefix_rounds,
    );
    messages.extend(tail);
    point.extend(remaining);
    (messages, point, value)
}

fn prove_product_direct(
    transcript: &mut Poseidon2Transcript,
    mut values: Vec<K>,
    lanes: &[K],
    blocks: &[K],
    mut claim: K,
    rounds: usize,
) -> (Vec<KeyRound>, Vec<K>, K) {
    let mut lanes = lanes.to_vec();
    let tabulate = |lanes: &[K]| -> Vec<K> { blocks.iter().map(|&block| lanes[0] * block).collect() };
    let mut weights = (lanes.len() == 1).then(|| tabulate(&lanes));
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let [zero, two, _] = match &weights {
            None => lane_sums(&values, &lanes, blocks),
            Some(weights) => {
                let pairs = values.len().min(weights.len()).div_ceil(2);
                sum_tasks(0..pairs, |pairs| plain(&values, weights, pairs))
            }
        };
        let sum = claim - zero.double();
        let quadratic = (two - zero - sum.double()).halve();
        let coefficients = [zero, sum - quadratic, quadratic];
        absorb_extension(transcript, &coefficients);
        let challenge = challenge(transcript);
        claim = poly_eval_k(&coefficients, challenge);
        match &mut weights {
            None => {
                bind_in_place(&mut values, challenge);
                bind_in_place(&mut lanes, challenge);
                if lanes.len() == 1 {
                    weights = Some(tabulate(&lanes));
                }
            }
            Some(weights) => {
                rayon::join(
                    || bind_in_place(&mut values, challenge),
                    || bind_in_place(weights, challenge),
                );
            }
        }
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

#[cfg(test)]
#[path = "../tests/unit/sumcheck.rs"]
mod tests;
