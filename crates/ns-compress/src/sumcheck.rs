//! The sum-checks of SPEC.md section 6.

use std::ops::Range;

use neo_math::{ProductSum, D, F, K};
use neo_reductions::sumcheck::poly_eval_k;
use neo_reductions::superneo_eval::EqualityWeights;
use neo_transcript::Poseidon2Transcript;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::lane_prefix::LaneStart;
use crate::layout::Slice;
use crate::layout::LANE_VARS;
use crate::transcript::{absorb_extension, challenge};
use crate::Error;

pub type Round = [K; 4];
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

/// SPEC.md Algorithm 9, step 8.
pub(crate) struct Lookup {
    pub(crate) slice: Slice,
    pub(crate) xi: K,
    pub(crate) lambda: K,
    pub(crate) inverse: Vec<K>,
    pub(crate) scaled: Vec<K>,
}

impl Lookup {
    fn values(&self, codes: &[u16]) -> (Line<K>, Line<K>) {
        let at = |values: &[K], code: u16| values.get(usize::from(code)).copied().unwrap_or(K::ZERO);
        (
            Line::new(at(&self.inverse, codes[0]), at(&self.inverse, codes[1])),
            Line::new(at(&self.scaled, codes[0]), at(&self.scaled, codes[1])),
        )
    }
}

/// SPEC.md Algorithm 9, step 8.
struct Check {
    pairs: Range<usize>,
    equality: EqualityWeights,
    line: Line<K>,
    lambda: K,
}

impl Check {
    fn new(lookup: &Lookup, beta: &[K], round: usize, scale: K) -> Self {
        let Slice { offset, vars } = lookup.slice;
        let (beta_j, xi) = (beta[round], lookup.xi * scale);
        Self {
            pairs: offset >> (round + 1)..(offset + lookup.slice.len()) >> (round + 1),
            equality: EqualityWeights::new(&beta[round + 1..vars]),
            line: Line::new(xi * (K::ONE - beta_j), xi * beta_j),
            lambda: lookup.lambda,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Terms {
    checked: [ProductSum; 3],
    inverse: [K; 2],
}

impl Terms {
    fn add<T: Cell>(&mut self, equality: K, witness: Line<T>, (inverse, scaled): (Line<K>, Line<K>)) {
        let [zero, two, cubic] = self.checked;
        self.checked = [
            zero.add(equality, scaled.zero - witness.zero.times(inverse.zero)),
            two.add(equality, scaled.two - witness.two.times(inverse.two)),
            cubic.add(equality, witness.slope.times(inverse.slope)),
        ];
        self.inverse[0] += inverse.zero;
        self.inverse[1] += inverse.two;
    }

    fn sums(&self, check: &Check) -> [K; 3] {
        let [zero, two, cubic] = self.checked.map(ProductSum::value);
        [
            check.line.zero * zero + check.lambda * self.inverse[0],
            check.line.two * two + check.lambda * self.inverse[1],
            -(check.line.slope * cubic),
        ]
    }
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

pub(crate) struct Start<'a> {
    pub(crate) witness: Vec<F>,
    pub(crate) weights: Vec<K>,
    pub(crate) codes: &'a [u16],
    pub(crate) beta: &'a [K],
    pub(crate) lookups: Vec<Lookup>,
}

impl<'a> Start<'a> {
    fn round(&self) -> [K; 3] {
        let pairs = self.witness.len().max(self.weights.len()).div_ceil(2);
        let plain = sum_tasks(0..pairs, |pairs| plain(&self.witness, &self.weights, pairs));
        self.lookups.iter().fold(plain, |sums, lookup| {
            let check = Check::new(lookup, self.beta, 0, K::ONE);
            let checked = sum_tasks(check.pairs.clone(), |pairs| {
                let mut terms = Terms::default();
                for pair in pairs {
                    let equality = check.equality.at(pair - check.pairs.start);
                    let values = lookup.values(&self.codes[2 * pair..]);
                    terms.add(equality, line(&self.witness, pair), values);
                }
                terms.sums(&check)
            });
            add(sums, checked)
        })
    }

    fn bind(self, challenge: K) -> Bound<'a> {
        let Start {
            mut witness,
            mut weights,
            codes,
            beta,
            lookups,
        } = self;
        let ((witness, (inverse, scaled)), ()) = rayon::join(
            || {
                rayon::join(
                    || {
                        if witness.len() % 2 == 1 {
                            witness.push(F::ZERO);
                        }
                        witness
                            .par_chunks_mut(2)
                            .with_min_len(PAIRS_PER_TASK)
                            .for_each(|pair| {
                                let value = Line::new(pair[0], pair[1]).at(challenge);
                                pair.copy_from_slice(value.as_basis_coefficients_slice());
                            });
                        K::reconstitute_from_base(witness)
                    },
                    || {
                        let length = codes.len().div_ceil(2);
                        let (mut inverse, mut scaled) = (vec![K::ZERO; length], vec![K::ZERO; length]);
                        for lookup in &lookups {
                            let cells = lookup.slice.offset / 2..(lookup.slice.offset + lookup.slice.len()) / 2;
                            inverse[cells.clone()]
                                .par_iter_mut()
                                .zip(&mut scaled[cells.clone()])
                                .zip(codes[2 * cells.start..2 * cells.end].par_chunks(2))
                                .with_min_len(PAIRS_PER_TASK)
                                .for_each(|((inverse, scaled), codes)| {
                                    let (inverse_line, scaled_line) = lookup.values(codes);
                                    *inverse = inverse_line.at(challenge);
                                    *scaled = scaled_line.at(challenge);
                                });
                        }
                        (inverse, scaled)
                    },
                )
            },
            || bind_in_place(&mut weights, challenge),
        );
        let mut bound = Bound {
            tables: [witness, weights, inverse, scaled],
            beta,
            lookups,
            round: 0,
            scale: K::ONE,
        };
        bound.advance(challenge);
        bound
    }
}

struct Bound<'a> {
    tables: [Vec<K>; 4],
    beta: &'a [K],
    lookups: Vec<Lookup>,
    round: usize,
    scale: K,
}

fn sum_cells(checks: &[Check], first: usize, [witness, weights, inverse, scaled]: [&[K]; 4]) -> [K; 3] {
    let pairs = witness.len().max(weights.len()).div_ceil(2);
    let mut sums = plain(witness, weights, 0..pairs);
    for check in checks {
        let mut terms = Terms::default();
        for pair in check.pairs.start.max(first)..check.pairs.end.min(first + pairs) {
            let (local, equality) = (pair - first, check.equality.at(pair - check.pairs.start));
            let lookup = (line(inverse, local), line(scaled, local));
            terms.add(equality, line(witness, local), lookup);
        }
        sums = add(sums, terms.sums(check));
    }
    sums
}

impl<'a> Bound<'a> {
    fn advance(&mut self, challenge: K) {
        if let Some(&beta) = self.beta.get(self.round) {
            self.scale *= equality(beta, challenge);
        }
        self.round += 1;
    }

    fn implicit(&self) -> bool {
        self.lookups
            .iter()
            .all(|lookup| self.round < lookup.slice.vars)
    }

    fn checks(&self) -> Vec<Check> {
        self.lookups
            .iter()
            .map(|lookup| Check::new(lookup, self.beta, self.round, self.scale))
            .collect()
    }

    fn explicit(self) -> Tables {
        let [witness, weights, inverse, scaled_inverse] = self.tables;
        let (mut check, mut sum) = (vec![K::ZERO; inverse.len()], vec![K::ZERO; inverse.len()]);
        for lookup in &self.lookups {
            let Slice { offset, vars } = lookup.slice;
            let equality = EqualityWeights::new(&self.beta[self.round..vars]);
            let xi = lookup.xi * self.scale;
            let cells = offset >> self.round..(offset + lookup.slice.len()) >> self.round;
            for (cell, (check, sum)) in check[cells.clone()]
                .iter_mut()
                .zip(&mut sum[cells])
                .enumerate()
            {
                *check = xi * equality.at(cell);
                *sum = lookup.lambda;
            }
        }
        Tables {
            witness,
            weights,
            check,
            sum,
            inverse,
            scaled_inverse,
        }
    }

    fn settle(self) -> (Stage<'a>, [K; 3]) {
        if !self.implicit() {
            let tables = self.explicit();
            let sums = tables.round();
            return (Stage::Tables(tables), sums);
        }
        let checks = self.checks();
        let pairs = self.tables[0].len().max(self.tables[1].len()).div_ceil(2);
        let sums = sum_tasks(0..pairs, |pairs| {
            let cells = self
                .tables
                .each_ref()
                .map(|table| &table[(2 * pairs.start).min(table.len())..(2 * pairs.end).min(table.len())]);
            sum_cells(&checks, pairs.start, cells)
        });
        (Stage::Bound(self), sums)
    }

    fn bind(mut self, challenge: K) -> (Stage<'a>, [K; 3]) {
        self.advance(challenge);
        self.tables
            .par_iter_mut()
            .for_each(|table| bind_in_place(table, challenge));
        self.settle()
    }
}

enum Stage<'a> {
    Start(Start<'a>),
    Bound(Bound<'a>),
    Tables(Tables),
}

impl<'a> Stage<'a> {
    fn bind(self, challenge: K) -> (Self, [K; 3]) {
        match self {
            Stage::Start(start) => start.bind(challenge).settle(),
            Stage::Bound(bound) => bound.bind(challenge),
            Stage::Tables(mut tables) => {
                tables.bind(challenge);
                let sums = tables.round();
                (Stage::Tables(tables), sums)
            }
        }
    }

    fn witness(&self) -> K {
        let table = match self {
            Stage::Start(start) => return start.witness.first().map_or(K::ZERO, |&cell| K::from(cell)),
            Stage::Bound(bound) => &bound.tables[0],
            Stage::Tables(tables) => &tables.witness,
        };
        table.first().copied().unwrap_or(K::ZERO)
    }
}

struct Tables {
    witness: Vec<K>,
    weights: Vec<K>,
    check: Vec<K>,
    sum: Vec<K>,
    inverse: Vec<K>,
    scaled_inverse: Vec<K>,
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
            .reduce(|| [K::ZERO; 3], add)
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

/// SPEC.md Algorithm 9, step 8.
pub(crate) fn prove(
    transcript: &mut Poseidon2Transcript,
    start: Start<'_>,
    claim: K,
    rounds: usize,
) -> (Vec<Round>, Vec<K>, K) {
    let sums = start.round();
    prove_stages(transcript, Stage::Start(start), sums, claim, rounds)
}

fn prove_stages(
    transcript: &mut Poseidon2Transcript,
    mut stage: Stage<'_>,
    mut sums: [K; 3],
    mut claim: K,
    rounds: usize,
) -> (Vec<Round>, Vec<K>, K) {
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let (coefficients, challenge) = prove_round(transcript, sums, &mut claim);
        (stage, sums) = stage.bind(challenge);
        messages.push(coefficients);
        point.push(challenge);
    }
    (messages, point, stage.witness())
}

pub(crate) struct ZeroPrefix<'a> {
    pub(crate) beta: &'a [K],
    pub(crate) xi: K,
    pub(crate) lambda: K,
    pub(crate) inverse: K,
    pub(crate) scaled: K,
}

pub(crate) fn prove_with_zero_prefix(
    transcript: &mut Poseidon2Transcript,
    witness: Vec<F>,
    weights: Vec<K>,
    mut lane_start: Option<LaneStart<'_>>,
    mut claim: K,
    rounds: usize,
    lookup: ZeroPrefix<'_>,
    weight_at: impl FnOnce(&[K]) -> Result<K, Error>,
) -> Result<(Vec<Round>, Vec<K>, K), Error> {
    let start = Start {
        witness,
        weights,
        codes: &[],
        beta: &[],
        lookups: Vec::new(),
    };
    let mut sums = start.round();
    let mut stage = Stage::Start(start);
    let vars = lookup.beta.len();
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut check = lookup.xi;
    let constant = lookup.lambda * lookup.inverse;
    for (bit, &beta) in lookup.beta.iter().enumerate() {
        if let Some(lanes) = &lane_start {
            sums = add(sums, lanes.round());
        }
        let sum = constant * K::from_u64(1 << (vars - bit - 1));
        let scaled = check * lookup.scaled;
        sums[0] += scaled * (K::ONE - beta) + sum;
        sums[1] += scaled * (beta.double() - (K::ONE - beta)) + sum;
        let (coefficients, challenge) = prove_round(transcript, sums, &mut claim);
        check *= equality(beta, challenge);
        (stage, sums) = stage.bind(challenge);
        messages.push(coefficients);
        point.push(challenge);
        if let Some(lanes) = &mut lane_start {
            lanes.bind(challenge);
        }
        if point.len() == LANE_VARS {
            if let Some(lanes) = lane_start.take() {
                let Stage::Bound(mut tail) = stage else {
                    unreachable!("the count tail has no lookup tables")
                };
                let [witness, weights, _, _] = &mut tail.tables;
                (*witness, *weights) = lanes.materialize(&point, witness, weights);
                (stage, sums) = tail.settle();
            }
        }
    }
    let Stage::Bound(bound) = stage else {
        unreachable!("the zero prefix has lane variables")
    };
    let [mut witness, mut weights, _, _] = bound.tables;
    witness.insert(0, K::ZERO);
    weights.insert(0, weight_at(&point)?);
    let tables = Tables {
        witness,
        weights,
        check: vec![check],
        sum: vec![lookup.lambda],
        inverse: vec![lookup.inverse],
        scaled_inverse: vec![lookup.scaled],
    };
    let sums = tables.round();
    let (later, remaining, witness) = prove_stages(transcript, Stage::Tables(tables), sums, claim, rounds - vars);
    messages.extend(later);
    point.extend(remaining);
    Ok((messages, point, witness))
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
