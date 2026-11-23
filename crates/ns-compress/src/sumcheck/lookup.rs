use neo_math::{KExtensions, ProductSum, F, K};
use neo_reductions::sumcheck::poly_eval_k;
use neo_reductions::superneo_eval::EqualityWeights;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use super::{add, bind_in_place, equality, line, Round, WitnessCells, PAIRS_PER_TASK};
use crate::lane_prefix::LaneStart;
use crate::layout::{Layout, Slice, LANE_VARS};
use crate::lookup::LookupPolynomial;
use crate::transcript::{absorb_extension, challenge};
use crate::weights::Challenges;
use crate::Error;

#[derive(Clone, Copy, PartialEq)]
struct LookupCell {
    value: K,
    inverse: K,
}

impl LookupCell {
    fn bind(self, b: Self, r: K) -> Self {
        if self == b {
            return self;
        }
        Self {
            value: self.value + (b.value - self.value) * r,
            inverse: self.inverse + (b.inverse - self.inverse) * r,
        }
    }
}

#[derive(Clone, Copy)]
struct RepeatedLine {
    a: LookupCell,
    b: LookupCell,
    count: u32,
    eq: K,
}

#[derive(Default)]
struct Samples {
    low: [ProductSum; 4],
    offset: K,
    quartic: K,
    quintic: K,
}

impl Samples {
    fn evaluations(self) -> Round {
        let [a, b, c, d] = self.low.map(|term| term.value() + self.offset);
        let mut last = d;
        let mut first = d - c;
        let mut second = first - (c - b);
        let mut third = second - (c - b - (b - a));
        let mut fourth = (self.quartic + self.quintic * F::from_u8(10)) * F::from_u8(24);
        let fifth = self.quintic * F::from_u8(120);
        let mut out = [a, b, c, d, K::ZERO, K::ZERO, K::ZERO];
        for value in &mut out[4..] {
            third += fourth;
            second += third;
            first += second;
            last += first;
            *value = last;
            fourth += fifth;
        }
        out
    }
}

impl RepeatedLine {
    fn flush(
        self,
        polynomial: &LookupPolynomial,
        default_sum: K,
        inverse_checks: &mut Samples,
        lookup_sums: &mut Samples,
    ) {
        let count = F::from_u32(self.count);
        let (mut value, inverse) = (self.a.value, self.a.inverse);
        if self.a == self.b {
            let (denominator, numerator) = polynomial.at(value);
            let inverse_check = self.eq * (inverse * denominator - K::ONE);
            let lookup_sum = (inverse * numerator - default_sum) * count;
            inverse_checks.offset += inverse_check;
            lookup_sums.offset += lookup_sum;
            return;
        }
        let (dz, du) = (self.b.value - value, self.b.inverse - inverse);
        let square = dz.square();
        let cubic = square * dz;
        let fourth = square.square();
        let third = value.double().double() + polynomial.denominator_cubic();
        let (mut checked, d_checked) = (inverse * self.eq, du * self.eq);
        let (mut counted, d_counted) = (inverse * count, du * count);
        let checked_cubic = d_checked * cubic;
        inverse_checks.quartic += checked * fourth + checked_cubic * third;
        inverse_checks.quintic += checked_cubic * dz;
        lookup_sums.quartic += d_counted * cubic;
        inverse_checks.offset -= self.eq;
        lookup_sums.offset -= default_sum * count;
        for i in 0..4 {
            let (denominator, numerator) = polynomial.at(value);
            inverse_checks.low[i] = inverse_checks.low[i].add(checked, denominator);
            lookup_sums.low[i] = lookup_sums.low[i].add(counted, numerator);
            value += dz;
            checked += d_checked;
            counted += d_counted;
        }
    }
}

struct CodePairs {
    pairs: Vec<u16>,
    classes: Vec<(usize, RepeatedLine)>,
    width: usize,
}

impl CodePairs {
    fn new(codes: &[u16], bound: u16, inverses: &[F], beta: &[K]) -> Option<Self> {
        if codes.len() < 2 * PAIRS_PER_TASK {
            return None;
        }
        let (min, max) = codes
            .par_iter()
            .with_min_len(PAIRS_PER_TASK)
            .fold(|| (bound, bound), |(min, max), &v| (min.min(v), max.max(v)))
            .reduce(|| (bound, bound), |(a, b), (c, d)| (a.min(c), b.max(d)));
        let width = usize::from(max - min) + 1;
        if max == crate::lookup::OUTSIDE || width > 256 || width * width > codes.len() / 8 {
            return None;
        }
        let equality = EqualityWeights::new(&beta[1..]);
        let pairs_per_tile = (width * width * 2).max(PAIRS_PER_TASK);
        let index = |a: u16, b: u16| (usize::from(a - min) * width + usize::from(b - min)) as u16;
        let default = index(bound, bound);
        let pairs: Vec<u16> = codes
            .par_chunks(2)
            .with_min_len(PAIRS_PER_TASK)
            .map(|pair| index(pair[0], pair.get(1).copied().unwrap_or(bound)))
            .collect();
        let counts = pairs
            .par_chunks(pairs_per_tile)
            .enumerate()
            .map(|(tile, pairs)| {
                let mut counts = vec![(0u32, K::ZERO); width * width];
                for (i, &pair) in pairs.iter().enumerate() {
                    if pair != default {
                        let entry = &mut counts[usize::from(pair)];
                        entry.0 += 1;
                        entry.1 += equality.at(tile * pairs_per_tile + i);
                    }
                }
                counts
            })
            .reduce_with(|mut left, right| {
                for ((n, e), (m, f)) in left.iter_mut().zip(right) {
                    *n += m;
                    *e += f;
                }
                left
            })
            .unwrap_or_default();
        let cell = |i: usize| {
            let code = usize::from(min) + i;
            LookupCell {
                value: K::from(F::from_usize(code) - F::from_u16(bound)),
                inverse: K::from(inverses[code]),
            }
        };
        let classes = counts
            .into_iter()
            .enumerate()
            .filter(|(_, (count, _))| *count != 0)
            .map(|(index, (count, eq))| {
                (
                    index,
                    RepeatedLine {
                        a: cell(index / width),
                        b: cell(index % width),
                        count,
                        eq,
                    },
                )
            })
            .collect();
        Some(Self { pairs, classes, width })
    }
    fn bind(self, base: LookupCell, challenge: K) -> Vec<LookupCell> {
        let mut cells = vec![base; self.width * self.width];
        for (i, class) in self.classes {
            cells[i] = class.a.bind(class.b, challenge);
        }
        self.pairs
            .par_iter()
            .map(|&pair| cells[usize::from(pair)])
            .collect()
    }
}

pub(crate) struct Slot {
    slice: Slice,
    base: LookupCell,
    cells: Vec<LookupCell>,
    first: Option<CodePairs>,
    xi: K,
    lambda: K,
}

impl Slot {
    pub(crate) fn new(
        slice: Slice,
        bound: u64,
        codes: &[u16],
        inverses: &[F],
        [xi, lambda]: [K; 2],
        witness: &[F],
        beta: &[K],
    ) -> Self {
        let base = LookupCell {
            value: K::ZERO,
            inverse: K::from(inverses[bound as usize]),
        };
        let len = codes
            .par_iter()
            .position_last(|&code| code != bound as u16)
            .map_or(0, |i| i + 1);
        let first = CodePairs::new(&codes[..len], bound as u16, inverses, &beta[..slice.vars]);
        let cells = if first.is_some() {
            Vec::new()
        } else {
            codes[..len]
                .par_iter()
                .zip(&witness[..len])
                .map(|(&code, &value)| LookupCell {
                    value: K::from(value),
                    inverse: K::from(inverses.get(usize::from(code)).copied().unwrap_or(F::ZERO)),
                })
                .collect()
        };
        Self {
            slice,
            base,
            cells,
            first,
            xi,
            lambda,
        }
    }
    fn get(&self, i: usize) -> LookupCell {
        self.cells.get(i).copied().unwrap_or(self.base)
    }
    fn round(&self, bit: usize, beta: &[K], scale: K, polynomial: &LookupPolynomial) -> [K; 7] {
        let (_, numerator_at_zero) = polynomial.at(K::ZERO);
        let default_sum = self.base.inverse * numerator_at_zero;
        let mut eq = K::ONE - beta[bit];
        let step = beta[bit] - eq;
        let xi = self.xi * scale;
        let mut base = [K::ZERO; 7];
        let count = K::from_u64(1 << (self.slice.vars - bit - 1));
        for e in &mut base {
            *e = xi * eq + self.lambda * count * default_sum;
            eq += step;
        }
        if self.cells.is_empty() && self.first.is_none() {
            return base;
        }
        let finish = |inverse: Samples, mut lookup: Samples| {
            lookup.quartic *= polynomial.numerator_leading();
            let (inverse_checks, lookup_sums) = (inverse.evaluations(), lookup.evaluations());
            let mut e = K::ONE - beta[bit];
            std::array::from_fn(|k| {
                let v = xi * e * inverse_checks[k] + self.lambda * lookup_sums[k];
                e += step;
                v
            })
        };
        if let Some(first) = &self.first {
            let sums = first
                .classes
                .par_chunks(PAIRS_PER_TASK)
                .map(|classes| {
                    let mut inverse = Samples::default();
                    let mut lookup = Samples::default();
                    for &(_, class) in classes {
                        class.flush(polynomial, default_sum, &mut inverse, &mut lookup);
                    }
                    finish(inverse, lookup)
                })
                .reduce(|| [K::ZERO; 7], |a, b| std::array::from_fn(|i| a[i] + b[i]));
            return std::array::from_fn(|i| base[i] + sums[i]);
        }
        let equality = EqualityWeights::new(&beta[bit + 1..self.slice.vars]);
        let sums = self
            .cells
            .par_chunks(2 * PAIRS_PER_TASK)
            .enumerate()
            .map(|(tile, cells)| {
                let mut inverse_checks = Samples::default();
                let mut lookup_sums = Samples::default();
                let empty = RepeatedLine {
                    a: self.base,
                    b: self.base,
                    count: 0,
                    eq: K::ZERO,
                };
                let mut cache = [empty; 128];
                for (i, pair) in cells.chunks(2).enumerate() {
                    let a = pair[0];
                    let b = pair.get(1).copied().unwrap_or(self.base);
                    if a == self.base && b == self.base {
                        continue;
                    }
                    let weight = equality.at(tile * PAIRS_PER_TASK + i);
                    let hash = a
                        .value
                        .to_limbs_u64()
                        .0
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(b.value.to_limbs_u64().0.wrapping_mul(0xd1342543de82ef95));
                    let entry = &mut cache[(hash >> 57) as usize];
                    if entry.count > 0 && entry.a == a && entry.b == b {
                        entry.count += 1;
                        entry.eq += weight;
                    } else {
                        if entry.count > 0 {
                            entry.flush(polynomial, default_sum, &mut inverse_checks, &mut lookup_sums);
                        }
                        *entry = RepeatedLine {
                            a,
                            b,
                            count: 1,
                            eq: weight,
                        };
                    }
                }
                for class in cache {
                    if class.count > 0 {
                        class.flush(polynomial, default_sum, &mut inverse_checks, &mut lookup_sums);
                    }
                }
                finish(inverse_checks, lookup_sums)
            })
            .reduce(|| [K::ZERO; 7], |a, b| std::array::from_fn(|i| a[i] + b[i]));
        std::array::from_fn(|i| base[i] + sums[i])
    }
    fn bind(&mut self, x: K) {
        if let Some(first) = self.first.take() {
            self.cells = first.bind(self.base, x);
            return;
        }
        self.cells = self
            .cells
            .par_chunks(2)
            .with_min_len(PAIRS_PER_TASK)
            .map(|p| p[0].bind(p.get(1).copied().unwrap_or(self.base), x))
            .collect();
    }
}

fn interpolate(mut values: [K; 7]) -> [K; 7] {
    let mut out = [K::ZERO; 7];
    let mut basis = [K::ZERO; 7];
    basis[0] = K::ONE;
    let mut factorial = K::ONE;
    for j in 0..7 {
        let weight = values[0] / factorial;
        for k in 0..=j {
            out[k] += weight * basis[k];
        }
        for i in 0..6 - j {
            values[i] = values[i + 1] - values[i];
        }
        let root = K::from_u64(j as u64);
        for k in (0..=(j + 1).min(6)).rev() {
            basis[k] = if k == 0 { K::ZERO } else { basis[k - 1] } - root * basis[k];
        }
        factorial *= K::from_u64((j + 1) as u64);
    }
    out
}
#[derive(Clone, Copy, Default)]
struct RestrictedLookup {
    inverse: K,
    check: K,
    sum: K,
}

impl RestrictedLookup {
    fn line(pair: &[Self]) -> (Self, Self) {
        let low = pair[0];
        let high = pair.get(1).copied().unwrap_or_default();
        (
            low,
            Self {
                inverse: high.inverse - low.inverse,
                check: high.check - low.check,
                sum: high.sum - low.sum,
            },
        )
    }

    fn bind(self, slope: Self, r: K) -> Self {
        Self {
            inverse: self.inverse + r * slope.inverse,
            check: self.check + r * slope.check,
            sum: self.sum + r * slope.sum,
        }
    }

    fn round(table: &[Self], witness: &[K], polynomial: &LookupPolynomial) -> Round {
        let mut samples = [K::ZERO; 7];
        for (j, pair) in table.chunks(2).enumerate() {
            let w = line(witness, j);
            let (mut row, slope) = Self::line(pair);
            let mut value = w.zero;
            for sample in &mut samples {
                let (denominator, numerator) = polynomial.at(value);
                *sample += row.inverse * (row.check * denominator + row.sum * numerator);
                value += w.slope;
                row.inverse += slope.inverse;
                row.check += slope.check;
                row.sum += slope.sum;
            }
        }
        samples
    }
}

fn round_message(mut samples: Round, claim: K, [zero, two, _]: [K; 3]) -> Round {
    let one = claim - samples[0] - samples[1] - zero;
    let mut quadratic = zero;
    let mut difference = one - zero;
    let second_difference = two - one - difference;
    for sample in &mut samples {
        *sample += quadratic;
        quadratic += difference;
        difference += second_difference;
    }
    interpolate(samples)
}

pub(crate) fn prove(
    transcript: &mut Poseidon2Transcript,
    witness: Vec<F>,
    mut weights: Vec<K>,
    mut slots: Vec<Slot>,
    challenges: &Challenges,
    layout: &Layout,
    mut claim: K,
    mut lanes: Option<LaneStart<'_>>,
    mut restore: Option<impl FnOnce(&[K]) -> Result<K, Error>>,
) -> Result<(Vec<Round>, Vec<K>, K, K), Error> {
    let mut witness = WitnessCells::Base(witness);
    let first_slice_end = slots.iter().map(|s| s.slice.vars).min().unwrap();
    let polynomial = LookupPolynomial::new(&challenges.taus, &challenges.psi);
    let mut scale = K::ONE;
    let mut tail: Vec<RestrictedLookup> = Vec::new();
    let mut messages = Vec::with_capacity(layout.witness_vars);
    let mut rho = Vec::with_capacity(layout.witness_vars);
    let mut inverse_value = None;
    for bit in 0..layout.witness_vars {
        if bit == first_slice_end {
            if let Some(restore_weight) = restore.take() {
                witness.bound().insert(0, K::ZERO);
                weights.insert(0, restore_weight(&rho)?);
            }
            tail = vec![RestrictedLookup::default(); layout.table_witness_len >> bit];
            for s in &slots {
                let eq = EqualityWeights::new(&challenges.beta[bit..s.slice.vars]);
                for i in 0..(s.slice.len() >> bit) {
                    tail[(s.slice.offset >> bit) + i] = RestrictedLookup {
                        inverse: s.get(i).inverse,
                        check: s.xi * scale * eq.at(i),
                        sum: s.lambda,
                    };
                }
            }
        }
        if bit == layout.table_witness_vars {
            inverse_value = Some(tail[0].inverse);
        }
        let sums = if bit < first_slice_end {
            slots.iter().fold([K::ZERO; 7], |v, s| {
                let a = s.round(bit, &challenges.beta, scale, &polynomial);
                std::array::from_fn(|i| a[i] + v[i])
            })
        } else {
            RestrictedLookup::round(&tail, witness.bound(), &polynomial)
        };
        let mut plain = witness.samples(&weights);
        if let Some(lanes) = lanes.as_ref() {
            plain = add(plain, lanes.round());
        }
        let msg = round_message(sums, claim, plain);
        absorb_extension(transcript, &msg);
        let r = challenge(transcript);
        claim = poly_eval_k(&msg, r);
        rayon::join(|| witness.bind(r), || bind_in_place(&mut weights, r));
        rho.push(r);
        if let Some(lanes) = lanes.as_mut() {
            lanes.bind(r);
        }
        if rho.len() == LANE_VARS {
            if let Some(lanes) = lanes.take() {
                let cells = witness.bound();
                (*cells, weights) = lanes.materialize(&rho, cells, &weights);
            }
        }
        if bit < first_slice_end {
            for s in &mut slots {
                s.bind(r);
            }
            scale *= equality(challenges.beta[bit], r);
        } else {
            tail = tail
                .chunks(2)
                .map(|pair| {
                    let (low, slope) = RestrictedLookup::line(pair);
                    low.bind(slope, r)
                })
                .collect();
        }
        messages.push(msg);
    }
    Ok((
        messages,
        rho,
        witness.bound()[0],
        inverse_value.unwrap_or_else(|| tail[0].inverse),
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/lookup_slot.rs"]
mod tests;
