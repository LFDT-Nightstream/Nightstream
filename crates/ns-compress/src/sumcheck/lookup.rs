use neo_math::{KExtensions, F, K};
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

impl RepeatedLine {
    fn flush(
        self,
        polynomial: &LookupPolynomial,
        default_sum: K,
        inverse_checks: &mut [K; 7],
        lookup_sums: &mut [K; 7],
    ) {
        let count = F::from_u32(self.count);
        let (mut value, mut inverse) = (self.a.value, self.a.inverse);
        if self.a == self.b {
            let (denominator, numerator) = polynomial.at(value);
            let inverse_check = self.eq * (inverse * denominator - K::ONE);
            let lookup_sum = (inverse * numerator - default_sum) * count;
            for i in 0..7 {
                inverse_checks[i] += inverse_check;
                lookup_sums[i] += lookup_sum;
            }
            return;
        }
        let (dz, du) = (self.b.value - value, self.b.inverse - inverse);
        for i in 0..7 {
            let (denominator, numerator) = polynomial.at(value);
            inverse_checks[i] += self.eq * (inverse * denominator - K::ONE);
            lookup_sums[i] += (inverse * numerator - default_sum) * count;
            value += dz;
            inverse += du;
        }
    }
}

pub(crate) struct Slot {
    slice: Slice,
    base: LookupCell,
    cells: Vec<LookupCell>,
    xi: K,
    lambda: K,
}

impl Slot {
    pub(crate) fn new(
        slice: Slice,
        bound: u64,
        codes: &[u16],
        inverses: &[F],
        xi: K,
        lambda: K,
        witness: &[F],
    ) -> Self {
        let base = LookupCell {
            value: K::ZERO,
            inverse: K::from(inverses[bound as usize]),
        };
        let len = codes
            .par_iter()
            .position_last(|&code| code != bound as u16)
            .map_or(0, |i| i + 1);
        let cells = codes[..len]
            .par_iter()
            .zip(&witness[..len])
            .map(|(&code, &value)| LookupCell {
                value: K::from(value),
                inverse: K::from(inverses.get(usize::from(code)).copied().unwrap_or(F::ZERO)),
            })
            .collect();
        Self {
            slice,
            base,
            cells,
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
        if self.cells.is_empty() {
            return base;
        }
        let equality = EqualityWeights::new(&beta[bit + 1..self.slice.vars]);
        let sums = self
            .cells
            .par_chunks(2 * PAIRS_PER_TASK)
            .enumerate()
            .map(|(tile, cells)| {
                let mut inverse_checks = [K::ZERO; 7];
                let mut lookup_sums = [K::ZERO; 7];
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
                let mut e = K::ONE - beta[bit];
                std::array::from_fn(|k| {
                    let v = xi * e * inverse_checks[k] + self.lambda * lookup_sums[k];
                    e += step;
                    v
                })
            })
            .reduce(|| [K::ZERO; 7], |a, b| std::array::from_fn(|i| a[i] + b[i]));
        std::array::from_fn(|i| base[i] + sums[i])
    }
    fn bind(&mut self, x: K) {
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
