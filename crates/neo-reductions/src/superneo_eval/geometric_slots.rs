use neo_math::{sum_of_scaled, KExtensions, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{FactoredSlots, RealBlockStorage, SuperneoZBlocks};

const SLOTS_PER_TASK: usize = 1 << 14;
const DIRECT_LIMIT: usize = 4;

pub(super) fn project(witness: &SuperneoZBlocks, slots: FactoredSlots<'_>, ratio: K) -> Option<Vec<K>> {
    let RealBlockStorage::Dense { blocks, nonzero } = &witness.re else {
        return None;
    };
    let weights = Weights::new(ratio, slots.ratio)?;
    let mut result = vec![K::ZERO; slots.starts.len()];
    let project = |(task, values): (usize, &mut [K])| {
        let first = SLOTS_PER_TASK * task;
        let last = first + values.len();
        let mut block = Block {
            digits: blocks,
            ring: Rq::zero(),
            nonzero,
            weights: &weights,
            index: usize::MAX,
            cursor: 0,
            value: K::ZERO,
            at_end: K::ZERO,
        };
        for ((output, &start), &len) in values
            .iter_mut()
            .zip(&slots.starts[first..last])
            .zip(&slots.widths[first..last])
        {
            if len != 0 {
                *output = block.slot(start as usize, usize::from(len));
            }
        }
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    result
        .par_chunks_mut(SLOTS_PER_TASK)
        .enumerate()
        .for_each(project);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    result
        .chunks_mut(SLOTS_PER_TASK)
        .enumerate()
        .for_each(project);
    Some(result)
}

struct Weights {
    ring: [K; D],
    inverse: [K; D + 1],
    slot: [F; u8::MAX as usize + 1],
    delta: K,
    correction: K,
    divisor: K,
}

impl Weights {
    fn new(ratio: K, slot_ratio: F) -> Option<Self> {
        let inverse_ratio = ratio.try_inverse()?;
        let divisor = (K::ONE - inverse_ratio.scale_base(slot_ratio)).try_inverse()?;
        let mut powers = ratio.powers();
        let ring = std::array::from_fn(|_| powers.next().expect("ring powers"));
        let delta = K::ONE + powers.next().expect("ring power at degree D");
        let mut powers = inverse_ratio.powers();
        let inverse = std::array::from_fn(|_| powers.next().expect("inverse powers"));
        let mut powers = slot_ratio.powers();
        let slot = std::array::from_fn(|_| powers.next().expect("slot powers"));
        Some(Self {
            ring,
            inverse,
            slot,
            delta,
            correction: delta * inverse_ratio,
            divisor,
        })
    }
}

struct Block<'a> {
    digits: &'a super::DenseBlocks,
    ring: Rq,
    nonzero: &'a [bool],
    weights: &'a Weights,
    index: usize,
    cursor: usize,
    value: K,
    at_end: K,
}

impl Block<'_> {
    fn slot(&mut self, start: usize, len: usize) -> K {
        let end = start + len;
        let mut total = K::ZERO;
        let mut first = start;
        while first < end {
            let index = first / D;
            let lane = first % D;
            let upper = (first + D - lane).min(end);
            let count = upper - first;
            if self.nonzero[index] {
                if self.index != index || self.cursor > lane {
                    self.index = index;
                    self.cursor = 0;
                    self.ring = self.digits.ring(index).into_owned();
                    self.value = sum_of_scaled(self.ring.0.into_iter().zip(self.weights.ring));
                    self.at_end = -self.value;
                }
                if self.cursor < lane {
                    let _ = self.span(lane);
                }
                let value = self.span(lane + count);
                if first == start {
                    total = value;
                } else {
                    total += value.scale_base(self.weights.slot[first - start]);
                }
            }
            first = upper;
        }
        total
    }

    fn span(&mut self, end: usize) -> K {
        let digits = &self.ring.0[self.cursor..end];
        let len = digits.len();
        self.cursor = end;
        if let [digit] = digits {
            let value = self.value;
            self.value = value * self.weights.inverse[1] - self.weights.correction.scale_base(*digit);
            return value;
        }
        if len < DIRECT_LIMIT {
            let mut sum = K::ZERO;
            for (&digit, &weight) in digits.iter().zip(&self.weights.slot) {
                sum += self.value.scale_base(weight);
                self.value = self.value * self.weights.inverse[1] - self.weights.correction.scale_base(digit);
            }
            return sum;
        }
        let slot_sum = F::sum_of_products(
            digits
                .iter()
                .copied()
                .zip(self.weights.slot.iter().copied()),
        );
        let finish = if end == D {
            self.at_end
        } else {
            let ring_sum = sum_of_scaled(digits.iter().copied().zip(self.weights.ring));
            (self.value - self.weights.delta * ring_sum) * self.weights.inverse[len]
        };
        let sum = self.value
            - finish.scale_base(self.weights.slot[len])
            - self
                .weights
                .correction
                .scale_base(self.weights.slot[1] * slot_sum);
        self.value = finish;
        sum * self.weights.divisor
    }
}

#[cfg(test)]
#[path = "../../tests/unit/geometric_slots.rs"]
mod tests;
