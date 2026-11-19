//! `M_j = L_j · G` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).

use core::ops::Range;

use neo_ccs::V1_2Evaluations;
use neo_math::{KExtensions, Rq, D, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::block_sums::{add_pair_sums, to_extension, zero_pair_sums, TaskSums, TASK_BLOCKS};
use super::{EqualityWeights, FactoredRows, SuperneoZBlocks};

const BAND_COLUMNS: usize = TASK_BLOCKS * D;

pub(super) struct OpeningBands<'a> {
    rows: FactoredRows<'a>,
    slots: Vec<Range<usize>>,
}

impl<'a> OpeningBands<'a> {
    pub(super) fn new(rows: FactoredRows<'a>, workspace_bytes: usize) -> Option<Self> {
        let columns = rows.shape().columns;
        let band_count = columns.div_ceil(BAND_COLUMNS);
        if rows.slot_count() * size_of::<K>() + band_count * size_of::<Range<usize>>() > workspace_bytes {
            return None;
        }
        let layout = rows.slots();
        let mut slots = vec![rows.slot_count()..rows.slot_count(); band_count];
        let mut previous_end = 0;
        for (slot, (&start, &width)) in layout.starts.iter().zip(layout.widths).enumerate() {
            if width == 0 {
                continue;
            }
            let start = start as usize;
            let end = start + usize::from(width);
            if start < previous_end || end > columns {
                return None;
            }
            for band in &mut slots[start / BAND_COLUMNS..=(end - 1) / BAND_COLUMNS] {
                band.start = band.start.min(slot);
                band.end = slot + 1;
            }
            previous_end = end;
        }
        Some(Self { rows, slots })
    }

    pub(super) fn evaluate(
        self,
        weights: &EqualityWeights,
        witnesses: &[&SuperneoZBlocks],
        active: &[usize],
        result: &mut [V1_2Evaluations<K>],
        mut storage: Vec<K>,
    ) {
        let layout = self.rows.slots();
        storage.resize(self.rows.slot_count(), K::ZERO);
        for matrix in 0..self.rows.shape().matrices {
            storage.fill(K::ZERO);
            self.rows
                .transpose_weights(&|row| weights.at(row), core::iter::once((matrix, K::ONE)), &mut storage);
            let task = |(band, slots): (usize, &Range<usize>)| {
                let first = band * BAND_COLUMNS;
                let limit = (first + BAND_COLUMNS).min(self.rows.shape().columns);
                let mut spans = slots.clone().filter_map(|slot| {
                    let mut value = storage[slot];
                    if layout.widths[slot] == 0 || value == K::ZERO {
                        return None;
                    }
                    let start = layout.starts[slot] as usize;
                    let end = start + usize::from(layout.widths[slot]);
                    let column = start.max(first);
                    if column != start {
                        value = value.scale_base(layout.ratio.exp_u64((column - start) as u64));
                    }
                    Some((column, end, value))
                });
                let mut span = spans.next();
                let mut sums = TaskSums::new(witnesses.len());
                for base in (first..limit).step_by(D) {
                    let Some((start, _, _)) = span else { break };
                    let end = base + D;
                    if start >= end {
                        continue;
                    }
                    let mut real = Rq::zero();
                    let mut imaginary = Rq::zero();
                    while let Some((column, span_end, value)) = span.as_mut().filter(|(column, _, _)| *column < end) {
                        for local in (*column - base)..((*span_end).min(end) - base) {
                            let (lane, coefficient) = if local == 0 { (0, *value) } else { (D - local, -*value) };
                            [real.0[lane], imaginary.0[lane]] = coefficient.as_coeffs();
                            *value = value.scale_base(layout.ratio);
                        }
                        if *span_end > end {
                            *column = end;
                            break;
                        }
                        span = spans.next();
                    }
                    sums.add_block(witnesses, base / D, || (real, imaginary));
                }
                sums.finish()
            };
            let zero = || zero_pair_sums(witnesses.len());
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let sums = self
                .slots
                .par_iter()
                .enumerate()
                .map(task)
                .reduce(zero, add_pair_sums);
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let sums = self
                .slots
                .iter()
                .enumerate()
                .map(task)
                .fold(zero(), add_pair_sums);
            for (&index, value) in active.iter().zip(to_extension(sums)) {
                result[index].eval_a[matrix].copy_from_slice(&value);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/factored_openings.rs"]
mod tests;
