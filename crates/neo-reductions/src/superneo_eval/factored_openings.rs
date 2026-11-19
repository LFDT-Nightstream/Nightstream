//! `M_j = L_j · G` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).

use core::ops::Range;

use neo_ccs::V1_2Evaluations;
use neo_math::{KExtensions, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::block_sums::{add_pair_sums, to_extension, zero_pair_sums, TaskSums, TASK_BLOCKS};
use super::{EqualityWeights, FactoredRows, SuperneoZBlocks};

const BAND_COLUMNS: usize = TASK_BLOCKS * D;

pub(super) struct OpeningBands<'a> {
    rows: FactoredRows<'a>,
    slots: Vec<Range<usize>>,
    index_budget: Option<usize>,
    parallel_transpose: bool,
}

struct OpeningDifference {
    ratio: F,
    powers: [F; D + 1],
    inverse: F,
}

impl OpeningDifference {
    fn new(ratio: F) -> Option<Self> {
        let mut powers = [F::ONE; D + 1];
        for lane in 1..=D {
            powers[lane] = powers[lane - 1] * ratio;
        }
        let inverse = (F::ONE + powers[D]).try_inverse()?;
        Some(Self { ratio, powers, inverse })
    }

    fn resolve(&self, coefficients: &mut [K; D]) {
        let origin = K::from_coeffs(core::array::from_fn(|part| {
            F::sum_of_products(
                self.powers
                    .iter()
                    .zip(&*coefficients)
                    .map(|(&power, value)| (power, value.as_coeffs()[part])),
            ) * self.inverse
        }));
        let mut following = -origin;
        for coefficient in coefficients.iter_mut().skip(1).rev() {
            *coefficient += following.scale_base(self.ratio);
            following = *coefficient;
        }
        coefficients[0] = origin;
    }
}

#[inline]
fn add_event(real: &mut Rq, imaginary: &mut Rq, lane: usize, value: K) {
    let [r, i] = value.as_coeffs();
    real.0[lane] += r;
    imaginary.0[lane] += i;
}

impl<'a> OpeningBands<'a> {
    pub(super) fn new(
        rows: FactoredRows<'a>,
        workspace_bytes: usize,
        witnesses: usize,
        storage_capacity: usize,
    ) -> Option<Self> {
        let columns = rows.shape().columns;
        let band_count = columns.div_ceil(BAND_COLUMNS);
        let common = storage_capacity
            .max(rows.slot_count())
            .checked_mul(size_of::<K>())?
            .checked_add(band_count.checked_mul(size_of::<Range<usize>>())?)?
            .checked_add(witnesses.checked_mul(size_of::<&SuperneoZBlocks>())?)?;
        let projection = common.checked_add(TaskSums::working_bytes(band_count, witnesses)?)?;
        let remaining = workspace_bytes.checked_sub(projection)?;
        let index_budget = rows
            .shape()
            .rows
            .checked_mul(size_of::<K>())
            .and_then(|weights| remaining.checked_sub(weights));
        let parallel_transpose = rows
            .transpose_working_bytes()
            .is_some_and(|bytes| bytes <= workspace_bytes - common);
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
        Some(Self {
            rows,
            slots,
            index_budget,
            parallel_transpose,
        })
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
        let columns = self
            .index_budget
            .and_then(|budget| self.rows.opening_columns(budget));
        let slot_count = columns.map_or(self.rows.slot_count(), |columns| {
            columns
                .iter()
                .map(|matrix| matrix.slots.len())
                .max()
                .unwrap_or(0)
        });
        if storage.capacity() < slot_count {
            drop(storage);
            storage = Vec::with_capacity(slot_count);
        }
        let row_weights = columns.map(|_| {
            let mut rows = vec![K::ZERO; self.rows.shape().rows];
            super::factored::fill(&mut rows, |row| weights.at(row));
            rows
        });
        let difference = OpeningDifference::new(layout.ratio);
        for matrix in 0..self.rows.shape().matrices {
            let ids = match (columns, &row_weights) {
                (Some(columns), Some(row_weights)) => {
                    columns[matrix].weights(row_weights, &mut storage);
                    Some(columns[matrix].slots.as_slice())
                }
                _ => {
                    storage.resize(self.rows.slot_count(), K::ZERO);
                    storage.fill(K::ZERO);
                    if self.parallel_transpose {
                        self.rows.transpose_weights(
                            &|row| weights.at(row),
                            core::iter::once((matrix, K::ONE)),
                            &mut storage,
                        );
                    } else {
                        self.rows
                            .transpose_one_serial(matrix, weights, &mut storage);
                    }
                    None
                }
            };
            let task = |(band, slots): (usize, &Range<usize>)| {
                let first = band * BAND_COLUMNS;
                let limit = (first + BAND_COLUMNS).min(self.rows.shape().columns);
                let range = ids.map_or_else(
                    || slots.clone(),
                    |ids| {
                        ids.partition_point(|&slot| (slot as usize) < slots.start)
                            ..ids.partition_point(|&slot| (slot as usize) < slots.end)
                    },
                );
                let mut spans = range.filter_map(|index| {
                    let slot = ids.map_or(index, |ids| ids[index] as usize);
                    let mut value = storage[index];
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
                    let mut events = 0;
                    while let Some((column, span_end, value)) = span.as_mut().filter(|(column, _, _)| *column < end) {
                        let start = *column - base;
                        let stop = (*span_end).min(end) - base;
                        if let Some(difference) = &difference {
                            let first_lane = (D - start) % D;
                            let last_lane = (D - stop) % D;
                            add_event(
                                &mut real,
                                &mut imaginary,
                                first_lane,
                                if start == 0 { *value } else { -*value },
                            );
                            *value = value.scale_base(difference.powers[stop - start]);
                            add_event(&mut real, &mut imaginary, last_lane, *value);
                            events |= (1u64 << first_lane) | (1u64 << last_lane);
                        } else {
                            for local in start..stop {
                                let (lane, coefficient) = if local == 0 { (0, *value) } else { (D - local, -*value) };
                                [real.0[lane], imaginary.0[lane]] = coefficient.as_coeffs();
                                *value = value.scale_base(layout.ratio);
                            }
                        }
                        if *span_end > end {
                            *column = end;
                            break;
                        }
                        span = spans.next();
                    }
                    sums.add_opening_block(witnesses, base / D, difference.is_some().then_some(events), || {
                        (real, imaginary)
                    });
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
            for (&index, mut value) in active.iter().zip(to_extension(sums)) {
                if let Some(difference) = &difference {
                    difference.resolve(&mut value);
                }
                result[index].eval_a[matrix].copy_from_slice(&value);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/factored_openings.rs"]
mod tests;
