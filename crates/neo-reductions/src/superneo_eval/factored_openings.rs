//! `M_j = L_j · G` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).

use core::ops::Range;

use neo_ccs::V1_2Evaluations;
use neo_math::{KExtensions, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::block_sums::{add_pair_sums, to_extension, zero_pair_sums, OpeningEvents, PairSums, TaskSums, TASK_BLOCKS};
use super::opening_columns::Columns;
use super::{EqualityWeights, FactoredRows, SuperneoZBlocks};

const BAND_COLUMNS: usize = TASK_BLOCKS * D;

pub(super) struct OpeningBands<'a> {
    rows: FactoredRows<'a>,
    slots: Vec<Range<usize>>,
    index_budget: Option<usize>,
    parallel_transpose: bool,
    #[cfg(test)]
    separate_matrices: bool,
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
        let runs = rows.existing_slot_runs();
        let workspace_bytes = workspace_bytes.checked_sub(runs.map_or(0, |r| r.storage_bytes()))?;
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
        let slots = if let Some(runs) = runs {
            if runs.last_column() > columns {
                return None;
            }
            (0..band_count)
                .map(|band| {
                    runs.slots_in(
                        band * BAND_COLUMNS,
                        ((band + 1) * BAND_COLUMNS).min(columns),
                        rows.slot_count(),
                    )
                })
                .collect()
        } else {
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
            slots
        };
        Some(Self {
            rows,
            slots,
            index_budget,
            parallel_transpose,
            #[cfg(test)]
            separate_matrices: false,
        })
    }

    fn evaluate_device(
        &self,
        device: &dyn super::TerminalOpeningDevice,
        budget: usize,
        columns: &[Columns],
        weights: &[K],
        witnesses: &[&SuperneoZBlocks],
        active: &[usize],
        result: &mut [V1_2Evaluations<K>],
        storage: &mut Vec<K>,
    ) -> Option<()> {
        let blocks = self.rows.shape().columns / D;
        if !device.supports(blocks, witnesses.len()) {
            return None;
        }
        let inputs = witnesses
            .iter()
            .map(|w| super::OpeningWitness::from_blocks(w, blocks))
            .collect::<Option<Vec<_>>>()?;
        drop(std::mem::take(storage));
        let fixed = self
            .rows
            .existing_slot_runs()
            .map_or(0, |runs| runs.storage_bytes())
            .checked_add(size_of_val(weights))?
            .checked_add(
                self.slots
                    .capacity()
                    .checked_mul(size_of::<Range<usize>>())?,
            )?
            .checked_add(
                columns
                    .len()
                    .checked_mul(inputs.len())?
                    .checked_mul(size_of::<[K; D]>())?,
            )?;
        let remaining = || {
            columns
                .iter()
                .try_fold(budget.checked_sub(fixed)?, |bytes, c| {
                    bytes.checked_sub(c.storage_bytes() + size_of::<Columns>())
                })
        };
        let mut completed = Vec::with_capacity(columns.len());
        for c in columns {
            let matrix = c.device_matrix(self.rows.slots(), blocks, weights.len(), remaining()?)?;
            let mut out = vec![[K::ZERO; D]; inputs.len()];
            if !device.evaluate(&matrix, weights, &inputs, remaining()?, &mut out) {
                return None;
            }
            completed.push(out);
        }
        for (matrix, rings) in completed.into_iter().enumerate() {
            for (&index, ring) in active.iter().zip(rings) {
                result[index].eval_a[matrix].copy_from_slice(&ring);
            }
        }
        Some(())
    }

    pub(super) fn evaluate(
        self,
        weights: &EqualityWeights,
        witnesses: &[&SuperneoZBlocks],
        active: &[usize],
        result: &mut [V1_2Evaluations<K>],
        mut storage: Vec<K>,
        device: Option<(&dyn super::TerminalOpeningDevice, usize)>,
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
        let row_weights = columns.map(|_| {
            let mut rows = vec![K::ZERO; self.rows.shape().rows];
            super::factored::fill(&mut rows, |row| weights.at(row));
            rows
        });
        if let (Some((device, budget)), Some(columns), Some(weights)) = (device, columns, row_weights.as_deref()) {
            if self
                .evaluate_device(
                    device,
                    budget,
                    columns,
                    weights,
                    witnesses,
                    active,
                    result,
                    &mut storage,
                )
                .is_some()
            {
                return;
            }
        }
        let difference = OpeningDifference::new(layout.ratio);
        #[cfg(test)]
        let fuse = !self.separate_matrices;
        #[cfg(not(test))]
        let fuse = true;
        if let (true, Some(columns), Some(row_weights), Some(difference)) = (fuse, columns, &row_weights, &difference) {
            if let Ok(columns) = <&[Columns; 4]>::try_from(columns) {
                let replaced = storage.capacity().max(slot_count) * size_of::<K>();
                if witnesses
                    .iter()
                    .any(|witness| matches!(&witness.re, super::RealBlockStorage::Dense { .. }))
                    && TaskSums::working_bytes(self.slots.len(), witnesses.len())
                        .and_then(|bytes| bytes.checked_mul(3))
                        .is_some_and(|extra| extra <= replaced)
                {
                    drop(storage);
                    self.evaluate_four(columns, row_weights, witnesses, active, result, difference);
                    return;
                }
            }
        }
        if storage.capacity() < slot_count {
            drop(storage);
            storage = Vec::with_capacity(slot_count);
        }
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

    fn evaluate_four(
        &self,
        columns: &[Columns; 4],
        row_weights: &[K],
        witnesses: &[&SuperneoZBlocks],
        active: &[usize],
        result: &mut [V1_2Evaluations<K>],
        difference: &OpeningDifference,
    ) {
        let layout = self.rows.slots();
        let zero = || std::array::from_fn::<_, 4, _>(|_| zero_pair_sums(witnesses.len()));
        let add = |left: [PairSums; 4], right: [PairSums; 4]| {
            let mut pairs = left.into_iter().zip(right);
            std::array::from_fn(|_| {
                let (left, right) = pairs.next().unwrap();
                add_pair_sums(left, right)
            })
        };
        let task = |(band, slots): (usize, &Range<usize>)| {
            let first = band * BAND_COLUMNS;
            let limit = (first + BAND_COLUMNS).min(self.rows.shape().columns);
            let mut cursors = columns.each_ref().map(|matrix| {
                let ids = &matrix.slots;
                let range = ids.partition_point(|&slot| (slot as usize) < slots.start)
                    ..ids.partition_point(|&slot| (slot as usize) < slots.end);
                let mut spans = range.filter_map(move |index| {
                    let slot = ids[index] as usize;
                    let start = layout.starts[slot] as usize;
                    let end = start + usize::from(layout.widths[slot]);
                    if start == end {
                        return None;
                    }
                    let mut value = matrix.weight_with_products(row_weights, index);
                    if value == K::ZERO {
                        return None;
                    }
                    let column = start.max(first);
                    if column != start {
                        value = value.scale_base(layout.ratio.exp_u64((column - start) as u64));
                    }
                    Some((column, end, value))
                });
                (spans.next(), spans)
            });
            let mut sums = std::array::from_fn::<_, 4, _>(|_| TaskSums::new(witnesses.len()));
            for base in (first..limit).step_by(D) {
                if cursors.iter().all(|(span, _)| span.is_none()) {
                    break;
                }
                let mut forms = std::array::from_fn(|_| OpeningEvents {
                    real: Rq::zero(),
                    imaginary: Rq::zero(),
                    mask: 0,
                });
                for ((span, spans), form) in cursors.iter_mut().zip(&mut forms) {
                    events_for_block(base, span, spans, difference, form);
                }
                TaskSums::add_opening_blocks(&mut sums, witnesses, base / D, &forms);
            }
            sums.map(TaskSums::finish)
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let sums = self
            .slots
            .par_iter()
            .enumerate()
            .map(task)
            .reduce(zero, add);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let sums = self.slots.iter().enumerate().map(task).fold(zero(), add);
        for (matrix, sums) in sums.into_iter().enumerate() {
            for (&index, mut value) in active.iter().zip(to_extension(sums)) {
                difference.resolve(&mut value);
                result[index].eval_a[matrix].copy_from_slice(&value);
            }
        }
    }
}

fn events_for_block(
    base: usize,
    span: &mut Option<(usize, usize, K)>,
    spans: &mut impl Iterator<Item = (usize, usize, K)>,
    difference: &OpeningDifference,
    result: &mut OpeningEvents,
) {
    let end = base + D;
    while let Some((column, span_end, value)) = span.as_mut().filter(|(column, _, _)| *column < end) {
        let start = *column - base;
        let stop = (*span_end).min(end) - base;
        let first_lane = (D - start) % D;
        let last_lane = (D - stop) % D;
        add_event(
            &mut result.real,
            &mut result.imaginary,
            first_lane,
            if start == 0 { *value } else { -*value },
        );
        *value = value.scale_base(difference.powers[stop - start]);
        add_event(&mut result.real, &mut result.imaginary, last_lane, *value);
        result.mask |= (1u64 << first_lane) | (1u64 << last_lane);
        if *span_end > end {
            *column = end;
            break;
        }
        *span = spans.next();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/factored_openings.rs"]
mod tests;
