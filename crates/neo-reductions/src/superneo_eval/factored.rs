//! Matrix rows in factored form `M_j = L_j · G` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).

use core::ops::Range;
use std::sync::{Mutex, PoisonError};

use neo_ccs::GeometricRowRun;
use neo_math::{sum_of_scaled, KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{MatrixRowSink, MatrixShape, SuperneoZBlocks};
use crate::PiCcsError;

#[derive(Clone, Copy)]
pub struct FactoredEntries<'a> {
    pub offsets: &'a [usize],
    pub slots: &'a [u32],
    pub coefficients: &'a [F],
}

#[derive(Clone, Copy)]
pub struct FactoredSlots<'a> {
    pub starts: &'a [u32],
    pub widths: &'a [u8],
    pub ratio: F,
}

const SLOTS_PER_BAND: usize = 1 << 14;
const QUEUED_PER_BAND: usize = 512;
const ROWS_PER_TASK: usize = 1 << 12;

#[derive(Clone, Copy)]
pub struct FactoredRows<'a> {
    shape: MatrixShape,
    entries: FactoredEntries<'a>,
    slots: FactoredSlots<'a>,
}

impl<'a> FactoredRows<'a> {
    pub fn new(shape: MatrixShape, entries: FactoredEntries<'a>, slots: FactoredSlots<'a>) -> Result<Self, PiCcsError> {
        Self::new_unchecked(shape, entries, slots).checked()
    }

    pub fn new_unchecked(shape: MatrixShape, entries: FactoredEntries<'a>, slots: FactoredSlots<'a>) -> Self {
        Self { shape, entries, slots }
    }

    pub fn checked(self) -> Result<Self, PiCcsError> {
        let Self { shape, entries, slots } = self;
        shape.validate(&(0..shape.rows))?;
        let invalid = |reason: &str| Err(PiCcsError::InvalidInput(format!("factored matrix rows: {reason}")));
        let FactoredEntries {
            offsets,
            slots: read,
            coefficients,
        } = entries;
        if shape
            .matrices
            .checked_mul(shape.rows)
            .and_then(|count| count.checked_add(1))
            != Some(offsets.len())
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&read.len())
            || coefficients.len() != read.len()
            || slots.starts.len() != slots.widths.len()
        {
            return invalid("table lengths");
        }
        if !offsets.windows(2).all(|pair| pair[0] <= pair[1]) {
            return invalid("entry offsets");
        }
        if !slots
            .starts
            .iter()
            .zip(slots.widths)
            .all(|(&start, &width)| start as usize + usize::from(width) <= shape.columns)
        {
            return invalid("slot columns");
        }
        if !all(read, |&slot| {
            slots
                .widths
                .get(slot as usize)
                .is_some_and(|&width| width != 0)
        }) {
            return invalid("an entry reads no slot");
        }
        Ok(self)
    }

    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    pub fn slot_count(&self) -> usize {
        self.slots.starts.len()
    }

    pub fn slots(&self) -> FactoredSlots<'a> {
        self.slots
    }

    fn range(&self, row: usize, matrix: usize) -> Range<usize> {
        let index = self.shape.matrices * row + matrix;
        self.entries.offsets[index]..self.entries.offsets[index + 1]
    }

    fn columns(&self, slot: usize) -> Range<usize> {
        let start = self.slots.starts[slot] as usize;
        start..start + usize::from(self.slots.widths[slot])
    }

    pub fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.shape.validate(&rows)?;
        for row in rows {
            for matrix in 0..self.shape.matrices {
                for entry in self.range(row, matrix) {
                    let columns = self.columns(self.entries.slots[entry] as usize);
                    let run = GeometricRowRun::new(
                        row,
                        columns.start,
                        columns.len(),
                        self.entries.coefficients[entry],
                        self.slots.ratio,
                    );
                    sink.push_run(row, matrix, run)?;
                }
                if sink.finish_matrix_row(row, matrix)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub fn slot_values(&self, witness: &SuperneoZBlocks) -> Vec<F> {
        let mut slot_values = vec![F::ZERO; self.slot_count()];
        fill(&mut slot_values, |slot| {
            self.columns(slot).rev().fold(F::ZERO, |value, column| {
                value * self.slots.ratio + witness.real_coefficient(column / D, column % D)
            })
        });
        slot_values
    }

    pub fn row_value(&self, matrix: usize, row: usize, slot_values: &[F]) -> F {
        self.range(row, matrix)
            .map(|entry| self.entries.coefficients[entry] * slot_values[self.entries.slots[entry] as usize])
            .sum()
    }

    pub fn projected_slot_values(&self, projection: &[K]) -> Vec<K> {
        let mut slot_values = vec![K::ZERO; self.slot_count()];
        fill(&mut slot_values, |slot| {
            projection[self.columns(slot)]
                .iter()
                .rev()
                .fold(K::ZERO, |value, &digit| value.scale_base(self.slots.ratio) + digit)
        });
        slot_values
    }

    pub fn slot_weights(&self, row_weight: &(dyn Fn(usize) -> K + Sync), matrix_coefficients: &[K]) -> Vec<K> {
        let mut weights = vec![K::ZERO; self.slot_count()];
        self.transpose_weights(
            row_weight,
            matrix_coefficients.iter().copied().enumerate(),
            &mut weights,
        );
        weights
    }

    pub(super) fn transpose_weights(
        &self,
        row_weight_for: &(dyn Fn(usize) -> K + Sync),
        matrices: impl Iterator<Item = (usize, K)> + Clone + Sync,
        weights: &mut [K],
    ) {
        let bands: Vec<Mutex<&mut [K]>> = weights.chunks_mut(SLOTS_PER_BAND).map(Mutex::new).collect();
        let add = |band: usize, queue: &mut Vec<(u32, K)>| {
            if queue.is_empty() {
                return;
            }
            let mut weights = bands[band].lock().unwrap_or_else(PoisonError::into_inner);
            for (slot, term) in queue.drain(..) {
                weights[slot as usize % SLOTS_PER_BAND] += term;
            }
        };
        let queue_row = |mut queues: Vec<Vec<(u32, K)>>, row: usize| {
            let row_weight = std::cell::OnceCell::new();
            for (matrix, coefficient) in matrices.clone() {
                let entries = self.range(row, matrix);
                if entries.is_empty() || coefficient == K::ZERO {
                    continue;
                }
                let weight = *row_weight.get_or_init(|| row_weight_for(row));
                let weight = if coefficient == K::ONE {
                    weight
                } else {
                    weight * coefficient
                };
                for entry in entries {
                    let slot = self.entries.slots[entry];
                    let band = slot as usize / SLOTS_PER_BAND;
                    queues[band].push((slot, weight.scale_base(self.entries.coefficients[entry])));
                    if queues[band].len() == QUEUED_PER_BAND {
                        add(band, &mut queues[band]);
                    }
                }
            }
            queues
        };
        let empty = || vec![Vec::new(); bands.len()];
        let add_all = |mut queues: Vec<Vec<(u32, K)>>| {
            for (band, queue) in queues.iter_mut().enumerate() {
                add(band, queue);
            }
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        (0..self.shape.rows)
            .into_par_iter()
            .with_min_len(ROWS_PER_TASK)
            .fold(empty, queue_row)
            .for_each(add_all);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        add_all((0..self.shape.rows).fold(empty(), queue_row));
    }

    pub fn weighted_row(&self, row: usize, matrix_coefficients: &[K], slot_values: &[K]) -> K {
        matrix_coefficients
            .iter()
            .enumerate()
            .filter_map(|(matrix, &weight)| {
                let entries = self.range(row, matrix);
                if entries.is_empty() {
                    return None;
                }
                let slots = self.entries.slots[entries.clone()].iter();
                let coefficients = &self.entries.coefficients[entries];
                let pairs = slots
                    .zip(coefficients)
                    .map(|(&slot, &coefficient)| (coefficient, slot_values[slot as usize]));
                Some(weight * sum_of_scaled(pairs))
            })
            .sum()
    }
}

pub(crate) fn fill<T: Send>(out: &mut [T], value: impl Fn(usize) -> T + Sync + Send) {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    out.par_iter_mut()
        .enumerate()
        .for_each(|(index, slot)| *slot = value(index));
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    out.iter_mut()
        .enumerate()
        .for_each(|(index, slot)| *slot = value(index));
}

fn all<T: Sync>(items: &[T], test: impl Fn(&T) -> bool + Sync + Send) -> bool {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    return items.par_iter().all(test);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    return items.iter().all(test);
}
