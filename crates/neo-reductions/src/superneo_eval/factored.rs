//! Matrix rows in factored form `M_j = L_j · G` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).

use core::ops::Range;

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

const SLOT_WEIGHT_PARTS: usize = 8;

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
        let part = |rows: Range<usize>| {
            let mut weights = vec![K::ZERO; self.slot_count()];
            for row in rows {
                let row_weight = row_weight(row);
                for (matrix, &coefficient) in matrix_coefficients.iter().enumerate() {
                    let weight = row_weight * coefficient;
                    for entry in self.range(row, matrix) {
                        weights[self.entries.slots[entry] as usize] +=
                            weight.scale_base(self.entries.coefficients[entry]);
                    }
                }
            }
            weights
        };
        let parts = SLOT_WEIGHT_PARTS.min(self.shape.rows);
        let rows = |part: usize| part * self.shape.rows / parts..(part + 1) * self.shape.rows / parts;
        let add = |mut left: Vec<K>, right: Vec<K>| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        return (0..parts)
            .into_par_iter()
            .map(|index| part(rows(index)))
            .reduce(|| vec![K::ZERO; self.slot_count()], add);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        return (0..parts)
            .map(|index| part(rows(index)))
            .fold(vec![K::ZERO; self.slot_count()], add);
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
