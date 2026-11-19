use std::hash::{BuildHasher, RandomState};

use super::factored::fill;
use super::{FactoredEntries, MatrixShape};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

#[derive(Default)]
pub struct OpeningColumns(std::sync::OnceLock<Option<Vec<Columns>>>);

const MAX_FACTORS: usize = u16::MAX as usize + 1;
const EMPTY: u32 = u32::MAX;

impl OpeningColumns {
    pub(super) fn get(
        &self,
        shape: MatrixShape,
        slots: usize,
        entries: FactoredEntries<'_>,
        available: usize,
    ) -> Option<&[Columns]> {
        let columns = if let Some(kept) = self.0.get() {
            kept
        } else {
            if construction_bytes(shape, slots, entries.slots.len())? > available {
                return None;
            }
            self.0.get_or_init(|| {
                let mut columns = Vec::with_capacity(shape.matrices);
                for matrix in 0..shape.matrices {
                    columns.push(Columns::new(matrix, shape, slots, entries)?);
                }
                Some(columns)
            })
        }
        .as_ref()?;
        let bytes = columns
            .iter()
            .try_fold(columns.capacity().checked_mul(size_of::<Columns>())?, |sum, matrix| {
                sum.checked_add(matrix.storage_bytes())
            })?;
        (bytes <= available).then_some(columns.as_slice())
    }
}

fn construction_bytes(shape: MatrixShape, slots: usize, entries: usize) -> Option<usize> {
    u32::try_from(shape.rows).ok()?;
    u32::try_from(slots).ok()?;
    u32::try_from(entries).ok()?;
    let matrices = shape.matrices;
    let occupied = slots.checked_mul(matrices)?.min(entries);
    let factors = entries
        .checked_add(2usize.checked_mul(matrices)?)?
        .min(MAX_FACTORS.checked_mul(matrices)?);
    let matrix_factors = entries.saturating_add(2).min(MAX_FACTORS);
    let factor_table = 2 * matrix_factors.next_power_of_two();
    let stored = matrices
        .checked_mul(size_of::<Columns>() + size_of::<u32>())?
        .checked_add(entries.checked_mul(size_of::<u32>() + size_of::<u16>())?)?
        .checked_add(occupied.checked_mul(2 * size_of::<u32>())?)?
        .checked_add(factors.checked_mul(size_of::<F>())?)?;
    stored
        .checked_add(
            slots
                .checked_add(factor_table)?
                .checked_mul(size_of::<u32>())?,
        )?
        .checked_add(matrix_factors * size_of::<F>())
}

pub(super) struct Columns {
    pub(super) slots: Vec<u32>,
    offsets: Vec<u32>,
    rows: Vec<u32>,
    tags: Vec<u16>,
    factors: Vec<F>,
}

impl Columns {
    fn new(matrix: usize, shape: MatrixShape, count: usize, entries: FactoredEntries<'_>) -> Option<Self> {
        let mut counts = vec![0u32; count];
        let ranges = || {
            (0..shape.rows).map(|row| {
                let port = row * shape.matrices + matrix;
                entries.offsets[port]..entries.offsets[port + 1]
            })
        };
        for range in ranges() {
            for &slot in &entries.slots[range] {
                counts[slot as usize] += 1;
            }
        }
        let occupied = counts.iter().filter(|&&count| count != 0).count();
        let mut offsets = Vec::with_capacity(occupied + 1);
        offsets.push(0);
        let mut slots = Vec::with_capacity(occupied);
        let mut total = 0;
        for (slot, count) in counts.iter_mut().enumerate() {
            let n = *count;
            *count = total;
            if n > 0 {
                total += n;
                slots.push(slot as u32);
                offsets.push(total);
            }
        }
        let mut rows = vec![0; total as usize];
        let mut tags = vec![0; total as usize];
        let mut ids = FactorIds::new(total as usize);
        for (row, range) in ranges().enumerate() {
            for entry in range {
                let next = &mut counts[entries.slots[entry] as usize];
                rows[*next as usize] = row as u32;
                tags[*next as usize] = ids.tag(entries.coefficients[entry])?;
                *next += 1;
            }
        }
        let mut factors = ids.factors;
        factors.shrink_to_fit();
        Some(Self {
            slots,
            offsets,
            rows,
            tags,
            factors,
        })
    }

    fn storage_bytes(&self) -> usize {
        (self.slots.capacity() + self.offsets.capacity() + self.rows.capacity()) * size_of::<u32>()
            + self.tags.capacity() * size_of::<u16>()
            + self.factors.capacity() * size_of::<F>()
    }

    pub(super) fn weights(&self, row_weights: &[K], out: &mut Vec<K>) {
        out.resize(self.slots.len(), K::ZERO);
        fill(out, |slot| self.weight(row_weights, slot));
    }

    pub(super) fn weight(&self, row_weights: &[K], slot: usize) -> K {
        let range = self.offsets[slot] as usize..self.offsets[slot + 1] as usize;
        self.rows[range.clone()]
            .iter()
            .zip(&self.tags[range])
            .map(|(&row, &tag)| {
                let w = row_weights[row as usize];
                match tag {
                    0 => w,
                    1 => -w,
                    _ => w.scale_base(self.factors[tag as usize]),
                }
            })
            .sum()
    }

    pub(super) fn weight_with_products(&self, row_weights: &[K], slot: usize) -> K {
        let range = self.offsets[slot] as usize..self.offsets[slot + 1] as usize;
        if range.len() < 4 {
            return self.weight(row_weights, slot);
        }
        let terms = self.rows[range.clone()].iter().zip(&self.tags[range]);
        neo_math::sum_of_scaled(terms.map(|(&row, &tag)| (self.factors[tag as usize], row_weights[row as usize])))
    }
}

struct FactorIds {
    hasher: RandomState,
    table: Vec<u32>,
    factors: Vec<F>,
}

impl FactorIds {
    fn new(entries: usize) -> Self {
        let factors = entries.saturating_add(2).min(MAX_FACTORS);
        let mut values = Vec::with_capacity(factors);
        values.extend([F::ONE, F::NEG_ONE]);
        Self {
            hasher: RandomState::new(),
            table: vec![EMPTY; 2 * factors.next_power_of_two()],
            factors: values,
        }
    }

    fn tag(&mut self, factor: F) -> Option<u16> {
        if factor == F::ONE {
            return Some(0);
        }
        if factor == F::NEG_ONE {
            return Some(1);
        }
        let mask = self.table.len() - 1;
        let mut index = (self.hasher.hash_one(factor) as usize) & mask;
        loop {
            let tag = self.table[index];
            if tag == EMPTY {
                let tag = u16::try_from(self.factors.len()).ok()?;
                self.factors.push(factor);
                self.table[index] = u32::from(tag);
                return Some(tag);
            }
            if self.factors[tag as usize] == factor {
                return Some(tag as u16);
            }
            index = (index + 1) & mask;
        }
    }
}
