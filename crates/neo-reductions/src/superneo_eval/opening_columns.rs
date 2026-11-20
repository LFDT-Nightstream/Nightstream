use std::hash::{BuildHasher, RandomState};

use super::factored::fill;
use super::{FactoredEntries, MatrixShape};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

#[derive(Default)]
pub struct OpeningColumns {
    columns: std::sync::OnceLock<Option<Vec<Columns>>>,
    pub(super) runs: std::sync::OnceLock<Option<super::SlotRuns>>,
}

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
        let columns = if let Some(kept) = self.columns.get() {
            kept
        } else {
            if construction_bytes(shape, slots, entries.slots.len())? > available {
                return None;
            }
            self.columns.get_or_init(|| {
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
    row_count: usize,
    tiles: std::sync::OnceLock<Option<Box<super::OpeningTiles>>>,
    pub(super) slots: Vec<u32>,
    offsets: Vec<u32>,
    rows: PackedRows,
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
        let mut rows = PackedRows::new(shape.rows, total as usize)?;
        let mut tags = vec![0; total as usize];
        let mut ids = FactorIds::new(total as usize);
        for (row, range) in ranges().enumerate() {
            for entry in range {
                let next = &mut counts[entries.slots[entry] as usize];
                rows.set(*next as usize, row as u32);
                tags[*next as usize] = ids.tag(entries.coefficients[entry])?;
                *next += 1;
            }
        }
        let mut factors = ids.factors;
        factors.shrink_to_fit();
        Some(Self {
            tiles: Default::default(),
            row_count: shape.rows,
            slots,
            offsets,
            rows,
            tags,
            factors,
        })
    }

    pub(super) fn device_matrix<'a>(
        &'a self,
        layout: super::FactoredSlots<'a>,
        blocks: usize,
        row_count: usize,
        available: usize,
    ) -> Option<super::OpeningMatrix<'a>> {
        if row_count != self.row_count
            || layout.starts.len() != layout.widths.len()
            || self
                .slots
                .last()
                .is_some_and(|&s| s as usize >= layout.widths.len())
        {
            return None;
        }
        let tiles = if let Some(kept) = self.tiles.get() {
            kept
        } else {
            let sizes = super::OpeningTiles::sizes(blocks, &self.offsets)?;
            if sizes.0 > available {
                return None;
            }
            self.tiles.get_or_init(|| {
                super::OpeningTiles::prepare(layout, blocks, &self.slots, &self.offsets, sizes.1, sizes.2)
            })
        }
        .as_deref()?;
        if tiles.first.len() != blocks + 1 || tiles.powers[1] != layout.ratio {
            return None;
        }
        Some(super::OpeningMatrix {
            layout,
            slots: &self.slots,
            offsets: &self.offsets,
            rows: &self.rows.bytes,
            row_bytes: self.rows.stride,
            row_count,
            tags: &self.tags,
            factors: &self.factors,
            tiles,
        })
    }

    pub(super) fn storage_bytes(&self) -> usize {
        self.tiles
            .get()
            .and_then(Option::as_ref)
            .map_or(0, |tiles| tiles.bytes())
            + (self.slots.capacity() + self.offsets.capacity()) * size_of::<u32>()
            + self.rows.bytes.capacity()
            + self.tags.capacity() * size_of::<u16>()
            + self.factors.capacity() * size_of::<F>()
    }

    pub(super) fn weights(&self, row_weights: &[K], out: &mut Vec<K>) {
        out.resize(self.slots.len(), K::ZERO);
        fill(out, |slot| self.weight(row_weights, slot));
    }

    pub(super) fn weight(&self, row_weights: &[K], slot: usize) -> K {
        let range = self.offsets[slot] as usize..self.offsets[slot + 1] as usize;
        self.rows
            .iter(range.clone())
            .zip(&self.tags[range])
            .map(|(row, &tag)| {
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
        let terms = self.rows.iter(range.clone()).zip(&self.tags[range]);
        neo_math::sum_of_scaled(terms.map(|(row, &tag)| (self.factors[tag as usize], row_weights[row as usize])))
    }
}

struct PackedRows {
    bytes: Vec<u8>,
    stride: usize,
    mask: u32,
}

impl PackedRows {
    fn new(rows: usize, n: usize) -> Option<Self> {
        let max = u32::try_from(rows.saturating_sub(1)).ok()?;
        let stride = (usize::try_from(u32::BITS - max.leading_zeros())
            .ok()?
            .div_ceil(8))
        .max(1);
        let mask = u32::MAX >> ((4 - stride) * 8);
        let size = n.checked_mul(stride)?.checked_add(4 - stride)?;
        Some(Self {
            bytes: vec![0; size],
            stride,
            mask,
        })
    }
    fn set(&mut self, index: usize, row: u32) {
        self.bytes[index * self.stride..][..self.stride].copy_from_slice(&row.to_le_bytes()[..self.stride]);
    }
    fn iter(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = u32> + '_ {
        let mut data = &self.bytes[range.start * self.stride..range.end * self.stride + (4 - self.stride)];
        std::iter::from_fn(move || {
            let word = u32::from_le_bytes(*data.first_chunk::<4>()?) & self.mask;
            data = &data[self.stride..];
            Some(word)
        })
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

#[cfg(test)]
#[path = "../../tests/unit/opening_rows.rs"]
mod tests;
