use std::num::NonZeroUsize;

use neo_math::Fq as F;
use p3_field::PrimeCharacteristicRing;

use super::R1csApplicationError;

/// SPEC.md section 7.6. A group reads one copy of these one-row columns.
#[derive(Clone, Debug)]
pub struct SharedColumns {
    pub columns: Vec<usize>,
    pub rows: NonZeroUsize,
}

impl SharedColumns {
    pub const NONE: Self = Self {
        columns: Vec::new(),
        rows: NonZeroUsize::MIN,
    };

    /// SPEC.md section 7.6. Pack local words by row, then shared words by group.
    /// `source` selects
    /// the row that supplies a group's shared values.
    pub fn step_witness(
        &self,
        width: usize,
        one: usize,
        rows: &[&[F]],
        source: impl Fn(&[&[F]]) -> usize,
    ) -> Result<Vec<F>, R1csApplicationError> {
        let count = NonZeroUsize::new(rows.len()).ok_or(R1csApplicationError::Dimension)?;
        let layout = StepLayout::new(width, one, count, self)?;
        if rows.iter().any(|row| row.len() != width) {
            return Err(R1csApplicationError::Dimension);
        }
        let mut z = vec![F::ZERO; layout.words()];
        for (row, witness) in rows.iter().enumerate() {
            for (column, value) in witness.iter().enumerate() {
                if !layout.is_shared(column) && (column != one || row == 0) {
                    z[layout.column(row, column)] = *value;
                }
            }
        }
        for (group_index, group) in rows.chunks(self.rows.get()).enumerate() {
            let witness = group
                .get(source(group))
                .ok_or(R1csApplicationError::Dimension)?;
            for &column in &self.columns {
                z[layout.column(group_index * self.rows.get(), column)] = witness[column];
            }
        }
        Ok(z)
    }
}

#[derive(Clone, Debug)]
pub(super) struct StepLayout {
    pub(super) rows: NonZeroUsize,
    pub(super) one: usize,
    pub(super) group_rows: NonZeroUsize,
    block: usize,
    shared: usize,
    columns: Vec<usize>,
    words: usize,
}

impl StepLayout {
    pub(super) fn new(
        width: usize,
        one: usize,
        rows: NonZeroUsize,
        shared: &SharedColumns,
    ) -> Result<Self, R1csApplicationError> {
        width
            .checked_mul(rows.get())
            .ok_or(R1csApplicationError::Dimension)?;
        if one >= width {
            return Err(R1csApplicationError::Column);
        }
        let block = width
            .checked_sub(shared.columns.len())
            .ok_or(R1csApplicationError::SharedColumn)?;
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(width)
            .map_err(|_| R1csApplicationError::Dimension)?;
        columns.resize(width, usize::MAX);
        for (index, &column) in shared.columns.iter().enumerate() {
            if column >= width || column == one || columns[column] != usize::MAX {
                return Err(R1csApplicationError::SharedColumn);
            }
            columns[column] = block + index;
        }
        let mut next = 0;
        for column in &mut columns {
            if *column == usize::MAX {
                *column = next;
                next += 1;
            }
        }
        let words = rows
            .get()
            .checked_mul(block)
            .and_then(|own| {
                rows.get()
                    .div_ceil(shared.rows.get())
                    .checked_mul(shared.columns.len())
                    .and_then(|blocks| own.checked_add(blocks))
            })
            .ok_or(R1csApplicationError::Dimension)?;
        Ok(Self {
            rows,
            one: columns[one],
            group_rows: shared.rows,
            block,
            shared: shared.columns.len(),
            columns,
            words,
        })
    }

    pub(super) fn is_shared(&self, column: usize) -> bool {
        self.columns[column] >= self.block
    }

    pub(super) fn words(&self) -> usize {
        self.words
    }

    pub(super) fn checked_column(&self, row: usize, column: usize) -> Option<usize> {
        (row < self.rows.get() && column < self.columns.len()).then(|| self.column(row, column))
    }

    pub(super) fn column(&self, row: usize, column: usize) -> usize {
        let column = self.columns[column];
        if column == self.one {
            column
        } else if column < self.block {
            row * self.block + column
        } else {
            self.rows.get() * self.block + row / self.group_rows.get() * self.shared + column - self.block
        }
    }
}
