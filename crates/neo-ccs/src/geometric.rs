//! Compact storage for contiguous geometric coefficients in one matrix row.

use p3_field::{Field, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};

/// The row terms `initial * ratio^i` over
/// `column_start..column_start + len`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometricRowRun<F> {
    row: usize,
    column_start: usize,
    len: usize,
    initial: F,
    ratio: F,
}

impl<F> GeometricRowRun<F> {
    /// Construct one nonempty run.
    pub fn new(row: usize, column_start: usize, len: usize, initial: F, ratio: F) -> Self {
        assert!(len > 0, "geometric matrix run must be nonempty");
        Self {
            row,
            column_start,
            len,
            initial,
            ratio,
        }
    }

    /// Matrix row containing the run.
    pub fn row(&self) -> usize {
        self.row
    }

    /// First matrix column in the run.
    pub fn column_start(&self) -> usize {
        self.column_start
    }

    /// Number of consecutive columns in the run.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Geometric runs are always nonempty by construction.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Coefficient at the first column.
    pub fn initial(&self) -> &F {
        &self.initial
    }

    /// Ratio between adjacent coefficients.
    pub fn ratio(&self) -> &F {
        &self.ratio
    }

    /// Shift the run into a block-diagonal matrix.
    pub fn shifted(&self, row_offset: usize, column_offset: usize) -> Self
    where
        F: Clone,
    {
        Self {
            row: self.row + row_offset,
            column_start: self.column_start + column_offset,
            len: self.len,
            initial: self.initial.clone(),
            ratio: self.ratio.clone(),
        }
    }

    /// Check that this run lies inside an enclosing matrix.
    pub fn validate_shape(&self, rows: usize, columns: usize) -> bool {
        self.row < rows
            && self
                .column_start
                .checked_add(self.len)
                .is_some_and(|end| end <= columns)
    }
}

impl<F> GeometricRowRun<F>
where
    F: Field + PrimeCharacteristicRing + Copy,
{
    /// Return one matrix entry contributed by this run.
    pub fn entry(&self, row: usize, column: usize) -> F {
        if row != self.row || !(self.column_start..self.column_start + self.len).contains(&column) {
            return F::ZERO;
        }
        let mut coefficient = self.initial;
        for _ in self.column_start..column {
            coefficient *= self.ratio;
        }
        coefficient
    }

    /// Visit the expanded nonzero matrix terms in column order.
    pub fn for_each_term(&self, mut visit: impl FnMut(usize, usize, F)) {
        let mut coefficient = self.initial;
        for offset in 0..self.len {
            if coefficient != F::ZERO {
                visit(self.row, self.column_start + offset, coefficient);
            }
            coefficient *= self.ratio;
        }
    }

    /// Accumulate this run into `y += A*x`.
    pub fn add_mul_into<K>(&self, x: &[K], y: &mut [K], n_eff: usize)
    where
        K: Copy + core::ops::AddAssign + core::ops::Mul<Output = K> + From<F>,
    {
        if self.row >= n_eff || self.row >= y.len() {
            return;
        }
        let mut coefficient = self.initial;
        for column in self.column_start..self.column_start + self.len {
            y[self.row] += K::from(coefficient) * x[column];
            coefficient *= self.ratio;
        }
    }

    /// Accumulate this run into `y += A^T*x`.
    pub fn add_mul_transpose_into<K>(&self, x: &[K], y: &mut [K], n_eff: usize)
    where
        K: Copy + core::ops::AddAssign + core::ops::Mul<Output = K> + From<F>,
    {
        if self.row >= n_eff || self.row >= x.len() {
            return;
        }
        let mut coefficient = self.initial;
        for column in self.column_start..self.column_start + self.len {
            y[column] += K::from(coefficient) * x[self.row];
            coefficient *= self.ratio;
        }
    }
}
