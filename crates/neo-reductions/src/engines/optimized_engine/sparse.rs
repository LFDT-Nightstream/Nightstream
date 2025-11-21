//! Compressed Sparse Row (CSR) and Column (CSC) matrix formats for efficient sparse operations.
//!
//! This module provides both CSR and CSC formats:
//! - CSR: Efficient for row-wise operations (y = A·x)
//! - CSC: Efficient for column-wise operations (y = A^T·x) - better cache locality

#![allow(non_snake_case)]

use neo_ccs::Mat;
use p3_field::{Field, PrimeCharacteristicRing};

/// Compressed Sparse Row format (row-major iteration).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct CsrMat<Ff> {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<Ff>,
}

/// Compressed Sparse Column format (column-major iteration).
/// Better for A^T·x operations with dense x (contiguous accumulation into y).
#[derive(Clone, Debug)]
pub struct CscMat<Ff> {
    pub nrows: usize,
    pub ncols: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub vals: Vec<Ff>,
}

#[allow(dead_code)]
impl<Ff: Field + PrimeCharacteristicRing + Copy> CsrMat<Ff> {
    /// Build CSR from a dense row-major Mat<Ff>, skipping exact zeros.
    pub fn from_dense_row_major(a: &Mat<Ff>) -> Self {
        let (nrows, ncols) = (a.rows(), a.cols());
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        let mut nnz = 0usize;

        row_ptr.push(0);
        for r in 0..nrows {
            for c in 0..ncols {
                let v = a[(r, c)];
                if v != Ff::ZERO {
                    col_idx.push(c);
                    vals.push(v);
                    nnz += 1;
                }
            }
            row_ptr.push(nnz);
        }

        CsrMat {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            vals,
        }
    }

    /// y += A^T * x (x: nrows, y: ncols)
    pub fn add_mul_transpose_into<Kf>(&self, x: &[Kf], y: &mut [Kf])
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        debug_assert_eq!(x.len(), self.nrows);
        debug_assert_eq!(y.len(), self.ncols);

        for r in 0..self.nrows {
            let xr = x[r];
            let s = self.row_ptr[r];
            let e = self.row_ptr[r + 1];
            for k in s..e {
                let c = self.col_idx[k];
                y[c] += Kf::from(self.vals[k]) * xr;
            }
        }
    }
}

impl<Ff: Field + PrimeCharacteristicRing + Copy> CscMat<Ff> {
    /// Build CSC from a dense row-major Mat<Ff>, skipping exact zeros.
    pub fn from_dense_row_major(a: &Mat<Ff>) -> Self {
        let (nrows, ncols) = (a.rows(), a.cols());

        // Count nnz per column
        let mut col_counts = vec![0usize; ncols];
        for r in 0..nrows {
            for c in 0..ncols {
                if a[(r, c)] != Ff::ZERO {
                    col_counts[c] += 1;
                }
            }
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for c in 0..ncols {
            col_ptr.push(col_ptr[c] + col_counts[c]);
        }

        let nnz = col_ptr[ncols];
        let mut row_idx = vec![0usize; nnz];
        let mut vals = vec![Ff::ZERO; nnz];

        // Temp write heads
        let mut next = col_ptr.clone();
        for r in 0..nrows {
            for c in 0..ncols {
                let v = a[(r, c)];
                if v != Ff::ZERO {
                    let k = next[c];
                    row_idx[k] = r;
                    vals[k] = v;
                    next[c] += 1;
                }
            }
        }

        CscMat {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            vals,
        }
    }

    /// y += A^T * x (x: nrows, y: ncols)
    /// Only processes rows < n_eff to avoid allocating padded x vector.
    pub fn add_mul_transpose_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        debug_assert!(n_eff <= self.nrows, "n_eff must be <= nrows");
        debug_assert!(n_eff <= x.len(), "n_eff must be <= x.len()");
        debug_assert_eq!(y.len(), self.ncols);

        for c in 0..self.ncols {
            let s = self.col_ptr[c];
            let e = self.col_ptr[c + 1];

            // Accumulate into a register: y[c] += dot(A[:,c], x)
            let mut acc = y[c];
            for k in s..e {
                let r = self.row_idx[k];
                if r < n_eff {
                    acc += Kf::from(self.vals[k]) * x[r];
                }
            }
            y[c] = acc;
        }
    }
}


