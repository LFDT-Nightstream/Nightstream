//! Sparse matrix utilities for CCS.
//!
//! Neo circuits often have extremely sparse CCS matrices (e.g. exported from R1CS). Materializing
//! dense `n×m` matrices is prohibitively expensive for large circuits, so we provide a compact
//! representation based on Compressed Sparse Column (CSC).
//!
//! This module is shared by higher-level crates (folding engines) for efficient M·x and Mᵀ·x
//! operations without scanning dense zeros.
#![allow(non_snake_case)]

use crate::matrix::Mat;
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Compressed Sparse Column (CSC) format for sparse matrices.
///
/// This layout is efficient for column-wise operations and for computing `y += Aᵀ·x`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CscMat<Ff> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Column pointers (length `ncols + 1`).
    pub col_ptr: Vec<usize>,
    /// Row indices for non-zero entries (length = nnz).
    pub row_idx: Vec<usize>,
    /// Non-zero values (length = nnz).
    pub vals: Vec<Ff>,
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> CscMat<Ff> {
    /// Build a CSC matrix from (row, col, val) triplets.
    ///
    /// - Skips exact zeros.
    /// - Sorts by (col, row).
    /// - Combines duplicates by summing coefficients.
    pub fn from_triplets(mut triplets: Vec<(usize, usize, Ff)>, nrows: usize, ncols: usize) -> Self {
        triplets.retain(|&(_, _, v)| v != Ff::ZERO);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            if rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() && triplets.len() >= 16_384 {
                triplets.par_sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            } else {
                triplets.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            triplets.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        }

        let mut entries: Vec<(usize, usize, Ff)> = Vec::with_capacity(triplets.len());
        for (r, c, v) in triplets {
            assert!(r < nrows, "triplet row out of bounds");
            assert!(c < ncols, "triplet col out of bounds");
            if let Some(last) = entries.last_mut() {
                if last.0 == r && last.1 == c {
                    last.2 += v;
                    if last.2 == Ff::ZERO {
                        entries.pop();
                    }
                    continue;
                }
            }
            entries.push((r, c, v));
        }

        let mut col_counts = vec![0usize; ncols];
        let mut row_idx = Vec::with_capacity(entries.len());
        let mut vals = Vec::with_capacity(entries.len());
        for (r, c, v) in entries {
            col_counts[c] += 1;
            row_idx.push(r);
            vals.push(v);
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for c in 0..ncols {
            col_ptr.push(col_ptr[c] + col_counts[c]);
        }

        Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            vals,
        }
    }

    /// Build CSC from a dense row-major matrix, skipping exact zeros.
    ///
    /// This is parallel over rows because scans are memory-bound for large matrices.
    pub fn from_dense_row_major(a: &Mat<Ff>) -> Self {
        let (nrows, ncols) = (a.rows(), a.cols());

        let (col_counts, triplets): (Vec<usize>, Vec<(usize, usize, Ff)>) = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..nrows)
                    .into_par_iter()
                    .fold(
                        || (vec![0usize; ncols], Vec::<(usize, usize, Ff)>::new()),
                        |(mut col_counts, mut triplets), r| {
                            let row = a.row(r);
                            for (c, &v) in row.iter().enumerate() {
                                if v != Ff::ZERO {
                                    col_counts[c] += 1;
                                    triplets.push((r, c, v));
                                }
                            }
                            (col_counts, triplets)
                        },
                    )
                    .reduce(
                        || (vec![0usize; ncols], Vec::<(usize, usize, Ff)>::new()),
                        |(mut a_counts, mut a_trips), (b_counts, mut b_trips)| {
                            for c in 0..ncols {
                                a_counts[c] += b_counts[c];
                            }
                            a_trips.reserve(b_trips.len());
                            a_trips.append(&mut b_trips);
                            (a_counts, a_trips)
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                let mut col_counts = vec![0usize; ncols];
                let mut triplets = Vec::<(usize, usize, Ff)>::new();
                for r in 0..nrows {
                    let row = a.row(r);
                    for (c, &v) in row.iter().enumerate() {
                        if v != Ff::ZERO {
                            col_counts[c] += 1;
                            triplets.push((r, c, v));
                        }
                    }
                }
                (col_counts, triplets)
            }
        };

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for c in 0..ncols {
            col_ptr.push(col_ptr[c] + col_counts[c]);
        }

        let nnz = col_ptr[ncols];
        let mut row_idx = vec![0usize; nnz];
        let mut vals = vec![Ff::ZERO; nnz];

        let mut next = col_ptr.clone();
        for (r, c, v) in triplets {
            let k = next[c];
            row_idx[k] = r;
            vals[k] = v;
            next[c] += 1;
        }

        Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            vals,
        }
    }

    /// Accumulate `y += Aᵀ·x`, reading only `x[..n_eff]` and only contributing rows `< n_eff`.
    pub fn add_mul_transpose_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        debug_assert!(n_eff <= self.nrows, "n_eff must be <= nrows");
        debug_assert!(x.len() >= n_eff, "x.len() must be >= n_eff");
        debug_assert_eq!(y.len(), self.ncols);

        for c in 0..self.ncols {
            let s = self.col_ptr[c];
            let e = self.col_ptr[c + 1];
            for k in s..e {
                let r = self.row_idx[k];
                if r < n_eff {
                    y[c] += Kf::from(self.vals[k]) * x[r];
                }
            }
        }
    }

    /// Accumulate `y += A·x`, updating only `y[..n_eff]`.
    pub fn add_mul_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        debug_assert!(n_eff <= self.nrows, "n_eff must be <= nrows");
        debug_assert!(y.len() >= n_eff, "y.len() must be >= n_eff");
        debug_assert_eq!(x.len(), self.ncols);

        for c in 0..self.ncols {
            let xc = x[c];
            let s = self.col_ptr[c];
            let e = self.col_ptr[c + 1];
            for k in s..e {
                let r = self.row_idx[k];
                if r < n_eff {
                    y[r] += Kf::from(self.vals[k]) * xc;
                }
            }
        }
    }
}

/// A simple per-matrix CSC cache.
///
/// By convention, `None` can be used to represent an identity matrix `I_n` (when square).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseCache<Ff> {
    csc: Vec<Option<CscMat<Ff>>>,
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> SparseCache<Ff> {
    /// Construct from a fully prepared CSC list (one per CCS matrix).
    pub fn from_csc(csc: Vec<Option<CscMat<Ff>>>) -> Self {
        Self { csc }
    }

    /// Construct from per-matrix triplets.
    pub fn from_triplets(nrows: usize, ncols: usize, matrices: Vec<Option<Vec<(usize, usize, Ff)>>>) -> Self {
        let csc = matrices
            .into_iter()
            .map(|m| m.map(|triplets| CscMat::from_triplets(triplets, nrows, ncols)))
            .collect();
        Self::from_csc(csc)
    }

    /// Number of matrices.
    #[inline]
    pub fn len(&self) -> usize {
        self.csc.len()
    }

    /// Returns `true` if the cache contains no matrices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.csc.is_empty()
    }

    /// Get the CSC for matrix `j` (returns `None` if the matrix is an identity sentinel).
    #[inline]
    pub fn csc(&self, j: usize) -> Option<&CscMat<Ff>> {
        self.csc.get(j).and_then(|m| m.as_ref())
    }
}

/// A CCS matrix representation.
///
/// CCS matrices are typically extremely sparse. For large circuits we avoid materializing dense
/// matrices and instead keep a CSC form, with an explicit identity variant to represent `I_n`
/// without storing `n` diagonal entries.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CcsMatrix<Ff> {
    /// Identity matrix `I_n` (only valid for square CCS).
    Identity {
        /// Dimension `n` of `I_n`.
        n: usize,
    },
    /// A sparse matrix stored in CSC form.
    Csc(CscMat<Ff>),
}

impl<Ff> CcsMatrix<Ff> {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        match self {
            CcsMatrix::Identity { n } => *n,
            CcsMatrix::Csc(m) => m.nrows,
        }
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        match self {
            CcsMatrix::Identity { n } => *n,
            CcsMatrix::Csc(m) => m.ncols,
        }
    }

    /// Borrow the underlying CSC matrix, if present.
    pub fn as_csc(&self) -> Option<&CscMat<Ff>> {
        match self {
            CcsMatrix::Identity { .. } => None,
            CcsMatrix::Csc(m) => Some(m),
        }
    }
}

impl<Ff> CcsMatrix<Ff>
where
    Ff: PrimeCharacteristicRing + Copy + Eq,
{
    /// Check whether this matrix is exactly the identity matrix `I_n`.
    pub fn is_identity(&self) -> bool {
        match self {
            CcsMatrix::Identity { .. } => true,
            CcsMatrix::Csc(m) => {
                if m.nrows != m.ncols {
                    return false;
                }
                for col in 0..m.ncols {
                    let s = m.col_ptr[col];
                    let e = m.col_ptr[col + 1];
                    if e != s + 1 {
                        return false;
                    }
                    let k = s;
                    if m.row_idx[k] != col {
                        return false;
                    }
                    if m.vals[k] != Ff::ONE {
                        return false;
                    }
                }
                true
            }
        }
    }
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> CcsMatrix<Ff> {
    /// Accumulate `y += Aᵀ·x`, reading only `x[..n_eff]` and only contributing rows `< n_eff`.
    pub fn add_mul_transpose_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        match self {
            CcsMatrix::Identity { n } => {
                debug_assert_eq!(*n, y.len(), "I_n: y must have length n");
                let limit = core::cmp::min(n_eff, core::cmp::min(*n, x.len()));
                for i in 0..limit {
                    // For identity: (I^T·x)[i] = x[i]
                    y[i] += x[i];
                }
            }
            CcsMatrix::Csc(m) => m.add_mul_transpose_into(x, y, n_eff),
        }
    }

    /// Accumulate `y += A·x`, updating only `y[..n_eff]`.
    pub fn add_mul_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff>,
    {
        match self {
            CcsMatrix::Identity { n } => {
                debug_assert_eq!(*n, x.len(), "I_n: x must have length n");
                let limit = core::cmp::min(n_eff, core::cmp::min(*n, y.len()));
                for i in 0..limit {
                    // For identity: (I·x)[i] = x[i]
                    y[i] += x[i];
                }
            }
            CcsMatrix::Csc(m) => m.add_mul_into(x, y, n_eff),
        }
    }
}
