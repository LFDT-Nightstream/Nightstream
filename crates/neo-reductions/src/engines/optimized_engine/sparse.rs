//! Compressed Sparse Column (CSC) matrix format for efficient sparse operations.
//!
//! This module provides CSC:
//! - CSC: Efficient for column-wise operations (y = A^T·x) - better cache locality

#![allow(non_snake_case)]

use neo_ccs::{CcsMatrix, CcsStructure, Mat};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Compressed Sparse Column format (column-major iteration).
/// Better for A^T·x operations with dense x (contiguous accumulation into y).
#[derive(Clone, Debug)]
pub struct CscMat<Ff> {
    pub nrows: usize,
    pub ncols: usize,
    pub col_ptr: Vec<u32>,
    pub row_idx: Vec<u32>,
    pub vals: Vec<Ff>,
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> CscMat<Ff> {
    /// Build CSC from a list of (row, col, val) triplets (skipping zeros and combining duplicates).
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
            col_ptr: compact_indices(col_ptr),
            row_idx: compact_indices(row_idx),
            vals,
        }
    }

    /// Build CSC from a dense row-major Mat<Ff>, skipping exact zeros.
    ///
    /// This is intentionally parallel over rows: large CCS matrices are often extremely sparse
    /// but stored densely (e.g. from exported R1CS). Scanning is memory-bound, so we want to
    /// saturate bandwidth and amortize branch mispredictions across cores.
    pub fn from_dense_row_major(a: &Mat<Ff>) -> Self {
        let (nrows, ncols) = (a.rows(), a.cols());

        // Count nnz per column and collect (row, col, val) triplets in one pass.
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

        // Temp write heads
        let mut next = col_ptr.clone();
        for (r, c, v) in triplets {
            let k = next[c];
            row_idx[k] = r;
            vals[k] = v;
            next[c] += 1;
        }

        CscMat {
            nrows,
            ncols,
            col_ptr: compact_indices(col_ptr),
            row_idx: compact_indices(row_idx),
            vals,
        }
    }

    /// y += A^T * x (x: nrows, y: ncols).
    /// Only reads rows < n_eff.
    pub fn add_mul_transpose_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff> + Send + Sync,
    {
        debug_assert!(n_eff <= self.nrows, "n_eff must be <= nrows");
        debug_assert!(
            x.len() >= n_eff,
            "x.len() must be >= n_eff (got {}, need {})",
            x.len(),
            n_eff
        );
        debug_assert_eq!(y.len(), self.ncols);

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            y.par_iter_mut().enumerate().for_each(|(c, yc)| {
                let s = self.col_ptr[c] as usize;
                let e = self.col_ptr[c + 1] as usize;
                if s == e {
                    return;
                }
                let mut sum = *yc;
                for k in s..e {
                    let r = self.row_idx[k] as usize;
                    if r < n_eff {
                        sum += Kf::from(self.vals[k]) * x[r];
                    }
                }
                *yc = sum;
            });
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            for c in 0..self.ncols {
                let s = self.col_ptr[c] as usize;
                let e = self.col_ptr[c + 1] as usize;
                if s == e {
                    continue;
                }
                let mut sum = y[c];
                for k in s..e {
                    let r = self.row_idx[k] as usize;
                    if r < n_eff {
                        sum += Kf::from(self.vals[k]) * x[r];
                    }
                }
                y[c] = sum;
            }
        }
    }

    /// y += A * x (x: ncols, y: nrows).
    /// Only updates rows < n_eff.
    pub fn add_mul_into<Kf>(&self, x: &[Kf], y: &mut [Kf], n_eff: usize)
    where
        Kf: Copy + core::ops::AddAssign + core::ops::Mul<Output = Kf> + From<Ff> + PartialEq,
    {
        debug_assert!(n_eff <= self.nrows, "n_eff must be <= nrows");
        debug_assert!(y.len() >= n_eff, "y.len() must be >= n_eff");
        debug_assert_eq!(x.len(), self.ncols);

        let zero = Kf::from(Ff::ZERO);
        for c in 0..self.ncols {
            let xc = x[c];
            if xc == zero {
                continue;
            }
            let s = self.col_ptr[c] as usize;
            let e = self.col_ptr[c + 1] as usize;
            for k in s..e {
                let r = self.row_idx[k] as usize;
                if r < n_eff {
                    y[r] += Kf::from(self.vals[k]) * xc;
                }
            }
        }
    }
}

fn compact_indices(indices: Vec<usize>) -> Vec<u32> {
    indices
        .into_iter()
        .map(|index| u32::try_from(index).unwrap_or_else(|_| panic!("CSC index exceeds u32: {index}")))
        .collect()
}

/// Cache of CSC matrix formats used by optimized routines.
#[derive(Clone)]
pub struct SparseCache<Ff> {
    source: SparseCacheSource<Ff>,
}

#[derive(Clone)]
enum SparseCacheSource<Ff> {
    Owned(Vec<Option<CscMat<Ff>>>),
    Shared(std::sync::Arc<CcsStructure<Ff>>),
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> SparseCache<Ff> {
    pub fn from_csc(csc: Vec<Option<CscMat<Ff>>>) -> Self {
        Self {
            source: SparseCacheSource::Owned(csc),
        }
    }

    pub fn from_shared_structure(structure: std::sync::Arc<CcsStructure<Ff>>) -> Self {
        Self {
            source: SparseCacheSource::Shared(structure),
        }
    }

    pub fn from_triplets(nrows: usize, ncols: usize, matrices: Vec<Option<Vec<(usize, usize, Ff)>>>) -> Self {
        let csc = matrices
            .into_iter()
            .map(|m| m.map(|triplets| CscMat::from_triplets(triplets, nrows, ncols)))
            .collect();
        Self::from_csc(csc)
    }

    pub fn build(s: &CcsStructure<Ff>) -> Self {
        let t = s.t();

        // `CcsStructure` stores matrices sparsely (CSC/identity). Build the cache by cloning the
        // existing CSC data (no dense scanning).
        let mut csc: Vec<Option<CscMat<Ff>>> = Vec::with_capacity(t);
        for j in 0..t {
            match &s.matrices[j] {
                CcsMatrix::Identity { .. } => csc.push(None),
                CcsMatrix::Csc(m) => csc.push(Some(CscMat {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    col_ptr: m.col_ptr.clone(),
                    row_idx: m.row_idx.clone(),
                    vals: m.vals.clone(),
                })),
                CcsMatrix::CscWithSeededPhi81 { csc: m, .. } => csc.push(Some(CscMat {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    col_ptr: m.col_ptr.clone(),
                    row_idx: m.row_idx.clone(),
                    vals: m.vals.clone(),
                })),
            }
        }

        Self::from_csc(csc)
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.source {
            SparseCacheSource::Owned(csc) => csc.len(),
            SparseCacheSource::Shared(structure) => structure.matrices.len(),
        }
    }

    #[inline]
    pub fn csc(&self, j: usize) -> Option<&CscMat<Ff>> {
        match &self.source {
            SparseCacheSource::Owned(csc) => csc.get(j).and_then(|matrix| matrix.as_ref()),
            // Shared structures already expose their canonical neo-ccs CSC
            // arrays to the digest path. Returning `None` selects those
            // arrays without cloning them into this engine-local CSC type.
            SparseCacheSource::Shared(_) => None,
        }
    }
}
