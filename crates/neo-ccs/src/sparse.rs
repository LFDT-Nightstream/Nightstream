//! Sparse matrix utilities for CCS.
//!
//! Neo circuits often have extremely sparse CCS matrices (e.g. exported from R1CS). Materializing
//! dense `n×m` matrices is prohibitively expensive for large circuits, so we provide a compact
//! representation based on Compressed Sparse Column (CSC).
//!
//! This module is shared by higher-level crates (folding engines) for efficient M·x and Mᵀ·x
//! operations without scanning dense zeros.
#![allow(non_snake_case)]

use crate::geometric::GeometricRowRun;
use crate::matrix::Mat;
use crate::seeded_phi81::{SeededPhi81Error, SeededPhi81LinearBlock};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::collections::BTreeMap;

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
    pub col_ptr: Vec<u32>,
    /// Row indices for non-zero entries (length = nnz).
    pub row_idx: Vec<u32>,
    /// Non-zero values (length = nnz).
    pub vals: Vec<Ff>,
}

impl<Ff> CscMat<Ff> {
    /// Return the compact-entry range for one column.
    #[inline]
    pub fn column_range(&self, column: usize) -> core::ops::Range<usize> {
        self.col_ptr[column] as usize..self.col_ptr[column + 1] as usize
    }

    /// Return one compact row index as a native slice index.
    #[inline]
    pub fn row_index(&self, entry: usize) -> usize {
        self.row_idx[entry] as usize
    }
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
            col_ptr: compact_csc_indices(col_ptr, "column pointer"),
            row_idx: compact_csc_indices(row_idx, "row index"),
            vals,
        }
    }

    /// Build canonical CSC by counting entries into columns, then sorting only
    /// the rows inside each column. This avoids one global `(column, row)` sort
    /// while producing the same canonical arrays as [`Self::from_triplets`].
    pub fn from_counted_triplets(triplets: Vec<(usize, usize, Ff)>, nrows: usize, ncols: usize) -> Self {
        let mut column_counts = vec![0usize; ncols];
        let mut nonzero_count = 0usize;
        for &(row, column, value) in &triplets {
            assert!(row < nrows, "triplet row out of bounds");
            assert!(column < ncols, "triplet col out of bounds");
            if value != Ff::ZERO {
                column_counts[column] += 1;
                nonzero_count += 1;
            }
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for count in column_counts {
            col_ptr.push(col_ptr.last().copied().expect("CSC pointer") + count);
        }
        let mut next = col_ptr[..ncols].to_vec();
        let mut entries = vec![(0usize, Ff::ZERO); nonzero_count];
        for (row, column, value) in triplets {
            if value == Ff::ZERO {
                continue;
            }
            let index = next[column];
            entries[index] = (row, value);
            next[column] += 1;
        }
        for column in 0..ncols {
            entries[col_ptr[column]..col_ptr[column + 1]].sort_unstable_by_key(|&(row, _)| row);
        }

        let mut write = 0usize;
        for column in 0..ncols {
            let read_start = col_ptr[column];
            let read_end = col_ptr[column + 1];
            col_ptr[column] = write;
            let mut read = read_start;
            while read < read_end {
                let row = entries[read].0;
                let mut value = entries[read].1;
                read += 1;
                while read < read_end && entries[read].0 == row {
                    value += entries[read].1;
                    read += 1;
                }
                if value != Ff::ZERO {
                    entries[write] = (row, value);
                    write += 1;
                }
            }
        }
        col_ptr[ncols] = write;
        entries.truncate(write);
        let (row_idx, vals) = entries.into_iter().unzip();

        Self {
            nrows,
            ncols,
            col_ptr: compact_csc_indices(col_ptr, "column pointer"),
            row_idx: compact_csc_indices(row_idx, "row index"),
            vals,
        }
    }

    /// Build canonical CSC directly from explicit terms plus compact
    /// geometric row runs.
    ///
    /// The final arrays are identical to expanding every run and calling
    /// [`Self::from_triplets`], but no expanded triplet vector is allocated.
    pub fn from_triplets_and_geometric_runs(
        triplets: Vec<(usize, usize, Ff)>,
        runs: &[GeometricRowRun<Ff>],
        nrows: usize,
        ncols: usize,
    ) -> Self {
        let mut column_counts = vec![0usize; ncols];
        let mut nonzero_count = 0usize;
        for &(row, column, value) in &triplets {
            assert!(row < nrows, "triplet row out of bounds");
            assert!(column < ncols, "triplet col out of bounds");
            if value != Ff::ZERO {
                column_counts[column] += 1;
                nonzero_count += 1;
            }
        }
        for run in runs {
            assert!(run.validate_shape(nrows, ncols), "geometric run out of bounds");
            run.for_each_term(|_, column, _| {
                column_counts[column] += 1;
                nonzero_count += 1;
            });
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for count in column_counts {
            col_ptr.push(col_ptr.last().copied().expect("CSC pointer") + count);
        }
        let mut next = col_ptr[..ncols].to_vec();
        let mut entries = vec![(0usize, Ff::ZERO); nonzero_count];
        for (row, column, value) in triplets {
            if value == Ff::ZERO {
                continue;
            }
            let index = next[column];
            entries[index] = (row, value);
            next[column] += 1;
        }
        for run in runs {
            run.for_each_term(|row, column, value| {
                let index = next[column];
                entries[index] = (row, value);
                next[column] += 1;
            });
        }
        for column in 0..ncols {
            entries[col_ptr[column]..col_ptr[column + 1]].sort_unstable_by_key(|&(row, _)| row);
        }

        let mut write = 0usize;
        for column in 0..ncols {
            let read_start = col_ptr[column];
            let read_end = col_ptr[column + 1];
            col_ptr[column] = write;
            let mut read = read_start;
            while read < read_end {
                let row = entries[read].0;
                let mut value = entries[read].1;
                read += 1;
                while read < read_end && entries[read].0 == row {
                    value += entries[read].1;
                    read += 1;
                }
                if value != Ff::ZERO {
                    entries[write] = (row, value);
                    write += 1;
                }
            }
        }
        col_ptr[ncols] = write;
        entries.truncate(write);
        let (row_idx, vals) = entries.into_iter().unzip();
        Self {
            nrows,
            ncols,
            col_ptr: compact_csc_indices(col_ptr, "column pointer"),
            row_idx: compact_csc_indices(row_idx, "row index"),
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
            col_ptr: compact_csc_indices(col_ptr, "column pointer"),
            row_idx: compact_csc_indices(row_idx, "row index"),
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
            for k in self.column_range(c) {
                let r = self.row_index(k);
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
            for k in self.column_range(c) {
                let r = self.row_index(k);
                if r < n_eff {
                    y[r] += Kf::from(self.vals[k]) * xc;
                }
            }
        }
    }
}

fn compact_csc_indices(indices: Vec<usize>, kind: &str) -> Vec<u32> {
    indices
        .into_iter()
        .map(|index| u32::try_from(index).unwrap_or_else(|_| panic!("CSC {kind} exceeds u32: {index}")))
        .collect()
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
    /// A sparse CSC base plus compact seeded Phi81 linear blocks.
    ///
    /// The blocks are part of the matrix, not auxiliary advice. Their public
    /// chunk seeds deterministically define every omitted coefficient.
    CscWithSeededPhi81 {
        /// Ordinary sparse terms not owned by a compact block.
        csc: CscMat<Ff>,
        /// Compact seeded blocks, each occupying disjoint constraint rows.
        blocks: Vec<SeededPhi81LinearBlock>,
        /// Compact contiguous radix expansions in individual rows.
        geometric_runs: Vec<GeometricRowRun<Ff>>,
    },
}

impl<Ff> CcsMatrix<Ff> {
    /// Build a matrix from an ordinary CSC base and compact seeded blocks.
    pub fn csc_with_seeded_phi81(
        csc: CscMat<Ff>,
        blocks: Vec<SeededPhi81LinearBlock>,
    ) -> Result<Self, SeededPhi81Error> {
        if blocks.is_empty() {
            return Ok(Self::Csc(csc));
        }
        for block in &blocks {
            block.validate_matrix_shape(csc.nrows, csc.ncols)?;
        }
        Ok(Self::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs: Vec::new(),
        })
    }

    /// Build a matrix from ordinary CSC terms and compact structured terms.
    pub fn csc_with_compact_rows(
        csc: CscMat<Ff>,
        blocks: Vec<SeededPhi81LinearBlock>,
        mut geometric_runs: Vec<GeometricRowRun<Ff>>,
    ) -> Result<Self, String> {
        if blocks.is_empty() && geometric_runs.is_empty() {
            return Ok(Self::Csc(csc));
        }
        for block in &blocks {
            block
                .validate_matrix_shape(csc.nrows, csc.ncols)
                .map_err(|error| error.to_string())?;
        }
        for (index, run) in geometric_runs.iter().enumerate() {
            if !run.validate_shape(csc.nrows, csc.ncols) {
                return Err(format!(
                    "geometric row run {index} lies outside {}x{} matrix",
                    csc.nrows, csc.ncols
                ));
            }
        }
        geometric_runs.sort_unstable_by_key(|run| (run.row(), run.column_start(), run.len()));
        Ok(Self::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        match self {
            CcsMatrix::Identity { n } => *n,
            CcsMatrix::Csc(m) => m.nrows,
            CcsMatrix::CscWithSeededPhi81 { csc, .. } => csc.nrows,
        }
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        match self {
            CcsMatrix::Identity { n } => *n,
            CcsMatrix::Csc(m) => m.ncols,
            CcsMatrix::CscWithSeededPhi81 { csc, .. } => csc.ncols,
        }
    }

    /// Borrow the underlying CSC matrix, if present.
    pub fn as_csc(&self) -> Option<&CscMat<Ff>> {
        match self {
            CcsMatrix::Identity { .. } => None,
            CcsMatrix::Csc(m) => Some(m),
            CcsMatrix::CscWithSeededPhi81 { .. } => None,
        }
    }

    /// Borrow the ordinary sparse component, excluding compact blocks.
    pub fn sparse_component(&self) -> Option<&CscMat<Ff>> {
        match self {
            CcsMatrix::Identity { .. } => None,
            CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => Some(csc),
        }
    }

    /// Borrow the compact seeded blocks in this matrix.
    pub fn seeded_phi81_blocks(&self) -> &[SeededPhi81LinearBlock] {
        match self {
            CcsMatrix::CscWithSeededPhi81 { blocks, .. } => blocks,
            CcsMatrix::Identity { .. } | CcsMatrix::Csc(_) => &[],
        }
    }

    /// Borrow compact geometric row runs in this matrix.
    pub fn geometric_runs(&self) -> &[GeometricRowRun<Ff>] {
        match self {
            CcsMatrix::CscWithSeededPhi81 { geometric_runs, .. } => geometric_runs,
            CcsMatrix::Identity { .. } | CcsMatrix::Csc(_) => &[],
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
                    let range = m.column_range(col);
                    if range.end != range.start + 1 {
                        return false;
                    }
                    let k = range.start;
                    if m.row_index(k) != col {
                        return false;
                    }
                    if m.vals[k] != Ff::ONE {
                        return false;
                    }
                }
                true
            }
            CcsMatrix::CscWithSeededPhi81 { .. } => false,
        }
    }
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> CcsMatrix<Ff> {
    /// Materialize one exact sparse row from every additive matrix component.
    ///
    /// The result is sorted by column, contains no duplicate columns or zero
    /// coefficients, and includes ordinary CSC, seeded Phi81, and geometric
    /// contributions after field addition. `None` means only that `row` is
    /// outside the matrix.
    pub fn materialize_row(&self, row: usize) -> Option<Vec<(usize, Ff)>> {
        if row >= self.rows() {
            return None;
        }
        let mut terms = BTreeMap::<usize, Ff>::new();
        match self {
            CcsMatrix::Identity { .. } => accumulate_row_term(&mut terms, row, Ff::ONE),
            CcsMatrix::Csc(csc) => accumulate_csc_row(&mut terms, csc, row),
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                accumulate_csc_row(&mut terms, csc, row);
                for block in blocks {
                    block.for_each_row_term::<Ff, _>(row, |column, coefficient| {
                        accumulate_row_term(&mut terms, column, coefficient);
                    });
                }
                for run in geometric_runs.iter().filter(|run| run.row() == row) {
                    run.for_each_term(|_, column, coefficient| {
                        accumulate_row_term(&mut terms, column, coefficient);
                    });
                }
            }
        }
        Some(
            terms
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != Ff::ZERO)
                .collect(),
        )
    }

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
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                csc.add_mul_transpose_into(x, y, n_eff);
                for block in blocks {
                    block.add_mul_transpose_into::<Ff, Kf>(x, y, n_eff);
                }
                for run in geometric_runs {
                    run.add_mul_transpose_into(x, y, n_eff);
                }
            }
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
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                csc.add_mul_into(x, y, n_eff);
                for block in blocks {
                    block.add_mul_into::<Ff, Kf>(x, y, n_eff);
                }
                for run in geometric_runs {
                    run.add_mul_into(x, y, n_eff);
                }
            }
        }
    }
}

fn accumulate_row_term<Ff>(terms: &mut BTreeMap<usize, Ff>, column: usize, coefficient: Ff)
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if coefficient != Ff::ZERO {
        *terms.entry(column).or_insert(Ff::ZERO) += coefficient;
    }
}

fn accumulate_csc_row<Ff>(terms: &mut BTreeMap<usize, Ff>, csc: &CscMat<Ff>, row: usize)
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    for (entry, &candidate_row) in csc.row_idx.iter().enumerate() {
        if candidate_row as usize != row {
            continue;
        }
        let pointer = csc
            .col_ptr
            .partition_point(|&start| start as usize <= entry);
        let column = pointer
            .checked_sub(1)
            .filter(|&column| column < csc.ncols)
            .expect("well-formed CSC entry must have one owning column");
        accumulate_row_term(terms, column, csc.vals[entry]);
    }
}
