//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Spartan.
use crate::errors::SpartanError;
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<F: PrimeField> {
  /// all non-zero values in the matrix
  pub data: Vec<F>,
  /// column indices
  pub indices: Vec<usize>,
  /// row information
  pub indptr: Vec<usize>,
  /// number of columns
  pub cols: usize,
}

impl<F: PrimeField> SparseMatrix<F> {
  /// 0x0 empty matrix
  pub fn empty() -> Self {
    SparseMatrix {
      data: vec![],
      indices: vec![],
      indptr: vec![0],
      cols: 0,
    }
  }

  /// Number of rows in the matrix.
  pub fn rows(&self) -> usize {
    self.indptr.len().saturating_sub(1)
  }

  /// Number of columns in the matrix.
  pub fn cols(&self) -> usize {
    self.cols
  }

  /// Number of non-zero entries in the matrix.
  pub fn nnz(&self) -> usize {
    self.indptr.last().copied().unwrap_or(0)
  }

  /// Construct a sparse matrix from canonical CSR arrays.
  ///
  /// Each row must contain strictly increasing column indices. Zero
  /// coefficients are rejected so that one relation has one canonical sparse
  /// representation.
  pub fn from_csr(
    rows: usize,
    cols: usize,
    data: Vec<F>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
  ) -> Result<Self, SpartanError> {
    if indptr.len() != rows.saturating_add(1)
      || indptr.first().copied() != Some(0)
      || data.len() != indices.len()
      || indptr.last().copied() != Some(data.len())
      || indptr.windows(2).any(|window| window[0] > window[1])
    {
      return Err(SpartanError::InvalidInputLength {
        reason: "invalid CSR dimensions or row pointers".to_string(),
      });
    }

    for row in indptr.windows(2) {
      let row_indices = &indices[row[0]..row[1]];
      if row_indices.iter().any(|&column| column >= cols)
        || row_indices.windows(2).any(|window| window[0] >= window[1])
      {
        return Err(SpartanError::InvalidIndex);
      }
    }
    if data.iter().any(|value| *value == F::ZERO) {
      return Err(SpartanError::InvalidInputLength {
        reason: "canonical CSR matrices cannot contain zero coefficients".to_string(),
      });
    }

    Ok(Self {
      data,
      indices,
      indptr,
      cols,
    })
  }

  /// Retrieves the data for row slice [i..j] from `ptrs`.
  /// We assume that `ptrs` is indexed from `indptrs` and do not check if the
  /// returned slice is actually a valid row.
  pub fn get_row_unchecked(&self, ptrs: &[usize; 2]) -> impl Iterator<Item = (&F, &usize)> {
    self.data[ptrs[0]..ptrs[1]]
      .iter()
      .zip(&self.indices[ptrs[0]..ptrs[1]])
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  ///
  /// # Errors
  /// Returns `SpartanError::InvalidInputLength` if the vector length doesn't match the matrix dimensions.
  pub fn multiply_vec(&self, vector: &[F]) -> Result<Vec<F>, SpartanError> {
    if self.cols != vector.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SparseMatrix multiply_vec: Expected {} elements in vector, got {}",
          self.cols,
          vector.len()
        ),
      });
    }

    Ok(self.multiply_vec_unchecked(vector))
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  /// This does not check that the shape of the matrix/vector are compatible.
  pub fn multiply_vec_unchecked(&self, vector: &[F]) -> Vec<F> {
    if crate::parallel::parallelism_enabled() {
      self
        .indptr
        .par_windows(2)
        .map(|ptrs| {
          // par_windows(2) guarantees ptrs has exactly 2 elements
          let row_ptrs = [ptrs[0], ptrs[1]];
          self
            .get_row_unchecked(&row_ptrs)
            .map(|(val, col_idx)| *val * vector[*col_idx])
            .sum()
        })
        .collect()
    } else {
      self
        .indptr
        .windows(2)
        .map(|ptrs| {
          let row_ptrs = [ptrs[0], ptrs[1]];
          self
            .get_row_unchecked(&row_ptrs)
            .map(|(val, col_idx)| *val * vector[*col_idx])
            .sum()
        })
        .collect()
    }
  }

  /// returns a custom iterator
  pub fn iter(&self) -> Iter<'_, F> {
    let mut row = 0;
    while row + 1 < self.indptr.len() && self.indptr[row + 1] == 0 {
      row += 1;
    }
    let nnz = if self.indptr.is_empty() {
      0
    } else {
      self.indptr[self.indptr.len() - 1]
    };
    Iter {
      matrix: self,
      row,
      i: 0,
      nnz,
    }
  }
}

/// Iterator for sparse matrix
pub struct Iter<'a, F: PrimeField> {
  matrix: &'a SparseMatrix<F>,
  row: usize,
  i: usize,
  nnz: usize,
}

impl<'a, F: PrimeField> Iterator for Iter<'a, F> {
  type Item = (usize, usize, F);

  fn next(&mut self) -> Option<Self::Item> {
    // are we at the end?
    if self.i == self.nnz {
      return None;
    }

    // compute current item
    let curr_item = (
      self.row,
      self.matrix.indices[self.i],
      self.matrix.data[self.i],
    );

    // advance the iterator
    self.i += 1;
    // edge case at the end
    if self.i == self.nnz {
      return Some(curr_item);
    }
    // if `i` has moved to next row
    while self.i >= self.matrix.indptr[self.row + 1] {
      self.row += 1;
    }

    Some(curr_item)
  }
}
