//! Sparse matrix operations using CSR (Compressed Sparse Row) format.
//!
//! Optimized CSR operations for efficient M·z and M^T·χ computations,
//! avoiding O(n*m) dense operations for highly sparse matrices.

use neo_ccs::Mat;
use p3_field::Field;
use rayon::prelude::*;

/// Minimal CSR (Compressed Sparse Row) format for sparse matrix operations.
///
/// This enables O(nnz) operations instead of O(n*m) for our extremely sparse matrices.
#[derive(Clone)]
pub struct Csr<F: Field> {
    pub rows: usize,
    pub cols: usize,
    pub indptr: Vec<usize>,  // len = rows + 1
    pub indices: Vec<usize>, // len = nnz  
    pub data: Vec<F>,        // len = nnz
}

/// Convert dense matrix to CSR format - O(nm) but done once.
pub fn to_csr<F: Field + Copy>(m: &Mat<F>, rows: usize, cols: usize) -> Csr<F> {
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for r in 0..rows {
        let row = m.row(r);
        for (c, &v) in row.iter().enumerate() {
            if v != F::ZERO {
                indices.push(c);
                data.push(v);
            }
        }
        indptr.push(indices.len());
    }
    Csr { rows, cols, indptr, indices, data }
}

/// Sparse matrix-vector multiply: y = A * x  (CSR SpMV, O(nnz)).
pub(crate) fn spmv_csr_ff<F: Field + Send + Sync + Copy>(a: &Csr<F>, x: &[F]) -> Vec<F> {
    let mut y = vec![F::ZERO; a.rows];
    y.par_iter_mut().enumerate().for_each(|(r, yr)| {
        let start = a.indptr[r];
        let end = a.indptr[r + 1];
        let mut acc = F::ZERO;
        for k in start..end {
            let c = a.indices[k];
            acc += a.data[k] * x[c];
        }
        *yr = acc;
    });
    y
}

