use core::ops::{Index, IndexMut};
use p3_field::PrimeCharacteristicRing;

/// A dense row-major matrix over a field-like type `T`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Mat<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Mat<T> {
    /// Create a matrix from row-major data; panics if `data.len() != rows*cols`.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    /// Zero-initialized matrix (caller provides zero element).
    pub fn zero(rows: usize, cols: usize, zero: T) -> Self {
        Self { rows, cols, data: vec![zero; rows * cols] }
    }

    /// Rows.
    pub fn rows(&self) -> usize { self.rows }

    /// Cols.
    pub fn cols(&self) -> usize { self.cols }

    /// Underlying row-major slice.
    pub fn as_slice(&self) -> &[T] { &self.data }

    /// Mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }

    /// Row i as a slice.
    pub fn row(&self, i: usize) -> &[T] {
        let start = i * self.cols;
        &self.data[start .. start + self.cols]
    }

    /// Row i as a mutable slice.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.cols;
        &mut self.data[start .. start + self.cols]
    }

    /// Append `k` zero rows to the matrix in-place.
    /// The caller must provide the zero element for the field type.
    pub fn append_zero_rows(&mut self, k: usize, zero: T) {
        if k == 0 { return; }
        let extra = k * self.cols;
        self.data.resize(self.data.len() + extra, zero);
        self.rows += k;
    }

    /// Set a single entry at (row, col) to the provided value.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        debug_assert!(row < self.rows, "row out of bounds");
        debug_assert!(col < self.cols, "col out of bounds");
        self.data[row * self.cols + col] = val;
    }
}

impl<F> Mat<F>
where
    F: p3_field::PrimeCharacteristicRing + Copy + Eq,
{
    /// Construct an identity matrix I_n over field F.
    pub fn identity(n: usize) -> Self {
        let mut m = Mat::zero(n, n, F::ZERO);
        for i in 0..n { m.set(i, i, F::ONE); }
        m
    }

    /// Check whether this matrix is exactly the identity matrix (I_n).
    pub fn is_identity(&self) -> bool {
        if self.rows != self.cols { return false; }
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = self[(r, c)];
                if r == c {
                    if v != F::ONE { return false; }
                } else if v != F::ZERO { return false; }
            }
        }
        true
    }

    /// Check whether this matrix is a column selector: each column has exactly one 1 and zeros elsewhere.
    /// Rows and cols can be different. This recognizes matrices used to expose Ajtai digits
    /// via v = M^T * chi_r, where v[c] = chi_r[row_map(c)].
    pub fn is_column_selector(&self) -> bool {
        for c in 0..self.cols {
            let mut ones = 0usize;
            for r in 0..self.rows {
                let v = self[(r, c)];
                if v == F::ONE {
                    ones += 1;
                } else if v != F::ZERO {
                    return false;
                }
            }
            if ones != 1 { return false; }
        }
        true
    }
}

/// TRUE Compressed Sparse Row (CSR) format - only stores non-zeros!
/// Specialized for neo_math::F for simplicity and performance
#[derive(Clone, Debug)]
pub struct CsrMatrix<F> {
    /// Number of rows in the matrix
    pub rows: usize,
    /// Number of columns in the matrix  
    pub cols: usize,
    /// row_ptrs[i] = start index in indices/values for row i
    pub row_ptrs: Vec<usize>,
    /// Column indices of non-zeros
    pub col_indices: Vec<usize>,
    /// Non-zero values (same length as col_indices) 
    pub values: Vec<F>,
}

impl<F: PrimeCharacteristicRing + Copy + Eq> CsrMatrix<F> {
    /// Convert dense matrix to CSR format - HUGE memory and performance win for sparse matrices
    pub fn from_dense(dense: &Mat<F>) -> Self {
        let zero = &F::ZERO;
        let mut row_ptrs = vec![0; dense.rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        
        for row in 0..dense.rows {
            row_ptrs[row] = col_indices.len();
            for col in 0..dense.cols {
                let val = &dense[(row, col)];
                if val != zero {
                    col_indices.push(col);
                    values.push(*val);
                }
            }
        }
        row_ptrs[dense.rows] = col_indices.len();
        
        #[cfg(feature = "neo-logs")]
        tracing::info!(
            "CSR conversion: {}×{} → {} non-zeros ({:.1}% density)", 
            dense.rows, dense.cols, values.len(), 
            100.0 * values.len() as f64 / (dense.rows * dense.cols) as f64
        );
        
        Self {
            rows: dense.rows,
            cols: dense.cols,
            row_ptrs,
            col_indices,
            values,
        }
    }
    
    /// TRUE O(nnz) sparse matrix-vector multiply: v = M^T * r
    /// Simple, working version - no features, no complexity
    #[inline]
    pub fn spmv_transpose(&self, r: &[F]) -> Vec<F> {
        debug_assert_eq!(r.len(), self.rows, "r length must equal matrix rows");
        let mut v = vec![F::ZERO; self.cols];
        
        for row in 0..self.rows {
            let r_val = r[row];
            if r_val == F::ZERO { continue; }
            
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            
            for idx in start..end {
                let col = self.col_indices[idx];
                let a = self.values[idx];
                v[col] += a * r_val;
            }
        }
        v
    }

    /// TRUE O(nnz) sparse matrix-vector multiply: v = M * r
    #[inline]
    pub fn spmv(&self, r: &[F]) -> Vec<F> {
        debug_assert_eq!(r.len(), self.cols, "r length must equal matrix cols");
        let mut v = vec![F::ZERO; self.rows];
        
        for row in 0..self.rows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            
            let mut acc = F::ZERO;
            for idx in start..end {
                let col = self.col_indices[idx];
                let a = self.values[idx];
                acc += a * r[col];
            }
            v[row] = acc;
        }
        v
    }
    
    /// Get non-zero elements in a row (TRUE sparse - no scanning!)
    #[inline]
    pub fn row_nz(&self, row: usize) -> (&[usize], &[F]) {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        (&self.col_indices[start..end], &self.values[start..end])
    }
    
    /// Number of non-zeros in matrix
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Number of non-zeros in specific row
    #[inline]
    pub fn row_nnz(&self, row: usize) -> usize {
        self.row_ptrs[row + 1] - self.row_ptrs[row]
    }
}

// Sparse matrix operations for performance optimization
impl<F: PrimeCharacteristicRing + Copy + Eq> Mat<F> {
    /// Convert to CSR format for REAL sparse operations
    pub fn to_csr(&self) -> CsrMatrix<F> {
        CsrMatrix::from_dense(self)
    }
    
    /// Iterator over non-zero elements in a specific row.
    /// Returns (column_index, value) pairs for elements that are not zero.
    /// 
    /// WARNING: This is O(m) per row! Use to_csr() for real performance.
    #[inline]
    pub fn row_nz<'a>(&'a self, row: usize) -> impl Iterator<Item=(usize, &'a F)> + 'a {
        self.row(row)
            .iter()
            .enumerate()
              .filter(move |(_, val)| **val != F::ZERO)
    }
    
    /// Count non-zeros in a specific row (useful for allocation sizing)
    #[inline] 
    pub fn row_nnz(&self, row: usize) -> usize {
        let zero = &F::ZERO;
        self.row(row).iter().filter(|val| *val != zero).count()
    }
    
    /// Total non-zeros in the matrix
    #[inline]
    pub fn nnz(&self) -> usize {
        let zero = &F::ZERO;
        self.data.iter().filter(|val| *val != zero).count()
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        let (r, c) = idx;
        &self.data[r * self.cols + c]
    }
}
impl<T> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        let (r, c) = idx;
        &mut self.data[r * self.cols + c]
    }
}

/// A borrowed view into a row-major matrix.
#[derive(Clone, Copy)]
pub struct MatRef<'a, T> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Row-major matrix data
    pub data: &'a [T],
}

impl<'a, T> MatRef<'a, T> {
    /// Make a `MatRef` from a full matrix.
    pub fn from_mat(m: &'a Mat<T>) -> Self {
        Self { rows: m.rows, cols: m.cols, data: &m.data }
    }

    /// Get a row slice.
    pub fn row(&self, i: usize) -> &'a [T] {
        let start = i * self.cols;
        &self.data[start .. start + self.cols]
    }
}

//
// P3 adapters (SHOULD)
//
use p3_matrix::dense::RowMajorMatrix as P3RowMajor;

impl<T: Clone + Send + Sync> From<&P3RowMajor<T>> for Mat<T> {
    fn from(m: &P3RowMajor<T>) -> Self {
        use p3_matrix::Matrix;
        let rows = m.height();
        let cols = m.width();
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row = m.row(r).expect("p3 row out-of-bounds");
            data.extend(row);
        }
        Self { rows, cols, data }
    }
}

impl<T: Clone + Send + Sync> From<&Mat<T>> for P3RowMajor<T> {
    fn from(m: &Mat<T>) -> Self {
        // p3_matrix wants a Vec<T> in row-major
        P3RowMajor::new(m.data.clone(), m.cols)
    }
}
