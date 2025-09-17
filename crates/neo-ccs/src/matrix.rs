use core::ops::{Index, IndexMut};
use p3_field::PrimeCharacteristicRing;

/// A dense row-major matrix over a field-like type `T`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

/// TRUE Compressed Sparse Row (CSR) format - only stores non-zeros!
/// Specialized for neo_math::F for simplicity and performance
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    /// Number of rows in the matrix
    pub rows: usize,
    /// Number of columns in the matrix  
    pub cols: usize,
    /// row_ptrs[i] = start index in indices/values for row i
    pub row_ptrs: Vec<usize>,
    /// Column indices of non-zeros
    pub col_indices: Vec<usize>,
    /// Non-zero values (same length as col_indices) 
    pub values: Vec<neo_math::F>,
}

impl CsrMatrix {
    /// Convert dense matrix to CSR format - HUGE memory and performance win for sparse matrices
    pub fn from_dense(dense: &Mat<neo_math::F>) -> Self {
        let zero = &neo_math::F::ZERO;
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
    pub fn spmv_transpose(&self, r_pairs: &[(neo_math::F, neo_math::F)]) -> (Vec<neo_math::F>, Vec<neo_math::F>) {
        // SECURITY: Ensure r_pairs length matches matrix rows to prevent panics
        debug_assert_eq!(r_pairs.len(), self.rows, 
            "r_pairs length ({}) must equal matrix rows ({})", 
            r_pairs.len(), self.rows);
        
        let mut v_re = vec![neo_math::F::ZERO; self.cols];
        let mut v_im = vec![neo_math::F::ZERO; self.cols];
        
        // CRITICAL: Only iterate actual non-zeros - THIS IS THE HUGE WIN!
        for row in 0..self.rows {
            let (r_re, r_im) = r_pairs[row];
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            
            // Process only non-zero elements in this row - skips all zeros!
            for idx in start..end {
                let col = self.col_indices[idx];
                let a = self.values[idx];
                
                // Simple accumulation - no features, just working code
                v_re[col] += a * r_re;
                v_im[col] += a * r_im;
            }
        }
        
        (v_re, v_im)
    }
    
    /// Get non-zero elements in a row (TRUE sparse - no scanning!)
    #[inline]
    pub fn row_nz(&self, row: usize) -> (&[usize], &[neo_math::F]) {
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
impl Mat<neo_math::F> {
    /// Convert to CSR format for REAL sparse operations
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_dense(self)
    }
    
    /// Iterator over non-zero elements in a specific row.
    /// Returns (column_index, value) pairs for elements that are not zero.
    /// 
    /// WARNING: This is O(m) per row! Use to_csr() for real performance.
    #[inline]
    pub fn row_nz<'a>(&'a self, row: usize) -> impl Iterator<Item=(usize, &'a neo_math::F)> + 'a {
        let zero = &neo_math::F::ZERO;
        self.row(row)
            .iter()
            .enumerate()
              .filter(move |(_, val)| *val != zero)
    }
    
    /// Count non-zeros in a specific row (useful for allocation sizing)
    #[inline] 
    pub fn row_nnz(&self, row: usize) -> usize {
        let zero = &neo_math::F::ZERO;
        self.row(row).iter().filter(|val| *val != zero).count()
    }
    
    /// Total non-zeros in the matrix
    #[inline]
    pub fn nnz(&self) -> usize {
        let zero = &neo_math::F::ZERO;
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
