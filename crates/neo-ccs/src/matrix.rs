use core::ops::{Index, IndexMut};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PackedSignedUnitBits {
    RowMajor {
        positive: Vec<u64>,
        negative: Vec<u64>,
    },
    ColumnMasks {
        positive: Vec<u64>,
        negative: Vec<u64>,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PackedSignedUnit<T> {
    bits: PackedSignedUnitBits,
    values: [T; 3],
    cols: usize,
}

impl<T> PackedSignedUnit<T> {
    #[inline]
    fn value(&self, index: usize) -> &T {
        let (positive, negative, word, bit) = match &self.bits {
            PackedSignedUnitBits::RowMajor { positive, negative } => (
                positive,
                negative,
                index / u64::BITS as usize,
                1u64 << (index % u64::BITS as usize),
            ),
            PackedSignedUnitBits::ColumnMasks { positive, negative } => {
                let row = index / self.cols;
                let column = index % self.cols;
                (positive, negative, column, 1u64 << row)
            }
        };
        if positive[word] & bit != 0 {
            &self.values[1]
        } else if negative[word] & bit != 0 {
            &self.values[2]
        } else {
            &self.values[0]
        }
    }

    #[inline]
    fn nonzero_count(&self) -> usize {
        let (positive, negative) = match &self.bits {
            PackedSignedUnitBits::RowMajor { positive, negative }
            | PackedSignedUnitBits::ColumnMasks { positive, negative } => (positive, negative),
        };
        positive
            .iter()
            .zip(negative)
            .map(|(&positive, &negative)| (positive | negative).count_ones() as usize)
            .sum()
    }

    #[inline]
    fn column_masks(&self) -> Option<(&[u64], &[u64])> {
        match &self.bits {
            PackedSignedUnitBits::ColumnMasks { positive, negative } => Some((positive, negative)),
            PackedSignedUnitBits::RowMajor { .. } => None,
        }
    }
}

/// A dense row-major matrix over a field-like type `T`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mat<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
    /// Compact representation for a matrix whose entries are all equal.
    ///
    /// This is used for structurally zero prover witnesses. Read-by-index is
    /// supported without materialization; mutable or slice access explicitly
    /// materializes the dense backing vector first.
    constant_hint: Option<T>,
    /// Bit-packed storage for matrices over the exact alphabet `{0, 1, -1}`.
    packed_signed_unit: Option<PackedSignedUnit<T>>,
    /// Fast-path marker for identity matrices created via `Mat::identity`.
    ///
    /// This is intentionally skipped for serde and ignored for equality: it is an optimization only.
    /// Any mutable access clears the marker to preserve correctness.
    #[serde(skip)]
    identity_hint: bool,
}

impl<T: PartialEq> PartialEq for Mat<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        if self.constant_hint.is_none()
            && self.packed_signed_unit.is_none()
            && other.constant_hint.is_none()
            && other.packed_signed_unit.is_none()
        {
            return self.data == other.data;
        }
        for row in 0..self.rows {
            for column in 0..self.cols {
                if self[(row, column)] != other[(row, column)] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Eq> Eq for Mat<T> {}

impl<T: Clone> Mat<T> {
    /// Create a matrix from row-major data; panics if `data.len() != rows*cols`.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self {
            rows,
            cols,
            data,
            constant_hint: None,
            packed_signed_unit: None,
            identity_hint: false,
        }
    }

    /// Zero-initialized matrix (caller provides zero element).
    pub fn zero(rows: usize, cols: usize, zero: T) -> Self {
        Self {
            rows,
            cols,
            data: vec![zero; rows * cols],
            constant_hint: None,
            packed_signed_unit: None,
            identity_hint: false,
        }
    }

    /// Constant-valued matrix with no dense allocation.
    pub fn virtual_constant(rows: usize, cols: usize, value: T) -> Self {
        Self {
            rows,
            cols,
            data: Vec::new(),
            constant_hint: Some(value),
            packed_signed_unit: None,
            identity_hint: false,
        }
    }

    /// Whether this matrix currently uses the compact constant representation.
    pub fn is_virtual_constant(&self) -> bool {
        self.constant_hint.is_some()
    }

    /// Compact constant value, when present.
    pub fn virtual_constant_value(&self) -> Option<&T> {
        self.constant_hint.as_ref()
    }

    /// Whether this matrix uses exact bit-packed `{0, 1, -1}` storage.
    pub fn is_packed_signed_unit(&self) -> bool {
        self.packed_signed_unit.is_some()
    }

    /// Number of nonzero entries when the matrix is bit-packed.
    pub fn packed_signed_unit_nonzero_count(&self) -> Option<usize> {
        self.packed_signed_unit
            .as_ref()
            .map(PackedSignedUnit::nonzero_count)
    }

    /// Borrow the validated per-column positive and negative row masks when
    /// this matrix was constructed directly from that representation.
    pub fn packed_signed_unit_column_masks(&self) -> Option<(&[u64], &[u64])> {
        self.packed_signed_unit.as_ref()?.column_masks()
    }

    /// Return the exact row-major values without changing the matrix storage.
    pub fn to_dense_vec(&self) -> Vec<T> {
        if let Some(value) = self.constant_hint.as_ref() {
            return vec![value.clone(); self.rows * self.cols];
        }
        if let Some(packed) = self.packed_signed_unit.as_ref() {
            return (0..self.rows * self.cols)
                .map(|index| packed.value(index).clone())
                .collect();
        }
        self.data.clone()
    }

    /// Materialize any compact representation into ordinary row-major data.
    fn materialize_compact(&mut self) {
        if let Some(value) = self.constant_hint.take() {
            self.data = vec![value; self.rows * self.cols];
            return;
        }
        if let Some(packed) = self.packed_signed_unit.take() {
            self.data = (0..self.rows * self.cols)
                .map(|index| packed.value(index).clone())
                .collect();
        }
    }

    /// Rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Cols.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Underlying row-major slice.
    #[track_caller]
    pub fn as_slice(&self) -> &[T] {
        assert!(
            self.constant_hint.is_none() && self.packed_signed_unit.is_none(),
            "Mat::as_slice requires dense storage; handle compact matrices explicitly"
        );
        &self.data
    }

    /// Mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.materialize_compact();
        self.identity_hint = false;
        &mut self.data
    }

    /// Row i as a slice.
    pub fn row(&self, i: usize) -> &[T] {
        assert!(
            self.constant_hint.is_none() && self.packed_signed_unit.is_none(),
            "Mat::row requires dense storage; handle compact matrices explicitly"
        );
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Row i as a mutable slice.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        self.materialize_compact();
        self.identity_hint = false;
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Append `k` zero rows to the matrix in-place.
    /// The caller must provide the zero element for the field type.
    pub fn append_zero_rows(&mut self, k: usize, zero: T) {
        if k == 0 {
            return;
        }
        self.materialize_compact();
        self.identity_hint = false;
        let extra = k * self.cols;
        self.data.resize(self.data.len() + extra, zero);
        self.rows += k;
    }

    /// Set a single entry at (row, col) to the provided value.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        self.materialize_compact();
        self.identity_hint = false;
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
        for i in 0..n {
            m.set(i, i, F::ONE);
        }
        m.identity_hint = true;
        m
    }

    /// Check whether this matrix is exactly the identity matrix (I_n).
    pub fn is_identity(&self) -> bool {
        if self.identity_hint {
            return self.rows == self.cols;
        }
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = self[(r, c)];
                if r == c {
                    if v != F::ONE {
                        return false;
                    }
                } else if v != F::ZERO {
                    return false;
                }
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
            if ones != 1 {
                return false;
            }
        }
        true
    }

    /// Fast-path marker for identity matrices created via `Mat::identity`.
    ///
    /// This is an optimization hint only; it is not serialized and is cleared on any mutable access.
    pub fn is_identity_hint(&self) -> bool {
        self.identity_hint
    }

    /// Store an exact signed-unit matrix in two bits per entry. Values
    /// outside `{0, 1, -1}` keep the ordinary dense representation.
    pub fn compact_signed_unit(rows: usize, cols: usize, data: Vec<F>) -> Self {
        assert_eq!(rows * cols, data.len());
        let neg_one = F::ZERO - F::ONE;
        let words = data.len().div_ceil(u64::BITS as usize);
        let mut positive = vec![0u64; words];
        let mut negative = vec![0u64; words];
        for (index, &value) in data.iter().enumerate() {
            let word = index / u64::BITS as usize;
            let bit = 1u64 << (index % u64::BITS as usize);
            if value == F::ONE {
                positive[word] |= bit;
            } else if value == neg_one {
                negative[word] |= bit;
            } else if value != F::ZERO {
                return Self::from_row_major(rows, cols, data);
            }
        }
        Self {
            rows,
            cols,
            data: Vec::new(),
            constant_hint: None,
            packed_signed_unit: Some(PackedSignedUnit {
                bits: PackedSignedUnitBits::RowMajor { positive, negative },
                values: [F::ZERO, F::ONE, neg_one],
                cols,
            }),
            identity_hint: false,
        }
    }

    /// Construct exact `{0, 1, -1}` storage from one positive and negative
    /// row mask per column. Bit `r` in a column mask represents `(r, column)`.
    pub fn compact_signed_unit_from_column_masks(
        rows: usize,
        cols: usize,
        positive_columns: &[u64],
        negative_columns: &[u64],
    ) -> Result<Self, &'static str> {
        if rows > u64::BITS as usize {
            return Err("signed-unit column masks support at most 64 rows");
        }
        if positive_columns.len() != cols || negative_columns.len() != cols {
            return Err("signed-unit column mask count does not match the matrix");
        }
        rows.checked_mul(cols)
            .ok_or("signed-unit matrix dimensions overflow")?;
        let valid_rows = match rows {
            0 => 0,
            64 => u64::MAX,
            rows => (1u64 << rows) - 1,
        };
        for column in 0..cols {
            let positive_column = positive_columns[column];
            let negative_column = negative_columns[column];
            if (positive_column | negative_column) & !valid_rows != 0 {
                return Err("signed-unit column mask sets a row outside the matrix");
            }
            if positive_column & negative_column != 0 {
                return Err("signed-unit column masks overlap");
            }
        }
        let neg_one = F::ZERO - F::ONE;
        Ok(Self {
            rows,
            cols,
            data: Vec::new(),
            constant_hint: None,
            packed_signed_unit: Some(PackedSignedUnit {
                bits: PackedSignedUnitBits::ColumnMasks {
                    positive: positive_columns.to_vec(),
                    negative: negative_columns.to_vec(),
                },
                values: [F::ZERO, F::ONE, neg_one],
                cols,
            }),
            identity_hint: false,
        })
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
            dense.rows,
            dense.cols,
            values.len(),
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
        debug_assert_eq!(
            r_pairs.len(),
            self.rows,
            "r_pairs length ({}) must equal matrix rows ({})",
            r_pairs.len(),
            self.rows
        );

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
    pub fn row_nz<'a>(&'a self, row: usize) -> impl Iterator<Item = (usize, &'a neo_math::F)> + 'a {
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
        if let Some(value) = self.constant_hint.as_ref() {
            return if *value == neo_math::F::ZERO {
                0
            } else {
                self.rows * self.cols
            };
        }
        if let Some(packed) = self.packed_signed_unit.as_ref() {
            return packed.nonzero_count();
        }
        let zero = &neo_math::F::ZERO;
        self.data.iter().filter(|val| *val != zero).count()
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        let (r, c) = idx;
        debug_assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        if let Some(value) = self.constant_hint.as_ref() {
            return value;
        }
        if let Some(packed) = self.packed_signed_unit.as_ref() {
            return packed.value(r * self.cols + c);
        }
        &self.data[r * self.cols + c]
    }
}
impl<T: Clone> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        self.materialize_compact();
        self.identity_hint = false;
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
        assert!(
            m.constant_hint.is_none() && m.packed_signed_unit.is_none(),
            "MatRef::from_mat requires dense storage"
        );
        Self {
            rows: m.rows,
            cols: m.cols,
            data: &m.data,
        }
    }

    /// Get a row slice.
    pub fn row(&self, i: usize) -> &'a [T] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
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
        Self {
            rows,
            cols,
            data,
            constant_hint: None,
            packed_signed_unit: None,
            identity_hint: false,
        }
    }
}

impl<T: Clone + Send + Sync> From<&Mat<T>> for P3RowMajor<T> {
    fn from(m: &Mat<T>) -> Self {
        // p3_matrix wants a Vec<T> in row-major
        let data = if let Some(value) = m.constant_hint.as_ref() {
            vec![value.clone(); m.rows * m.cols]
        } else if let Some(packed) = m.packed_signed_unit.as_ref() {
            (0..m.rows * m.cols)
                .map(|index| packed.value(index).clone())
                .collect()
        } else {
            m.data.clone()
        };
        P3RowMajor::new(data, m.cols)
    }
}
