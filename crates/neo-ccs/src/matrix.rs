use core::ops::{Index, IndexMut};

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
