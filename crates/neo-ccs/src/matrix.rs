use core::ops::{Index, IndexMut};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PackedSignedUnitBits {
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
        let PackedSignedUnitBits::ColumnMasks { positive, negative } = &self.bits;
        let column = index % self.cols;
        let bit = 1u64 << (index / self.cols);
        if positive[column] & bit != 0 {
            &self.values[1]
        } else if negative[column] & bit != 0 {
            &self.values[2]
        } else {
            &self.values[0]
        }
    }

    #[inline]
    fn nonzero_count(&self) -> usize {
        let PackedSignedUnitBits::ColumnMasks { positive, negative } = &self.bits;
        positive
            .iter()
            .zip(negative)
            .map(|(&positive, &negative)| (positive | negative).count_ones() as usize)
            .sum()
    }

    #[inline]
    fn column_masks(&self) -> (&[u64], &[u64]) {
        let PackedSignedUnitBits::ColumnMasks { positive, negative } = &self.bits;
        (positive, negative)
    }
}

#[derive(Clone, Debug)]
enum Storage<T> {
    Dense { data: Vec<T>, identity: bool },
    Constant(T),
    Packed(Arc<PackedSignedUnit<T>>),
}

#[derive(Clone, Debug)]
pub struct Mat<T> {
    rows: usize,
    cols: usize,
    storage: Storage<T>,
}

#[derive(serde::Deserialize)]
struct MatWire<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
    constant_hint: Option<T>,
    packed_signed_unit: Option<PackedSignedUnit<T>>,
}

impl<T: serde::Serialize> serde::Serialize for Mat<T> {
    fn serialize<SerializerT>(&self, serializer: SerializerT) -> Result<SerializerT::Ok, SerializerT::Error>
    where
        SerializerT: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let (data, constant_hint, packed_signed_unit): (&[T], Option<&T>, Option<&PackedSignedUnit<T>>) =
            match &self.storage {
                Storage::Dense { data, .. } => (data, None, None),
                Storage::Constant(value) => (&[], Some(value), None),
                Storage::Packed(packed) => (&[], None, Some(packed)),
            };
        let mut wire = serializer.serialize_struct("Mat", 5)?;
        wire.serialize_field("rows", &self.rows)?;
        wire.serialize_field("cols", &self.cols)?;
        wire.serialize_field("data", data)?;
        wire.serialize_field("constant_hint", &constant_hint)?;
        wire.serialize_field("packed_signed_unit", &packed_signed_unit)?;
        wire.end()
    }
}

impl<'de, T> serde::Deserialize<'de> for Mat<T>
where
    T: serde::Deserialize<'de> + PrimeCharacteristicRing + Clone + Eq,
{
    fn deserialize<DeserializerT>(deserializer: DeserializerT) -> Result<Self, DeserializerT::Error>
    where
        DeserializerT: serde::Deserializer<'de>,
    {
        let wire = <MatWire<T> as serde::Deserialize>::deserialize(deserializer)?;
        let element_count = wire
            .rows
            .checked_mul(wire.cols)
            .ok_or_else(|| serde::de::Error::custom("matrix dimensions overflow usize"))?;

        let storage = match (wire.constant_hint, wire.packed_signed_unit) {
            (None, None) => {
                if wire.data.len() != element_count {
                    return Err(serde::de::Error::custom(format!(
                        "dense matrix has {} elements, expected {element_count}",
                        wire.data.len()
                    )));
                }
                Storage::Dense {
                    data: wire.data,
                    identity: false,
                }
            }
            (Some(value), None) => {
                if !wire.data.is_empty() {
                    return Err(serde::de::Error::custom(
                        "constant matrix must not contain dense backing data",
                    ));
                }
                Storage::Constant(value)
            }
            (None, Some(packed)) => {
                if !wire.data.is_empty() {
                    return Err(serde::de::Error::custom(
                        "packed matrix must not contain dense backing data",
                    ));
                }
                if packed.cols != wire.cols || packed.values != [T::ZERO, T::ONE, T::ZERO - T::ONE] {
                    return Err(serde::de::Error::custom(
                        "packed matrix metadata does not describe signed-unit values",
                    ));
                }
                let (positive, negative) = packed.column_masks();
                if wire.rows > u64::BITS as usize || positive.len() != wire.cols || negative.len() != wire.cols {
                    return Err(serde::de::Error::custom(
                        "packed column-mask shape does not match the matrix",
                    ));
                }
                let valid_rows = match wire.rows {
                    0 => 0,
                    64 => u64::MAX,
                    rows => (1u64 << rows) - 1,
                };
                if positive
                    .iter()
                    .zip(negative)
                    .any(|(&pos, &neg)| pos & neg != 0 || (pos | neg) & !valid_rows != 0)
                {
                    return Err(serde::de::Error::custom(
                        "packed column masks overlap or address a row outside the matrix",
                    ));
                }
                Storage::Packed(Arc::new(packed))
            }
            (Some(_), Some(_)) => {
                return Err(serde::de::Error::custom(
                    "matrix cannot use constant and packed storage together",
                ));
            }
        };

        Ok(Self {
            rows: wire.rows,
            cols: wire.cols,
            storage,
        })
    }
}

impl<T: PartialEq> PartialEq for Mat<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        if let (Storage::Dense { data, .. }, Storage::Dense { data: other_data, .. }) = (&self.storage, &other.storage)
        {
            return data == other_data;
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
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        let element_count = rows
            .checked_mul(cols)
            .expect("matrix dimensions overflow usize");
        assert_eq!(element_count, data.len());
        Self {
            rows,
            cols,
            storage: Storage::Dense { data, identity: false },
        }
    }

    pub fn zero(rows: usize, cols: usize, zero: T) -> Self {
        let element_count = rows
            .checked_mul(cols)
            .expect("matrix dimensions overflow usize");
        Self {
            rows,
            cols,
            storage: Storage::Dense {
                data: vec![zero; element_count],
                identity: false,
            },
        }
    }

    pub fn virtual_constant(rows: usize, cols: usize, value: T) -> Self {
        rows.checked_mul(cols)
            .expect("matrix dimensions overflow usize");
        Self {
            rows,
            cols,
            storage: Storage::Constant(value),
        }
    }

    pub fn is_virtual_constant(&self) -> bool {
        matches!(self.storage, Storage::Constant(_))
    }

    pub fn virtual_constant_value(&self) -> Option<&T> {
        match &self.storage {
            Storage::Constant(value) => Some(value),
            Storage::Dense { .. } | Storage::Packed(_) => None,
        }
    }

    pub fn is_packed_signed_unit(&self) -> bool {
        matches!(self.storage, Storage::Packed(_))
    }

    pub fn packed_signed_unit_nonzero_count(&self) -> Option<usize> {
        match &self.storage {
            Storage::Packed(packed) => Some(packed.nonzero_count()),
            Storage::Dense { .. } | Storage::Constant(_) => None,
        }
    }

    pub fn packed_signed_unit_column_masks(&self) -> Option<(&[u64], &[u64])> {
        match &self.storage {
            Storage::Packed(packed) => Some(packed.column_masks()),
            Storage::Dense { .. } | Storage::Constant(_) => None,
        }
    }

    pub fn to_dense_vec(&self) -> Vec<T> {
        match &self.storage {
            Storage::Dense { data, .. } => data.clone(),
            Storage::Constant(value) => vec![value.clone(); self.rows * self.cols],
            Storage::Packed(packed) => (0..self.rows * self.cols)
                .map(|index| packed.value(index).clone())
                .collect(),
        }
    }

    fn dense_mut(&mut self) -> &mut Vec<T> {
        if !matches!(self.storage, Storage::Dense { .. }) {
            self.storage = Storage::Dense {
                data: self.to_dense_vec(),
                identity: false,
            };
        }
        match &mut self.storage {
            Storage::Dense { data, identity } => {
                *identity = false;
                data
            }
            Storage::Constant(_) | Storage::Packed(_) => unreachable!("compact storage was materialized"),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    #[track_caller]
    pub fn as_slice(&self) -> &[T] {
        let Storage::Dense { data, .. } = &self.storage else {
            panic!("Mat::as_slice requires dense storage; handle compact matrices explicitly");
        };
        data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.dense_mut()
    }

    pub fn row(&self, i: usize) -> &[T] {
        let Storage::Dense { data, .. } = &self.storage else {
            panic!("Mat::row requires dense storage; handle compact matrices explicitly");
        };
        let start = i * self.cols;
        &data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.cols;
        let end = start + self.cols;
        &mut self.dense_mut()[start..end]
    }

    pub fn append_zero_rows(&mut self, k: usize, zero: T) {
        if k == 0 {
            return;
        }
        let new_rows = self
            .rows
            .checked_add(k)
            .expect("matrix row count overflow usize");
        let new_len = new_rows
            .checked_mul(self.cols)
            .expect("matrix dimensions overflow usize");
        self.dense_mut().resize(new_len, zero);
        self.rows = new_rows;
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        debug_assert!(row < self.rows, "row out of bounds");
        debug_assert!(col < self.cols, "col out of bounds");
        let index = row * self.cols + col;
        self.dense_mut()[index] = val;
    }
}

impl<F> Mat<F>
where
    F: p3_field::PrimeCharacteristicRing + Copy + Eq,
{
    pub fn identity(n: usize) -> Self {
        let mut m = Mat::zero(n, n, F::ZERO);
        for i in 0..n {
            m.set(i, i, F::ONE);
        }
        if let Storage::Dense { identity, .. } = &mut m.storage {
            *identity = true;
        }
        m
    }

    pub fn is_identity(&self) -> bool {
        if self.is_identity_hint() {
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

    pub fn is_identity_hint(&self) -> bool {
        matches!(self.storage, Storage::Dense { identity: true, .. })
    }

    pub fn compact_signed_unit_from_column_major_prefix(
        rows: usize,
        cols: usize,
        values: &[i8],
    ) -> Result<Self, &'static str> {
        if rows == 0 || rows > u64::BITS as usize {
            return Err("signed-unit column masks support 1 to 64 rows");
        }
        if rows.checked_mul(cols).is_none_or(|len| values.len() > len) {
            return Err("signed-unit values exceed the matrix");
        }
        let (positive, negative): (Vec<u64>, Vec<u64>) = (0..cols)
            .into_par_iter()
            .map(|column| {
                let start = (column * rows).min(values.len());
                let end = (start + rows).min(values.len());
                values[start..end]
                    .iter()
                    .enumerate()
                    .try_fold((0u64, 0u64), |(positive, negative), (row, &value)| match value {
                        0 => Ok((positive, negative)),
                        1 => Ok((positive | 1 << row, negative)),
                        -1 => Ok((positive, negative | 1 << row)),
                        _ => Err("signed-unit value is outside -1, 0, 1"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        Self::compact_signed_unit_from_column_masks(rows, cols, &positive, &negative)
    }

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
            storage: Storage::Packed(Arc::new(PackedSignedUnit {
                bits: PackedSignedUnitBits::ColumnMasks {
                    positive: positive_columns.to_vec(),
                    negative: negative_columns.to_vec(),
                },
                values: [F::ZERO, F::ONE, neg_one],
                cols,
            })),
        })
    }
}

#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<neo_math::F>,
}

impl CsrMatrix {
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

    #[inline]
    pub fn spmv_transpose(&self, r_pairs: &[(neo_math::F, neo_math::F)]) -> (Vec<neo_math::F>, Vec<neo_math::F>) {
        debug_assert_eq!(
            r_pairs.len(),
            self.rows,
            "r_pairs length ({}) must equal matrix rows ({})",
            r_pairs.len(),
            self.rows
        );

        let mut v_re = vec![neo_math::F::ZERO; self.cols];
        let mut v_im = vec![neo_math::F::ZERO; self.cols];

        for row in 0..self.rows {
            let (r_re, r_im) = r_pairs[row];
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let a = self.values[idx];

                v_re[col] += a * r_re;
                v_im[col] += a * r_im;
            }
        }

        (v_re, v_im)
    }

    #[inline]
    pub fn row_nz(&self, row: usize) -> (&[usize], &[neo_math::F]) {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        (&self.col_indices[start..end], &self.values[start..end])
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn row_nnz(&self, row: usize) -> usize {
        self.row_ptrs[row + 1] - self.row_ptrs[row]
    }
}

impl Mat<neo_math::F> {
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_dense(self)
    }

    #[inline]
    pub fn row_nz<'a>(&'a self, row: usize) -> impl Iterator<Item = (usize, &'a neo_math::F)> + 'a {
        let zero = &neo_math::F::ZERO;
        self.row(row)
            .iter()
            .enumerate()
            .filter(move |(_, val)| *val != zero)
    }

    #[inline]
    pub fn row_nnz(&self, row: usize) -> usize {
        let zero = &neo_math::F::ZERO;
        self.row(row).iter().filter(|val| *val != zero).count()
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        match &self.storage {
            Storage::Constant(value) if *value == neo_math::F::ZERO => 0,
            Storage::Constant(_) => self.rows * self.cols,
            Storage::Packed(packed) => packed.nonzero_count(),
            Storage::Dense { data, .. } => data.iter().filter(|val| **val != neo_math::F::ZERO).count(),
        }
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        let (r, c) = idx;
        debug_assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        match &self.storage {
            Storage::Dense { data, .. } => &data[r * self.cols + c],
            Storage::Constant(value) => value,
            Storage::Packed(packed) => packed.value(r * self.cols + c),
        }
    }
}
impl<T: Clone> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        let (r, c) = idx;
        let index = r * self.cols + c;
        &mut self.dense_mut()[index]
    }
}

#[derive(Clone, Copy)]
pub struct MatRef<'a, T> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [T],
}

impl<'a, T> MatRef<'a, T> {
    pub fn from_mat(m: &'a Mat<T>) -> Self {
        let Storage::Dense { data, .. } = &m.storage else {
            panic!("MatRef::from_mat requires dense storage");
        };
        Self {
            rows: m.rows,
            cols: m.cols,
            data,
        }
    }

    pub fn row(&self, i: usize) -> &'a [T] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }
}

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
            storage: Storage::Dense { data, identity: false },
        }
    }
}

impl<T: Clone + Send + Sync> From<&Mat<T>> for P3RowMajor<T> {
    fn from(m: &Mat<T>) -> Self {
        P3RowMajor::new(m.to_dense_vec(), m.cols)
    }
}
