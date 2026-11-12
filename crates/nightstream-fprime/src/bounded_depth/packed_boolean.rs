use std::collections::BTreeSet;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::packed_product::{self, Product, PRODUCTS_PER_ROW};
use super::packed_zero::{self, ZeroProduct, ZEROS_PER_ROW};
use super::{layout::forced_column, Linear, Row};
use crate::package::PackageError;

const PAIRS_PER_ROW: usize = 31;

#[derive(Clone, Debug, Default)]
pub(super) struct PackedBooleans {
    pub(super) start: usize,
    pub(super) sources: Vec<usize>,
    pub(super) removed: Vec<(usize, Row)>,
    pub(super) products: Vec<Product>,
    pub(super) zeros: Vec<ZeroProduct>,
}

impl PackedBooleans {
    pub(super) fn lower(rows: &mut Vec<Row>, start: usize, eligible: impl Fn(usize) -> bool) -> Self {
        let sources = rows
            .iter()
            .filter_map(forced_column)
            .filter(|&column| eligible(column))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut packed = Self {
            start,
            sources,
            removed: Vec::new(),
            products: Product::checked(rows.iter(), &(0..start).map(&eligible).collect::<Vec<_>>()),
            zeros: ZeroProduct::checked(rows.iter(), &eligible),
        };
        *rows = std::mem::take(rows)
            .into_iter()
            .enumerate()
            .filter_map(|(index, row)| {
                if forced_column(&row).is_some_and(|column| packed.sources.binary_search(&column).is_ok())
                    || packed
                        .products
                        .binary_search_by_key(&index, |product| product.row)
                        .is_ok()
                    || packed
                        .zeros
                        .binary_search_by_key(&index, |zero| zero.row)
                        .is_ok()
                {
                    packed.removed.push((index, row));
                    None
                } else {
                    Some(row)
                }
            })
            .collect();
        rows.extend(packed.rows());
        packed
    }

    pub(super) fn end(&self) -> usize {
        self.zero_start() + self.zeros.len()
    }

    pub(super) fn zero_start(&self) -> usize {
        self.product_start() + 2 * self.products.len()
    }

    pub(super) fn product_start(&self) -> usize {
        self.start + self.sources.len()
    }

    pub(super) fn row_count(&self) -> usize {
        (self.sources.len() + self.products.len()).div_ceil(PAIRS_PER_ROW)
            + self.products.len().div_ceil(PRODUCTS_PER_ROW)
            + self.zeros.len().div_ceil(ZEROS_PER_ROW)
    }

    pub(super) fn rows(&self) -> impl Iterator<Item = Row> + '_ {
        let pairs = self
            .sources
            .iter()
            .enumerate()
            .map(|(index, &source)| (source, self.start + index))
            .chain((0..self.products.len()).map(|index| {
                let column = self.product_start() + 2 * index;
                (column, column + 1)
            }))
            .collect::<Vec<_>>();
        let mut rows = pairs
            .chunks(PAIRS_PER_ROW)
            .map(|sources| {
                let mut a = Linear::default();
                let mut coefficient = F::ONE;
                for &(source, companion) in sources {
                    a.constant -= coefficient;
                    a.terms.push((source, coefficient));
                    a.terms.push((companion, coefficient));
                    coefficient *= F::from_u64(4);
                }
                Row {
                    a,
                    b: Linear {
                        constant: F::ONE,
                        terms: Vec::new(),
                    },
                    c: Linear::default(),
                }
            })
            .collect::<Vec<_>>();
        rows.extend(packed_product::rows(&self.products, self.product_start()));
        rows.extend(packed_zero::rows(&self.zeros, self.zero_start()));
        rows.into_iter()
    }

    pub(super) fn original_rows<'a>(&'a self, rows: &'a [Row]) -> Result<Vec<&'a Row>, PackageError> {
        let kept = rows
            .len()
            .checked_sub(self.row_count())
            .ok_or(PackageError::Malformed("bounded-depth packed Boolean row count"))?;
        let count = kept + self.removed.len();
        let mut original = Vec::with_capacity(count);
        let mut kept = rows[..kept].iter();
        let mut removed = self.removed.iter().peekable();
        for index in 0..count {
            let row = match removed.peek() {
                Some((position, _)) if *position == index => &removed.next().expect("present row").1,
                _ => kept
                    .next()
                    .ok_or(PackageError::Malformed("bounded-depth removed Boolean row order"))?,
            };
            original.push(row);
        }
        if kept.next().is_some() || removed.next().is_some() {
            return Err(PackageError::Malformed("bounded-depth removed Boolean row order"));
        }
        Ok(original)
    }

    pub(super) fn validate(&self, rows: &[Row], start: usize, width: usize) -> Result<(), PackageError> {
        let invalid = || PackageError::Malformed("bounded-depth packed Boolean map");
        if self.start != start
            || self.sources.iter().any(|&column| column >= start)
            || !self.sources.windows(2).all(|pair| pair[0] < pair[1])
            || self
                .start
                .checked_add(self.sources.len() + 2 * self.products.len() + self.zeros.len())
                .is_none_or(|end| end > width)
            || rows.len() < self.row_count()
            || !rows[rows.len() - self.row_count()..]
                .iter()
                .eq(self.rows().collect::<Vec<_>>().iter())
            || !self
                .products
                .windows(2)
                .all(|pair| pair[0].row < pair[1].row)
            || !self.zeros.windows(2).all(|pair| pair[0].row < pair[1].row)
            || self.removed.iter().any(|(index, row)| {
                forced_column(row).is_none_or(|column| self.sources.binary_search(&column).is_err())
                    && self
                        .products
                        .binary_search_by_key(index, |product| product.row)
                        .is_err()
                    && self
                        .zeros
                        .binary_search_by_key(index, |zero| zero.row)
                        .is_err()
            })
        {
            return Err(invalid());
        }
        let original = self.original_rows(rows)?;
        let checked = Product::checked(original.iter().copied(), &vec![true; start]);
        for product in &self.products {
            if checked
                .binary_search_by_key(&product.row, |gate| gate.row)
                .ok()
                .is_none_or(|index| checked[index] != *product)
                || self
                    .removed
                    .binary_search_by_key(&product.row, |(index, _)| *index)
                    .is_err()
            {
                return Err(invalid());
            }
        }
        let checked = ZeroProduct::checked(original.iter().copied(), |column| column < start);
        for zero in &self.zeros {
            if checked
                .binary_search_by_key(&zero.row, |gate| gate.row)
                .ok()
                .is_none_or(|index| checked[index] != *zero)
                || self
                    .removed
                    .binary_search_by_key(&zero.row, |(index, _)| *index)
                    .is_err()
            {
                return Err(invalid());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_packed_boolean.rs"]
mod tests;
