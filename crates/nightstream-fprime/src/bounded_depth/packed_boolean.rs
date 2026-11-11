use std::collections::BTreeSet;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{layout::forced_column, Linear, Row};
use crate::package::PackageError;

const PAIRS_PER_ROW: usize = 31;

#[derive(Clone, Debug, Default)]
pub(super) struct PackedBooleans {
    pub(super) start: usize,
    pub(super) sources: Vec<usize>,
    pub(super) removed: Vec<(usize, Row)>,
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
        };
        *rows = std::mem::take(rows)
            .into_iter()
            .enumerate()
            .filter_map(|(index, row)| {
                if forced_column(&row).is_some_and(|column| packed.sources.binary_search(&column).is_ok()) {
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
        self.start + self.sources.len()
    }

    pub(super) fn row_count(&self) -> usize {
        self.sources.len().div_ceil(PAIRS_PER_ROW)
    }

    pub(super) fn rows(&self) -> impl Iterator<Item = Row> + '_ {
        self.sources
            .chunks(PAIRS_PER_ROW)
            .enumerate()
            .map(|(group, sources)| {
                let mut a = Linear::default();
                let mut coefficient = F::ONE;
                for (index, &source) in sources.iter().enumerate() {
                    a.constant -= coefficient;
                    a.terms.push((source, coefficient));
                    a.terms
                        .push((self.start + group * PAIRS_PER_ROW + index, coefficient));
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
                .checked_add(self.sources.len())
                .is_none_or(|end| end > width)
            || rows.len() < self.row_count()
            || !rows[rows.len() - self.row_count()..]
                .iter()
                .eq(self.rows().collect::<Vec<_>>().iter())
            || self
                .removed
                .iter()
                .any(|(_, row)| forced_column(row).is_none_or(|column| self.sources.binary_search(&column).is_err()))
        {
            return Err(invalid());
        }
        self.original_rows(rows)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_packed_boolean.rs"]
mod tests;
