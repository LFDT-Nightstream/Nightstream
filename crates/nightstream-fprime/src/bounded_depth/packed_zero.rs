use std::collections::BTreeSet;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{layout::forced_column, Linear, Row};

pub(super) const ZEROS_PER_ROW: usize = 31;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Input {
    pub(super) column: usize,
    pub(super) complement: bool,
}

impl Input {
    fn from_linear(linear: &Linear) -> Option<Self> {
        let [(column, coefficient)] = linear.terms.as_slice() else {
            return None;
        };
        let complement = match (linear.constant, *coefficient) {
            (constant, coefficient) if constant == F::ZERO && coefficient == F::ONE => false,
            (constant, coefficient) if constant == F::ONE && coefficient == F::NEG_ONE => true,
            _ => return None,
        };
        (*column != 0).then_some(Self {
            column: *column,
            complement,
        })
    }

    pub(super) fn value(&self, columns: &[F]) -> F {
        if self.complement {
            F::ONE - columns[self.column]
        } else {
            columns[self.column]
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ZeroProduct {
    pub(super) row: usize,
    pub(super) left: Input,
    pub(super) right: Input,
}

impl ZeroProduct {
    pub(super) fn checked<'a>(rows: impl IntoIterator<Item = &'a Row>, eligible: impl Fn(usize) -> bool) -> Vec<Self> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let forced = rows
            .iter()
            .filter_map(|row| forced_column(row))
            .collect::<BTreeSet<_>>();
        rows.into_iter()
            .enumerate()
            .filter_map(|(index, row)| {
                if forced_column(row).is_some() || row.c.constant != F::ZERO || !row.c.terms.is_empty() {
                    return None;
                }
                let left = Input::from_linear(&row.a)?;
                let right = Input::from_linear(&row.b)?;
                [left.column, right.column]
                    .into_iter()
                    .all(|column| eligible(column) && forced.contains(&column))
                    .then_some(Self {
                        row: index,
                        left,
                        right,
                    })
            })
            .collect()
    }
}

pub(super) fn rows(zeros: &[ZeroProduct], start: usize) -> impl Iterator<Item = Row> + '_ {
    zeros
        .chunks(ZEROS_PER_ROW)
        .enumerate()
        .map(move |(group, zeros)| {
            let mut a = Linear::default();
            let mut coefficient = F::ONE;
            for (index, zero) in zeros.iter().enumerate() {
                a.terms
                    .push((start + group * ZEROS_PER_ROW + index, coefficient));
                for input in [&zero.left, &zero.right] {
                    if input.complement {
                        a.constant -= coefficient;
                        a.terms.push((input.column, coefficient));
                    } else {
                        a.terms.push((input.column, -coefficient));
                    }
                }
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

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_packed_zero.rs"]
mod tests;
