use neo_math::Fq as F;
use p3_field::PrimeCharacteristicRing;

use super::{boolean::inferred_column, layout::forced_column, Linear, Row};

pub(super) const PRODUCTS_PER_ROW: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Product {
    pub(super) row: usize,
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) output: usize,
}

fn variable(linear: &Linear) -> Option<usize> {
    let [(column, coefficient)] = linear.terms.as_slice() else {
        return None;
    };
    (linear.constant == F::ZERO && *coefficient == F::ONE && *column != 0).then_some(*column)
}

impl Product {
    fn from_row(row: &Row, index: usize) -> Option<Self> {
        Some(Self {
            row: index,
            left: variable(&row.a)?,
            right: variable(&row.b)?,
            output: variable(&row.c)?,
        })
    }

    pub(super) fn checked<'a>(rows: impl IntoIterator<Item = &'a Row>, eligible: &[bool]) -> Vec<Self> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let mut known = vec![false; eligible.len()];
        for column in rows.iter().filter_map(|row| forced_column(row)) {
            if let Some(bit) = known.get_mut(column) {
                *bit = true;
            }
        }
        let mut products = Vec::new();
        for (index, row) in rows.into_iter().enumerate() {
            if forced_column(row).is_none() {
                if let Some(product) = Self::from_row(row, index).filter(|gate| {
                    [gate.left, gate.right, gate.output]
                        .into_iter()
                        .all(|column| eligible.get(column) == Some(&true))
                        && known[gate.left]
                        && known[gate.right]
                }) {
                    known[product.output] = true;
                    products.push(product);
                    continue;
                }
            }
            if let Some(column) = inferred_column(row, &known, eligible) {
                known[column] = true;
            }
        }
        products
    }
}

pub(super) fn rows(products: &[Product], start: usize) -> impl Iterator<Item = Row> + '_ {
    products
        .chunks(PRODUCTS_PER_ROW)
        .enumerate()
        .map(move |(group, products)| {
            let mut a = Linear::default();
            let mut coefficient = F::ONE;
            for (index, product) in products.iter().enumerate() {
                a.terms.extend([
                    (product.left, coefficient),
                    (product.right, coefficient),
                    (product.output, -F::from_u64(2) * coefficient),
                    (start + 2 * (group * PRODUCTS_PER_ROW + index), -coefficient),
                ]);
                coefficient *= F::from_u64(6);
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
#[path = "../../tests/unit/bounded_depth_packed_product.rs"]
mod tests;
