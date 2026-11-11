use std::collections::{BTreeMap, BTreeSet};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{layout::forced_column, Linear, Row};

struct Form {
    constant: F,
    terms: BTreeMap<usize, F>,
}

impl Form {
    fn canonical(source: &Linear) -> Self {
        let mut constant = source.constant;
        let mut terms = BTreeMap::new();
        for &(column, coefficient) in &source.terms {
            if column == 0 {
                constant += coefficient;
            } else {
                *terms.entry(column).or_insert(F::ZERO) += coefficient;
            }
        }
        terms.retain(|_, coefficient| *coefficient != F::ZERO);
        Self { constant, terms }
    }

    fn at(&self, inputs: &[usize], assignment: usize) -> F {
        inputs
            .iter()
            .enumerate()
            .fold(self.constant, |value, (bit, column)| {
                value + self.terms.get(column).copied().unwrap_or(F::ZERO) * F::from_usize((assignment >> bit) & 1)
            })
    }
}

fn inferred_column(row: &Row, known: &[bool], eligible: &[bool]) -> Option<usize> {
    let mut c = Form::canonical(&row.c);
    let mut unknown = c
        .terms
        .keys()
        .copied()
        .filter(|&column| known.get(column) != Some(&true));
    let output = unknown.next()?;
    if unknown.next().is_some() || eligible.get(output) != Some(&true) {
        return None;
    }
    let coefficient = c.terms.remove(&output)?;
    let a = Form::canonical(&row.a);
    let b = Form::canonical(&row.b);
    let inputs = a
        .terms
        .keys()
        .chain(b.terms.keys())
        .chain(c.terms.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    if inputs.len() > 2
        || inputs
            .iter()
            .any(|&column| known.get(column) != Some(&true))
    {
        return None;
    }
    let inputs = inputs.into_iter().collect::<Vec<_>>();
    for assignment in 0..1 << inputs.len() {
        let product = a.at(&inputs, assignment) * b.at(&inputs, assignment);
        let rest = c.at(&inputs, assignment);
        if product != rest && product != rest + coefficient {
            return None;
        }
    }
    Some(output)
}

pub(super) fn inferred_columns(rows: &[Row], eligible: &[bool]) -> Vec<usize> {
    let mut known = vec![false; eligible.len()];
    for column in rows.iter().filter_map(forced_column) {
        if let Some(value) = known.get_mut(column) {
            *value = true;
        }
    }
    let mut inferred = Vec::new();
    for row in rows {
        if let Some(column) = inferred_column(row, &known, eligible) {
            known[column] = true;
            inferred.push(column);
        }
    }
    inferred
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_boolean.rs"]
mod tests;
