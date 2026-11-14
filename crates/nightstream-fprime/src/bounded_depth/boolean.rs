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

pub(super) fn inferred_column(row: &Row, known: &[bool], eligible: &[bool]) -> Option<usize> {
    let mut a = Form::canonical(&row.a);
    let mut b = Form::canonical(&row.b);
    let mut c = Form::canonical(&row.c);
    if a.terms.is_empty() || b.terms.is_empty() {
        let (factor, form) = if a.terms.is_empty() {
            (a.constant, &b)
        } else {
            (b.constant, &a)
        };
        c.constant -= factor * form.constant;
        for (&column, &coefficient) in &form.terms {
            *c.terms.entry(column).or_insert(F::ZERO) -= factor * coefficient;
        }
        c.terms.retain(|_, coefficient| *coefficient != F::ZERO);
        a.constant = F::ZERO;
        a.terms.clear();
        b.constant = F::ZERO;
        b.terms.clear();
    }
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

#[cfg(test)]
pub(super) fn inferred_columns(rows: &[Row], eligible: &[bool]) -> Vec<usize> {
    inferred_from_rows(rows.iter(), eligible, &[])
}

pub(super) fn inferred_from_rows<'a>(
    rows: impl IntoIterator<Item = &'a Row>,
    eligible: &[bool],
    seeds: &[usize],
) -> Vec<usize> {
    let rows = rows.into_iter().collect::<Vec<_>>();
    let mut known = vec![false; eligible.len()];
    for column in rows
        .iter()
        .filter_map(|row| forced_column(row))
        .chain(seeds.iter().copied())
    {
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
