use std::collections::{BTreeMap, BTreeSet};

use neo_math::Fq as F;
use p3_field::{Field, PrimeCharacteristicRing};

use super::{Linear, Piece, Row, StepPackage};
use crate::PackageError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Power7 {
    pub input: Linear,
    pub powers: [usize; 4],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct FieldLowering {
    pub powers: BTreeMap<usize, Power7>,
    pub affine: BTreeMap<usize, (usize, Linear)>,
    pub removed: BTreeSet<usize>,
    pub dead: BTreeSet<usize>,
}

pub(super) fn normalized(mut form: Linear) -> Linear {
    form.terms.sort_unstable_by_key(|&(column, _)| column);
    let mut terms: Vec<(usize, F)> = Vec::with_capacity(form.terms.len());
    for (column, coefficient) in form.terms {
        if column == 0 {
            form.constant += coefficient;
        } else if let Some(last) = terms.last_mut().filter(|last| last.0 == column) {
            last.1 += coefficient;
        } else {
            terms.push((column, coefficient));
        }
    }
    terms.retain(|&(_, coefficient)| coefficient != F::ZERO);
    form.terms = terms;
    form
}

fn variable(form: &Linear) -> Option<usize> {
    match form.terms.as_slice() {
        &[(column, coefficient)] if form.constant == F::ZERO && coefficient == F::ONE => Some(column),
        _ => None,
    }
}

fn unit(column: usize) -> Linear {
    Linear {
        constant: F::ZERO,
        terms: vec![(column, F::ONE)],
    }
}

fn product(row: &Row, left: &Linear, right: &Linear) -> bool {
    (&row.a == left && &row.b == right) || (&row.a == right && &row.b == left)
}

fn checked_index(mask: &mut [bool], column: usize) -> Result<(), PackageError> {
    *mask
        .get_mut(column)
        .ok_or(PackageError::Malformed("field lowering source column"))? = true;
    Ok(())
}

impl FieldLowering {
    pub(super) fn checked(package: &StepPackage) -> Result<Self, PackageError> {
        let Some(Piece::Rows(source)) = package.pieces.first() else {
            return Err(PackageError::Malformed("field lowering application rows"));
        };
        let end = package.source_end();
        let mut protected = vec![false; end];
        for &(start, count, kind) in &package.slots {
            if kind != 2 {
                for column in start..start + count {
                    checked_index(&mut protected, column)?;
                }
            }
        }
        for &column in &package.packed_booleans.sources {
            checked_index(&mut protected, column)?;
        }
        for piece in package.pieces.iter().skip(1) {
            for column in super::application::piece_columns(piece) {
                checked_index(&mut protected, column)?;
            }
        }
        let region = package.application;
        let eligible = |column: usize| {
            ((super::application::WITNESS_START..super::application::WITNESS_START + region.witness_words)
                .contains(&column)
                || (region.local_start..region.local_start + region.locals).contains(&column))
                && protected.get(column) == Some(&false)
        };
        let mut uses = vec![0usize; end];
        for row in source {
            for side in [&row.a, &row.b, &row.c] {
                for &(column, coefficient) in &side.terms {
                    let count = uses
                        .get_mut(column)
                        .ok_or(PackageError::Malformed("field lowering row column"))?;
                    *count += usize::from(coefficient != F::ZERO);
                }
            }
        }
        let rows = source
            .iter()
            .map(|row| Row {
                a: normalized(row.a.clone()),
                b: normalized(row.b.clone()),
                c: normalized(row.c.clone()),
            })
            .collect::<Vec<_>>();
        let mut result = Self::default();
        let mut first = 0;
        while first + 4 <= rows.len() {
            let chain = &rows[first..first + 4];
            let found = (|| {
                let powers = [
                    variable(&chain[0].c)?,
                    variable(&chain[1].c)?,
                    variable(&chain[2].c)?,
                    variable(&chain[3].c)?,
                ];
                let [x2, x4, x6, _] = powers;
                let input = &chain[0].a;
                if chain[0].a != chain[0].b
                    || !product(&chain[1], &unit(x2), &unit(x2))
                    || !product(&chain[2], &unit(x4), &unit(x2))
                    || !product(&chain[3], &unit(x6), input)
                    || powers.iter().copied().collect::<BTreeSet<_>>().len() != 4
                    || input
                        .terms
                        .iter()
                        .any(|&(column, _)| powers.contains(&column))
                    || ![x2, x4, x6].into_iter().all(eligible)
                    || [uses[x2], uses[x4], uses[x6]] != [4, 2, 2]
                {
                    return None;
                }
                Some(Power7 {
                    input: input.clone(),
                    powers,
                })
            })();
            if let Some(power) = found {
                result.dead.extend(power.powers[..3].iter().copied());
                result.removed.extend(first..first + 4);
                result.powers.insert(first, power);
                first += 4;
            } else {
                first += 1;
            }
        }
        let mut candidates = BTreeMap::new();
        for (index, row) in rows.iter().enumerate() {
            if result.removed.contains(&index) {
                continue;
            }
            let equation = if row.a.terms.is_empty() {
                row.b
                    .clone()
                    .scale(row.a.constant)
                    .plus(row.c.clone().scale(-F::ONE))
            } else if row.b.terms.is_empty() {
                row.a
                    .clone()
                    .scale(row.b.constant)
                    .plus(row.c.clone().scale(-F::ONE))
            } else {
                continue;
            };
            let mut equation = normalized(equation);
            let Some((column, coefficient)) = equation.terms.pop() else {
                continue;
            };
            if !eligible(column) || result.dead.contains(&column) {
                continue;
            }
            candidates
                .entry(column)
                .or_insert((index, equation.scale(-coefficient.inverse())));
        }
        for (column, (index, form)) in candidates {
            let mut substituted = Linear {
                constant: form.constant,
                terms: Vec::new(),
            };
            for (input, coefficient) in form.terms {
                if let Some((_, replacement)) = result.affine.get(&input) {
                    substituted = substituted.plus(replacement.clone().scale(coefficient));
                } else {
                    substituted.terms.push((input, coefficient));
                }
            }
            let substituted = normalized(substituted);
            if substituted.terms.len() > 128 || substituted.terms.iter().any(|&(input, _)| input >= column) {
                continue;
            }
            result.affine.insert(column, (index, substituted));
            result.removed.insert(index);
            result.dead.insert(column);
        }
        Ok(result)
    }

    pub(super) fn reconstruct(&self, columns: &mut [F]) -> Result<(), PackageError> {
        let evaluate = |form: &Linear, values: &[F]| {
            form.terms
                .iter()
                .try_fold(form.constant, |sum, &(column, coefficient)| {
                    Ok::<_, PackageError>(
                        sum + coefficient
                            * *values
                                .get(column)
                                .ok_or(PackageError::Input("field reconstruction input"))?,
                    )
                })
        };
        for (&column, (_, form)) in &self.affine {
            let value = evaluate(form, columns)?;
            *columns
                .get_mut(column)
                .ok_or(PackageError::Input("field reconstruction output"))? = value;
        }
        for power in self.powers.values() {
            let x = evaluate(&power.input, columns)?;
            let x2 = x * x;
            let x4 = x2 * x2;
            let x6 = x4 * x2;
            for (&column, value) in power.powers[..3].iter().zip([x2, x4, x6]) {
                *columns
                    .get_mut(column)
                    .ok_or(PackageError::Input("power reconstruction output"))? = value;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_field_lowering.rs"]
mod tests;
