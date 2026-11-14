use std::collections::{BTreeMap, BTreeSet, HashMap};

use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use super::{field_lowering::normalized, layout::forced_column, Linear, Piece, Row, StepPackage};
use crate::{package::Entries, PackageError};

type Key = Vec<(usize, u64)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Group {
    pub sources: Vec<usize>,
    pub certificates: Vec<usize>,
    pub source_records: Vec<(usize, usize)>,
    pub row: Row,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct GuardFusion {
    pub groups: Vec<Group>,
    pub replacements: BTreeMap<usize, Row>,
    pub removed: BTreeSet<usize>,
}

#[derive(Clone)]
struct Family {
    source: usize,
    selectors: BTreeSet<usize>,
    total_one: bool,
}

#[derive(Clone)]
struct Guarded {
    source: usize,
    gate: Linear,
    residual: Linear,
    certificate: Option<usize>,
}

fn key(form: &Linear) -> Key {
    std::iter::once((0, form.constant.as_canonical_u64()))
        .chain(
            form.terms
                .iter()
                .map(|&(column, value)| (column, value.as_canonical_u64())),
        )
        .collect()
}

fn one() -> Linear {
    Linear {
        constant: F::ONE,
        terms: Vec::new(),
    }
}

fn canonical(form: Linear) -> Linear {
    let form = normalized(form);
    let coefficient = form.terms.first().map_or(F::ONE, |term| term.1);
    form.scale(coefficient.inverse())
}

pub(super) fn resolved(form: &Linear, forms: &HashMap<usize, Entries>) -> Linear {
    let mut result = Linear {
        constant: form.constant,
        terms: Vec::new(),
    };
    for &(column, coefficient) in &form.terms {
        if let Some(replacement) = forms.get(&column) {
            result.terms.extend(
                replacement
                    .iter()
                    .map(|&(input, value)| (input, value * coefficient)),
            );
        } else {
            result.terms.push((column, coefficient));
        }
    }
    normalized(result)
}

fn linear_equation(row: &Row) -> Option<Linear> {
    let equation = if row.a.terms.is_empty() {
        row.b.clone().scale(row.a.constant)
    } else if row.b.terms.is_empty() {
        row.a.clone().scale(row.b.constant)
    } else {
        return None;
    };
    Some(canonical(equation.plus(row.c.clone().scale(-F::ONE))))
}

fn families(rows: &[(usize, Row)], known: &BTreeSet<usize>) -> Vec<Family> {
    let mut found = BTreeMap::new();
    for (source, row) in rows {
        let Some(equation) = linear_equation(row) else { continue };
        for equation in [equation.clone(), equation.scale(-F::ONE)] {
            if equation
                .terms
                .iter()
                .any(|(column, _)| !known.contains(column))
            {
                continue;
            }
            let selectors = equation
                .terms
                .iter()
                .filter_map(|&(column, value)| (value == F::ONE).then_some(column))
                .collect::<BTreeSet<_>>();
            let rest = equation
                .terms
                .iter()
                .filter(|(_, value)| *value != F::ONE)
                .collect::<Vec<_>>();
            let total_one = equation.constant == -F::ONE && rest.is_empty();
            let total_bit = equation.constant == F::ZERO && rest.len() == 1 && rest[0].1 == -F::ONE;
            if selectors.len() >= 2 && (selectors.len() as u128) < F::ORDER_U64 as u128 && (total_one || total_bit) {
                found.entry(selectors.clone()).or_insert(Family {
                    source: *source,
                    selectors,
                    total_one,
                });
            }
        }
    }
    found.into_values().collect()
}

fn gate(form: &Linear, known: &BTreeSet<usize>, families: &[Family]) -> Option<(Linear, Option<usize>)> {
    let form = canonical(form.clone());
    for form in [form.clone(), form.scale(-F::ONE)] {
        if form.terms.is_empty() {
            continue;
        }
        let positive = form.constant == F::ZERO && form.terms.iter().all(|(_, value)| *value == F::ONE);
        let complement = form.constant == F::ONE && form.terms.iter().all(|(_, value)| *value == -F::ONE);
        if !positive && !complement {
            continue;
        }
        if form.terms.len() == 1 && known.contains(&form.terms[0].0) {
            return Some((form, None));
        }
        if let Some(family) = families.iter().find(|family| {
            form.terms
                .iter()
                .all(|(column, _)| family.selectors.contains(column))
        }) {
            return Some((form, Some(family.source)));
        }
    }
    None
}

impl GuardFusion {
    pub(super) fn checked(package: &StepPackage) -> Result<Self, PackageError> {
        let Some(Piece::Rows(source)) = package.pieces.first() else {
            return Err(PackageError::Malformed("guard fusion application rows"));
        };
        let forms = package.resolved_forms()?;
        let forcing = source
            .iter()
            .enumerate()
            .filter_map(|(index, row)| forced_column(row).map(|column| (index, column)))
            .collect::<BTreeMap<_, _>>();
        let known =
            package
                .packed_booleans
                .sources
                .iter()
                .copied()
                .chain(forcing.iter().filter_map(|(&index, &column)| {
                    (!package.field_lowering.removed.contains(&index)).then_some(column)
                }))
                .filter(|column| !forms.contains_key(column))
                .collect::<BTreeSet<_>>();
        let rows = source
            .iter()
            .enumerate()
            .filter(|(index, _)| !package.field_lowering.removed.contains(index))
            .map(|(index, row)| {
                (
                    index,
                    Row {
                        a: resolved(&row.a, &forms),
                        b: resolved(&row.b, &forms),
                        c: resolved(&row.c, &forms),
                    },
                )
            })
            .collect::<Vec<_>>();
        let families = families(&rows, &known);
        let certificates = families
            .iter()
            .map(|family| family.source)
            .collect::<BTreeSet<_>>();
        let mut candidates = Vec::new();
        for (index, row) in &rows {
            if row.c != Linear::default() || forcing.contains_key(index) || certificates.contains(index) {
                continue;
            }
            for (input, residual) in [(&row.a, &row.b), (&row.b, &row.a)] {
                if let Some((gate, certificate)) = gate(input, &known, &families) {
                    candidates.push(Guarded {
                        source: *index,
                        gate,
                        residual: canonical(residual.clone()),
                        certificate,
                    });
                }
            }
        }
        let mut groups: BTreeMap<(usize, Key), Vec<usize>> = BTreeMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.gate.constant != F::ZERO {
                continue;
            }
            for (family_index, family) in families.iter().enumerate() {
                if candidate
                    .gate
                    .terms
                    .iter()
                    .all(|(column, _)| family.selectors.contains(column))
                {
                    let mut common = candidate.residual.clone();
                    common.constant = F::ZERO;
                    groups
                        .entry((family_index, key(&common)))
                        .or_default()
                        .push(index);
                }
            }
        }
        let mut groups = groups.into_iter().collect::<Vec<_>>();
        groups.sort_by(|left, right| {
            right
                .1
                .len()
                .cmp(&left.1.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut result = Self::default();
        let mut used = BTreeSet::new();
        for ((family_index, _), members) in groups {
            let mut selected = Vec::new();
            let mut selectors = BTreeSet::new();
            for index in members {
                let candidate = &candidates[index];
                if used.contains(&candidate.source)
                    || candidate
                        .gate
                        .terms
                        .iter()
                        .any(|(column, _)| selectors.contains(column))
                {
                    continue;
                }
                selectors.extend(candidate.gate.terms.iter().map(|&(column, _)| column));
                selected.push(candidate);
            }
            if selected.len() < 2 {
                continue;
            }
            let family = &families[family_index];
            let mut a = Linear::default();
            let mut b = selected[0].residual.clone();
            b.constant = F::ZERO;
            let mut c = Linear::default();
            for candidate in &selected {
                a = a.plus(candidate.gate.clone());
                c = c.plus(candidate.gate.clone().scale(-candidate.residual.constant));
            }
            if family.total_one && selectors == family.selectors {
                a = one();
            }
            let sources = selected
                .iter()
                .map(|candidate| candidate.source)
                .collect::<Vec<_>>();
            used.extend(sources.iter().copied());
            result.add(Group {
                sources,
                certificates: vec![family.source],
                source_records: Vec::new(),
                row: Row {
                    a: normalized(a),
                    b,
                    c: normalized(c),
                },
            });
        }
        let mut pending: BTreeMap<Key, Vec<usize>> = BTreeMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            if used.contains(&candidate.source) {
                continue;
            }
            let complement = normalized(one().plus(candidate.gate.clone().scale(-F::ONE)));
            let others = pending.entry(key(&complement)).or_default();
            let matched = loop {
                match others.pop() {
                    Some(other) if used.contains(&candidates[other].source) => continue,
                    Some(other) if candidates[other].source != candidate.source => break Some(other),
                    _ => break None,
                }
            };
            if let Some(other) = matched {
                let other = &candidates[other];
                used.extend([candidate.source, other.source]);
                let mut certificates = candidate
                    .certificate
                    .into_iter()
                    .chain(other.certificate)
                    .collect::<Vec<_>>();
                certificates.sort_unstable();
                certificates.dedup();
                result.add(Group {
                    sources: vec![candidate.source, other.source],
                    certificates,
                    source_records: Vec::new(),
                    row: Row {
                        a: candidate.gate.clone(),
                        b: normalized(
                            candidate
                                .residual
                                .clone()
                                .plus(other.residual.clone().scale(-F::ONE)),
                        ),
                        c: other.residual.clone().scale(-F::ONE),
                    },
                });
            } else {
                pending.entry(key(&candidate.gate)).or_default().push(index);
            }
        }
        result.groups.sort_by_key(|group| group.sources[0]);
        if !result.groups.is_empty() {
            let records = package
                .native
                .as_ref()
                .ok_or(PackageError::Malformed("guard fusion source records"))?;
            let mut dropped = BTreeSet::new();
            for block in package.blocks().take(package.application.pieces - 1) {
                let first = block.start() - package.application.local_start;
                for recipe in first..first + 1096 {
                    dropped.insert(records.recipe_row(recipe)?);
                }
            }
            let packed = package
                .packed_booleans
                .removed
                .iter()
                .map(|(index, _)| *index)
                .collect::<BTreeSet<_>>();
            let origins = (0..records.row_count())
                .filter(|index| !dropped.contains(index))
                .enumerate()
                .filter_map(|(index, record)| (!packed.contains(&index)).then_some(record))
                .collect::<Vec<_>>();
            for group in &mut result.groups {
                group.source_records = group
                    .sources
                    .iter()
                    .chain(&group.certificates)
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|index| {
                        origins
                            .get(index)
                            .copied()
                            .map(|record| (index, record))
                            .ok_or(PackageError::Malformed("guard fusion original row index"))
                    })
                    .collect::<Result<_, _>>()?;
            }
        }
        Ok(result)
    }

    fn add(&mut self, mut group: Group) {
        group.sources.sort_unstable();
        self.replacements
            .insert(group.sources[0], group.row.clone());
        self.removed.extend(group.sources[1..].iter().copied());
        self.groups.push(group);
    }

    pub(super) fn bind(&self, source: &[Row], update: &mut impl FnMut(u64)) {
        fn row(row: &Row, update: &mut impl FnMut(u64)) {
            for form in [&row.a, &row.b, &row.c] {
                update(form.constant.as_canonical_u64());
                update(form.terms.len() as u64);
                for &(column, value) in &form.terms {
                    update(column as u64);
                    update(value.as_canonical_u64());
                }
            }
        }
        update(self.groups.len() as u64);
        for group in &self.groups {
            update(group.source_records.len() as u64);
            for &(index, record) in &group.source_records {
                update(index as u64);
                update(record as u64);
            }
            for indices in [&group.sources, &group.certificates] {
                update(indices.len() as u64);
                for &index in indices {
                    update(index as u64);
                    row(&source[index], update);
                }
            }
            row(&group.row, update);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_guard_fusion.rs"]
mod tests;
