use std::collections::{BTreeMap, BTreeSet};

use neo_math::Fq as F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use super::{
    field_lowering::normalized, guard_fusion::resolved, layout::forced_column, Linear, Piece, Row, StepPackage,
};
use crate::PackageError;

const SOURCE_BITS: u32 = 32;
const PAIRS_PER_ROW: usize = 63;
type Key = Vec<(usize, u64)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Bit {
    Zero,
    One,
    Column(usize),
}

type Pair = (Bit, Bit);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Group {
    sources: Vec<usize>,
    certificates: Vec<usize>,
    guard: Linear,
    pairs: Vec<Pair>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct WordPacking {
    groups: Vec<Group>,
}

fn binary_pairs(form: &Linear, known: &BTreeMap<usize, usize>) -> Option<Vec<Pair>> {
    let mut digits = BTreeMap::<u32, (Option<Bit>, Option<Bit>)>::new();
    for (column, coefficient) in std::iter::once((0, form.constant)).chain(form.terms.iter().copied()) {
        let value = coefficient.as_canonical_u64();
        let positive = value <= F::ORDER_U64 / 2;
        let magnitude = if positive { value } else { F::ORDER_U64 - value };
        if magnitude == 0 {
            continue;
        }
        if magnitude >= 1u64 << SOURCE_BITS {
            return None;
        }
        let positions = if column == 0 {
            (0..SOURCE_BITS)
                .filter(|position| magnitude & (1u64 << position) != 0)
                .collect::<Vec<_>>()
        } else {
            if !known.contains_key(&column) || !magnitude.is_power_of_two() {
                return None;
            }
            vec![magnitude.trailing_zeros()]
        };
        for position in positions {
            let digit = digits.entry(position).or_default();
            let side = if positive { &mut digit.0 } else { &mut digit.1 };
            if side
                .replace(if column == 0 { Bit::One } else { Bit::Column(column) })
                .is_some()
            {
                return None;
            }
        }
    }
    let pairs = digits
        .into_values()
        .filter_map(|(left, right)| {
            let left = left.unwrap_or(Bit::Zero);
            let right = right.unwrap_or(Bit::Zero);
            (left != right).then_some(if left < right { (left, right) } else { (right, left) })
        })
        .collect::<Vec<_>>();
    (!pairs.is_empty()).then_some(pairs)
}

fn guard_key(form: &Linear) -> Option<Key> {
    let terms = std::iter::once((0, form.constant))
        .chain(form.terms.iter().copied())
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect::<Vec<_>>();
    let scale = terms.first()?.1.inverse();
    Some(
        terms
            .into_iter()
            .map(|(column, value)| (column, (value * scale).as_canonical_u64()))
            .collect(),
    )
}

fn form(key: Key) -> Linear {
    normalized(Linear {
        constant: F::ZERO,
        terms: key
            .into_iter()
            .map(|(column, value)| (column, F::from_u64(value)))
            .collect(),
    })
}

impl Group {
    fn rows(&self) -> Vec<Row> {
        self.pairs
            .chunks(PAIRS_PER_ROW)
            .map(|pairs| {
                let mut packed = Linear::default();
                for (position, &(left, right)) in pairs.iter().enumerate() {
                    let coefficient = F::from_u64(1u64 << position);
                    for (bit, coefficient) in [(left, coefficient), (right, -coefficient)] {
                        match bit {
                            Bit::Zero => {}
                            Bit::One => packed.constant += coefficient,
                            Bit::Column(column) => packed.terms.push((column, coefficient)),
                        }
                    }
                }
                Row {
                    a: self.guard.clone(),
                    b: normalized(packed),
                    c: Linear::default(),
                }
            })
            .collect()
    }
}

impl WordPacking {
    pub(super) fn checked(package: &StepPackage) -> Result<Self, PackageError> {
        let Some(Piece::Rows(source)) = package.pieces.first() else {
            return Err(PackageError::Malformed("word packing application rows"));
        };
        let forms = package.resolved_forms()?;
        let retained = |index: usize| {
            !package.field_lowering.removed.contains(&index)
                && !package.guard_fusion.removed.contains(&index)
                && !package.guard_fusion.replacements.contains_key(&index)
        };
        let source_bits = package
            .packed_booleans
            .original_rows(source)?
            .into_iter()
            .filter_map(forced_column)
            .collect::<BTreeSet<_>>();
        let rows = source
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                !package.field_lowering.removed.contains(index) && !package.guard_fusion.removed.contains(index)
            })
            .map(|(index, row)| {
                let row = package.guard_fusion.replacements.get(&index).unwrap_or(row);
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
        let mut known = BTreeMap::new();
        let mut protected = BTreeSet::new();
        for (index, row) in &rows {
            if let Some(column) = forced_column(row) {
                protected.insert(*index);
                if source_bits.contains(&column) && retained(*index) {
                    known.insert(column, *index);
                }
            }
        }
        let first_packed = source
            .len()
            .checked_sub(package.packed_booleans.row_count())
            .ok_or(PackageError::Malformed("word packing certificate rows"))?;
        let layout = package.layout()?;
        let pairs = package.packed_booleans.pairs().collect::<Vec<_>>();
        for (offset, pairs) in pairs
            .chunks(super::packed_boolean::PAIRS_PER_ROW)
            .enumerate()
        {
            let certificate = first_packed + offset;
            protected.insert(certificate);
            if retained(certificate)
                && pairs
                    .iter()
                    .flat_map(|&(left, right)| [left, right])
                    .all(|column| {
                        !forms.contains_key(&column)
                            && layout
                                .slot(column)
                                .is_some_and(|(_, kind)| kind == super::layout::Kind::Bit)
                    })
            {
                for &(column, _) in pairs {
                    if source_bits.contains(&column) {
                        known.insert(column, certificate);
                    }
                }
            }
        }
        let mut candidates = BTreeMap::<Key, Vec<(usize, Vec<Pair>)>>::new();
        for (index, row) in rows {
            if protected.contains(&index) || row.c != Linear::default() {
                continue;
            }
            let mut choices = [(&row.a, &row.b), (&row.b, &row.a)]
                .into_iter()
                .filter_map(|(guard, residual)| Some((guard_key(guard)?, binary_pairs(residual, &known)?)))
                .collect::<Vec<_>>();
            choices.sort_by_key(|(_, pairs)| pairs.len());
            if let Some((guard, pairs)) = choices.into_iter().next() {
                candidates.entry(guard).or_default().push((index, pairs));
            }
        }
        let mut groups = Vec::new();
        for (guard, candidates) in candidates {
            let pairs = candidates
                .iter()
                .flat_map(|(_, pairs)| pairs.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if pairs.len().div_ceil(PAIRS_PER_ROW) >= candidates.len() {
                continue;
            }
            let certificates = pairs
                .iter()
                .flat_map(|&(left, right)| [left, right])
                .filter_map(|bit| match bit {
                    Bit::Column(column) => Some(known[&column]),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            groups.push(Group {
                sources: candidates.into_iter().map(|(index, _)| index).collect(),
                certificates,
                guard: form(guard),
                pairs,
            });
        }
        groups.sort_by_key(|group| group.sources[0]);
        Ok(Self { groups })
    }

    pub(super) fn edits(&self) -> BTreeMap<usize, Vec<Row>> {
        let mut edits = BTreeMap::new();
        for group in &self.groups {
            edits.insert(group.sources[0], group.rows());
            for &index in &group.sources[1..] {
                edits.insert(index, Vec::new());
            }
        }
        edits
    }

    pub(super) fn validate(&self, package: &StepPackage) -> Result<(), PackageError> {
        if self != &Self::checked(package)? {
            return Err(PackageError::Malformed("word packing source map"));
        }
        Ok(())
    }

    pub(super) fn bind(&self, source: &[Row], update: &mut impl FnMut(u64)) {
        update(self.groups.len() as u64);
        for group in &self.groups {
            for indices in [&group.sources, &group.certificates] {
                update(indices.len() as u64);
                for &index in indices {
                    update(index as u64);
                    for side in [&source[index].a, &source[index].b, &source[index].c] {
                        update(side.constant.as_canonical_u64());
                        update(side.terms.len() as u64);
                        for &(column, value) in &side.terms {
                            update(column as u64);
                            update(value.as_canonical_u64());
                        }
                    }
                }
            }
            update(group.guard.constant.as_canonical_u64());
            update(group.guard.terms.len() as u64);
            for &(column, value) in &group.guard.terms {
                update(column as u64);
                update(value.as_canonical_u64());
            }
            update(group.pairs.len() as u64);
            for &(left, right) in &group.pairs {
                for bit in [left, right] {
                    let (tag, column) = match bit {
                        Bit::Zero => (0, 0),
                        Bit::One => (1, 0),
                        Bit::Column(column) => (2, column),
                    };
                    update(tag);
                    update(column as u64);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_word_packing.rs"]
mod tests;
