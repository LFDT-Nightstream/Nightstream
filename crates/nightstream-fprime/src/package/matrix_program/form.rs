//! Linear matrix expressions keep retained fields as geometric coefficient runs.
//! Expansion is reserved for the scalar-row reference interface.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{Entry, PackageError};

/// Add `coefficient * ratio^i` at `column + i` for each coordinate in this run.
/// Runs in a row can overlap; their contributions add in the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixRun {
    column: usize,
    count: usize,
    coefficient: Goldilocks,
    ratio: Goldilocks,
}

impl MatrixRun {
    pub fn column(&self) -> usize {
        self.column
    }
    pub fn column_count(&self) -> usize {
        self.count
    }
    pub fn coefficient(&self) -> u64 {
        self.coefficient.as_canonical_u64()
    }
    pub fn ratio(&self) -> u64 {
        self.ratio.as_canonical_u64()
    }

    fn key(&self) -> (usize, usize, u64) {
        (self.column, self.count, self.ratio.as_canonical_u64())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::package) enum Form {
    #[default]
    Empty,
    One(MatrixRun),
    Many(Vec<MatrixRun>),
}

impl Form {
    pub(in crate::package) fn singleton(column: usize, coefficient: Goldilocks) -> Self {
        if coefficient == Goldilocks::ZERO {
            return Self::Empty;
        }
        Self::One(MatrixRun {
            column,
            count: 1,
            coefficient,
            ratio: Goldilocks::ONE,
        })
    }

    pub(super) fn retained(column: usize, count: usize) -> Self {
        Self::One(MatrixRun {
            column,
            count,
            coefficient: Goldilocks::ONE,
            ratio: if count == 1 {
                Goldilocks::ONE
            } else {
                Goldilocks::from_u64(3)
            },
        })
    }

    pub(super) fn from_entries(entries: Vec<Entry>) -> Self {
        let mut terms = entries
            .into_iter()
            .map(|entry| MatrixRun {
                column: entry.column,
                count: 1,
                coefficient: entry.coefficient,
                ratio: Goldilocks::ONE,
            })
            .collect::<Vec<_>>();
        terms.sort_unstable_by_key(MatrixRun::key);
        let mut combined: Vec<MatrixRun> = Vec::with_capacity(terms.len());
        for term in terms {
            if let Some(last) = combined.last_mut() {
                if last.key() == term.key() {
                    last.coefficient += term.coefficient;
                    if last.coefficient == Goldilocks::ZERO {
                        combined.pop();
                    }
                    continue;
                }
            }
            if term.coefficient != Goldilocks::ZERO {
                combined.push(term);
            }
        }
        Self::from_terms(combined)
    }

    fn from_terms(mut terms: Vec<MatrixRun>) -> Self {
        match terms.len() {
            0 => Self::Empty,
            1 => Self::One(terms.pop().expect("one term")),
            _ => Self::Many(terms),
        }
    }

    pub(in crate::package) fn terms(&self) -> &[MatrixRun] {
        match self {
            Self::Empty => &[],
            Self::One(term) => std::slice::from_ref(term),
            Self::Many(terms) => terms,
        }
    }

    pub(in crate::package) fn into_terms(self) -> Vec<MatrixRun> {
        match self {
            Self::Empty => Vec::new(),
            Self::One(term) => vec![term],
            Self::Many(terms) => terms,
        }
    }

    pub(in crate::package) fn entries(&self) -> Vec<Entry> {
        let mut entries = Vec::new();
        for term in self.terms() {
            let mut coefficient = term.coefficient;
            for offset in 0..term.count {
                entries.push(Entry {
                    column: term.column + offset,
                    coefficient,
                });
                coefficient *= term.ratio;
            }
        }
        entries.sort_unstable_by_key(|entry| entry.column);
        let mut combined: Vec<Entry> = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(last) = combined.last_mut() {
                if last.column == entry.column {
                    last.coefficient += entry.coefficient;
                    if last.coefficient == Goldilocks::ZERO {
                        combined.pop();
                    }
                    continue;
                }
            }
            if entry.coefficient != Goldilocks::ZERO {
                combined.push(entry);
            }
        }
        combined
    }

    pub(in crate::package) fn into_entries(self) -> Vec<Entry> {
        self.entries()
    }

    pub(in crate::package) fn append(self, other: Self) -> Self {
        let (left, right) = match (self, other) {
            (Self::Empty, other) => return other,
            (form, Self::Empty) => return form,
            (Self::One(mut left), Self::One(right)) => {
                return match left.key().cmp(&right.key()) {
                    std::cmp::Ordering::Less => Self::Many(vec![left, right]),
                    std::cmp::Ordering::Greater => Self::Many(vec![right, left]),
                    std::cmp::Ordering::Equal => {
                        left.coefficient += right.coefficient;
                        if left.coefficient == Goldilocks::ZERO {
                            Self::Empty
                        } else {
                            Self::One(left)
                        }
                    }
                };
            }
            (Self::Many(mut left), Self::Many(right)) if left.last().unwrap().key() < right.first().unwrap().key() => {
                left.extend(right);
                return Self::Many(left);
            }
            (Self::Many(left), Self::Many(mut right)) if right.last().unwrap().key() < left.first().unwrap().key() => {
                right.extend(left);
                return Self::Many(right);
            }
            (Self::Many(mut terms), Self::One(term)) | (Self::One(term), Self::Many(mut terms))
                if terms.last().unwrap().key() < term.key() =>
            {
                terms.push(term);
                return Self::Many(terms);
            }
            (left, right) => (left.into_terms(), right.into_terms()),
        };
        let mut left = left.into_iter().peekable();
        let mut right = right.into_iter().peekable();
        let mut terms = Vec::with_capacity(left.len() + right.len());
        while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
            match a.key().cmp(&b.key()) {
                std::cmp::Ordering::Less => terms.push(left.next().unwrap()),
                std::cmp::Ordering::Greater => terms.push(right.next().unwrap()),
                std::cmp::Ordering::Equal => {
                    let mut a = left.next().unwrap();
                    a.coefficient += right.next().unwrap().coefficient;
                    if a.coefficient != Goldilocks::ZERO {
                        terms.push(a);
                    }
                }
            }
        }
        terms.extend(left);
        terms.extend(right);
        Self::from_terms(terms)
    }

    pub(in crate::package) fn scaled(mut self, scalar: Goldilocks) -> Self {
        if scalar == Goldilocks::ZERO {
            return Self::Empty;
        }
        for term in match &mut self {
            Self::Empty => &mut [],
            Self::One(term) => std::slice::from_mut(term),
            Self::Many(terms) => terms,
        } {
            term.coefficient *= scalar;
        }
        self
    }

    pub(super) fn validate(&self, columns: usize) -> Result<(), PackageError> {
        if self.terms().iter().any(|term| {
            term.count == 0
                || term
                    .column
                    .checked_add(term.count)
                    .is_none_or(|end| end > columns)
        }) {
            return Err(PackageError::Invalid("matrix sparse column"));
        }
        Ok(())
    }
}
