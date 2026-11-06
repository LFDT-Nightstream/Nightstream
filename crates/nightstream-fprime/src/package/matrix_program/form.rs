//! Linear matrix expressions keep retained fields as geometric coefficient runs.
//! Expansion is reserved for the scalar-row reference interface.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{Entry, PackageError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatrixRun {
    column: usize,
    count: usize,
    coefficient: Goldilocks,
    ratio: Goldilocks,
}

impl MatrixRun {
    fn key(&self) -> (usize, usize, u64) {
        (self.column, self.count, self.ratio.as_canonical_u64())
    }

    pub(super) fn scaled(mut self, scalar: Goldilocks) -> Self {
        self.coefficient *= scalar;
        self
    }
}

pub(super) fn normalize_terms(terms: &mut Vec<MatrixRun>) {
    terms.sort_unstable_by_key(MatrixRun::key);
    let mut written = 0;
    for read in 0..terms.len() {
        let term = terms[read];
        if written != 0 && terms[written - 1].key() == term.key() {
            terms[written - 1].coefficient += term.coefficient;
            if terms[written - 1].coefficient == Goldilocks::ZERO {
                written -= 1;
            }
        } else if term.coefficient != Goldilocks::ZERO {
            terms[written] = term;
            written += 1;
        }
    }
    terms.truncate(written);
}

pub(super) fn entries(terms: &[MatrixRun]) -> Vec<Entry> {
    let mut entries = Vec::new();
    for term in terms {
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

    pub(super) fn from_terms(mut terms: Vec<MatrixRun>) -> Self {
        debug_assert!(
            terms.windows(2).all(|pair| pair[0].key() < pair[1].key())
                && terms
                    .iter()
                    .all(|term| term.coefficient != Goldilocks::ZERO),
            "matrix form terms must be sorted, unique and nonzero"
        );
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

    pub(in crate::package) fn entries(&self) -> Vec<Entry> {
        entries(self.terms())
    }

    pub(super) fn validate(&self, columns: usize) -> Result<(), PackageError> {
        if self.terms().iter().any(|term| {
            term.count == 0
                || term
                    .column
                    .checked_add(term.count)
                    .is_none_or(|end| end > columns)
        }) {
            return Err(PackageError::Malformed("matrix sparse column"));
        }
        Ok(())
    }
}
