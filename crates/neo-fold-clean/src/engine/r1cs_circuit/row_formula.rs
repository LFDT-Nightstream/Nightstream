//! Pure sparse-row constructors shared by synthesis and indexed artifacts.
//!
//! These functions allocate nothing and carry no acceptance meaning. They
//! make the exact A/B/C encoding of the builder's primitive multiplication
//! and equality rows available to the active row-at compiler.

use std::collections::BTreeMap;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{Lc, Var};

pub(crate) type ConstraintRow = (Lc, Lc, Lc);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalSparseRow {
    pub a: Vec<(usize, F)>,
    pub b: Vec<(usize, F)>,
    pub c: Vec<(usize, F)>,
}

pub(crate) fn multiplication_constraint_row(a: &Lc, b: &Lc, output: Var) -> ConstraintRow {
    (a.clone(), b.clone(), Lc::from_var(output))
}

pub(crate) fn equality_constraint_row(left: &Lc, right: &Lc) -> ConstraintRow {
    (
        left.clone().add_scaled(right, -F::ONE),
        Lc::from_var(Var::ONE),
        Lc::zero(),
    )
}

pub(crate) fn canonical_sparse_row(row: &ConstraintRow) -> CanonicalSparseRow {
    CanonicalSparseRow {
        a: canonical_terms(&row.0),
        b: canonical_terms(&row.1),
        c: canonical_terms(&row.2),
    }
}

pub(crate) fn canonical_terms(lc: &Lc) -> Vec<(usize, F)> {
    let mut coefficients = BTreeMap::<usize, F>::new();
    if lc.constant != F::ZERO {
        coefficients.insert(Var::ONE.col(), lc.constant);
    }
    for &(column, coefficient) in &lc.terms {
        *coefficients.entry(column).or_insert(F::ZERO) += coefficient;
    }
    coefficients
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}
