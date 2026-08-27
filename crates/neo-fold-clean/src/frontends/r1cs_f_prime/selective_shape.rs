//! Exact shape discovery for selective low-norm lowering.
//!
//! Owns: selective row/column totals, column alignment, sparse-polynomial shape,
//! and width-audit return data.
//!
//! Does not own: matrix coefficients, witness encoding, trace validity, or
//! relation acceptance.
//!
//! Emits constraints: no. It computes the shape consumed by the row emitter.
//!
//! Authority boundary: shape counts are derived metadata; only the emitted
//! matrices and verified assignment establish the relation.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Layout preparation | shape-audit entrypoints | no | Prepared selective arm plans |
//! | Exact rows | `PreparedSelectiveRows::total_rows` | no | Prepared source-row plan |
//! | CCS polynomial | `selective_polynomial` | no | Fixed selective gate vocabulary |

use std::collections::BTreeMap;

use neo_ccs::{SparsePoly, Term};
use neo_math::{D, F};
use p3_field::{Field, PrimeCharacteristicRing};

use super::super::lowering::LowNormR1csError;
use super::{
    prepare_selective_layout, SelectiveCompilerAudit, SparseR1cs, A, B, BIT, C, CANON_CHUNK_CLASS_SELECTORS,
    CENTERED_UNIT, EVAL_PAIRS, EVAL_SELECTOR, GENERAL_SELECTOR, SBOX_INPUT, SELECTIVE_ARITY,
};

type BorrowMonomial = [u32; 4];
type BorrowPolynomial = BTreeMap<BorrowMonomial, F>;

fn borrow_constant(value: F) -> BorrowPolynomial {
    BTreeMap::from([([0; 4], value)])
}

fn borrow_variable(index: usize) -> BorrowPolynomial {
    let mut powers = [0; 4];
    powers[index] = 1;
    BTreeMap::from([(powers, F::ONE)])
}

fn borrow_add_scaled(target: &mut BorrowPolynomial, source: &BorrowPolynomial, scale: F) {
    for (&monomial, &coefficient) in source {
        let entry = target.entry(monomial).or_insert(F::ZERO);
        *entry += scale * coefficient;
        if *entry == F::ZERO {
            target.remove(&monomial);
        }
    }
}

fn borrow_mul(left: &BorrowPolynomial, right: &BorrowPolynomial) -> BorrowPolynomial {
    let mut product = BorrowPolynomial::new();
    for (&left_monomial, &left_coefficient) in left {
        for (&right_monomial, &right_coefficient) in right {
            let monomial = core::array::from_fn(|index| left_monomial[index] + right_monomial[index]);
            let entry = product.entry(monomial).or_insert(F::ZERO);
            *entry += left_coefficient * right_coefficient;
            if *entry == F::ZERO {
                product.remove(&monomial);
            }
        }
    }
    product
}

fn fixed_borrow_step(bound: usize, digit: &BorrowPolynomial, borrow: &BorrowPolynomial) -> BorrowPolynomial {
    let half = F::from_u64(2).inverse();
    let mut digit_minus_one = digit.clone();
    borrow_add_scaled(&mut digit_minus_one, &borrow_constant(F::ONE), -F::ONE);
    let negative = {
        let mut value = borrow_mul(digit, &digit_minus_one);
        for coefficient in value.values_mut() {
            *coefficient *= half;
        }
        value
    };
    let mut positive = digit.clone();
    borrow_add_scaled(&mut positive, &negative, F::ONE);
    let mut zero = borrow_constant(F::ONE);
    borrow_add_scaled(&mut zero, digit, -F::ONE);
    borrow_add_scaled(&mut zero, &negative, -F::from_u64(2));

    match bound {
        0 => {
            let mut one_minus_borrow = borrow_constant(F::ONE);
            borrow_add_scaled(&mut one_minus_borrow, borrow, -F::ONE);
            let mut result = borrow_constant(F::ONE);
            borrow_add_scaled(&mut result, &borrow_mul(&negative, &one_minus_borrow), -F::ONE);
            result
        }
        1 => {
            let mut result = positive;
            borrow_add_scaled(&mut result, &borrow_mul(&zero, borrow), F::ONE);
            result
        }
        2 => borrow_mul(&positive, borrow),
        _ => unreachable!("base-3 bound digit"),
    }
}

/// `borrow_out - step(h₁, d₁, step(h₀, d₀, borrow_in))`.
fn fixed_two_trit_borrow_relation(bound: usize) -> BorrowPolynomial {
    let digit_zero = borrow_variable(0);
    let digit_one = borrow_variable(1);
    let borrow_in = borrow_variable(2);
    let borrow_out = borrow_variable(3);
    let first = fixed_borrow_step(bound % 3, &digit_zero, &borrow_in);
    let second = fixed_borrow_step((bound / 3) % 3, &digit_one, &first);
    let mut relation = borrow_out;
    borrow_add_scaled(&mut relation, &second, -F::ONE);
    relation
}

pub(crate) struct SelectiveLowNormShape {
    pub rows: usize,
    pub columns: usize,
    pub public_input_len: usize,
    pub polynomial: SparsePoly<F>,
    pub compiler_audit: SelectiveCompilerAudit,
}

pub(crate) struct SelectiveLowNormShapeSummary {
    pub rows: usize,
    pub columns: usize,
    pub public_input_len: usize,
    pub polynomial: SparsePoly<F>,
    pub total_coordinates: usize,
}

#[cfg(test)]
impl SelectiveLowNormShapeSummary {
    fn matches(&self, shape: &SelectiveLowNormShape) -> bool {
        self.rows == shape.rows
            && self.columns == shape.columns
            && self.public_input_len == shape.public_input_len
            && self.total_coordinates == shape.compiler_audit.width().total_coordinates
            && self.polynomial.arity() == shape.polynomial.arity()
            && self.polynomial.terms().len() == shape.polynomial.terms().len()
            && self
                .polynomial
                .terms()
                .iter()
                .zip(shape.polynomial.terms())
                .all(|(left, right)| left.coeff == right.coeff && left.exps == right.exps)
    }
}

pub(crate) fn audit_multi_branch_selective_low_norm_shape_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormShape, LowNormR1csError> {
    audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(
        arms,
        shared_private_fields,
        shared_private_fields,
        modulus,
        residue,
    )
}

pub(crate) fn audit_multi_branch_selective_low_norm_shape_with_shared_bit_prefix(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
) -> Result<SelectiveLowNormShape, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?;
    let columns = layout.columns.next_multiple_of(D);
    let rows = layout.prepared_rows.total_rows();
    Ok(SelectiveLowNormShape {
        rows,
        columns,
        public_input_len: layout.public_input_len,
        polynomial: selective_polynomial(),
        compiler_audit: layout.compiler_audit,
    })
}

#[doc(hidden)]
pub fn selective_polynomial() -> SparsePoly<F> {
    let term = |coefficient: F, powers: &[(usize, u32)]| {
        let mut exps = vec![0u32; SELECTIVE_ARITY];
        for &(index, power) in powers {
            exps[index] = power;
        }
        Term {
            coeff: coefficient,
            exps,
        }
    };
    let mut terms = vec![
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (BIT, 2)]),
        term(-F::ONE, &[(GENERAL_SELECTOR, 1), (BIT, 1)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (A, 1), (B, 1)]),
        term(-F::ONE, &[(GENERAL_SELECTOR, 1), (C, 1)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (SBOX_INPUT, 7)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (CENTERED_UNIT, 3)]),
        term(-F::ONE, &[(GENERAL_SELECTOR, 1), (CENTERED_UNIT, 1)]),
        term(
            -F::ONE,
            &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (CENTERED_UNIT, 3)],
        ),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (CENTERED_UNIT, 1)]),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (CENTERED_UNIT, 6)]),
        term(
            -F::from_u64(2),
            &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (CENTERED_UNIT, 4)],
        ),
        term(F::ONE, &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (CENTERED_UNIT, 2)]),
        term(-F::from_u64(7), &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (A, 6)]),
        term(F::from_u64(14), &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (A, 4)]),
        term(-F::from_u64(7), &[(GENERAL_SELECTOR, 1), (EVAL_SELECTOR, 1), (A, 2)]),
        term(-F::ONE, &[(EVAL_SELECTOR, 1), (C, 1)]),
    ];
    for &(left, right) in &EVAL_PAIRS {
        terms.push(term(F::ONE, &[(EVAL_SELECTOR, 1), (left, 1), (right, 1)]));
    }

    // Pair adjacent radix-3 transitions. Bounds 5..=8 are normalized to
    // 3..=0 by complementing both trits and the endpoint borrows. The five
    // class ports are zero on ordinary GENERAL rows, while GENERAL is zero on
    // evaluation rows. Each term therefore carries both selectors. The
    // composed relation has degree five, hence degree seven after gating.
    let variables = [CENTERED_UNIT, A, BIT, C];
    for (bound, &class_selector) in CANON_CHUNK_CLASS_SELECTORS.iter().enumerate() {
        for (monomial, coefficient) in fixed_two_trit_borrow_relation(bound) {
            let mut powers = vec![(GENERAL_SELECTOR, 1), (class_selector, 1)];
            for (variable, power) in variables.into_iter().zip(monomial) {
                if power != 0 {
                    powers.push((variable, power));
                }
            }
            terms.push(term(coefficient, &powers));
        }
    }
    SparsePoly::new(SELECTIVE_ARITY, terms)
}

/// Report whether a polynomial is the exact selective low-norm gate
/// polynomial used by this frontend.
#[doc(hidden)]
pub fn is_canonical_selective_low_norm_polynomial(polynomial: &SparsePoly<F>) -> bool {
    let expected = selective_polynomial();
    polynomial.arity() == expected.arity()
        && polynomial.terms().len() == expected.terms().len()
        && polynomial
            .terms()
            .iter()
            .zip(expected.terms())
            .all(|(actual, expected)| actual.coeff == expected.coeff && actual.exps == expected.exps)
}

#[cfg(test)]
#[path = "tests/selective_shape.rs"]
mod tests;
