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

use neo_ccs::{SparsePoly, Term};
use neo_math::{D, F};
use p3_field::{Field, PrimeCharacteristicRing};

use super::super::lowering::LowNormR1csError;
use super::{
    prepare_selective_layout, SelectiveCompilerAudit, SparseR1cs, A, B, BIT, C, CANON_BORROW, CANON_BOUND_DIGIT,
    CANON_DIGIT, CANON_NEXT_BORROW, CENTERED_UNIT, EVAL_PAIRS, EVAL_SELECTOR, GENERAL_SELECTOR, SBOX_INPUT,
    SELECTIVE_ARITY,
};

pub(crate) struct SelectiveLowNormShape {
    pub rows: usize,
    pub columns: usize,
    pub public_input_len: usize,
    pub polynomial: SparsePoly<F>,
    pub compiler_audit: SelectiveCompilerAudit,
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

pub(super) fn selective_polynomial() -> SparsePoly<F> {
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
        term(-F::ONE, &[(EVAL_SELECTOR, 1), (C, 1)]),
    ];
    for &(left, right) in &EVAL_PAIRS {
        terms.push(term(F::ONE, &[(EVAL_SELECTOR, 1), (left, 1), (right, 1)]));
    }

    // Exact shifted-base-3 transition over
    // d,h in {-1,0,1}, b in {0,1}:
    //
    //   b' = [d + 1 + b > h + 1].
    //
    // This is the Lagrange interpolation of the 18-point transition table.
    // GENERAL_SELECTOR isolates these rows from the evaluation ports reused
    // below. Its degree is six, within the existing degree-eight relation.
    let half = F::from_u64(2).inverse();
    let quarter = half * half;
    let transition = [
        (half, vec![(CANON_BOUND_DIGIT, 1)]),
        (F::ONE, vec![(CANON_NEXT_BORROW, 1)]),
        (-F::ONE, vec![(CANON_BORROW, 1)]),
        (-half, vec![(CANON_DIGIT, 1)]),
        (-half, vec![(CANON_BOUND_DIGIT, 2)]),
        (quarter, vec![(CANON_DIGIT, 1), (CANON_BOUND_DIGIT, 1)]),
        (-half, vec![(CANON_DIGIT, 2)]),
        (F::ONE, vec![(CANON_BORROW, 1), (CANON_BOUND_DIGIT, 2)]),
        (quarter, vec![(CANON_DIGIT, 1), (CANON_BOUND_DIGIT, 2)]),
        (-half, vec![(CANON_DIGIT, 1), (CANON_BORROW, 1), (CANON_BOUND_DIGIT, 1)]),
        (-quarter, vec![(CANON_DIGIT, 2), (CANON_BOUND_DIGIT, 1)]),
        (F::ONE, vec![(CANON_DIGIT, 2), (CANON_BORROW, 1)]),
        (F::from_u64(3) * quarter, vec![(CANON_DIGIT, 2), (CANON_BOUND_DIGIT, 2)]),
        (
            -F::from_u64(3) * half,
            vec![(CANON_DIGIT, 2), (CANON_BORROW, 1), (CANON_BOUND_DIGIT, 2)],
        ),
    ];
    for (coefficient, mut powers) in transition {
        powers.push((GENERAL_SELECTOR, 1));
        terms.push(term(coefficient, &powers));
    }
    SparsePoly::new(SELECTIVE_ARITY, terms)
}
