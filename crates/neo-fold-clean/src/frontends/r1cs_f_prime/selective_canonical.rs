//! Shifted-ternary canonicality rows for the selective CCS image.
//!
//! Owns: per-digit centered-unit rows and borrow-chain rows against the
//! Goldilocks canonical bound.
//!
//! Does not own: witness digit generation, slot planning, selector allocation,
//! or the surrounding selective relation.
//!
//! Emits constraints: yes, by appending terms to the selective CCS matrices.
//!
//! Authority boundary: the parent must bind selector and source field slots;
//! these rows only prove canonicality of that retained encoding.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Centered digits | [`emit_shifted_ternary_trace_rows`] | yes | Bound retained digit slots |
//! | Canonical bound | borrow-chain portion of the same emitter | yes | Goldilocks modulus digits |

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::ShiftedTernaryCanonicalTrace;

use super::emit::append_field;
use super::terms::MatrixTerms;
use super::{
    LinearDefinitions, LowNormR1csError, BALANCED_FIELD_WIDTH, CANON_BORROW, CANON_BOUND_DIGIT, CANON_DIGIT,
    CANON_NEXT_BORROW, CENTERED_UNIT, GENERAL_SELECTOR,
};
use neo_math::F;

pub(super) fn emit_shifted_ternary_trace_rows(
    trace: &ShiftedTernaryCanonicalTrace,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
    selector: usize,
    matrix_terms: &mut [MatrixTerms],
    row_cursor: &mut usize,
) -> Result<(), LowNormR1csError> {
    let mut bound = F::ORDER_U64 - 1;
    for index in 0..BALANCED_FIELD_WIDTH {
        let digit = trace.digit_columns_start + index;

        matrix_terms[GENERAL_SELECTOR].push((*row_cursor, selector, F::ONE));
        append_field(
            &mut matrix_terms[CENTERED_UNIT],
            *row_cursor,
            digit,
            F::ONE,
            slots,
            definitions,
        )?;
        *row_cursor += 1;

        matrix_terms[GENERAL_SELECTOR].push((*row_cursor, selector, F::ONE));
        append_field(
            &mut matrix_terms[CANON_DIGIT],
            *row_cursor,
            digit,
            F::ONE,
            slots,
            definitions,
        )?;
        if index != 0 {
            append_field(
                &mut matrix_terms[CANON_BORROW],
                *row_cursor,
                trace.borrow_columns_start + index - 1,
                F::ONE,
                slots,
                definitions,
            )?;
        }
        if index + 1 != BALANCED_FIELD_WIDTH {
            append_field(
                &mut matrix_terms[CANON_NEXT_BORROW],
                *row_cursor,
                trace.borrow_columns_start + index,
                F::ONE,
                slots,
                definitions,
            )?;
        }
        let centered_bound = match bound % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("base-3 digit"),
        };
        bound /= 3;
        if centered_bound != F::ZERO {
            matrix_terms[CANON_BOUND_DIGIT].push((*row_cursor, 0, centered_bound));
        }
        *row_cursor += 1;
    }
    debug_assert_eq!(bound, 0, "41 base-3 digits must cover p-1");
    Ok(())
}
