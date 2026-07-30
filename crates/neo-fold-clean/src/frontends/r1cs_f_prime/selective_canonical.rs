//! Shifted-ternary canonicality rows for the selective CCS image.
//!
//! Owns: paired borrow-chain rows against the Goldilocks canonical bound.
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
//! | Centered digits | outer Split-NC channel | no | Complete low-norm assignment |
//! | Canonical bound | [`emit_shifted_ternary_trace_rows`] | yes | Goldilocks modulus digits |

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::ShiftedTernaryCanonicalTrace;

use super::emit::append_field;
use super::terms::MatrixTerms;
use super::{
    LinearDefinitions, LowNormR1csError, A, BALANCED_FIELD_WIDTH, BIT, C, CANON_CHUNK_CLASS_SELECTORS,
    CANON_CHUNK_COUNT, CANON_CHUNK_WIDTH, CENTERED_UNIT, GENERAL_SELECTOR, SBOX_INPUT,
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
    for chunk in 0..CANON_CHUNK_COUNT {
        let digit_index = chunk * CANON_CHUNK_WIDTH;
        let first_bound = bound % 3;
        bound /= 3;
        let second_bound = if digit_index + 1 < BALANCED_FIELD_WIDTH {
            let value = bound % 3;
            bound /= 3;
            value
        } else {
            0
        };
        let chunk_bound = first_bound + 3 * second_bound;
        let complemented = chunk_bound > 4;
        let normalized_bound = if complemented { 8 - chunk_bound } else { chunk_bound } as usize;
        let scale = if complemented { -F::ONE } else { F::ONE };
        matrix_terms[GENERAL_SELECTOR].push((*row_cursor, selector, F::ONE));
        matrix_terms[CANON_CHUNK_CLASS_SELECTORS[normalized_bound]].push((*row_cursor, selector, F::ONE));

        append_field(
            &mut matrix_terms[CENTERED_UNIT],
            *row_cursor,
            trace.digit_columns_start + digit_index,
            scale,
            slots,
            definitions,
        )?;
        if digit_index + 1 < BALANCED_FIELD_WIDTH {
            append_field(
                &mut matrix_terms[A],
                *row_cursor,
                trace.digit_columns_start + digit_index + 1,
                scale,
                slots,
                definitions,
            )?;
        } else {
            matrix_terms[A].push((*row_cursor, 0, -scale));
        }

        if chunk != 0 {
            append_field(
                &mut matrix_terms[BIT],
                *row_cursor,
                trace.borrow_columns_start + digit_index - 1,
                scale,
                slots,
                definitions,
            )?;
        }
        if complemented {
            matrix_terms[BIT].push((*row_cursor, 0, F::ONE));
        }

        if chunk + 1 != CANON_CHUNK_COUNT {
            let output = trace.borrow_columns_start + digit_index + 1;
            append_field(&mut matrix_terms[C], *row_cursor, output, scale, slots, definitions)?;
            append_field(
                &mut matrix_terms[SBOX_INPUT],
                *row_cursor,
                output,
                scale,
                slots,
                definitions,
            )?;
        }
        if complemented {
            matrix_terms[C].push((*row_cursor, 0, F::ONE));
            matrix_terms[SBOX_INPUT].push((*row_cursor, 0, F::ONE));
        }
        *row_cursor += 1;
    }
    debug_assert_eq!(bound, 0, "41 base-3 digits must cover p-1");
    Ok(())
}
