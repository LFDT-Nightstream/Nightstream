//! Linear-form emission for selective low-norm slots.
//!
//! Owns: recursive substitution of validated linear definitions and expansion of
//! retained low-norm slots into CCS matrix terms.
//!
//! Does not own: slot allocation, definition discovery, row-family semantics, or
//! witness encoding.
//!
//! Emits constraints: yes, by appending coefficients to caller-owned matrix
//! terms.
//!
//! Authority boundary: slot maps and definitions are parent-validated inputs;
//! references to an unencoded temporary are rejected rather than trusted.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Linear combination | [`append_lc`] | yes | Source LC and validated definitions |
//! | Field substitution | [`append_field`] | yes | Retained slot map |
//! | Slot expansion | [`append_slot`] | yes | Fixed binary or balanced radix |

use neo_ccs::GeometricRowRun;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::terms::MatrixTerms;
use super::{Lc, LinearDefinitions, LowNormR1csError, BALANCED_FIELD_WIDTH};

pub(super) fn append_lc(
    terms: &mut MatrixTerms,
    row: usize,
    lc: &Lc,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
) -> Result<(), LowNormR1csError> {
    append_lc_scaled(terms, row, lc, F::ONE, slots, definitions)
}

pub(super) fn append_lc_scaled(
    terms: &mut MatrixTerms,
    row: usize,
    lc: &Lc,
    scale: F,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
) -> Result<(), LowNormR1csError> {
    if lc.constant != F::ZERO {
        terms.push((row, 0, lc.constant * scale));
    }
    for &(field_col, coefficient) in &lc.terms {
        append_field(terms, row, field_col, coefficient * scale, slots, definitions)?;
    }
    Ok(())
}

pub(super) fn append_field(
    terms: &mut MatrixTerms,
    row: usize,
    field_col: usize,
    coefficient: F,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
) -> Result<(), LowNormR1csError> {
    if coefficient == F::ZERO {
        return Ok(());
    }
    let mut stack = vec![(field_col, coefficient)];
    while let Some((column, scale)) = stack.pop() {
        if column == 0 {
            terms.push((row, 0, scale));
            continue;
        }
        if let Some(rhs) = definitions.get(column) {
            if rhs.constant != F::ZERO {
                terms.push((row, 0, rhs.constant * scale));
            }
            stack.extend(
                rhs.terms
                    .iter()
                    .map(|&(rhs_column, rhs_coefficient)| (rhs_column, rhs_coefficient * scale)),
            );
            continue;
        }
        let (start, width) =
            slots[column].ok_or_else(|| trace_error("retained row references an unencoded selective temporary"))?;
        append_slot(terms, row, (start, width), scale);
    }
    Ok(())
}

pub(super) fn append_slot(terms: &mut MatrixTerms, row: usize, slot: (usize, usize), coefficient: F) {
    let (start, width) = slot;
    let radix = if width == BALANCED_FIELD_WIDTH {
        F::from_u64(3)
    } else {
        F::from_u64(2)
    };
    if width > 1 {
        terms
            .geometric_runs
            .push(GeometricRowRun::new(row, start, width, coefficient, radix));
        return;
    }
    let mut power = coefficient;
    for bit in 0..width {
        terms.push((row, start + bit, power));
        power *= radix;
    }
}

pub(super) fn lc_from_column(column: usize) -> Lc {
    Lc {
        terms: vec![(column, F::ONE)],
        constant: F::ZERO,
    }
}

pub(super) fn trace_error(message: &str) -> LowNormR1csError {
    LowNormR1csError::SelectiveTrace(message.to_owned())
}
