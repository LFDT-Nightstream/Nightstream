//! Exact balanced-ternary opening refinement for gadget-native lowering.
//!
//! Owns: validation and column-role classification for one canonical
//! Goldilocks opening in 41 centered base-3 digits.
//!
//! Does not own: SIS matrix semantics, source-row emission, or assignment
//! materialization.
//!
//! Emits constraints: no. The parent lowering emits the replacement rows.
//!
//! Authority boundary: every composite trace is accepted only after all 124
//! source R1CS rows are reconstructed and compared exactly. The reduction
//! plan may then omit only the 41 support rows and one reconstruction source
//! row, plus 41 negative and 40 borrow coordinate-bitness gates. The
//! reconstruction is discharged by a checked structural field-to-digit slot
//! alias before emission.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---:|---|---|
//! | `shifted_ternary.centered` | `d^3-d=0` fixes `d` to `{-1,0,1}` | yes, in parent lowering | `gadget_native.rs` | `ShiftedTernaryReducedCore.CenteredUnitGateHolds` |
//! | `shifted_ternary.definition` | `d(d-1)=2n` defines the negative indicator | yes, retained source row | `gadget_native.rs` | `ShiftedTernaryReducedCore.Accepts.negativeDefinition` |
//! | `shifted_ternary.transition` | shifted base-3 borrow transition below `p` | yes, retained source row | `gadget_native.rs` | `ShiftedTernaryReducedCore.Accepts.borrowTransition` |
//! | `shifted_ternary.omitted` | bitness, support, and reconstruction follow from the retained core plus the structural alias | no | `balanced_ternary.rs` | `ShiftedTernaryReducedCore.reduced_iff_canonicalRows` |

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{BalancedTernaryOpeningTraceEntry, Lc, R1csEncodingTrace, R1csSnapshot, Var};

use super::{validate_row, GadgetNativeError};

const BORROW_DIGITS: usize = BALANCED_TERNARY_DIGITS - 1;
const DIGIT_SOURCE_ROWS: usize = 2 * BALANCED_TERNARY_DIGITS;
pub(super) const RETAINED_SOURCE_ROWS_PER_OPENING: usize = 2 * BALANCED_TERNARY_DIGITS;
pub(super) const OMITTED_SOURCE_ROWS_PER_OPENING: usize = BALANCED_TERNARY_DIGITS + 1;
pub(super) const RETAINED_GATES_PER_OPENING: usize = 3 * BALANCED_TERNARY_DIGITS;
pub(super) const OMITTED_GATES_PER_OPENING: usize = 3 * BALANCED_TERNARY_DIGITS;

/// Validated source-column roles for all traced balanced-ternary openings.
pub(super) struct ValidatedBalancedTernary {
    field_opening: Vec<Option<usize>>,
    digit_alias: Vec<Option<(usize, usize)>>,
    binary_columns: Vec<bool>,
    structural_columns: Vec<bool>,
    retained_source_rows: Vec<bool>,
    omitted_source_rows: Vec<bool>,
}

impl ValidatedBalancedTernary {
    pub(super) fn validate(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        covered_rows: &[bool],
    ) -> Result<Self, GadgetNativeError> {
        let mut field_opening = vec![None; source.cols()];
        let mut digit_alias = vec![None; source.cols()];
        let mut binary_columns = vec![false; source.cols()];
        let mut structural_columns = vec![false; source.cols()];
        let mut owned_columns = vec![false; source.cols()];
        let mut owned_rows = vec![false; source.rows()];
        let mut retained_source_rows = vec![false; source.rows()];
        let mut omitted_source_rows = vec![false; source.rows()];

        for (opening_index, opening) in trace.balanced_ternary_openings().iter().enumerate() {
            validate_geometry(source, opening_index, opening, &mut owned_columns)?;
            validate_rows(source, opening_index, opening)?;

            for row in opening.digit_rows.start..opening.transition_rows.end {
                if std::mem::replace(&mut owned_rows[row], true) {
                    return Err(geometry(opening_index, "overlapping opening row schedule"));
                }
            }
            for digit in 0..BALANCED_TERNARY_DIGITS {
                retained_source_rows[opening.digit_rows.start + 2 * digit] = true;
                omitted_source_rows[opening.digit_rows.start + 2 * digit + 1] = true;
                retained_source_rows[opening.transition_rows.start + digit] = true;
            }
            omitted_source_rows[opening.reconstruction_row] = true;

            field_opening[opening.field_col] = Some(opening_index);
            structural_columns[opening.field_col] = true;
            for (digit, &column) in opening.digit_cols.iter().enumerate() {
                digit_alias[column] = Some((opening.field_col, digit));
                structural_columns[column] = true;
            }
            for &column in &opening.borrow_cols {
                binary_columns[column] = true;
                structural_columns[column] = true;
            }
            for &column in &opening.negative_cols {
                binary_columns[column] = true;
                structural_columns[column] = true;
            }
            let rows = opening
                .digit_rows
                .clone()
                .chain(std::iter::once(opening.reconstruction_row))
                .chain(opening.transition_rows.clone());
            if rows.into_iter().any(|row| covered_rows[row]) {
                return Err(geometry(opening_index, "row overlap with another replacement"));
            }
        }

        Ok(Self {
            field_opening,
            digit_alias,
            binary_columns,
            structural_columns,
            retained_source_rows,
            omitted_source_rows,
        })
    }

    pub(super) fn opening_for_field(&self, column: usize) -> Option<usize> {
        self.field_opening[column]
    }

    pub(super) fn digit_alias(&self, column: usize) -> Option<(usize, usize)> {
        self.digit_alias[column]
    }

    pub(super) fn is_binary(&self, column: usize) -> bool {
        self.binary_columns[column]
    }

    pub(super) fn is_structural(&self, column: usize) -> bool {
        self.structural_columns[column]
    }

    pub(super) fn retained_source_rows(&self) -> &[bool] {
        &self.retained_source_rows
    }

    pub(super) fn omitted_source_rows(&self) -> &[bool] {
        &self.omitted_source_rows
    }

    pub(super) fn reject_public_columns(&self, is_public: &[bool]) -> Result<(), GadgetNativeError> {
        if let Some(column) = (1..is_public.len()).find(|&column| is_public[column] && self.is_structural(column)) {
            return Err(GadgetNativeError::PublicBalancedTernaryColumn { column });
        }
        Ok(())
    }

    pub(super) fn reduction_removed_rows(
        &self,
        other_removed_rows: &[bool],
        redundant_boolean_rows: &[bool],
    ) -> Result<Vec<bool>, GadgetNativeError> {
        if other_removed_rows.len() != self.omitted_source_rows.len()
            || redundant_boolean_rows.len() != self.omitted_source_rows.len()
        {
            return Err(geometry(0, "reduction mask width"));
        }
        let mut removed = other_removed_rows.to_vec();
        for row in 0..removed.len() {
            let owned = self.retained_source_rows[row] || self.omitted_source_rows[row];
            if owned && (removed[row] || redundant_boolean_rows[row]) {
                return Err(geometry(0, "planned reduction ownership overlap"));
            }
            removed[row] |= self.omitted_source_rows[row];
        }
        Ok(removed)
    }
}

fn validate_geometry(
    source: &R1csSnapshot,
    opening_index: usize,
    opening: &BalancedTernaryOpeningTraceEntry,
    owned_columns: &mut [bool],
) -> Result<(), GadgetNativeError> {
    if opening.digit_rows.len() != DIGIT_SOURCE_ROWS
        || opening.transition_rows.len() != BALANCED_TERNARY_DIGITS
        || opening.digit_rows.end != opening.reconstruction_row
        || opening.reconstruction_row.checked_add(1) != Some(opening.transition_rows.start)
        || opening.transition_rows.end > source.rows()
    {
        return Err(geometry(opening_index, "row schedule"));
    }
    if opening.field_col == Var::ONE.col() || opening.field_col >= source.cols() {
        return Err(geometry(opening_index, "field column"));
    }

    let mut claim_column = |column: usize, role: &'static str| {
        if column == Var::ONE.col() || column >= source.cols() || std::mem::replace(&mut owned_columns[column], true) {
            return Err(geometry(opening_index, role));
        }
        Ok(())
    };
    claim_column(opening.field_col, "duplicate field column")?;
    for &column in &opening.digit_cols {
        if column <= opening.field_col {
            return Err(geometry(opening_index, "non-topological digit column"));
        }
        claim_column(column, "duplicate digit column")?;
    }
    for (&digit, &negative) in opening.digit_cols.iter().zip(&opening.negative_cols) {
        if negative <= digit {
            return Err(geometry(opening_index, "non-topological negative column"));
        }
        claim_column(negative, "duplicate negative column")?;
    }
    for &column in &opening.borrow_cols {
        if column <= opening.field_col {
            return Err(geometry(opening_index, "non-topological borrow column"));
        }
        claim_column(column, "duplicate borrow column")?;
    }
    Ok(())
}

fn validate_rows(
    source: &R1csSnapshot,
    opening_index: usize,
    opening: &BalancedTernaryOpeningTraceEntry,
) -> Result<(), GadgetNativeError> {
    let two = F::from_u64(2);
    for digit in 0..BALANCED_TERNARY_DIGITS {
        let digit_var = var(opening.digit_cols[digit]);
        let negative_var = var(opening.negative_cols[digit]);
        let digit_lc = Lc::from_var(digit_var);
        let minus_one = digit_lc
            .clone()
            .add_scaled(&Lc::from_const(F::ONE), -F::ONE);
        let twice_negative = Lc::zero().add_scaled(&Lc::from_var(negative_var), two);
        validate_row(
            source,
            "balanced-ternary digit alphabet",
            opening.digit_rows.start + 2 * digit,
            &digit_lc,
            &minus_one,
            &twice_negative,
        )?;
        let plus_one = digit_lc.add_scaled(&Lc::from_const(F::ONE), F::ONE);
        validate_row(
            source,
            "balanced-ternary digit alphabet",
            opening.digit_rows.start + 2 * digit + 1,
            &Lc::from_var(negative_var),
            &plus_one,
            &Lc::zero(),
        )?;
    }

    let mut reconstruction = Lc::from_var(var(opening.field_col));
    let mut power = F::ONE;
    for &digit in &opening.digit_cols {
        reconstruction.add_term(var(digit), -power);
        power *= F::from_u64(3);
    }
    validate_row(
        source,
        "balanced-ternary reconstruction",
        opening.reconstruction_row,
        &reconstruction,
        &Lc::from_var(Var::ONE),
        &Lc::zero(),
    )?;

    let mut bound = F::ORDER_U64 - 1;
    for digit in 0..BALANCED_TERNARY_DIGITS {
        let bound_digit = bound % 3;
        bound /= 3;
        let digit_lc = Lc::from_var(var(opening.digit_cols[digit]));
        let negative = Lc::from_var(var(opening.negative_cols[digit]));
        let borrow = if digit == 0 {
            Lc::zero()
        } else {
            Lc::from_var(var(opening.borrow_cols[digit - 1]))
        };
        let next_borrow = if digit == BORROW_DIGITS {
            Lc::zero()
        } else {
            Lc::from_var(var(opening.borrow_cols[digit]))
        };
        let positive = digit_lc.clone().add_scaled(&negative, F::ONE);
        let zero = Lc::from_const(F::ONE)
            .add_scaled(&digit_lc, -F::ONE)
            .add_scaled(&negative, -two);
        let (left, right, out) = match bound_digit {
            0 => (
                negative,
                Lc::from_const(F::ONE).add_scaled(&borrow, -F::ONE),
                Lc::from_const(F::ONE).add_scaled(&next_borrow, -F::ONE),
            ),
            1 => (zero, borrow, next_borrow.add_scaled(&positive, -F::ONE)),
            2 => (positive, borrow, next_borrow),
            _ => unreachable!("base-3 digit"),
        };
        validate_row(
            source,
            "balanced-ternary canonicality",
            opening.transition_rows.start + digit,
            &left,
            &right,
            &out,
        )?;
    }
    if bound != 0 {
        return Err(geometry(opening_index, "Goldilocks bound width"));
    }
    Ok(())
}

fn geometry(opening: usize, detail: &'static str) -> GadgetNativeError {
    GadgetNativeError::BalancedTernaryGeometry { opening, detail }
}

fn var(column: usize) -> Var {
    // This module has already range-checked every column. `Var` deliberately
    // has no public constructor, so use an LC round trip through this narrow
    // crate-owned helper exposed by the builder.
    Var::from_column_for_trace(column)
}
