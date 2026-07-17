//! Exact duplicate source-Boolean row classification.
//!
//! Owns: structural recognition of normalized source bit rows and validation
//! that their source column decodes from one encoded Boolean coordinate.
//!
//! Does not own: Boolean column discovery, slot allocation, or common gate
//! emission.
//!
//! Emits constraints: no. It may authorize omission of one generic fallback
//! row already enforced by the common encoded-coordinate bitness gate.
//!
//! Authority boundary: source-row structure and the concrete source-to-slot
//! map are both required. Boolean metadata alone never authorizes omission.
//!
//! | Check | Exact condition | Failure behavior | Lean theorem |
//! |---|---|---|---|
//! | Source row | `v * (v - 1) = 0`, allowing only A/B exchange | retain | `BooleanRowDedup.substituted_bitRow_eq_slot_bitRow` |
//! | Source map | `v` expands to exactly `[(slot, 1)]` with a Boolean slot | retain | singleton substitution hypothesis |
//! | Existing owner | row is not already claimed by a traced replacement | retain | disjoint ownership |

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{R1csSnapshot, Var};

use super::slots::{slot_terms, ValueEncoding};
use super::{GadgetNativeError, SourceColumn, TraceMarks};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ExactBooleanRows {
    redundant: Vec<bool>,
}

impl ExactBooleanRows {
    /// Estimator/profile classification from the exact source relation and
    /// deterministic slot-allocation facts.
    pub(super) fn from_plan(
        source: &R1csSnapshot,
        is_public: &[bool],
        explicit_bits: &[bool],
        linearly_derived: &[bool],
        marks: &TraceMarks,
    ) -> Self {
        let singleton = planned_singleton_columns(source, is_public, explicit_bits, linearly_derived, marks);
        Self::from_planned_columns(source, &singleton, &marks.covered_rows)
    }

    /// Estimator-side classification from the exact source relation and the
    /// deterministic slot plan. Production materialization checks this result
    /// against the concrete source-column map before omitting any row.
    pub(super) fn from_planned_columns(
        source: &R1csSnapshot,
        singleton_boolean_columns: &[bool],
        covered_rows: &[bool],
    ) -> Self {
        debug_assert_eq!(singleton_boolean_columns.len(), source.cols());
        debug_assert_eq!(covered_rows.len(), source.rows());
        Self {
            redundant: (0..source.rows())
                .map(|row| {
                    !covered_rows[row]
                        && exact_bit_row_column(source, row).is_some_and(|column| singleton_boolean_columns[column])
                })
                .collect(),
        }
    }

    /// Materializer-side classification. This reads the concrete decoded
    /// column map and therefore proves that the common bitness gate constrains
    /// the same singleton slot as the exact source row.
    pub(super) fn from_encoded_columns(
        source: &R1csSnapshot,
        source_columns: &[SourceColumn],
        covered_rows: &[bool],
    ) -> Self {
        debug_assert_eq!(source_columns.len(), source.cols());
        debug_assert_eq!(covered_rows.len(), source.rows());
        Self {
            redundant: (0..source.rows())
                .map(|row| {
                    !covered_rows[row]
                        && exact_bit_row_column(source, row)
                            .is_some_and(|column| is_singleton_boolean_slot(&source_columns[column]))
                })
                .collect(),
        }
    }

    pub(super) fn rows(&self) -> &[bool] {
        &self.redundant
    }

    pub(super) fn count(&self) -> usize {
        self.redundant
            .iter()
            .filter(|&&redundant| redundant)
            .count()
    }

    pub(super) fn first_mismatch(&self, other: &Self) -> Option<usize> {
        self.redundant
            .iter()
            .zip(&other.redundant)
            .position(|(planned, concrete)| planned != concrete)
            .or_else(|| (self.redundant.len() != other.redundant.len()).then_some(0))
    }

    pub(super) fn first_overlap(&self, other_rows: &[bool]) -> Option<usize> {
        self.redundant
            .iter()
            .zip(other_rows)
            .position(|(&redundant, &other)| redundant && other)
    }

    /// Fail closed if another source-row replacement also claims a row that
    /// this classifier would omit.
    pub(super) fn require_disjoint(&self, other_rows: &[bool]) -> Result<(), GadgetNativeError> {
        if let Some(row) = self.first_overlap(other_rows) {
            return Err(GadgetNativeError::BooleanDedupOwnershipOverlap { row });
        }
        Ok(())
    }

    /// Recheck the estimator's plan against the concrete decoded-column map
    /// before production materialization omits a row.
    pub(super) fn checked_concrete(
        source: &R1csSnapshot,
        source_columns: &[SourceColumn],
        covered_rows: &[bool],
        planned: &Self,
        other_removed_rows: &[bool],
    ) -> Result<Self, GadgetNativeError> {
        let concrete = Self::from_encoded_columns(source, source_columns, covered_rows);
        if let Some(row) = planned.first_mismatch(&concrete) {
            return Err(GadgetNativeError::BooleanDedupPlanMismatch { row });
        }
        concrete.require_disjoint(other_removed_rows)?;
        Ok(concrete)
    }

    pub(super) fn retained_fallback_count(&self, covered_rows: &[bool], other_removed_rows: &[bool]) -> usize {
        self.redundant
            .iter()
            .enumerate()
            .filter(|&(row, &redundant)| !covered_rows[row] && !other_removed_rows[row] && !redundant)
            .count()
    }
}

/// Validate the caller-selected public Boolean columns against source
/// relation evidence. The returned masks drive deterministic singleton-slot
/// planning; no caller metadata is trusted without this source check.
pub(super) fn validate_public_columns(
    source: &R1csSnapshot,
    public_bit_columns: &[usize],
) -> Result<(Vec<bool>, Vec<bool>), GadgetNativeError> {
    let explicit_bits = source.explicitly_boolean_columns();
    let mut is_public = vec![false; source.cols()];
    for &column in public_bit_columns {
        if column == 0 || column >= source.cols() {
            return Err(GadgetNativeError::PublicColumnOutOfRange {
                column,
                cols: source.cols(),
            });
        }
        if is_public[column] {
            return Err(GadgetNativeError::DuplicatePublicColumn { column });
        }
        if !explicit_bits[column] {
            return Err(GadgetNativeError::PublicColumnNotBoolean { column });
        }
        is_public[column] = true;
    }
    Ok((is_public, explicit_bits))
}

/// Source columns whose deterministic allocation path is exactly one Boolean
/// coordinate. This is only a planning fact;
/// [`ExactBooleanRows::checked_concrete`] validates the materialized
/// source-to-slot map before omission.
fn planned_singleton_columns(
    source: &R1csSnapshot,
    is_public: &[bool],
    explicit_bits: &[bool],
    linearly_derived: &[bool],
    marks: &TraceMarks,
) -> Vec<bool> {
    let mut singleton = vec![false; source.cols()];
    for column in 1..source.cols() {
        if marks.gadget_columns[column]
            || linearly_derived[column]
            || marks.balanced_ternary.digit_alias(column).is_some()
            || marks.balanced_ternary.opening_for_field(column).is_some()
        {
            continue;
        }
        singleton[column] = is_public[column] || marks.balanced_ternary.is_binary(column) || explicit_bits[column];
    }
    singleton
}

pub(super) fn exact_bit_row_column(source: &R1csSnapshot, row: usize) -> Option<usize> {
    if !source.c_row(row).is_empty() {
        return None;
    }
    exact_bit_sides(source.a_row(row), source.b_row(row))
        .or_else(|| exact_bit_sides(source.b_row(row), source.a_row(row)))
}

fn exact_bit_sides(variable: &[(usize, F)], variable_minus_one: &[(usize, F)]) -> Option<usize> {
    let [(column, coefficient)] = variable else {
        return None;
    };
    if *column == Var::ONE.col() || *coefficient != F::ONE {
        return None;
    }
    (variable_minus_one == [(Var::ONE.col(), -F::ONE), (*column, F::ONE)]).then_some(*column)
}

fn is_singleton_boolean_slot(source_column: &SourceColumn) -> bool {
    let SourceColumn::Encoded(slot) = source_column else {
        return false;
    };
    slot.encoding == ValueEncoding::Boolean && slot.width == 1 && slot_terms(*slot) == [(slot.start, F::ONE)]
}
