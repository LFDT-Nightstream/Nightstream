//! Auditable balanced-ternary source-to-CCS layout.
//!
//! Owns: the exact shared-slot aliases and retained source-to-encoded row
//! mapping produced by the parent gadget-native lowering.
//!
//! Does not own: balanced-ternary semantics, source-row validation, or any
//! row-removal argument.
//!
//! Emits constraints: no. It records and reads the rows emitted by the parent.
//!
//! Authority boundary: source R1CS rows remain the local implementation
//! arithmetic reference. A reduction plan is created only after all 124 rows,
//! the concrete shared-slot aliases, every retained product row, and every
//! retained centered-unit residual pair/tail agree. Paper-level necessity is a
//! separate theorem obligation.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---:|---|---|
//! | `shifted_ternary.shared.field` | `x = sum 3^i d_i` uses the same target slots as the 41 digits | no | `build_plan` | `SharedSlots.production_decoded_sharedAlias` |
//! | `shifted_ternary.shared.aliases` | digit, negative, and borrow source wires use exact singleton target slots | no | `build_plan` | `SharedSlots.productionLayout` |
//! | `shifted_ternary.shared.rows.retained` | 41 definitions and 41 transitions remain as one product gate each | no | `ReductionPlan::{before_emit,after_emit,finish}` | `SharedSlots.satisfies_retainedProductRows_iff` |
//! | `shifted_ternary.shared.rows.omitted` | 41 support and one reconstruction source row are omitted only after exact validation | no | `ReductionPlan::checked` | `SharedSlots.productionAccepts_iff_canonicalRows` |

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_ccs::{CcsMatrix, CscMat, SparsePoly};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{BalancedTernaryOpeningTraceEntry, R1csEncodingTrace};

use super::balanced_ternary::{
    ValidatedBalancedTernary, OMITTED_GATES_PER_OPENING, OMITTED_SOURCE_ROWS_PER_OPENING, RETAINED_GATES_PER_OPENING,
    RETAINED_SOURCE_ROWS_PER_OPENING,
};
use super::coordinate_gates::{
    GadgetNativeCenteredFamily, GadgetNativeCoordinateGateSchedule, GadgetNativeCoordinateRowAudit,
};
use super::slots::slot_terms;
use super::{gate, EncodedGadgetNativeR1cs, GadgetNativeError, GadgetNativePlan, SourceColumn};

const SOURCE_ROWS: usize = 3 * BALANCED_TERNARY_DIGITS + 1;

/// Exact production layout for one retained balanced-ternary opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BalancedTernarySharedSlotPlan {
    source_field_column: usize,
    source_digit_columns: [usize; BALANCED_TERNARY_DIGITS],
    source_negative_columns: [usize; BALANCED_TERNARY_DIGITS],
    source_borrow_columns: [usize; BALANCED_TERNARY_DIGITS - 1],
    one_column: usize,
    digit_columns: [usize; BALANCED_TERNARY_DIGITS],
    negative_columns: [usize; BALANCED_TERNARY_DIGITS],
    borrow_columns: [usize; BALANCED_TERNARY_DIGITS - 1],
    field_terms: Vec<(usize, F)>,
    source_rows: Range<usize>,
    retained_source_rows: Vec<usize>,
    retained_encoded_rows: Vec<usize>,
    centered_unit_rows: [usize; BALANCED_TERNARY_DIGITS],
}

impl BalancedTernarySharedSlotPlan {
    pub fn source_field_column(&self) -> usize {
        self.source_field_column
    }

    pub fn source_digit_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS] {
        &self.source_digit_columns
    }

    pub fn source_negative_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS] {
        &self.source_negative_columns
    }

    pub fn source_borrow_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS - 1] {
        &self.source_borrow_columns
    }

    pub fn one_column(&self) -> usize {
        self.one_column
    }

    pub fn digit_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS] {
        &self.digit_columns
    }

    pub fn negative_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS] {
        &self.negative_columns
    }

    pub fn borrow_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS - 1] {
        &self.borrow_columns
    }

    pub fn field_terms(&self) -> &[(usize, F)] {
        &self.field_terms
    }

    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn retained_source_rows(&self) -> &[usize] {
        &self.retained_source_rows
    }

    pub fn retained_encoded_rows(&self) -> &[usize] {
        &self.retained_encoded_rows
    }

    pub fn encoded_row_for_source(&self, source_row: usize) -> Option<usize> {
        self.retained_source_rows
            .iter()
            .position(|&candidate| candidate == source_row)
            .map(|index| self.retained_encoded_rows[index])
    }

    pub fn indicator_definition_source_rows(&self) -> Vec<usize> {
        (0..BALANCED_TERNARY_DIGITS)
            .map(|digit| self.source_rows.start + 2 * digit)
            .collect()
    }

    pub fn indicator_support_source_rows(&self) -> Vec<usize> {
        (0..BALANCED_TERNARY_DIGITS)
            .map(|digit| self.source_rows.start + 2 * digit + 1)
            .collect()
    }

    pub fn reconstruction_source_row(&self) -> usize {
        self.source_rows.start + 2 * BALANCED_TERNARY_DIGITS
    }

    pub fn transition_source_rows(&self) -> Range<usize> {
        self.reconstruction_source_row() + 1..self.source_rows.end
    }

    pub fn centered_unit_rows(&self) -> Vec<usize> {
        self.centered_unit_rows.to_vec()
    }

    pub fn omitted_negative_bitness_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS] {
        &self.negative_columns
    }

    pub fn omitted_borrow_bitness_columns(&self) -> &[usize; BALANCED_TERNARY_DIGITS - 1] {
        &self.borrow_columns
    }

    pub fn retained_obligation_count(&self) -> usize {
        self.centered_unit_rows.len() + self.retained_encoded_rows.len()
    }

    pub fn retained_physical_row_count(&self) -> usize {
        self.centered_unit_rows
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            + self.retained_encoded_rows.len()
    }

    pub fn omitted_obligation_count(&self) -> usize {
        self.negative_columns.len() + self.borrow_columns.len() + self.indicator_support_source_rows().len() + 1
    }
}

/// One R1CS-shaped row read back from the actual gadget-native CCS matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeConstraintRow {
    a: Vec<(usize, F)>,
    b: Vec<(usize, F)>,
    c: Vec<(usize, F)>,
}

impl GadgetNativeConstraintRow {
    pub fn a(&self) -> &[(usize, F)] {
        &self.a
    }

    pub fn b(&self) -> &[(usize, F)] {
        &self.b
    }

    pub fn c(&self) -> &[(usize, F)] {
        &self.c
    }
}

impl GadgetNativePlan {
    pub fn balanced_ternary_openings(&self) -> &[BalancedTernarySharedSlotPlan] {
        &self.balanced_ternary_openings
    }
}

impl EncodedGadgetNativeR1cs {
    /// Read the exact retained rows back from the production CCS matrices.
    pub fn balanced_ternary_rows(&self, opening: usize) -> Result<Vec<GadgetNativeConstraintRow>, GadgetNativeError> {
        let plan = self
            .plan
            .balanced_ternary_openings
            .get(opening)
            .ok_or_else(|| geometry(opening, "missing production plan"))?;
        if self.structure.matrices.len() != gate::ARITY {
            return Err(geometry(opening, "CCS gate arity"));
        }
        validate_gate_polynomial(&self.structure.f, opening)?;
        validate_alphabet_rows(&self.structure.matrices, plan, &self.plan.coordinate_gates, opening)?;
        plan.retained_encoded_rows
            .iter()
            .map(|&row| extract_product_row(&self.structure.matrices, row, opening))
            .collect()
    }
}

fn validate_gate_polynomial(polynomial: &SparsePoly<F>, opening: usize) -> Result<(), GadgetNativeError> {
    if polynomial.arity() != gate::ARITY
        || polynomial
            .terms()
            .iter()
            .any(|term| term.exps.len() != gate::ARITY)
    {
        return Err(geometry(opening, "CCS gate polynomial arity"));
    }

    validate_polynomial_specialization(
        polynomial,
        &[gate::SELECTOR, gate::CENTERED_UNIT_TAIL],
        &[
            (F::ONE, &[(gate::SELECTOR, 1), (gate::CENTERED_UNIT_TAIL, 3)]),
            (-F::ONE, &[(gate::SELECTOR, 1), (gate::CENTERED_UNIT_TAIL, 1)]),
        ],
        opening,
        "centered-unit tail polynomial",
    )?;
    validate_polynomial_specialization(
        polynomial,
        &[gate::SELECTOR, gate::CENTERED_PAIR_LEFT, gate::CENTERED_PAIR_RIGHT],
        &[
            (F::ONE, &[(gate::SELECTOR, 1), (gate::CENTERED_PAIR_LEFT, 6)]),
            (-F::from_u64(2), &[(gate::SELECTOR, 1), (gate::CENTERED_PAIR_LEFT, 4)]),
            (F::ONE, &[(gate::SELECTOR, 1), (gate::CENTERED_PAIR_LEFT, 2)]),
            (-F::from_u64(7), &[(gate::SELECTOR, 1), (gate::CENTERED_PAIR_RIGHT, 6)]),
            (F::from_u64(14), &[(gate::SELECTOR, 1), (gate::CENTERED_PAIR_RIGHT, 4)]),
            (-F::from_u64(7), &[(gate::SELECTOR, 1), (gate::CENTERED_PAIR_RIGHT, 2)]),
        ],
        opening,
        "centered-unit residual-pair polynomial",
    )?;
    validate_polynomial_specialization(
        polynomial,
        &[
            gate::SELECTOR,
            gate::PRODUCT_LEFT,
            gate::PRODUCT_RIGHT,
            gate::PRODUCT_OUT,
        ],
        &[
            (
                F::ONE,
                &[(gate::SELECTOR, 1), (gate::PRODUCT_LEFT, 1), (gate::PRODUCT_RIGHT, 1)],
            ),
            (-F::ONE, &[(gate::SELECTOR, 1), (gate::PRODUCT_OUT, 1)]),
        ],
        opening,
        "single-product gate polynomial",
    )
}

fn validate_polynomial_specialization(
    polynomial: &SparsePoly<F>,
    active_variables: &[usize],
    expected: &[(F, &[(usize, u32)])],
    opening: usize,
    detail: &'static str,
) -> Result<(), GadgetNativeError> {
    let actual = polynomial
        .terms()
        .iter()
        .filter(|term| {
            term.coeff != F::ZERO
                && term
                    .exps
                    .iter()
                    .enumerate()
                    .all(|(variable, &power)| power == 0 || active_variables.contains(&variable))
        })
        .collect::<Vec<_>>();
    let exact_term = |term: &&neo_ccs::Term<F>, coefficient: F, powers: &[(usize, u32)]| {
        term.coeff == coefficient
            && term.exps.iter().enumerate().all(|(variable, &power)| {
                power
                    == powers
                        .iter()
                        .find_map(|&(expected_variable, expected_power)| {
                            (expected_variable == variable).then_some(expected_power)
                        })
                        .unwrap_or(0)
            })
    };
    if actual.len() != expected.len()
        || expected.iter().any(|&(coefficient, powers)| {
            !actual
                .iter()
                .any(|term| exact_term(term, coefficient, powers))
        })
    {
        return Err(geometry(opening, detail));
    }
    Ok(())
}

pub(super) struct ReductionPlan {
    openings: Vec<BalancedTernarySharedSlotPlan>,
    retained_rows: BTreeMap<usize, (usize, usize)>,
    retained_encoded_rows: Vec<Vec<Option<usize>>>,
    omitted_source_rows: Vec<bool>,
    omitted_coordinate_columns: Vec<bool>,
}

impl ReductionPlan {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn checked(
        trace: &R1csEncodingTrace,
        source_columns: &[SourceColumn],
        validated: &ValidatedBalancedTernary,
        other_removed_rows: &[bool],
        redundant_boolean_rows: &[bool],
        encoded_columns: usize,
    ) -> Result<Self, GadgetNativeError> {
        let source_row_count = validated.omitted_source_rows().len();
        if other_removed_rows.len() != source_row_count
            || redundant_boolean_rows.len() != source_row_count
            || validated.retained_source_rows().len() != source_row_count
        {
            return Err(geometry(0, "reduction mask width"));
        }

        let mut openings = Vec::with_capacity(trace.balanced_ternary_openings().len());
        let mut concrete_retained = vec![false; source_row_count];
        let mut concrete_omitted = vec![false; source_row_count];
        let mut omitted_coordinate_columns = vec![false; encoded_columns];

        for (opening_index, opening) in trace.balanced_ternary_openings().iter().enumerate() {
            let source_rows = source_range(opening);
            if source_rows.len() != SOURCE_ROWS {
                return Err(geometry(opening_index, "reduction source row count"));
            }
            let retained_source_rows = retained_source_rows(opening);
            let omitted_source_rows = omitted_source_rows(opening);
            if retained_source_rows.len() != RETAINED_SOURCE_ROWS_PER_OPENING
                || omitted_source_rows.len() != OMITTED_SOURCE_ROWS_PER_OPENING
            {
                return Err(geometry(opening_index, "reduction family count"));
            }
            for &row in &retained_source_rows {
                if std::mem::replace(&mut concrete_retained[row], true) {
                    return Err(geometry(opening_index, "overlapping retained row family"));
                }
            }
            for &row in &omitted_source_rows {
                if std::mem::replace(&mut concrete_omitted[row], true) {
                    return Err(geometry(opening_index, "overlapping omitted row family"));
                }
            }

            let mut plan = build_plan(
                opening,
                source_columns,
                source_rows,
                retained_source_rows,
                Vec::new(),
                [0; BALANCED_TERNARY_DIGITS],
                opening_index,
            )?;
            for &column in plan.negative_columns.iter().chain(&plan.borrow_columns) {
                let Some(omitted) = omitted_coordinate_columns.get_mut(column) else {
                    return Err(geometry(opening_index, "omitted coordinate column"));
                };
                if std::mem::replace(omitted, true) {
                    return Err(geometry(opening_index, "overlapping omitted coordinate"));
                }
            }
            if plan.omitted_obligation_count() != OMITTED_GATES_PER_OPENING {
                return Err(geometry(opening_index, "omitted gate count"));
            }
            plan.retained_encoded_rows = Vec::with_capacity(RETAINED_SOURCE_ROWS_PER_OPENING);
            openings.push(plan);
        }

        if concrete_retained != validated.retained_source_rows() || concrete_omitted != validated.omitted_source_rows()
        {
            return Err(geometry(0, "planned/materialized source-row disagreement"));
        }
        for row in 0..source_row_count {
            let owned = concrete_retained[row] || concrete_omitted[row];
            if owned && (other_removed_rows[row] || redundant_boolean_rows[row]) {
                return Err(geometry(
                    opening_for_row(trace, row).unwrap_or(0),
                    "reduction ownership overlap",
                ));
            }
        }

        let mut retained_rows = BTreeMap::new();
        let retained_encoded_rows = openings
            .iter()
            .enumerate()
            .map(|(opening, plan)| {
                for (position, &row) in plan.retained_source_rows.iter().enumerate() {
                    if retained_rows.insert(row, (opening, position)).is_some() {
                        return Err(geometry(opening, "duplicate retained source row"));
                    }
                }
                Ok(vec![None; plan.retained_source_rows.len()])
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            openings,
            retained_rows,
            retained_encoded_rows,
            omitted_source_rows: concrete_omitted,
            omitted_coordinate_columns,
        })
    }

    pub(super) fn omits_source_row(&self, row: usize) -> bool {
        self.omitted_source_rows[row]
    }

    pub(super) fn omits_coordinate_column(&self, column: usize) -> bool {
        self.omitted_coordinate_columns[column]
    }

    pub(super) fn install_coordinate_rows(
        &mut self,
        schedule: &GadgetNativeCoordinateGateSchedule,
    ) -> Result<(), GadgetNativeError> {
        for (opening_index, plan) in self.openings.iter_mut().enumerate() {
            for (digit, &column) in plan.digit_columns.iter().enumerate() {
                let row = schedule
                    .row_for_column(column)
                    .ok_or_else(|| geometry(opening_index, "retained centered-unit coordinate"))?;
                if !matches!(
                    schedule.rows().get(row),
                    Some(GadgetNativeCoordinateRowAudit::CenteredUnitPair {
                        row: recorded,
                        family: GadgetNativeCenteredFamily::SisOpening,
                        left,
                        right,
                    }) if *recorded == row && (*left == column || *right == column)
                ) && !matches!(
                    schedule.rows().get(row),
                    Some(GadgetNativeCoordinateRowAudit::CenteredUnitTail {
                        row: recorded,
                        family: GadgetNativeCenteredFamily::SisOpening,
                        coordinate,
                    }) if *recorded == row && *coordinate == column
                ) {
                    return Err(geometry(opening_index, "centered-unit row family"));
                }
                plan.centered_unit_rows[digit] = row;
            }
        }
        Ok(())
    }

    pub(super) fn before_emit(&mut self, source_row: usize, encoded_row: usize) -> Result<(), GadgetNativeError> {
        if self.omitted_source_rows[source_row] {
            return Err(geometry(
                self.opening_for_source_row(source_row).unwrap_or(0),
                "omitted source row reached emitter",
            ));
        }
        if let Some(&(opening, position)) = self.retained_rows.get(&source_row) {
            if self.retained_encoded_rows[opening][position]
                .replace(encoded_row)
                .is_some()
            {
                return Err(geometry(opening, "duplicate retained encoded row"));
            }
        }
        Ok(())
    }

    pub(super) fn after_emit(&self, source_row: usize, encoded_row: usize) -> Result<(), GadgetNativeError> {
        if let Some(&(opening, position)) = self.retained_rows.get(&source_row) {
            let before = self.retained_encoded_rows[opening][position]
                .ok_or_else(|| geometry(opening, "missing retained encoded row start"))?;
            if encoded_row != before + 1 {
                return Err(geometry(opening, "retained row emitted more than one gate"));
            }
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Vec<BalancedTernarySharedSlotPlan>, GadgetNativeError> {
        let mut all_encoded_rows = BTreeSet::new();
        for (opening, plan) in self.openings.iter_mut().enumerate() {
            let retained = self.retained_encoded_rows[opening]
                .iter()
                .map(|row| row.ok_or_else(|| geometry(opening, "source row was not retained")))
                .collect::<Result<Vec<_>, _>>()?;
            if retained.len() != RETAINED_SOURCE_ROWS_PER_OPENING
                || retained.iter().any(|&row| !all_encoded_rows.insert(row))
            {
                return Err(geometry(opening, "non-bijective retained row mapping"));
            }
            plan.retained_encoded_rows = retained;
            if plan.retained_obligation_count() != RETAINED_GATES_PER_OPENING
                || plan.retained_physical_row_count()
                    != RETAINED_SOURCE_ROWS_PER_OPENING + (BALANCED_TERNARY_DIGITS + 1) / 2
                || plan.omitted_obligation_count() != OMITTED_GATES_PER_OPENING
            {
                return Err(geometry(opening, "final reduction census"));
            }
        }
        Ok(self.openings)
    }

    fn opening_for_source_row(&self, row: usize) -> Option<usize> {
        self.openings
            .iter()
            .position(|opening| opening.source_rows.contains(&row))
    }
}

fn build_plan(
    opening: &BalancedTernaryOpeningTraceEntry,
    source_columns: &[SourceColumn],
    source_rows: Range<usize>,
    retained_source_rows: Vec<usize>,
    retained_encoded_rows: Vec<usize>,
    centered_unit_rows: [usize; BALANCED_TERNARY_DIGITS],
    opening_index: usize,
) -> Result<BalancedTernarySharedSlotPlan, GadgetNativeError> {
    let field_terms = encoded_terms(source_columns, opening.field_col, opening_index)?;
    if field_terms.len() != BALANCED_TERNARY_DIGITS {
        return Err(geometry(opening_index, "field expansion width"));
    }
    let mut power = F::ONE;
    for &(_, coefficient) in &field_terms {
        if coefficient != power {
            return Err(geometry(opening_index, "field expansion coefficient"));
        }
        power *= F::from_u64(3);
    }
    let digit_columns = singleton_targets(source_columns, &opening.digit_cols, opening_index)?;
    let negative_columns = singleton_targets(source_columns, &opening.negative_cols, opening_index)?;
    let borrow_columns = singleton_targets(source_columns, &opening.borrow_cols, opening_index)?;
    if digit_columns
        .iter()
        .zip(&field_terms)
        .any(|(&column, &(field_column, _))| column != field_column)
    {
        return Err(geometry(opening_index, "field/digit slot alias"));
    }
    let distinct = std::iter::once(0)
        .chain(digit_columns)
        .chain(negative_columns)
        .chain(borrow_columns)
        .collect::<BTreeSet<_>>();
    if distinct.len() != 1 + BALANCED_TERNARY_DIGITS * 2 + BALANCED_TERNARY_DIGITS - 1 {
        return Err(geometry(opening_index, "target slot overlap"));
    }
    Ok(BalancedTernarySharedSlotPlan {
        source_field_column: opening.field_col,
        source_digit_columns: opening.digit_cols,
        source_negative_columns: opening.negative_cols,
        source_borrow_columns: opening.borrow_cols,
        one_column: 0,
        digit_columns,
        negative_columns,
        borrow_columns,
        field_terms,
        source_rows,
        retained_source_rows,
        retained_encoded_rows,
        centered_unit_rows,
    })
}

fn retained_source_rows(opening: &BalancedTernaryOpeningTraceEntry) -> Vec<usize> {
    (0..BALANCED_TERNARY_DIGITS)
        .map(|digit| opening.digit_rows.start + 2 * digit)
        .chain(opening.transition_rows.clone())
        .collect()
}

fn omitted_source_rows(opening: &BalancedTernaryOpeningTraceEntry) -> Vec<usize> {
    (0..BALANCED_TERNARY_DIGITS)
        .map(|digit| opening.digit_rows.start + 2 * digit + 1)
        .chain(std::iter::once(opening.reconstruction_row))
        .collect()
}

fn opening_for_row(trace: &R1csEncodingTrace, row: usize) -> Option<usize> {
    trace
        .balanced_ternary_openings()
        .iter()
        .position(|opening| source_range(opening).contains(&row))
}

fn encoded_terms(
    source_columns: &[SourceColumn],
    source_column: usize,
    opening: usize,
) -> Result<Vec<(usize, F)>, GadgetNativeError> {
    match source_columns.get(source_column) {
        Some(SourceColumn::Encoded(slot)) => Ok(slot_terms(*slot)),
        _ => Err(geometry(opening, "source column is not directly encoded")),
    }
}

fn singleton_targets<const N: usize>(
    source_columns: &[SourceColumn],
    columns: &[usize; N],
    opening: usize,
) -> Result<[usize; N], GadgetNativeError> {
    let mut targets = Vec::with_capacity(N);
    for &column in columns {
        match encoded_terms(source_columns, column, opening)?.as_slice() {
            [(target, coefficient)] if *coefficient == F::ONE => targets.push(*target),
            _ => return Err(geometry(opening, "non-singleton source alias")),
        }
    }
    targets
        .try_into()
        .map_err(|_| geometry(opening, "source alias width"))
}

fn source_range(opening: &BalancedTernaryOpeningTraceEntry) -> Range<usize> {
    opening.digit_rows.start..opening.transition_rows.end
}

fn extract_product_row(
    matrices: &[CcsMatrix<F>],
    row: usize,
    opening: usize,
) -> Result<GadgetNativeConstraintRow, GadgetNativeError> {
    let rows = matrices
        .iter()
        .map(|matrix| matrix_row(matrix, row).ok_or_else(|| geometry(opening, "non-CSC production matrix")))
        .collect::<Result<Vec<_>, _>>()?;
    if rows[gate::SELECTOR] != [(0, F::ONE)]
        || rows[gate::BITNESS].len() != 0
        || rows[gate::CENTERED_UNIT_TAIL].len() != 0
        || rows[(gate::PRODUCT_LEFT + 1)..gate::PRODUCT_RIGHT]
            .iter()
            .any(|terms| !terms.is_empty())
        || rows[(gate::PRODUCT_RIGHT + 1)..gate::PRODUCT_OUT]
            .iter()
            .any(|terms| !terms.is_empty())
        || rows[(gate::PRODUCT_OUT + 1)..]
            .iter()
            .any(|terms| !terms.is_empty())
    {
        return Err(geometry(opening, "retained row is not a single-product gate"));
    }
    Ok(GadgetNativeConstraintRow {
        a: rows[gate::PRODUCT_LEFT].clone(),
        b: rows[gate::PRODUCT_RIGHT].clone(),
        c: rows[gate::PRODUCT_OUT].clone(),
    })
}

fn validate_alphabet_rows(
    matrices: &[CcsMatrix<F>],
    plan: &BalancedTernarySharedSlotPlan,
    schedule: &GadgetNativeCoordinateGateSchedule,
    opening: usize,
) -> Result<(), GadgetNativeError> {
    for (&column, row) in plan.digit_columns.iter().zip(plan.centered_unit_rows()) {
        let audit = schedule
            .rows()
            .get(row)
            .ok_or_else(|| geometry(opening, "missing centered coordinate audit row"))?;
        if schedule.row_for_column(column) != Some(row) {
            return Err(geometry(opening, "centered coordinate audit map"));
        }
        validate_centered_coordinate_gate(matrices, *audit, column, opening)?;
    }
    Ok(())
}

fn validate_centered_coordinate_gate(
    matrices: &[CcsMatrix<F>],
    audit: GadgetNativeCoordinateRowAudit,
    column: usize,
    opening: usize,
) -> Result<(), GadgetNativeError> {
    let row = audit.row();
    let rows = matrices
        .iter()
        .map(|matrix| matrix_row(matrix, row).ok_or_else(|| geometry(opening, "non-CSC production matrix")))
        .collect::<Result<Vec<_>, _>>()?;
    let exact = match audit {
        GadgetNativeCoordinateRowAudit::CenteredUnitPair {
            family: GadgetNativeCenteredFamily::SisOpening,
            left,
            right,
            ..
        } => {
            (column == left || column == right)
                && rows[gate::CENTERED_PAIR_LEFT] == [(left, F::ONE)]
                && rows[gate::CENTERED_PAIR_RIGHT] == [(right, F::ONE)]
                && rows[gate::CENTERED_UNIT_TAIL].is_empty()
        }
        GadgetNativeCoordinateRowAudit::CenteredUnitTail {
            family: GadgetNativeCenteredFamily::SisOpening,
            coordinate,
            ..
        } => {
            column == coordinate
                && rows[gate::CENTERED_UNIT_TAIL] == [(coordinate, F::ONE)]
                && rows[gate::CENTERED_PAIR_LEFT].is_empty()
                && rows[gate::CENTERED_PAIR_RIGHT].is_empty()
        }
        _ => false,
    };
    let only_centered_roles = rows.iter().enumerate().all(|(matrix, terms)| {
        matches!(
            matrix,
            gate::SELECTOR | gate::CENTERED_UNIT_TAIL | gate::CENTERED_PAIR_LEFT | gate::CENTERED_PAIR_RIGHT
        ) || terms.is_empty()
    });
    if rows[gate::SELECTOR] != [(0, F::ONE)] || !exact || !only_centered_roles {
        return Err(geometry(opening, "coordinate gate row"));
    }
    Ok(())
}

pub(super) fn matrix_row(matrix: &CcsMatrix<F>, row: usize) -> Option<Vec<(usize, F)>> {
    let matrix = matrix.as_csc()?;
    Some(csc_row(matrix, row))
}

fn csc_row(matrix: &CscMat<F>, row: usize) -> Vec<(usize, F)> {
    let mut terms = Vec::new();
    for column in 0..matrix.ncols {
        for entry in matrix.column_range(column) {
            if matrix.row_index(entry) == row {
                terms.push((column, matrix.vals[entry]));
            }
        }
    }
    terms
}

fn geometry(opening: usize, detail: &'static str) -> GadgetNativeError {
    GadgetNativeError::BalancedTernaryGeometry { opening, detail }
}
