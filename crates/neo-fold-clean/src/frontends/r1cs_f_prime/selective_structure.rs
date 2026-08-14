//! Exact matrix materialization for the selective CCS image.
//!
//! Owns: the ordered selective row families, source-matrix remapping, compact
//! seeded/geometric rows, and construction of the final 13-matrix CCS relation.
//!
//! Does not own: slot planning, trace discovery, witness encoding, semantic
//! sufficiency, or proof that a source row is necessary.
//!
//! Emits constraints: yes. Every row emitted here must be assigned to exactly
//! one cost-tree leaf and justified by a separate Lean obligation theorem.
//!
//! Authority boundary: this module materializes a previously prepared compiler
//! plan. Its matrices are refinement evidence, never the semantic definition of
//! the SuperNeo/NIFS transition.
//!
//! | Child path | Mathematical obligation | Rust owner | Lean owner |
//! |---|---|---|---|
//! | `f_prime.selective_ccs.encoding.domain` | retained binary/centered coordinates have the declared domain | `build_structure` prefix | open refinement |
//! | `f_prime.selective_ccs.branch.one_hot` | exactly one source arm is active | `build_structure` selector row | selective-CCS semantics |
//! | `f_prime.selective_ccs.padding.{public,private,ring}` | every structural padding coordinate is zero | `build_structure` padding rows | `SelectiveCcs.Padding.Refinement` |
//! | `f_prime.selective_ccs.source.retained` | each retained source equation holds under its arm selector | `append_source_matrix` | open source-row refinement |
//! | `f_prime.selective_ccs.trace.poseidon2` | recorded S-box and linear-output equations hold | `build_structure` Poseidon2 loop | open per-trace refinement |
//! | `f_prime.selective_ccs.trace.centered_unit` | eliminated centered values remain in `{-1,0,1}` | `build_structure` centered loop | open per-trace refinement |
//! | `f_prime.selective_ccs.encoding.canonical` | shifted-ternary digits encode one canonical field element | `canonical::emit_shifted_ternary_trace_rows` | open canonical refinement |
//! | `f_prime.selective_ccs.trace.evaluation` | recorded polynomial evaluations are recomposed | `build_structure` evaluation loop | open per-trace refinement |
//! | `f_prime.selective_ccs.trace.product_sum` | recorded bounded product sums equal their result | `build_structure` product-sum loop | open per-trace refinement |

use neo_ccs::{CcsMatrix, CcsStructure, CscMat};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::super::lowering::{DerivedProductSumEncoding, LowNormR1csError};
use super::super::selective_audit::SelectiveEmittedRowFamily;
use super::super::SparseR1cs;
use super::canonical;
use super::emit::{append_field, append_lc, append_lc_scaled, append_slot, trace_error};
use super::rows::PreparedSelectiveRows;
use super::shape::selective_polynomial;
use super::terms::MatrixTerms;
use super::{
    LinearDefinitions, SelectiveArmPlan, SelectiveEncoding, A, B, BALANCED_FIELD_WIDTH, BIT, C, CENTERED_UNIT,
    EVAL_GROUP_SIZE, EVAL_PAIRS, EVAL_SELECTOR, GENERAL_SELECTOR, SBOX_INPUT, SELECTIVE_ARITY,
};
use crate::paper::relations::Structure;

pub(super) struct EmittedStructureTerms {
    pub(super) matrix_terms: Vec<MatrixTerms>,
    pub(super) rows: usize,
    pub(super) columns: usize,
}

impl EmittedStructureTerms {
    fn into_structure(self) -> Result<Structure, LowNormR1csError> {
        let mut matrices = Vec::with_capacity(SELECTIVE_ARITY);
        for (_matrix_index, mut terms) in self.matrix_terms.into_iter().enumerate() {
            #[cfg(feature = "perf-timers")]
            let matrix_started = std::time::Instant::now();
            #[cfg(feature = "perf-timers")]
            let explicit_terms = terms.explicit.len();
            let csc = if terms.retain_geometric {
                CscMat::from_counted_triplets(core::mem::take(&mut terms.explicit), self.rows, self.columns)
            } else {
                CscMat::from_triplets_and_geometric_runs(
                    core::mem::take(&mut terms.explicit),
                    &terms.geometric_runs,
                    self.rows,
                    self.columns,
                )
            };
            if !terms.retain_geometric {
                terms.geometric_runs.clear();
            }
            matrices.push(
                CcsMatrix::csc_with_compact_rows(csc, terms.seeded, terms.geometric_runs)
                    .map_err(|error| trace_error(&error))?,
            );
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[selective-matrix] index={_matrix_index} explicit_terms={explicit_terms} total={:.3}s",
                matrix_started.elapsed().as_secs_f64(),
            );
        }
        CcsStructure::new_sparse(matrices, selective_polynomial()).map_err(|error| trace_error(&error.to_string()))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_structure(
    arms: &[SparseR1cs],
    encoding: SelectiveEncoding,
    plans: &[SelectiveArmPlan],
    slots: &[Vec<Option<(usize, usize)>>],
    aliases: &[Vec<Option<(usize, usize)>>],
    equal_aliases: &[Vec<Option<usize>>],
    shared_private_fields: usize,
    derived_product_sums: &[Vec<DerivedProductSumEncoding>],
    selectors: &[usize],
    public_padding_cols: &[usize],
    private_padding_cols: &[usize],
    cols: usize,
    prepared_rows: &PreparedSelectiveRows,
) -> Result<Structure, LowNormR1csError> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let emission_started = std::time::Instant::now();
    let emitted = emit_structure_terms(
        arms,
        encoding,
        plans,
        slots,
        aliases,
        equal_aliases,
        shared_private_fields,
        derived_product_sums,
        selectors,
        public_padding_cols,
        private_padding_cols,
        cols,
        prepared_rows,
    )?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[selective-structure] phase=emit total={:.3}s",
        emission_started.elapsed().as_secs_f64(),
    );
    let structure = emitted.into_structure()?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[selective-structure] phase=complete total={:.3}s",
        total_started.elapsed().as_secs_f64(),
    );
    Ok(structure)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_structure_terms(
    arms: &[SparseR1cs],
    encoding: SelectiveEncoding,
    plans: &[SelectiveArmPlan],
    slots: &[Vec<Option<(usize, usize)>>],
    aliases: &[Vec<Option<(usize, usize)>>],
    equal_aliases: &[Vec<Option<usize>>],
    shared_private_fields: usize,
    derived_product_sums: &[Vec<DerivedProductSumEncoding>],
    selectors: &[usize],
    public_padding_cols: &[usize],
    private_padding_cols: &[usize],
    cols: usize,
    prepared_rows: &PreparedSelectiveRows,
) -> Result<EmittedStructureTerms, LowNormR1csError> {
    let eval_pair = |pair_index: usize| EVAL_PAIRS[pair_index];

    let expected_rows = prepared_rows.total_rows();
    let mut matrix_terms = (0..SELECTIVE_ARITY)
        .map(|index| MatrixTerms::new(index == SBOX_INPUT))
        .collect::<Vec<_>>();
    let mut row_cursor = 0usize;
    let mut emission_plan = prepared_rows.emission_cursor();

    let family_start = row_cursor;
    for &selector in selectors {
        emit_bit_domain(&mut matrix_terms, &mut row_cursor, None, selector, false);
    }
    emission_plan.check(
        SelectiveEmittedRowFamily::SelectorDomain,
        None,
        None,
        family_start..row_cursor,
    )?;

    let family_start = row_cursor;
    let mut pending_centered = None;
    for source in 1..arms[0].m_in + shared_private_fields {
        if aliases[0][source].is_some() {
            continue;
        }
        if let Some((start, width)) = slots[0][source] {
            let source_proves_boolean = plans.iter().all(|plan| plan.source_boolean_rows[source]);
            if source_proves_boolean {
                continue;
            }
            if width == encoding.general_field_width()
                || ((plans[0].centered[source] || width == BALANCED_FIELD_WIDTH)
                    && encoding.outer_norm_proves_centered_unit())
            {
                continue;
            }
            for column in start..start + width {
                if plans[0].centered[source] || width == BALANCED_FIELD_WIDTH {
                    if let Some(left) = pending_centered.take() {
                        emit_centered_unit_pair(&mut matrix_terms, &mut row_cursor, None, left, Some(column));
                    } else {
                        pending_centered = Some(column);
                    }
                } else {
                    emit_bit_domain(&mut matrix_terms, &mut row_cursor, None, column, true);
                }
            }
        }
    }
    if let Some(left) = pending_centered {
        emit_centered_unit_pair(&mut matrix_terms, &mut row_cursor, None, left, None);
    }
    emission_plan.check(
        SelectiveEmittedRowFamily::SharedDomain,
        None,
        None,
        family_start..row_cursor,
    )?;

    for (arm_index, arm) in arms.iter().enumerate() {
        let family_start = row_cursor;
        let mut pending_centered = None;
        for source in arm.m_in + shared_private_fields..arm.m {
            if aliases[arm_index][source].is_some() || equal_aliases[arm_index][source].is_some() {
                continue;
            }
            if let Some((start, width)) = slots[arm_index][source] {
                if plans[arm_index].source_boolean_rows[source] {
                    continue;
                }
                if width == encoding.general_field_width()
                    || ((plans[arm_index].centered[source] || width == BALANCED_FIELD_WIDTH)
                        && encoding.outer_norm_proves_centered_unit())
                {
                    continue;
                }
                for column in start..start + width {
                    if plans[arm_index].centered[source] || width == BALANCED_FIELD_WIDTH {
                        if let Some(left) = pending_centered.take() {
                            emit_centered_unit_pair(
                                &mut matrix_terms,
                                &mut row_cursor,
                                Some(selectors[arm_index]),
                                left,
                                Some(column),
                            );
                        } else {
                            pending_centered = Some(column);
                        }
                    } else {
                        emit_bit_domain(
                            &mut matrix_terms,
                            &mut row_cursor,
                            Some(selectors[arm_index]),
                            column,
                            true,
                        );
                    }
                }
            }
        }
        if let Some(left) = pending_centered {
            emit_centered_unit_pair(
                &mut matrix_terms,
                &mut row_cursor,
                Some(selectors[arm_index]),
                left,
                None,
            );
        }
        emission_plan.check(
            SelectiveEmittedRowFamily::ArmDomain,
            Some(arm_index),
            None,
            family_start..row_cursor,
        )?;
    }

    let family_start = row_cursor;
    let selector_row = row_cursor;
    matrix_terms[GENERAL_SELECTOR].push((selector_row, 0, F::ONE));
    matrix_terms[C].push((selector_row, 0, -F::ONE));
    for &selector in selectors {
        matrix_terms[C].push((selector_row, selector, F::ONE));
    }
    row_cursor += 1;
    emission_plan.check(SelectiveEmittedRowFamily::OneHot, None, None, family_start..row_cursor)?;

    let family_start = row_cursor;
    for &col in public_padding_cols {
        matrix_terms[GENERAL_SELECTOR].push((row_cursor, 0, F::ONE));
        matrix_terms[C].push((row_cursor, col, F::ONE));
        row_cursor += 1;
    }
    emission_plan.check(
        SelectiveEmittedRowFamily::PublicPadding,
        None,
        None,
        family_start..row_cursor,
    )?;

    let family_start = row_cursor;
    for &col in private_padding_cols {
        matrix_terms[GENERAL_SELECTOR].push((row_cursor, 0, F::ONE));
        matrix_terms[C].push((row_cursor, col, F::ONE));
        row_cursor += 1;
    }
    emission_plan.check(
        SelectiveEmittedRowFamily::PrivatePadding,
        None,
        None,
        family_start..row_cursor,
    )?;

    for (arm_index, arm) in arms.iter().enumerate() {
        let definitions = &plans[arm_index].definitions;
        let prepared_arm = prepared_rows.arm(arm_index);
        let retained_rows = prepared_arm.retained_emitted_rows();
        if row_cursor != retained_rows.start {
            return Err(trace_error("selective retained-row prefix differs from prepared plan"));
        }
        let row_map = prepared_arm.expand_source_to_emitted(arm.n)?;
        for run in prepared_arm.source_runs() {
            let Some(emitted_start) = run.emitted_start() else {
                continue;
            };
            let family_start = row_cursor;
            for target_row in emitted_start..emitted_start + run.source_rows().len() {
                if target_row != row_cursor {
                    return Err(trace_error(
                        "selective retained-row mapping differs from emission order",
                    ));
                }
                matrix_terms[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                row_cursor += 1;
            }
            emission_plan.check(
                SelectiveEmittedRowFamily::Retained,
                Some(arm_index),
                None,
                family_start..row_cursor,
            )?;
        }
        if row_cursor != retained_rows.end {
            return Err(trace_error("selective retained-row count differs from prepared plan"));
        }
        append_source_matrix(&mut matrix_terms[A], &arm.a, &slots[arm_index], definitions, &row_map)?;
        append_source_matrix(&mut matrix_terms[B], &arm.b, &slots[arm_index], definitions, &row_map)?;
        append_source_matrix(&mut matrix_terms[C], &arm.c, &slots[arm_index], definitions, &row_map)?;
        for (trace_index, trace) in arm.poseidon2_traces().iter().enumerate() {
            let family_start = row_cursor;
            for sbox in &trace.sboxes {
                matrix_terms[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                append_lc(
                    &mut matrix_terms[SBOX_INPUT],
                    row_cursor,
                    &sbox.input,
                    &slots[arm_index],
                    definitions,
                )?;
                append_field(
                    &mut matrix_terms[C],
                    row_cursor,
                    sbox.output_col,
                    F::ONE,
                    &slots[arm_index],
                    definitions,
                )?;
                row_cursor += 1;
            }
            for lane in 0..trace.output_cols.len() {
                if definitions.get(trace.output_cols[lane]).is_some() {
                    continue;
                }
                matrix_terms[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                append_field(
                    &mut matrix_terms[C],
                    row_cursor,
                    trace.output_cols[lane],
                    F::ONE,
                    &slots[arm_index],
                    definitions,
                )?;
                append_lc_scaled(
                    &mut matrix_terms[C],
                    row_cursor,
                    &trace.output_linear_forms[lane],
                    -F::ONE,
                    &slots[arm_index],
                    definitions,
                )?;
                row_cursor += 1;
            }
            emission_plan.check(
                SelectiveEmittedRowFamily::Poseidon2,
                Some(arm_index),
                Some(prepared_arm.poseidon2_rewrite(trace_index)),
                family_start..row_cursor,
            )?;
        }
        for (trace_index, trace) in arm.centered_unit_traces().iter().enumerate() {
            let family_start = row_cursor;
            if plans[arm_index].widths[trace.value_col] != 0 {
                emission_plan.check(
                    SelectiveEmittedRowFamily::CenteredUnit,
                    Some(arm_index),
                    Some(prepared_arm.centered_unit_rewrite(trace_index)),
                    family_start..row_cursor,
                )?;
                continue;
            }
            matrix_terms[GENERAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
            append_field(
                &mut matrix_terms[CENTERED_UNIT],
                row_cursor,
                trace.value_col,
                F::ONE,
                &slots[arm_index],
                definitions,
            )?;
            row_cursor += 1;
            emission_plan.check(
                SelectiveEmittedRowFamily::CenteredUnit,
                Some(arm_index),
                Some(prepared_arm.centered_unit_rewrite(trace_index)),
                family_start..row_cursor,
            )?;
        }
        for (trace_index, trace) in arm.shifted_ternary_canonical_traces().iter().enumerate() {
            let family_start = row_cursor;
            canonical::emit_shifted_ternary_trace_rows(
                trace,
                &slots[arm_index],
                definitions,
                selectors[arm_index],
                &mut matrix_terms,
                &mut row_cursor,
            )?;
            emission_plan.check(
                SelectiveEmittedRowFamily::ShiftedTernaryCanonical,
                Some(arm_index),
                Some(prepared_arm.shifted_ternary_rewrite(trace_index)),
                family_start..row_cursor,
            )?;
        }
        let mut derived_cursor = 0usize;
        for (trace_index, trace) in arm.polynomial_evaluation_traces().iter().enumerate() {
            let family_start = row_cursor;
            for limb in 0..2 {
                let product_indices = (1..trace.coefficient_cols.len()).collect::<Vec<_>>();
                let groups = product_indices.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                if groups.is_empty() {
                    matrix_terms[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    append_field(
                        &mut matrix_terms[C],
                        row_cursor,
                        trace.output_cols[limb],
                        F::ONE,
                        &slots[arm_index],
                        definitions,
                    )?;
                    if limb == 0 {
                        append_field(
                            &mut matrix_terms[C],
                            row_cursor,
                            trace.coefficient_cols[0],
                            -F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    row_cursor += 1;
                    continue;
                }
                let mut previous = None;
                for (group_index, group) in groups.iter().enumerate() {
                    matrix_terms[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    if group_index + 1 == groups.len() {
                        append_field(
                            &mut matrix_terms[C],
                            row_cursor,
                            trace.output_cols[limb],
                            F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                        if limb == 0 {
                            append_field(
                                &mut matrix_terms[C],
                                row_cursor,
                                trace.coefficient_cols[0],
                                -F::ONE,
                                &slots[arm_index],
                                definitions,
                            )?;
                        }
                    } else {
                        let derived = &derived_product_sums[arm_index][derived_cursor];
                        derived_cursor += 1;
                        append_slot(&mut matrix_terms[C], row_cursor, derived.slot, F::ONE);
                    }
                    if let Some(previous) = previous {
                        append_slot(&mut matrix_terms[C], row_cursor, previous, -F::ONE);
                    }
                    for (pair_index, &term_index) in group.iter().enumerate() {
                        let (left, right) = eval_pair(pair_index);
                        append_field(
                            &mut matrix_terms[left],
                            row_cursor,
                            trace.coefficient_cols[term_index],
                            F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                        append_field(
                            &mut matrix_terms[right],
                            row_cursor,
                            trace.power_cols[term_index][limb],
                            F::ONE,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    if group_index + 1 != groups.len() {
                        previous = Some(derived_product_sums[arm_index][derived_cursor - 1].slot);
                    }
                    row_cursor += 1;
                }
            }
            emission_plan.check(
                SelectiveEmittedRowFamily::PolynomialEvaluation,
                Some(arm_index),
                Some(prepared_arm.polynomial_evaluation_rewrite(trace_index)),
                family_start..row_cursor,
            )?;
        }
        for (batch_index, batch) in arm.product_sum_batch_traces().iter().enumerate() {
            let family_start = row_cursor;
            for identity in &batch.identities {
                if identity.factors.len() <= EVAL_GROUP_SIZE {
                    matrix_terms[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    append_lc(
                        &mut matrix_terms[C],
                        row_cursor,
                        &identity.result,
                        &slots[arm_index],
                        definitions,
                    )?;
                    for (pair_index, factor) in identity.factors.iter().enumerate() {
                        let (left, right) = eval_pair(pair_index);
                        append_lc_scaled(
                            &mut matrix_terms[left],
                            row_cursor,
                            &factor.left,
                            factor.coefficient,
                            &slots[arm_index],
                            definitions,
                        )?;
                        append_lc(
                            &mut matrix_terms[right],
                            row_cursor,
                            &factor.right,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    row_cursor += 1;
                    continue;
                }
                let groups = identity.factors.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
                let mut previous = None;
                for (group_index, group) in groups.iter().enumerate() {
                    matrix_terms[EVAL_SELECTOR].push((row_cursor, selectors[arm_index], F::ONE));
                    if group_index + 1 == groups.len() {
                        append_lc(
                            &mut matrix_terms[C],
                            row_cursor,
                            &identity.result,
                            &slots[arm_index],
                            definitions,
                        )?;
                    } else {
                        let derived = &derived_product_sums[arm_index][derived_cursor];
                        derived_cursor += 1;
                        append_slot(&mut matrix_terms[C], row_cursor, derived.slot, F::ONE);
                    }
                    if let Some(previous) = previous {
                        append_slot(&mut matrix_terms[C], row_cursor, previous, -F::ONE);
                    }
                    for (pair_index, factor) in group.iter().enumerate() {
                        let (left, right) = eval_pair(pair_index);
                        append_lc_scaled(
                            &mut matrix_terms[left],
                            row_cursor,
                            &factor.left,
                            factor.coefficient,
                            &slots[arm_index],
                            definitions,
                        )?;
                        append_lc(
                            &mut matrix_terms[right],
                            row_cursor,
                            &factor.right,
                            &slots[arm_index],
                            definitions,
                        )?;
                    }
                    if group_index + 1 != groups.len() {
                        previous = Some(derived_product_sums[arm_index][derived_cursor - 1].slot);
                    }
                    row_cursor += 1;
                }
            }
            emission_plan.check(
                SelectiveEmittedRowFamily::ProductSum,
                Some(arm_index),
                Some(prepared_arm.product_sum_rewrite(batch_index)),
                family_start..row_cursor,
            )?;
        }
        if derived_cursor != derived_product_sums[arm_index].len() {
            return Err(trace_error(
                "derived evaluation-product census drifted during row emission",
            ));
        }
        if row_cursor != prepared_arm.emitted_rows().end {
            return Err(trace_error("selective arm row count differs from prepared plan"));
        }
    }

    // The one-joint relation keeps rows and assignment separate. Add only D-alignment
    // coordinates, and constrain them to zero rather than witness slack.
    let columns = cols.next_multiple_of(D);
    let family_start = row_cursor;
    for column in cols..columns {
        matrix_terms[GENERAL_SELECTOR].push((row_cursor, 0, F::ONE));
        matrix_terms[C].push((row_cursor, column, F::ONE));
        row_cursor += 1;
    }
    emission_plan.check(
        SelectiveEmittedRowFamily::RingPadding,
        None,
        None,
        family_start..row_cursor,
    )?;
    emission_plan.finish()?;
    let rows = row_cursor;
    if rows != expected_rows {
        return Err(trace_error("selective row count differs from emitted structure"));
    }
    Ok(EmittedStructureTerms {
        matrix_terms,
        rows,
        columns,
    })
}

fn emit_bit_domain(
    matrix_terms: &mut [MatrixTerms],
    row_cursor: &mut usize,
    selector: Option<usize>,
    column: usize,
    combined_selector: bool,
) {
    let selector = selector.unwrap_or(0);
    matrix_terms[GENERAL_SELECTOR].push((*row_cursor, selector, F::ONE));
    if combined_selector {
        matrix_terms[EVAL_SELECTOR].push((*row_cursor, selector, F::ONE));
    }
    matrix_terms[BIT].push((*row_cursor, column, F::ONE));
    *row_cursor += 1;
}

fn emit_centered_unit_pair(
    matrix_terms: &mut [MatrixTerms],
    row_cursor: &mut usize,
    selector: Option<usize>,
    left: usize,
    right: Option<usize>,
) {
    let selector = selector.unwrap_or(0);
    matrix_terms[GENERAL_SELECTOR].push((*row_cursor, selector, F::ONE));
    matrix_terms[EVAL_SELECTOR].push((*row_cursor, selector, F::ONE));
    matrix_terms[CENTERED_UNIT].push((*row_cursor, left, F::ONE));
    if let Some(right) = right {
        matrix_terms[A].push((*row_cursor, right, F::ONE));
    }
    *row_cursor += 1;
}

fn append_source_matrix(
    terms: &mut MatrixTerms,
    matrix: &CcsMatrix<F>,
    slots: &[Option<(usize, usize)>],
    definitions: &LinearDefinitions,
    row_map: &[Option<usize>],
) -> Result<(), LowNormR1csError> {
    let mut append_csc = |csc: &CscMat<F>| -> Result<(), LowNormR1csError> {
        for field_col in 0..csc.ncols.min(slots.len()) {
            for index in csc.column_range(field_col) {
                if let Some(target_row) = row_map[csc.row_index(index)] {
                    append_field(terms, target_row, field_col, csc.vals[index], slots, definitions)?;
                }
            }
        }
        Ok(())
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for source_row in 0..(*n).min(row_map.len()).min(slots.len()) {
                if let Some(target_row) = row_map[source_row] {
                    append_field(terms, target_row, source_row, F::ONE, slots, definitions)?;
                }
            }
        }
        CcsMatrix::Csc(csc) => append_csc(csc)?,
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            append_csc(csc)?;
            for run in geometric_runs {
                let Some(target_row) = row_map[run.row()] else {
                    continue;
                };
                let mut coefficient = *run.initial();
                for field_col in run.column_start()..run.column_start() + run.len() {
                    append_field(terms, target_row, field_col, coefficient, slots, definitions)?;
                    coefficient *= *run.ratio();
                }
            }
            for block in blocks {
                let target_start = row_map[block.row_start()]
                    .ok_or_else(|| trace_error("seeded Phi81 block overlaps a removed Poseidon2 row"))?;
                for offset in 0..neo_math::D * block.kappa() {
                    if row_map[block.row_start() + offset] != Some(target_start + offset) {
                        return Err(trace_error(
                            "seeded Phi81 rows are not contiguous after selective lowering",
                        ));
                    }
                }
                let mut starts = Vec::with_capacity(block.word_starts().len());
                for &source_start in block.word_starts() {
                    let (target, width) =
                        slots[source_start].ok_or_else(|| trace_error("seeded Phi81 input bit was eliminated"))?;
                    if width != 1 {
                        return Err(trace_error("seeded Phi81 input is not a one-bit slot"));
                    }
                    for offset in 0..block.word_width() {
                        if slots[source_start + offset] != Some((target + offset, 1)) {
                            return Err(trace_error("seeded Phi81 input word is not contiguous"));
                        }
                    }
                    starts.push(target);
                }
                terms
                    .seeded
                    .push(block.with_geometry(target_start, starts)?);
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(trace_error(
                "selective source lowering requires materialized matrix content",
            ));
        }
    }
    Ok(())
}
