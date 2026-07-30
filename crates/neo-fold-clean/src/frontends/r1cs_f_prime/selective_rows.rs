//! Exclusive source-rewrite ledger and emitted-row plan for selective lowering.
//!
//! Owns: the run-compressed source partition, stable physical rewrite IDs,
//! stage-occurrence attribution, exact emitted family intervals, and final row
//! total consumed by both shape discovery and matrix emission.
//!
//! Does not own: trace semantics, theorem names, rewrite sufficiency, or
//! permission to remove a constraint family.
//!
//! Emits constraints: no. Matrix emission must consume these intervals in order.
//!
//! Authority boundary: validated trace descriptors and linear definitions are
//! physical compiler inputs. This ledger records what the compiler did; it is
//! never semantic or deletion authority.
//!
//! | Plan branch | Source disposition | Emitted family |
//! |---|---|---|
//! | Coordinate prefix | none | selector/shared/arm domain, one-hot, padding |
//! | Retained source | retained | retained |
//! | Trace rewrite | Poseidon2, centered, canonical, evaluation, product-sum | matching trace family |
//! | Definition rewrite | linear definition | empty |
//! | Ring alignment | none | ring padding |

use core::ops::Range;

use crate::engine::r1cs_circuit::PhysicalStageRange;

use super::super::lowering::DerivedProductSumEncoding;
use super::super::selective_audit::{
    SelectiveArmRowMappingAudit, SelectiveEmittedRowFamily, SelectiveEmittedRowRunAudit, SelectiveRewriteAudit,
    SelectiveRewriteId, SelectiveRewriteKind, SelectiveRowMappingAudit, SelectiveSourceRowDisposition,
    SelectiveSourceRowRunAudit,
};
use super::{
    trace_error, LowNormR1csError, SelectiveArmPlan, SparseR1cs, BALANCED_FIELD_WIDTH, CANON_CHUNK_COUNT,
    EVAL_GROUP_SIZE,
};

#[derive(Clone)]
struct SourceClaim {
    rows: Range<usize>,
    disposition: SelectiveSourceRowDisposition,
}

/// One arm's prepared source mapping and trace-rewrite sequence.
pub(super) struct PreparedSelectiveArmRows {
    source_runs: Vec<SelectiveSourceRowRunAudit>,
    retained_emitted_rows: Range<usize>,
    emitted_rows: Range<usize>,
    poseidon2_rewrites: Vec<SelectiveRewriteId>,
    centered_unit_rewrites: Vec<SelectiveRewriteId>,
    shifted_ternary_rewrites: Vec<SelectiveRewriteId>,
    polynomial_evaluation_rewrites: Vec<SelectiveRewriteId>,
    product_sum_rewrites: Vec<SelectiveRewriteId>,
}

impl PreparedSelectiveArmRows {
    pub(super) fn expand_source_to_emitted(
        &self,
        source_row_count: usize,
    ) -> Result<Vec<Option<usize>>, LowNormR1csError> {
        if self
            .source_runs
            .last()
            .map_or(source_row_count != 0, |run| run.source_rows().end != source_row_count)
        {
            return Err(trace_error("prepared source-row partition has the wrong boundary"));
        }
        let mut source_to_emitted = vec![None; source_row_count];
        let mut source_cursor = 0usize;
        for run in &self.source_runs {
            let source_rows = run.source_rows();
            if source_rows.start != source_cursor || source_rows.is_empty() {
                return Err(trace_error("prepared source-row partition has a gap or empty run"));
            }
            if let Some(emitted_start) = run.emitted_start() {
                for (offset, source_row) in source_rows.clone().enumerate() {
                    source_to_emitted[source_row] = Some(emitted_start + offset);
                }
            }
            source_cursor = source_rows.end;
        }
        Ok(source_to_emitted)
    }

    pub(super) fn source_runs(&self) -> &[SelectiveSourceRowRunAudit] {
        &self.source_runs
    }

    pub(super) fn retained_emitted_rows(&self) -> Range<usize> {
        self.retained_emitted_rows.clone()
    }

    pub(super) fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub(super) fn poseidon2_rewrite(&self, index: usize) -> SelectiveRewriteId {
        self.poseidon2_rewrites[index]
    }

    pub(super) fn centered_unit_rewrite(&self, index: usize) -> SelectiveRewriteId {
        self.centered_unit_rewrites[index]
    }

    pub(super) fn shifted_ternary_rewrite(&self, index: usize) -> SelectiveRewriteId {
        self.shifted_ternary_rewrites[index]
    }

    pub(super) fn polynomial_evaluation_rewrite(&self, index: usize) -> SelectiveRewriteId {
        self.polynomial_evaluation_rewrites[index]
    }

    pub(super) fn product_sum_rewrite(&self, index: usize) -> SelectiveRewriteId {
        self.product_sum_rewrites[index]
    }
}

/// Checked cursor over the exact emitted intervals planned before allocation.
pub(super) struct PreparedEmissionCursor<'a> {
    runs: &'a [SelectiveEmittedRowRunAudit],
    next: usize,
}

impl PreparedEmissionCursor<'_> {
    pub(super) fn check(
        &mut self,
        family: SelectiveEmittedRowFamily,
        arm: Option<usize>,
        rewrite_id: Option<SelectiveRewriteId>,
        emitted_rows: Range<usize>,
    ) -> Result<(), LowNormR1csError> {
        let Some(planned) = self.runs.get(self.next) else {
            return Err(trace_error("selective emitter exceeded the prepared row plan"));
        };
        if planned.family() != family
            || planned.arm() != arm
            || planned.rewrite_id() != rewrite_id
            || planned.emitted_rows() != emitted_rows
        {
            return Err(trace_error("selective emitted interval differs from prepared row plan"));
        }
        self.next += 1;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<(), LowNormR1csError> {
        if self.next != self.runs.len() {
            return Err(trace_error("selective emitter did not consume the complete row plan"));
        }
        Ok(())
    }
}

/// Single prepared source for selective row mapping, rewrite attribution, and totals.
pub(super) struct PreparedSelectiveRows {
    prefix_rows: Range<usize>,
    arms: Vec<PreparedSelectiveArmRows>,
    ring_padding_rows: Range<usize>,
    emitted_runs: Vec<SelectiveEmittedRowRunAudit>,
    rewrites: Vec<SelectiveRewriteAudit>,
    total_rows: usize,
}

impl PreparedSelectiveRows {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        arms: &[SparseR1cs],
        plans: &[SelectiveArmPlan],
        slots: &[Vec<Option<(usize, usize)>>],
        aliases: &[Vec<Option<(usize, usize)>>],
        equal_aliases: &[Vec<Option<usize>>],
        shared_private_fields: usize,
        derived_product_sums: &[Vec<DerivedProductSumEncoding>],
        selector_count: usize,
        public_padding_rows: usize,
        private_padding_rows: usize,
        columns_before_ring_padding: usize,
    ) -> Result<Self, LowNormR1csError> {
        let mut row_cursor = 0usize;
        let mut emitted_runs = Vec::new();
        push_emitted_run(
            &mut emitted_runs,
            &mut row_cursor,
            selector_count,
            SelectiveEmittedRowFamily::SelectorDomain,
            None,
            None,
            None,
        );

        let mut shared_domain_rows = 0usize;
        for source in 1..arms[0].m_in + shared_private_fields {
            if aliases[0][source].is_some() {
                continue;
            }
            if let Some((_, width)) = slots[0][source] {
                let source_proves_boolean = plans.iter().all(|plan| plan.source_boolean_rows[source]);
                if !source_proves_boolean && !plans[0].centered[source] && width != BALANCED_FIELD_WIDTH {
                    shared_domain_rows += width;
                }
            }
        }
        push_emitted_run(
            &mut emitted_runs,
            &mut row_cursor,
            shared_domain_rows,
            SelectiveEmittedRowFamily::SharedDomain,
            None,
            None,
            None,
        );

        for (arm_index, arm) in arms.iter().enumerate() {
            let mut arm_domain_rows = 0usize;
            for source in arm.m_in + shared_private_fields..arm.m {
                if aliases[arm_index][source].is_some() || equal_aliases[arm_index][source].is_some() {
                    continue;
                }
                if let Some((_, width)) = slots[arm_index][source] {
                    if !plans[arm_index].source_boolean_rows[source]
                        && !plans[arm_index].centered[source]
                        && width != BALANCED_FIELD_WIDTH
                    {
                        arm_domain_rows += width;
                    }
                }
            }
            push_emitted_run(
                &mut emitted_runs,
                &mut row_cursor,
                arm_domain_rows,
                SelectiveEmittedRowFamily::ArmDomain,
                Some(arm_index),
                None,
                None,
            );
        }
        push_emitted_run(
            &mut emitted_runs,
            &mut row_cursor,
            1,
            SelectiveEmittedRowFamily::OneHot,
            None,
            None,
            None,
        );
        push_emitted_run(
            &mut emitted_runs,
            &mut row_cursor,
            public_padding_rows,
            SelectiveEmittedRowFamily::PublicPadding,
            None,
            None,
            None,
        );
        push_emitted_run(
            &mut emitted_runs,
            &mut row_cursor,
            private_padding_rows,
            SelectiveEmittedRowFamily::PrivatePadding,
            None,
            None,
            None,
        );
        let prefix_rows = 0..row_cursor;

        let mut rewrites = Vec::new();
        let mut prepared_arms = Vec::with_capacity(arms.len());
        for (arm_index, arm) in arms.iter().enumerate() {
            validate_stage_schedule(arm.physical_stage_ranges(), arm.n)?;
            let definitions = &plans[arm_index].definitions;
            let stages = arm.physical_stage_ranges();
            let mut claims = Vec::new();

            let mut poseidon2_rewrites = Vec::with_capacity(arm.poseidon2_traces().len());
            for trace in arm.poseidon2_traces() {
                let source_rows = vec![trace.row_start..trace.row_end];
                let id = allocate_rewrite(
                    &mut rewrites,
                    arm_index,
                    SelectiveRewriteKind::Poseidon2,
                    source_rows.clone(),
                    stages,
                )?;
                claims.push(SourceClaim {
                    rows: source_rows[0].clone(),
                    disposition: SelectiveSourceRowDisposition::Poseidon2(id),
                });
                poseidon2_rewrites.push(id);
            }

            let mut centered_unit_rewrites = Vec::with_capacity(arm.centered_unit_traces().len());
            for trace in arm.centered_unit_traces() {
                let source_rows = vec![trace.row_start..trace.row_end];
                let id = allocate_rewrite(
                    &mut rewrites,
                    arm_index,
                    SelectiveRewriteKind::CenteredUnit,
                    source_rows.clone(),
                    stages,
                )?;
                claims.push(SourceClaim {
                    rows: source_rows[0].clone(),
                    disposition: SelectiveSourceRowDisposition::CenteredUnit(id),
                });
                centered_unit_rewrites.push(id);
            }

            let mut shifted_ternary_rewrites = Vec::with_capacity(arm.shifted_ternary_canonical_traces().len());
            for trace in arm.shifted_ternary_canonical_traces() {
                let source_rows = vec![
                    trace.digit_rows_start..trace.digit_rows_start + 2 * BALANCED_FIELD_WIDTH,
                    trace.reconstruction_row..trace.reconstruction_row + 1,
                    trace.transition_rows_start..trace.transition_rows_start + BALANCED_FIELD_WIDTH,
                ];
                let id = allocate_rewrite(
                    &mut rewrites,
                    arm_index,
                    SelectiveRewriteKind::ShiftedTernaryCanonical,
                    source_rows.clone(),
                    stages,
                )?;
                claims.extend(source_rows.into_iter().map(|rows| SourceClaim {
                    rows,
                    disposition: SelectiveSourceRowDisposition::ShiftedTernaryCanonical(id),
                }));
                shifted_ternary_rewrites.push(id);
            }

            let mut polynomial_evaluation_rewrites = Vec::with_capacity(arm.polynomial_evaluation_traces().len());
            for trace in arm.polynomial_evaluation_traces() {
                let source_rows = vec![trace.row_start..trace.row_end];
                let id = allocate_rewrite(
                    &mut rewrites,
                    arm_index,
                    SelectiveRewriteKind::PolynomialEvaluation,
                    source_rows.clone(),
                    stages,
                )?;
                claims.push(SourceClaim {
                    rows: source_rows[0].clone(),
                    disposition: SelectiveSourceRowDisposition::PolynomialEvaluation(id),
                });
                polynomial_evaluation_rewrites.push(id);
            }

            let mut product_sum_rewrites = Vec::with_capacity(arm.product_sum_batch_traces().len());
            for trace in arm.product_sum_batch_traces() {
                let source_rows = vec![trace.row_start..trace.row_end];
                let id = allocate_rewrite(
                    &mut rewrites,
                    arm_index,
                    SelectiveRewriteKind::ProductSum,
                    source_rows.clone(),
                    stages,
                )?;
                claims.push(SourceClaim {
                    rows: source_rows[0].clone(),
                    disposition: SelectiveSourceRowDisposition::ProductSum(id),
                });
                product_sum_rewrites.push(id);
            }

            for definition in &definitions.entries {
                let Some(row) = definition.row else {
                    continue;
                };
                let source_rows = vec![row..row + 1];
                let id = allocate_rewrite(
                    &mut rewrites,
                    arm_index,
                    SelectiveRewriteKind::LinearDefinition,
                    source_rows.clone(),
                    stages,
                )?;
                claims.push(SourceClaim {
                    rows: source_rows[0].clone(),
                    disposition: SelectiveSourceRowDisposition::LinearDefinition(id),
                });
            }

            claims.sort_by_key(|claim| (claim.rows.start, claim.rows.end));
            let source_runs = build_source_partition(arm.n, stages, &claims)?;
            let arm_start = row_cursor;
            let retained_start = row_cursor;
            let mut finalized_source_runs = Vec::with_capacity(source_runs.len());
            for run in source_runs {
                let source_rows = run.source_rows();
                let emitted_start = if run.disposition() == SelectiveSourceRowDisposition::Retained {
                    let start = row_cursor;
                    let stage_occurrence = run.stage_occurrence();
                    push_emitted_run(
                        &mut emitted_runs,
                        &mut row_cursor,
                        source_rows.len(),
                        SelectiveEmittedRowFamily::Retained,
                        Some(arm_index),
                        None,
                        stage_occurrence,
                    );
                    Some(start)
                } else {
                    None
                };
                finalized_source_runs.push(SelectiveSourceRowRunAudit::new(
                    source_rows,
                    run.disposition(),
                    run.stage_occurrence(),
                    emitted_start,
                ));
            }
            let retained_emitted_rows = retained_start..row_cursor;

            for (trace, &id) in arm.poseidon2_traces().iter().zip(&poseidon2_rewrites) {
                let count = trace.sboxes.len()
                    + trace
                        .output_cols
                        .iter()
                        .filter(|&&column| definitions.get(column).is_none())
                        .count();
                plan_rewrite_emission(
                    &mut emitted_runs,
                    &mut rewrites,
                    &mut row_cursor,
                    count,
                    SelectiveEmittedRowFamily::Poseidon2,
                    arm_index,
                    id,
                );
            }
            for (trace, &id) in arm
                .centered_unit_traces()
                .iter()
                .zip(&centered_unit_rewrites)
            {
                let count = usize::from(plans[arm_index].widths[trace.value_col] == 0);
                plan_rewrite_emission(
                    &mut emitted_runs,
                    &mut rewrites,
                    &mut row_cursor,
                    count,
                    SelectiveEmittedRowFamily::CenteredUnit,
                    arm_index,
                    id,
                );
            }
            for &id in &shifted_ternary_rewrites {
                plan_rewrite_emission(
                    &mut emitted_runs,
                    &mut rewrites,
                    &mut row_cursor,
                    CANON_CHUNK_COUNT,
                    SelectiveEmittedRowFamily::ShiftedTernaryCanonical,
                    arm_index,
                    id,
                );
            }

            let mut derived_count = 0usize;
            for (trace, &id) in arm
                .polynomial_evaluation_traces()
                .iter()
                .zip(&polynomial_evaluation_rewrites)
            {
                let groups = trace
                    .coefficient_cols
                    .len()
                    .saturating_sub(1)
                    .div_ceil(EVAL_GROUP_SIZE)
                    .max(1);
                plan_rewrite_emission(
                    &mut emitted_runs,
                    &mut rewrites,
                    &mut row_cursor,
                    2 * groups,
                    SelectiveEmittedRowFamily::PolynomialEvaluation,
                    arm_index,
                    id,
                );
                derived_count += 2 * groups.saturating_sub(1);
            }
            for (batch, &id) in arm
                .product_sum_batch_traces()
                .iter()
                .zip(&product_sum_rewrites)
            {
                let mut count = 0usize;
                for identity in &batch.identities {
                    let groups = identity.factors.len().div_ceil(EVAL_GROUP_SIZE).max(1);
                    count += groups;
                    derived_count += groups.saturating_sub(1);
                }
                plan_rewrite_emission(
                    &mut emitted_runs,
                    &mut rewrites,
                    &mut row_cursor,
                    count,
                    SelectiveEmittedRowFamily::ProductSum,
                    arm_index,
                    id,
                );
            }
            if derived_count != derived_product_sums[arm_index].len() {
                return Err(trace_error(
                    "derived evaluation-product census drifted during row preparation",
                ));
            }

            // Definition-only rewrites have no emitted family, but their exact
            // source-to-empty geometry remains in the rewrite ledger.
            for rewrite in rewrites.iter_mut().filter(|rewrite| {
                rewrite.arm() == arm_index && rewrite.kind() == SelectiveRewriteKind::LinearDefinition
            }) {
                rewrite.set_emitted_rows(row_cursor..row_cursor);
            }
            prepared_arms.push(PreparedSelectiveArmRows {
                source_runs: finalized_source_runs,
                retained_emitted_rows,
                emitted_rows: arm_start..row_cursor,
                poseidon2_rewrites,
                centered_unit_rewrites,
                shifted_ternary_rewrites,
                polynomial_evaluation_rewrites,
                product_sum_rewrites,
            });
        }

        let ring_padding_start = row_cursor;
        let ring_padding_count =
            columns_before_ring_padding.next_multiple_of(neo_math::D) - columns_before_ring_padding;
        push_emitted_run(
            &mut emitted_runs,
            &mut row_cursor,
            ring_padding_count,
            SelectiveEmittedRowFamily::RingPadding,
            None,
            None,
            None,
        );
        let ring_padding_rows = ring_padding_start..row_cursor;
        Ok(Self {
            prefix_rows,
            arms: prepared_arms,
            ring_padding_rows,
            emitted_runs,
            rewrites,
            total_rows: row_cursor,
        })
    }

    pub(super) fn arm(&self, index: usize) -> &PreparedSelectiveArmRows {
        &self.arms[index]
    }

    pub(super) fn total_rows(&self) -> usize {
        self.total_rows
    }

    pub(super) fn emission_cursor(&self) -> PreparedEmissionCursor<'_> {
        PreparedEmissionCursor {
            runs: &self.emitted_runs,
            next: 0,
        }
    }

    pub(super) fn audit(&self) -> SelectiveRowMappingAudit {
        SelectiveRowMappingAudit::new(
            self.prefix_rows.clone(),
            self.arms
                .iter()
                .map(|arm| {
                    SelectiveArmRowMappingAudit::new(
                        arm.source_runs.clone(),
                        arm.retained_emitted_rows.clone(),
                        arm.emitted_rows.clone(),
                    )
                })
                .collect(),
            self.ring_padding_rows.clone(),
            self.emitted_runs.clone(),
            self.rewrites.clone(),
            self.total_rows,
        )
    }
}

fn push_emitted_run(
    emitted_runs: &mut Vec<SelectiveEmittedRowRunAudit>,
    row_cursor: &mut usize,
    count: usize,
    family: SelectiveEmittedRowFamily,
    arm: Option<usize>,
    rewrite_id: Option<SelectiveRewriteId>,
    source_stage_occurrence: Option<usize>,
) -> Range<usize> {
    let rows = *row_cursor..*row_cursor + count;
    *row_cursor = rows.end;
    emitted_runs.push(SelectiveEmittedRowRunAudit::new(
        rows.clone(),
        family,
        arm,
        rewrite_id,
        source_stage_occurrence,
    ));
    rows
}

fn plan_rewrite_emission(
    emitted_runs: &mut Vec<SelectiveEmittedRowRunAudit>,
    rewrites: &mut [SelectiveRewriteAudit],
    row_cursor: &mut usize,
    count: usize,
    family: SelectiveEmittedRowFamily,
    arm: usize,
    id: SelectiveRewriteId,
) {
    let occurrence = rewrites[id.index()].source_stage_occurrence();
    let rows = push_emitted_run(emitted_runs, row_cursor, count, family, Some(arm), Some(id), occurrence);
    rewrites[id.index()].set_emitted_rows(rows);
}

fn allocate_rewrite(
    rewrites: &mut Vec<SelectiveRewriteAudit>,
    arm: usize,
    kind: SelectiveRewriteKind,
    source_rows: Vec<Range<usize>>,
    stages: &[PhysicalStageRange],
) -> Result<SelectiveRewriteId, LowNormR1csError> {
    let id = SelectiveRewriteId::from_index(rewrites.len())
        .ok_or_else(|| trace_error("selective rewrite identifier overflow"))?;
    let mut occurrence = None;
    for rows in &source_rows {
        let current = source_stage_occurrence(stages, rows).map_err(|(start, end)| {
            trace_error(&format!(
                "arm {arm} {kind:?} rewrite {} source rows {rows:?} crosses physical stage occurrences {start:?} and {end:?}",
                id.index()
            ))
        })?;
        if !stages.is_empty() {
            if let Some(previous) = occurrence {
                if current != Some(previous) {
                    return Err(trace_error(&format!(
                        "arm {arm} {kind:?} rewrite {} source intervals {source_rows:?} cross physical stage occurrences {previous} and {current:?}",
                        id.index()
                    )));
                }
            } else {
                occurrence = current;
            }
        }
    }
    let mut normalized_rows = Vec::<Range<usize>>::with_capacity(source_rows.len());
    for rows in source_rows {
        if let Some(previous) = normalized_rows.last_mut() {
            if previous.end == rows.start {
                previous.end = rows.end;
                continue;
            }
        }
        normalized_rows.push(rows);
    }
    rewrites.push(SelectiveRewriteAudit::new(id, arm, kind, normalized_rows, occurrence));
    Ok(id)
}

fn validate_stage_schedule(stages: &[PhysicalStageRange], source_rows: usize) -> Result<(), LowNormR1csError> {
    if stages.is_empty() {
        return Ok(());
    }
    let mut cursor = 0usize;
    for stage in stages {
        if stage.row_start() != cursor || stage.row_end() < stage.row_start() || stage.row_end() > source_rows {
            return Err(trace_error(
                "physical stage occurrences do not partition the source rows",
            ));
        }
        cursor = stage.row_end();
    }
    if cursor != source_rows {
        return Err(trace_error(
            "physical stage occurrences do not close at the source boundary",
        ));
    }
    Ok(())
}

fn source_stage_occurrence(
    stages: &[PhysicalStageRange],
    rows: &Range<usize>,
) -> Result<Option<usize>, (Option<usize>, Option<usize>)> {
    if rows.is_empty() {
        return Err((None, None));
    }
    if stages.is_empty() {
        return Ok(None);
    }
    let occurrence = |row| {
        let index = stages.partition_point(|stage| stage.row_end() <= row);
        stages
            .get(index)
            .filter(|stage| stage.contains_row(row))
            .map(|_| index)
    };
    let start = occurrence(rows.start);
    let end = rows.end.checked_sub(1).and_then(occurrence);
    if start != end || start.is_none() {
        return Err((start, end));
    }
    Ok(start)
}

fn build_source_partition(
    source_row_count: usize,
    stages: &[PhysicalStageRange],
    claims: &[SourceClaim],
) -> Result<Vec<SelectiveSourceRowRunAudit>, LowNormR1csError> {
    let mut runs = Vec::new();
    let mut cursor = 0usize;
    for claim in claims {
        if claim.rows.start < cursor || claim.rows.end > source_row_count || claim.rows.is_empty() {
            return Err(trace_error(
                "selective rewrite source intervals overlap or escape the source arm",
            ));
        }
        append_source_span(
            &mut runs,
            cursor..claim.rows.start,
            SelectiveSourceRowDisposition::Retained,
            stages,
        )?;
        append_source_span(&mut runs, claim.rows.clone(), claim.disposition, stages)?;
        cursor = claim.rows.end;
    }
    append_source_span(
        &mut runs,
        cursor..source_row_count,
        SelectiveSourceRowDisposition::Retained,
        stages,
    )?;
    Ok(runs)
}

fn append_source_span(
    runs: &mut Vec<SelectiveSourceRowRunAudit>,
    rows: Range<usize>,
    disposition: SelectiveSourceRowDisposition,
    stages: &[PhysicalStageRange],
) -> Result<(), LowNormR1csError> {
    if rows.is_empty() {
        return Ok(());
    }
    if stages.is_empty() {
        push_source_run(runs, rows, disposition, None);
        return Ok(());
    }
    let mut cursor = rows.start;
    while cursor < rows.end {
        let index = stages.partition_point(|stage| stage.row_end() <= cursor);
        let stage = stages
            .get(index)
            .filter(|stage| stage.contains_row(cursor))
            .ok_or_else(|| trace_error("source row has no physical stage occurrence"))?;
        let end = rows.end.min(stage.row_end());
        push_source_run(runs, cursor..end, disposition, Some(index));
        cursor = end;
    }
    Ok(())
}

fn push_source_run(
    runs: &mut Vec<SelectiveSourceRowRunAudit>,
    rows: Range<usize>,
    disposition: SelectiveSourceRowDisposition,
    stage_occurrence: Option<usize>,
) {
    if let Some(last) = runs.last_mut() {
        if last.source_rows().end == rows.start
            && last.disposition() == disposition
            && last.stage_occurrence() == stage_occurrence
        {
            let start = last.source_rows().start;
            *last = SelectiveSourceRowRunAudit::new(start..rows.end, disposition, stage_occurrence, None);
            return;
        }
    }
    runs.push(SelectiveSourceRowRunAudit::new(
        rows,
        disposition,
        stage_occurrence,
        None,
    ));
}

pub(super) fn skipped_selective_rows(arm: &SparseR1cs) -> Result<Vec<bool>, LowNormR1csError> {
    let mut skipped = vec![false; arm.n];
    let mut claim = |range: Range<usize>, overlap: &'static str| {
        for row in range {
            if row >= skipped.len() || core::mem::replace(&mut skipped[row], true) {
                return Err(trace_error(overlap));
            }
        }
        Ok(())
    };

    for trace in arm.poseidon2_traces() {
        claim(trace.row_start..trace.row_end, "Poseidon2 traces overlap")?;
    }
    for trace in arm.polynomial_evaluation_traces() {
        claim(trace.row_start..trace.row_end, "selective trace row ranges overlap")?;
    }
    for trace in arm.product_sum_batch_traces() {
        claim(
            trace.row_start..trace.row_end,
            "product-sum trace overlaps another selective trace",
        )?;
    }
    for trace in arm.centered_unit_traces() {
        claim(
            trace.row_start..trace.row_end,
            "centered-unit trace overlaps another selective trace",
        )?;
    }
    for trace in arm.shifted_ternary_canonical_traces() {
        validate_shifted_ternary_reconstruction_row(arm, trace)?;
        claim(
            trace.digit_rows_start..trace.digit_rows_start + 2 * BALANCED_FIELD_WIDTH,
            "shifted-ternary digit rows overlap another selective trace",
        )?;
        claim(
            trace.reconstruction_row..trace.reconstruction_row + 1,
            "shifted-ternary reconstruction row overlaps another selective trace",
        )?;
        claim(
            trace.transition_rows_start..trace.transition_rows_start + BALANCED_FIELD_WIDTH,
            "shifted-ternary transition rows overlap another selective trace",
        )?;
    }
    Ok(skipped)
}

fn validate_shifted_ternary_reconstruction_row(
    arm: &SparseR1cs,
    trace: &crate::engine::r1cs_circuit::builder::ShiftedTernaryCanonicalTrace,
) -> Result<(), LowNormR1csError> {
    if trace.reconstruction_row != trace.digit_rows_start + 2 * BALANCED_FIELD_WIDTH
        || trace.transition_rows_start != trace.reconstruction_row + 1
    {
        return Err(trace_error(
            "shifted-ternary reconstruction row is not between its digit and transition rows",
        ));
    }
    let decomposition = arm
        .balanced_ternary_decompositions()
        .iter()
        .find(|decomposition| decomposition.digit_cols[0] == trace.digit_columns_start)
        .ok_or_else(|| trace_error("shifted-ternary reconstruction has no source decomposition"))?;
    if decomposition.field_col != trace.field_column
        || decomposition
            .digit_cols
            .iter()
            .copied()
            .ne(trace.digit_columns_start..trace.digit_columns_start + BALANCED_FIELD_WIDTH)
    {
        return Err(trace_error(
            "shifted-ternary reconstruction digits are not one exact word",
        ));
    }
    Ok(())
}
