//! Bounded row projection from the exact selective emitter term stream.
//!
//! This path serves assurance fixtures whose complete fixed-point column
//! domain is intentionally too large to materialize as thirteen full CSC
//! matrices. It invokes the same emitter as `build_structure`, then
//! canonicalizes only caller-selected rows before allocating column-sized
//! arrays.
//!
//! Owns: bounded projection of the shared emitter's exact thirteen-port term
//! stream and source/compiler provenance for caller-selected rows.
//!
//! Does not own: source semantics, selector truth, protocol authority,
//! security reductions, or permission to remove constraints.
//!
//! Emits constraints: no new rows; it observes rows emitted by the shared
//! selective structure path.
//!
//! | Child path | Mathematical obligation | Authority class |
//! |---|---|---|
//! | `selective.projected_rows.final` | selected A/B/C terms equal the shared emitter stream | direct dataflow |
//! | `selective.projected_rows.source` | retained slots and substitutions cover every referenced source column | computed |
//! | `selective.projected_rows.rewrite` | each compact rewrite records its exact factors and output | computed |

use std::collections::{BTreeMap, BTreeSet};

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::ProductFactorTrace;
use crate::engine::r1cs_circuit::Lc;

use super::super::selective_audit::{SelectiveCompilerAudit, SelectiveEmittedRowRunAudit};
use super::super::SparseR1cs;
use super::emit::{append_field, append_lc, append_lc_scaled, append_slot};
use super::projected_decoder::{
    decoder_provenance, decoder_run_provenance, SelectiveProjectedDecoderProvenance,
    SelectiveProjectedDecoderRunProvenance,
};
use super::terms::MatrixTerms;
use super::{
    prepare_selective_layout, structure, trace_error, LowNormR1csError, A, B, C, EVAL_GROUP_SIZE, EVAL_PAIRS,
    EVAL_SELECTOR, GENERAL_SELECTOR, SELECTIVE_ARITY,
};

#[path = "selective_projected_rows/model.rs"]
mod model;
#[path = "selective_projected_rows/poseidon2.rs"]
mod poseidon2;
#[path = "selective_projected_rows/row_index.rs"]
mod row_index;

pub use model::{
    SelectiveProjectedDerivedProductSum, SelectiveProjectedExplicitRunCensus, SelectiveProjectedGeometricRun,
    SelectiveProjectedPort, SelectiveProjectedPoseidon2OutputStep, SelectiveProjectedPoseidon2SboxStep,
    SelectiveProjectedProductFactor, SelectiveProjectedPublicCoordinate, SelectiveProjectedPublicCoordinateSource,
    SelectiveProjectedRetainedStep, SelectiveProjectedRewriteOutput, SelectiveProjectedRewriteStep,
    SelectiveProjectedRowArtifact, SelectiveProjectedSourceDefinition, SelectiveProjectedSourceImage,
    SelectiveProjectedSourceLinearCombination, SelectiveProjectedSourceProvenance, SelectiveProjectedSourceSlot,
    SelectiveProjectedSourceTerm, SelectiveProjectedTerm,
};

/// Exact selected rows emitted from one prepared selective compiler plan.
#[derive(Debug)]
pub struct SelectiveProjectedRowsAudit {
    rows: usize,
    columns: usize,
    selector_columns: Vec<usize>,
    compiler_audit: SelectiveCompilerAudit,
    public_coordinates: Vec<SelectiveProjectedPublicCoordinate>,
    public_padding_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    selector_domain_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    one_hot_row_artifact: SelectiveProjectedRowArtifact,
    private_padding_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    ring_padding_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    source_provenance: Option<SelectiveProjectedSourceProvenance>,
    decoder_provenance: Option<SelectiveProjectedDecoderProvenance>,
    decoder_run_provenance: Option<SelectiveProjectedDecoderRunProvenance>,
    explicit_run_census: Vec<SelectiveProjectedExplicitRunCensus>,
}

impl SelectiveProjectedRowsAudit {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn selector_columns(&self) -> &[usize] {
        &self.selector_columns
    }

    pub fn compiler_audit(&self) -> &SelectiveCompilerAudit {
        &self.compiler_audit
    }

    /// Complete public-coordinate decoder validated against every arm of the
    /// same prepared layout used by the projected emitter.
    pub fn public_coordinates(&self) -> &[SelectiveProjectedPublicCoordinate] {
        &self.public_coordinates
    }

    /// Exact public-padding zero rows projected independently of the
    /// caller-selected semantic slice.
    pub fn public_padding_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.public_padding_row_artifacts
    }

    /// Exact selector-domain rows projected independently of the
    /// caller-selected semantic slice.
    pub fn selector_domain_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.selector_domain_row_artifacts
    }

    /// Exact selector-total row projected independently of the
    /// caller-selected semantic slice.
    pub fn one_hot_row_artifact(&self) -> &SelectiveProjectedRowArtifact {
        &self.one_hot_row_artifact
    }

    /// Exact private-alignment zero rows projected independently of the
    /// caller-selected semantic slice.
    pub fn private_padding_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.private_padding_row_artifacts
    }

    /// Exact final ring-alignment zero rows projected independently of the
    /// caller-selected semantic slice.
    pub fn ring_padding_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.ring_padding_row_artifacts
    }

    pub fn row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.row_artifacts
    }

    pub fn source_provenance(&self) -> Option<&SelectiveProjectedSourceProvenance> {
        self.source_provenance.as_ref()
    }

    /// Exact source-to-final-assignment decoder requested independently of
    /// the selected row certificate.
    pub fn decoder_provenance(&self) -> Option<&SelectiveProjectedDecoderProvenance> {
        self.decoder_provenance.as_ref()
    }

    /// Complete run-compressed decoder requested independently of selected
    /// row/source closure.  This remains layout data, not a value theorem.
    pub fn decoder_run_provenance(&self) -> Option<&SelectiveProjectedDecoderRunProvenance> {
        self.decoder_run_provenance.as_ref()
    }

    /// Per-port format-design census over the exact emitter-order explicit
    /// term stream. A run has at least three terms, one fixed field
    /// coefficient, and fixed signed row and column deltas.
    pub fn explicit_run_census(&self) -> &[SelectiveProjectedExplicitRunCensus] {
        &self.explicit_run_census
    }
}

fn signed_delta(next: usize, current: usize) -> i128 {
    next as i128 - current as i128
}

fn explicit_run_census(terms: &[(usize, usize, F)]) -> SelectiveProjectedExplicitRunCensus {
    let mut cursor = 0usize;
    let mut affine_run_count = 0usize;
    let mut affine_run_terms = 0usize;
    let mut literal_count = 0usize;
    while cursor < terms.len() {
        if cursor + 2 < terms.len() {
            let first = terms[cursor];
            let second = terms[cursor + 1];
            let row_delta = signed_delta(second.0, first.0);
            let column_delta = signed_delta(second.1, first.1);
            if second.2 == first.2
                && terms[cursor + 2].2 == first.2
                && signed_delta(terms[cursor + 2].0, second.0) == row_delta
                && signed_delta(terms[cursor + 2].1, second.1) == column_delta
            {
                let mut stop = cursor + 3;
                while stop < terms.len()
                    && terms[stop].2 == first.2
                    && signed_delta(terms[stop].0, terms[stop - 1].0) == row_delta
                    && signed_delta(terms[stop].1, terms[stop - 1].1) == column_delta
                {
                    stop += 1;
                }
                affine_run_count += 1;
                affine_run_terms += stop - cursor;
                cursor = stop;
                continue;
            }
        }
        literal_count += 1;
        cursor += 1;
    }
    SelectiveProjectedExplicitRunCensus {
        term_count: terms.len(),
        affine_run_count,
        affine_run_terms,
        literal_count,
    }
}

fn public_coordinate_decoder(
    arms: &[SparseR1cs],
    layout: &super::SelectiveLayout,
) -> Result<Vec<SelectiveProjectedPublicCoordinate>, LowNormR1csError> {
    let audit = layout.compiler_audit.layout();
    let logical = audit.logical_public_input_len();
    let public = audit.public_input_len();
    if logical == 0 || public < logical {
        return Err(trace_error("selective public decoder has an invalid public range"));
    }
    if audit
        .public_padding_columns()
        .iter()
        .copied()
        .ne(logical..public)
    {
        return Err(trace_error(
            "selective public decoder padding differs from the emitted public range",
        ));
    }
    for (arm, source) in arms.iter().enumerate() {
        if source.m_in != logical {
            return Err(trace_error(
                "selective public decoder source width differs from the encoded logical prefix",
            ));
        }
        if layout.slots[arm][0].is_some()
            || layout.aliases[arm][0].is_some()
            || layout.equal_aliases[arm][0].is_some()
            || layout.plans[arm].centered[0]
        {
            return Err(trace_error(
                "selective public decoder constant coordinate has source-owned encoding metadata",
            ));
        }
        for field in 1..logical {
            if layout.slots[arm][field] != Some((field, 1))
                || layout.aliases[arm][field].is_some()
                || layout.equal_aliases[arm][field].is_some()
                || layout.plans[arm].centered[field]
            {
                return Err(trace_error(
                    "selective public decoder field is not the canonical direct coordinate",
                ));
            }
        }
    }

    let mut decoded = Vec::with_capacity(public);
    decoded.push(SelectiveProjectedPublicCoordinate {
        column: 0,
        source: SelectiveProjectedPublicCoordinateSource::ConstantOne,
    });
    decoded.extend((1..logical).map(|field| SelectiveProjectedPublicCoordinate {
        column: field,
        source: SelectiveProjectedPublicCoordinateSource::SourceField(field),
    }));
    decoded.extend((logical..public).map(|column| SelectiveProjectedPublicCoordinate {
        column,
        source: SelectiveProjectedPublicCoordinateSource::FixedZero,
    }));
    Ok(decoded)
}

fn project_port(terms: &MatrixTerms, row: usize, columns: usize) -> Result<SelectiveProjectedPort, LowNormR1csError> {
    let mut canonical = BTreeMap::<usize, F>::new();
    let mut add = |column: usize, coefficient: F| -> Result<(), LowNormR1csError> {
        if column >= columns {
            return Err(trace_error("projected selective term exceeds the final column domain"));
        }
        *canonical.entry(column).or_insert(F::ZERO) += coefficient;
        Ok(())
    };

    for &(term_row, column, coefficient) in &terms.explicit {
        if term_row == row {
            add(column, coefficient)?;
        }
    }
    let geometric_runs = terms
        .geometric_runs
        .iter()
        .filter(|run| run.row() == row)
        .map(|run| {
            if run.column_start() + run.len() > columns {
                return Err(trace_error("projected geometric run exceeds the final column domain"));
            }
            Ok(SelectiveProjectedGeometricRun {
                column_start: run.column_start(),
                length: run.len(),
                initial: *run.initial(),
                ratio: *run.ratio(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for block in &terms.seeded {
        if block.row_start() <= row && row < block.row_start() + D * block.kappa() {
            return Err(trace_error(
                "bounded selective projection intersects a compact seeded row",
            ));
        }
    }
    canonical.retain(|_, coefficient| *coefficient != F::ZERO);
    Ok(SelectiveProjectedPort {
        explicit: canonical
            .into_iter()
            .map(|(column, coefficient)| SelectiveProjectedTerm { column, coefficient })
            .collect(),
        geometric_runs,
        seeded_blocks: Vec::new(),
    })
}

fn unique_owner(
    audit: &SelectiveCompilerAudit,
    row: usize,
) -> Result<(usize, &SelectiveEmittedRowRunAudit), LowNormR1csError> {
    let mut owners = audit
        .rows()
        .emitted_runs()
        .iter()
        .enumerate()
        .filter(|(_, run)| !run.emitted_rows().is_empty() && run.emitted_rows().contains(&row));
    let owner = owners
        .next()
        .ok_or_else(|| trace_error("projected selective row has no emitted-run owner"))?;
    if owners.next().is_some() {
        return Err(trace_error("projected selective row has multiple emitted-run owners"));
    }
    Ok(owner)
}

fn project_row_artifact(
    emitted: &structure::EmittedStructureTerms,
    audit: &SelectiveCompilerAudit,
    row: usize,
) -> Result<SelectiveProjectedRowArtifact, LowNormR1csError> {
    let ports = (0..SELECTIVE_ARITY)
        .map(|port| project_port(&emitted.matrix_terms[port], row, emitted.columns))
        .collect::<Result<Vec<_>, _>>()?;
    let ports: [SelectiveProjectedPort; SELECTIVE_ARITY] = ports
        .try_into()
        .expect("the selective port range has compile-time arity");
    let (run_index, owner) = unique_owner(audit, row)?;
    Ok(SelectiveProjectedRowArtifact {
        rows: emitted.rows,
        columns: emitted.columns,
        emitted_row: row,
        run_index,
        family: owner.family(),
        arm: owner.arm(),
        ports,
    })
}

fn source_terms(terms: &[(usize, F)]) -> Vec<SelectiveProjectedSourceTerm> {
    terms
        .iter()
        .map(|&(column, coefficient)| SelectiveProjectedSourceTerm { column, coefficient })
        .collect()
}

fn port_intersects_slot(port: &SelectiveProjectedPort, (start, width): (usize, usize)) -> bool {
    let end = start + width;
    port.explicit
        .iter()
        .any(|term| (start..end).contains(&term.column))
        || port.geometric_runs.iter().any(|run| {
            let run_end = run.column_start + run.length;
            start < run_end && run.column_start < end
        })
}

#[derive(Clone)]
enum PlannedRewriteOutput {
    Source(Lc),
    DerivedProductSum(usize),
}

#[derive(Clone)]
struct PlannedRewriteStep {
    emitted_row: usize,
    rewrite_id: usize,
    kind: super::super::selective_audit::SelectiveRewriteKind,
    source_rows: Vec<(usize, usize)>,
    output: PlannedRewriteOutput,
    base: Lc,
    previous: Option<usize>,
    factors: Vec<ProductFactorTrace>,
}

fn source_column_lc(column: usize) -> Lc {
    Lc {
        terms: vec![(column, F::ONE)],
        constant: F::ZERO,
    }
}

fn product_factor_traces_exact(left: &[ProductFactorTrace], right: &[ProductFactorTrace]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.coefficient == right.coefficient
                && left.left.constant == right.left.constant
                && left.left.terms == right.left.terms
                && left.right.constant == right.right.constant
                && left.right.terms == right.right.terms
        })
}

fn rewrite_geometry(
    layout: &super::SelectiveLayout,
    rewrite_id: super::super::selective_audit::SelectiveRewriteId,
    arm: usize,
    kind: super::super::selective_audit::SelectiveRewriteKind,
) -> Result<(Vec<(usize, usize)>, std::ops::Range<usize>), LowNormR1csError> {
    let rewrite = layout
        .compiler_audit
        .rows()
        .rewrites()
        .get(rewrite_id.index())
        .filter(|rewrite| rewrite.id() == rewrite_id)
        .ok_or_else(|| trace_error("projected rewrite identifier is absent from the compiler ledger"))?;
    if rewrite.arm() != arm || rewrite.kind() != kind {
        return Err(trace_error(
            "projected rewrite metadata differs from its compiler ledger owner",
        ));
    }
    Ok((
        rewrite
            .source_rows()
            .iter()
            .map(|rows| (rows.start, rows.end))
            .collect(),
        rewrite.emitted_rows(),
    ))
}

fn planned_rewrite_steps(
    source_arm: &SparseR1cs,
    layout: &super::SelectiveLayout,
    arm: usize,
) -> Result<Vec<PlannedRewriteStep>, LowNormR1csError> {
    use super::super::selective_audit::SelectiveRewriteKind;

    let prepared = layout.prepared_rows.arm(arm);
    let derived = &layout.derived_product_sums[arm];
    let mut derived_cursor = 0usize;
    let mut steps = Vec::new();

    for (trace_index, trace) in source_arm.polynomial_evaluation_traces().iter().enumerate() {
        let rewrite_id = prepared.polynomial_evaluation_rewrite(trace_index);
        let (source_rows, emitted_rows) =
            rewrite_geometry(layout, rewrite_id, arm, SelectiveRewriteKind::PolynomialEvaluation)?;
        let mut emitted_row = emitted_rows.start;
        for limb in 0..2 {
            let product_indices = (1..trace.coefficient_cols.len()).collect::<Vec<_>>();
            let groups = product_indices.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
            if groups.is_empty() {
                let base = if limb == 0 {
                    source_column_lc(trace.coefficient_cols[0])
                } else {
                    Lc::zero()
                };
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::PolynomialEvaluation,
                    source_rows: source_rows.clone(),
                    output: PlannedRewriteOutput::Source(source_column_lc(trace.output_cols[limb])),
                    base,
                    previous: None,
                    factors: Vec::new(),
                });
                emitted_row += 1;
                continue;
            }
            let mut previous = None;
            for (group_index, group) in groups.iter().enumerate() {
                let final_group = group_index + 1 == groups.len();
                let factors = group
                    .iter()
                    .map(|&term_index| ProductFactorTrace {
                        left: source_column_lc(trace.coefficient_cols[term_index]),
                        right: source_column_lc(trace.power_cols[term_index][limb]),
                        coefficient: F::ONE,
                    })
                    .collect::<Vec<_>>();
                let output = if final_group {
                    PlannedRewriteOutput::Source(source_column_lc(trace.output_cols[limb]))
                } else {
                    let Some(encoding) = derived.get(derived_cursor) else {
                        return Err(trace_error("projected polynomial rewrite exceeds derived-product plan"));
                    };
                    if encoding.previous != previous {
                        return Err(trace_error(
                            "projected polynomial predecessor differs from derived-product plan",
                        ));
                    }
                    if !product_factor_traces_exact(&factors, &encoding.factors) {
                        return Err(trace_error(
                            "projected polynomial factors differ from derived-product witness plan",
                        ));
                    }
                    let output = PlannedRewriteOutput::DerivedProductSum(derived_cursor);
                    derived_cursor += 1;
                    output
                };
                let base = if final_group && limb == 0 {
                    source_column_lc(trace.coefficient_cols[0])
                } else {
                    Lc::zero()
                };
                let next_previous = match &output {
                    PlannedRewriteOutput::DerivedProductSum(index) => Some(*index),
                    PlannedRewriteOutput::Source(_) => None,
                };
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::PolynomialEvaluation,
                    source_rows: source_rows.clone(),
                    output,
                    base,
                    previous,
                    factors,
                });
                previous = next_previous;
                emitted_row += 1;
            }
        }
        if emitted_row != emitted_rows.end {
            return Err(trace_error(
                "projected polynomial step count differs from compiler rewrite interval",
            ));
        }
    }

    for (batch_index, batch) in source_arm.product_sum_batch_traces().iter().enumerate() {
        let rewrite_id = prepared.product_sum_rewrite(batch_index);
        let (source_rows, emitted_rows) = rewrite_geometry(layout, rewrite_id, arm, SelectiveRewriteKind::ProductSum)?;
        let mut emitted_row = emitted_rows.start;
        for identity in &batch.identities {
            if identity.factors.len() <= EVAL_GROUP_SIZE {
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::ProductSum,
                    source_rows: source_rows.clone(),
                    output: PlannedRewriteOutput::Source(identity.result.clone()),
                    base: Lc::zero(),
                    previous: None,
                    factors: identity.factors.clone(),
                });
                emitted_row += 1;
                continue;
            }
            let groups = identity.factors.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
            let mut previous = None;
            for (group_index, group) in groups.iter().enumerate() {
                let final_group = group_index + 1 == groups.len();
                let factors = group.to_vec();
                let output = if final_group {
                    PlannedRewriteOutput::Source(identity.result.clone())
                } else {
                    let Some(encoding) = derived.get(derived_cursor) else {
                        return Err(trace_error(
                            "projected product-sum rewrite exceeds derived-product plan",
                        ));
                    };
                    if encoding.previous != previous {
                        return Err(trace_error(
                            "projected product-sum predecessor differs from derived-product plan",
                        ));
                    }
                    if !product_factor_traces_exact(&factors, &encoding.factors) {
                        return Err(trace_error(
                            "projected product-sum factors differ from derived-product witness plan",
                        ));
                    }
                    let output = PlannedRewriteOutput::DerivedProductSum(derived_cursor);
                    derived_cursor += 1;
                    output
                };
                let next_previous = match &output {
                    PlannedRewriteOutput::DerivedProductSum(index) => Some(*index),
                    PlannedRewriteOutput::Source(_) => None,
                };
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::ProductSum,
                    source_rows: source_rows.clone(),
                    output,
                    base: Lc::zero(),
                    previous,
                    factors,
                });
                previous = next_previous;
                emitted_row += 1;
            }
        }
        if emitted_row != emitted_rows.end {
            return Err(trace_error(
                "projected product-sum step count differs from compiler rewrite interval",
            ));
        }
    }
    if derived_cursor != derived.len() {
        return Err(trace_error(
            "projected rewrite plan did not consume every compiler-derived product sum",
        ));
    }
    Ok(steps)
}

fn verify_rewrite_step(
    step: &PlannedRewriteStep,
    artifact: &SelectiveProjectedRowArtifact,
    layout: &super::SelectiveLayout,
    arm: usize,
) -> Result<(), LowNormR1csError> {
    let emitted_run = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .get(artifact.run_index)
        .ok_or_else(|| trace_error("projected rewrite row has an invalid emitted-run owner"))?;
    if artifact.emitted_row != step.emitted_row
        || emitted_run.rewrite_id().map(|id| id.index()) != Some(step.rewrite_id)
        || emitted_run.family() != artifact.family
        || emitted_run.arm() != Some(arm)
    {
        return Err(trace_error(
            "projected executable rewrite step differs from its emitted-row owner",
        ));
    }

    let mut expected = (0..SELECTIVE_ARITY)
        .map(|_| MatrixTerms::new(false))
        .collect::<Vec<_>>();
    expected[EVAL_SELECTOR].push((0, layout.selector_cols[arm], F::ONE));
    match &step.output {
        PlannedRewriteOutput::Source(output) => {
            append_lc(
                &mut expected[C],
                0,
                output,
                &layout.slots[arm],
                &layout.plans[arm].definitions,
            )?;
        }
        PlannedRewriteOutput::DerivedProductSum(index) => {
            append_slot(
                &mut expected[C],
                0,
                layout.derived_product_sums[arm][*index].slot,
                F::ONE,
            );
        }
    }
    append_lc_scaled(
        &mut expected[C],
        0,
        &step.base,
        -F::ONE,
        &layout.slots[arm],
        &layout.plans[arm].definitions,
    )?;
    if let Some(previous) = step.previous {
        append_slot(
            &mut expected[C],
            0,
            layout.derived_product_sums[arm][previous].slot,
            -F::ONE,
        );
    }
    for (pair_index, factor) in step.factors.iter().enumerate() {
        let (left, right) = EVAL_PAIRS[pair_index];
        append_lc_scaled(
            &mut expected[left],
            0,
            &factor.left,
            factor.coefficient,
            &layout.slots[arm],
            &layout.plans[arm].definitions,
        )?;
        append_lc(
            &mut expected[right],
            0,
            &factor.right,
            &layout.slots[arm],
            &layout.plans[arm].definitions,
        )?;
    }
    for (port, terms) in expected.iter().enumerate() {
        if project_port(terms, 0, artifact.columns)? != artifact.ports[port] {
            return Err(trace_error(
                "projected executable rewrite step does not reproduce its exact emitted row",
            ));
        }
    }
    Ok(())
}

fn projected_factor(factor: &ProductFactorTrace) -> SelectiveProjectedProductFactor {
    SelectiveProjectedProductFactor {
        left_constant: factor.left.constant,
        left_terms: source_terms(&factor.left.terms),
        right_constant: factor.right.constant,
        right_terms: source_terms(&factor.right.terms),
        coefficient: factor.coefficient,
    }
}

fn verify_retained_step(
    step: &SelectiveProjectedRetainedStep,
    artifact: &SelectiveProjectedRowArtifact,
    layout: &super::SelectiveLayout,
    arm: usize,
) -> Result<(), LowNormR1csError> {
    use super::super::selective_audit::SelectiveEmittedRowFamily;

    let emitted_run = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .get(artifact.run_index)
        .ok_or_else(|| trace_error("projected retained row has an invalid emitted-run owner"))?;
    if artifact.emitted_row != step.emitted_row
        || artifact.family != SelectiveEmittedRowFamily::Retained
        || emitted_run.family() != SelectiveEmittedRowFamily::Retained
        || emitted_run.rewrite_id().is_some()
        || emitted_run.arm() != Some(arm)
    {
        return Err(trace_error(
            "projected retained source step differs from its emitted-row owner",
        ));
    }

    let mut expected = (0..SELECTIVE_ARITY)
        .map(|_| MatrixTerms::new(false))
        .collect::<Vec<_>>();
    expected[GENERAL_SELECTOR].push((0, layout.selector_cols[arm], F::ONE));
    for (port, source) in [A, B, C].into_iter().zip(&step.ports) {
        append_field(
            &mut expected[port],
            0,
            0,
            source.constant,
            &layout.slots[arm],
            &layout.plans[arm].definitions,
        )?;
        for term in &source.terms {
            append_field(
                &mut expected[port],
                0,
                term.column,
                term.coefficient,
                &layout.slots[arm],
                &layout.plans[arm].definitions,
            )?;
        }
    }
    for (port, terms) in expected.iter().enumerate() {
        if project_port(terms, 0, artifact.columns)? != artifact.ports[port] {
            return Err(trace_error(
                "projected retained source step does not reproduce its exact emitted row",
            ));
        }
    }
    Ok(())
}

fn source_provenance(
    source_arm: &SparseR1cs,
    layout: &super::SelectiveLayout,
    arm: usize,
    requested_source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
    row_artifacts: &[SelectiveProjectedRowArtifact],
) -> Result<SelectiveProjectedSourceProvenance, LowNormR1csError> {
    let Some(slots) = layout.slots.get(arm) else {
        return Err(trace_error("projected source-provenance arm is out of range"));
    };
    let plan = &layout.plans[arm];
    let (poseidon2_sbox_steps, poseidon2_output_steps) =
        poseidon2::project_steps(source_arm, layout, arm, row_artifacts)?;
    let retained_source_rows = retained_row_pairs
        .iter()
        .map(|&(source_row, _)| source_row)
        .collect::<BTreeSet<_>>();
    let retained_source_ports = row_index::source_rows(source_arm, &retained_source_rows)?;
    let mut closure = requested_source_columns
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for step in &poseidon2_sbox_steps {
        closure.extend(step.input.terms.iter().map(|term| term.column));
        closure.extend(step.output.terms.iter().map(|term| term.column));
    }
    for step in &poseidon2_output_steps {
        closure.extend(step.output.terms.iter().map(|term| term.column));
        closure.extend(step.linear_form.terms.iter().map(|term| term.column));
    }
    for ports in retained_source_ports.values() {
        closure.extend(
            ports
                .iter()
                .flat_map(|port| &port.terms)
                .map(|term| term.column),
        );
    }
    if closure.iter().any(|&column| column >= slots.len()) {
        return Err(trace_error("projected source-provenance column exceeds its source arm"));
    }

    loop {
        let mut added = false;
        for column in closure.iter().copied().collect::<Vec<_>>() {
            let Some(rhs) = plan.definitions.get(column) else {
                continue;
            };
            for &(dependency, _) in &rhs.terms {
                if dependency >= slots.len() {
                    return Err(trace_error(
                        "projected compiler definition references an out-of-range source column",
                    ));
                }
                added |= closure.insert(dependency);
            }
        }
        if !added {
            break;
        }
    }

    let mut retained_slots = Vec::new();
    let mut trace_eliminated_columns = Vec::new();
    for &column in &closure {
        if column == 0 || plan.definitions.get(column).is_some() {
            continue;
        }
        if let Some((start, width)) = slots[column] {
            if width == 0 || start + width > layout.columns.next_multiple_of(D) {
                return Err(trace_error("projected retained source slot is out of range"));
            }
            retained_slots.push(SelectiveProjectedSourceSlot { column, start, width });
        } else {
            trace_eliminated_columns.push(column);
        }
    }

    let linear_definitions = plan
        .definitions
        .entries
        .iter()
        .filter(|definition| closure.contains(&definition.target))
        .map(|definition| SelectiveProjectedSourceDefinition {
            target: definition.target,
            constant: definition.rhs.constant,
            terms: source_terms(&definition.rhs.terms),
        })
        .collect::<Vec<_>>();
    let requested_source_images = requested_source_columns
        .iter()
        .copied()
        .map(|column| {
            let mut terms = MatrixTerms::new(false);
            append_field(&mut terms, 0, column, F::ONE, slots, &plan.definitions)?;
            Ok(SelectiveProjectedSourceImage {
                column,
                port: project_port(&terms, 0, layout.columns.next_multiple_of(D))?,
            })
        })
        .collect::<Result<Vec<_>, LowNormR1csError>>()?;
    let definition_targets = linear_definitions
        .iter()
        .map(SelectiveProjectedSourceDefinition::target)
        .collect::<BTreeSet<_>>();
    if definition_targets.len() != linear_definitions.len()
        || definition_targets
            != closure
                .iter()
                .copied()
                .filter(|&column| plan.definitions.get(column).is_some())
                .collect()
    {
        return Err(trace_error(
            "projected source-provenance definition closure is incomplete",
        ));
    }

    let derived = &layout.derived_product_sums[arm];
    let mut selected_derived = derived
        .iter()
        .enumerate()
        .filter(|(_, encoding)| {
            row_artifacts
                .iter()
                .flat_map(|row| row.ports.iter())
                .any(|port| port_intersects_slot(port, encoding.slot))
        })
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    loop {
        let mut added = false;
        for index in selected_derived.iter().copied().collect::<Vec<_>>() {
            if let Some(previous) = derived[index].previous {
                if previous >= index {
                    return Err(trace_error(
                        "projected derived product predecessor is not earlier in compiler order",
                    ));
                }
                added |= selected_derived.insert(previous);
            }
        }
        if !added {
            break;
        }
    }
    let derived_product_sums = selected_derived
        .into_iter()
        .map(|compiler_index| {
            let encoding = &derived[compiler_index];
            SelectiveProjectedDerivedProductSum {
                compiler_index,
                start: encoding.slot.0,
                width: encoding.slot.1,
                factors: encoding
                    .factors
                    .iter()
                    .map(|factor| SelectiveProjectedProductFactor {
                        left_constant: factor.left.constant,
                        left_terms: source_terms(&factor.left.terms),
                        right_constant: factor.right.constant,
                        right_terms: source_terms(&factor.right.terms),
                        coefficient: factor.coefficient,
                    })
                    .collect(),
                previous: encoding.previous,
            }
        })
        .collect();

    let artifacts_by_row = row_artifacts
        .iter()
        .map(|artifact| (artifact.emitted_row, artifact))
        .collect::<BTreeMap<_, _>>();
    let retained_steps = retained_row_pairs
        .iter()
        .map(|&(source_row, emitted_row)| {
            let artifact = artifacts_by_row
                .get(&emitted_row)
                .copied()
                .ok_or_else(|| trace_error("projected retained emitted row is absent"))?;
            let ports = retained_source_ports
                .get(&source_row)
                .cloned()
                .ok_or_else(|| trace_error("projected retained source row is absent"))?;
            let step = SelectiveProjectedRetainedStep {
                emitted_row,
                source_row,
                ports,
            };
            verify_retained_step(&step, artifact, layout, arm)?;
            Ok(step)
        })
        .collect::<Result<Vec<_>, LowNormR1csError>>()?;
    let expected_retained_rows = row_artifacts
        .iter()
        .filter(|artifact| artifact.family == super::super::selective_audit::SelectiveEmittedRowFamily::Retained)
        .map(|artifact| artifact.emitted_row)
        .collect::<BTreeSet<_>>();
    let actual_retained_rows = retained_steps
        .iter()
        .map(|step| step.emitted_row)
        .collect::<BTreeSet<_>>();
    if retained_steps.len() != retained_row_pairs.len()
        || actual_retained_rows.len() != retained_steps.len()
        || actual_retained_rows != expected_retained_rows
    {
        return Err(trace_error(
            "projected retained source steps do not cover every selected retained row",
        ));
    }
    let selected_steps = planned_rewrite_steps(source_arm, layout, arm)?
        .into_iter()
        .filter(|step| artifacts_by_row.contains_key(&step.emitted_row))
        .collect::<Vec<_>>();
    let expected_step_count = row_artifacts
        .iter()
        .filter(|artifact| {
            matches!(
                artifact.family,
                super::super::selective_audit::SelectiveEmittedRowFamily::PolynomialEvaluation
                    | super::super::selective_audit::SelectiveEmittedRowFamily::ProductSum
            )
        })
        .count();
    if selected_steps.len() != expected_step_count {
        return Err(trace_error(
            "projected executable rewrite steps do not cover every selected rewrite row",
        ));
    }
    for step in &selected_steps {
        if matches!(step.output, PlannedRewriteOutput::DerivedProductSum(_))
            && (step.base.constant != F::ZERO || !step.base.terms.is_empty())
        {
            return Err(trace_error(
                "projected derived-product rewrite has a base term absent from the witness encoding",
            ));
        }
        verify_rewrite_step(step, artifacts_by_row[&step.emitted_row], layout, arm)?;
    }
    let rewrite_steps = selected_steps
        .into_iter()
        .map(|step| SelectiveProjectedRewriteStep {
            emitted_row: step.emitted_row,
            rewrite_id: step.rewrite_id,
            kind: step.kind,
            source_rows: step.source_rows,
            output: match step.output {
                PlannedRewriteOutput::Source(output) => SelectiveProjectedRewriteOutput::Source {
                    constant: output.constant,
                    terms: source_terms(&output.terms),
                },
                PlannedRewriteOutput::DerivedProductSum(compiler_index) => {
                    SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index }
                }
            },
            base_constant: step.base.constant,
            base_terms: source_terms(&step.base.terms),
            previous: step.previous,
            factors: step.factors.iter().map(projected_factor).collect(),
        })
        .collect::<Vec<_>>();

    let source_term_in_closure = |term: &SelectiveProjectedSourceTerm| closure.contains(&term.column);
    if rewrite_steps.iter().any(|step| {
        let output_outside = match &step.output {
            SelectiveProjectedRewriteOutput::Source { terms, .. } => {
                terms.iter().any(|term| !source_term_in_closure(term))
            }
            SelectiveProjectedRewriteOutput::DerivedProductSum { .. } => false,
        };
        output_outside
            || step
                .base_terms
                .iter()
                .any(|term| !source_term_in_closure(term))
            || step.factors.iter().any(|factor| {
                factor
                    .left_terms
                    .iter()
                    .chain(&factor.right_terms)
                    .any(|term| !source_term_in_closure(term))
            })
    }) {
        return Err(trace_error(
            "projected executable rewrite step references a source column outside its closure",
        ));
    }
    if retained_steps.iter().any(|step| {
        step.ports
            .iter()
            .flat_map(|port| &port.terms)
            .any(|term| !closure.contains(&term.column))
    }) {
        return Err(trace_error(
            "projected retained source step references a source column outside its closure",
        ));
    }
    if poseidon2_output_steps.iter().any(|step| {
        step.output
            .terms
            .iter()
            .chain(&step.linear_form.terms)
            .any(|term| !closure.contains(&term.column))
    }) {
        return Err(trace_error(
            "projected Poseidon2 output step references a source column outside its closure",
        ));
    }

    Ok(SelectiveProjectedSourceProvenance {
        arm,
        source_columns: closure.into_iter().collect(),
        retained_slots,
        requested_source_images,
        linear_definitions,
        trace_eliminated_columns,
        poseidon2_sbox_steps,
        poseidon2_output_steps,
        derived_product_sums,
        rewrite_steps,
        retained_steps,
    })
}

pub(crate) fn project_rows_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_inner(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        selected_rows,
        None,
        None,
    )
}

/// Project rows once and decode the exact transitive source-column closure
/// computed by that same provenance pass.
pub(crate) fn project_rows_with_complete_source_provenance_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_arm: usize,
    source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_inner(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        selected_rows,
        Some((source_arm, source_columns, retained_row_pairs, None)),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn project_rows_inner(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_request: Option<(usize, &[usize], &[(usize, usize)], Option<&[usize]>)>,
    decoder_run_request: Option<(usize, std::ops::Range<usize>)>,
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?;
    let public_coordinates = public_coordinate_decoder(arms, &layout)?;
    let emitted = structure::emit_structure_terms(
        arms,
        layout.encoding,
        &layout.plans,
        &layout.slots,
        &layout.aliases,
        &layout.equal_aliases,
        shared_private_fields,
        &layout.derived_product_sums,
        &layout.selector_cols,
        &layout.public_padding_cols,
        &layout.private_padding_cols,
        layout.columns,
        &layout.prepared_rows,
    )?;
    if emitted.rows != layout.compiler_audit.rows().total_rows() {
        return Err(trace_error(
            "projected emitter row count differs from its compiler audit",
        ));
    }
    let explicit_run_census = emitted
        .matrix_terms
        .iter()
        .map(|terms| explicit_run_census(&terms.explicit))
        .collect();

    let mut unique = BTreeSet::new();
    for &row in selected_rows {
        if row >= emitted.rows {
            return Err(trace_error("requested selective projection row is out of range"));
        }
        if !unique.insert(row) {
            return Err(trace_error("requested selective projection row is duplicated"));
        }
    }

    let row_artifacts = row_index::project_rows(&emitted, &layout.compiler_audit, selected_rows)?;

    let public_padding_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::PublicPadding)
        .collect::<Vec<_>>();
    let [public_padding_run] = public_padding_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one public-padding owner",
        ));
    };
    if public_padding_run.arm().is_some() || public_padding_run.emitted_rows().len() != layout.public_padding_cols.len()
    {
        return Err(trace_error(
            "projected public-padding owner differs from the prepared public range",
        ));
    }
    let public_padding_row_artifacts = public_padding_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let selector_domain_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::SelectorDomain)
        .collect::<Vec<_>>();
    let [selector_domain_run] = selector_domain_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one selector-domain owner",
        ));
    };
    if selector_domain_run.arm().is_some() || selector_domain_run.emitted_rows().len() != layout.selector_cols.len() {
        return Err(trace_error(
            "projected selector-domain owner differs from the prepared selector range",
        ));
    }
    let selector_domain_row_artifacts = selector_domain_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let one_hot_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::OneHot)
        .collect::<Vec<_>>();
    let [one_hot_run] = one_hot_runs.as_slice() else {
        return Err(trace_error("projected emitter must have exactly one one-hot owner"));
    };
    if one_hot_run.arm().is_some() || one_hot_run.emitted_rows().len() != 1 {
        return Err(trace_error(
            "projected one-hot owner differs from the prepared selector-total row",
        ));
    }
    let one_hot_row_artifact =
        project_row_artifact(&emitted, &layout.compiler_audit, one_hot_run.emitted_rows().start)?;

    let private_padding_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::PrivatePadding)
        .collect::<Vec<_>>();
    let [private_padding_run] = private_padding_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one private-padding owner",
        ));
    };
    if private_padding_run.arm().is_some()
        || private_padding_run.emitted_rows().len() != layout.private_padding_cols.len()
    {
        return Err(trace_error(
            "projected private-padding owner differs from the prepared alignment range",
        ));
    }
    let private_padding_row_artifacts = private_padding_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let ring_padding_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::RingPadding)
        .collect::<Vec<_>>();
    let [ring_padding_run] = ring_padding_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one ring-padding owner",
        ));
    };
    let expected_ring_padding_rows = layout.compiler_audit.rows().ring_padding_rows();
    if ring_padding_run.arm().is_some()
        || ring_padding_run.emitted_rows() != expected_ring_padding_rows
        || ring_padding_run.emitted_rows().len() != emitted.columns - layout.columns
    {
        return Err(trace_error(
            "projected ring-padding owner differs from the final alignment range",
        ));
    }
    let ring_padding_row_artifacts = ring_padding_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let source_provenance = source_request
        .map(|(arm, source_columns, retained_row_pairs, _)| {
            let source_arm = arms
                .get(arm)
                .ok_or_else(|| trace_error("projected source-provenance arm is out of range"))?;
            source_provenance(
                source_arm,
                &layout,
                arm,
                source_columns,
                retained_row_pairs,
                &row_artifacts,
            )
        })
        .transpose()?;
    let decoder_provenance = match source_request {
        None => None,
        Some((arm, _, _, requested)) => {
            let decoder_source_columns = match requested {
                Some(columns) => columns,
                None => source_provenance
                    .as_ref()
                    .ok_or_else(|| trace_error("projected complete decoder omitted source provenance"))?
                    .source_columns(),
            };
            Some(decoder_provenance(&layout, arm, decoder_source_columns)?)
        }
    };
    let decoder_run_provenance = decoder_run_request
        .map(|(arm, source_range)| {
            let source_arm = arms
                .get(arm)
                .ok_or_else(|| trace_error("complete decoder arm is out of range"))?;
            decoder_run_provenance(&layout, arm, source_range, source_arm)
        })
        .transpose()?;

    Ok(SelectiveProjectedRowsAudit {
        rows: emitted.rows,
        columns: emitted.columns,
        selector_columns: layout.selector_cols,
        compiler_audit: layout.compiler_audit,
        public_coordinates,
        public_padding_row_artifacts,
        selector_domain_row_artifacts,
        one_hot_row_artifact,
        private_padding_row_artifacts,
        ring_padding_row_artifacts,
        row_artifacts,
        source_provenance,
        decoder_provenance,
        decoder_run_provenance,
        explicit_run_census,
    })
}
