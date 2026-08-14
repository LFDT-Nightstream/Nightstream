//! Exact source-family binding to the final selective row stream.

use std::collections::BTreeSet;
use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveCompilerAudit, SelectiveEmittedRowFamily, SelectiveProjectedRowArtifact, SelectiveRewriteKind,
    SelectiveSourceRowDisposition, R1CS_F_PRIME_COMPILER_ID,
};
use p3_field::PrimeField64;
use recursive_constraint_minimizer::Problem;
use sha2::{Digest, Sha256};

use super::{finish_digest, hash_bytes, hash_physical_stages, hash_sparse_matrix, hash_usize, ExportError};

const PLAN_DIGEST_DOMAIN: &[u8] = b"nightstream/selective-fixed-point-plan/v1";
const SLICE_DIGEST_DOMAIN: &[u8] = b"nightstream/selective-fixed-point-slice/v1";

/// One source row copied monotonically into the selective relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveRetainedRowBinding {
    source_row: usize,
    emitted_row: usize,
    stage_occurrence: Option<usize>,
}

impl SelectiveRetainedRowBinding {
    pub fn source_row(&self) -> usize {
        self.source_row
    }

    pub fn emitted_row(&self) -> usize {
        self.emitted_row
    }

    pub fn stage_occurrence(&self) -> Option<usize> {
        self.stage_occurrence
    }
}

/// One complete compiler rewrite touched by the requested source family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveRewriteBinding {
    rewrite_id: usize,
    kind: SelectiveRewriteKind,
    source_rows: Vec<Range<usize>>,
    emitted_rows: Range<usize>,
    stage_occurrence: Option<usize>,
}

impl SelectiveRewriteBinding {
    pub fn rewrite_id(&self) -> usize {
        self.rewrite_id
    }

    pub fn kind(&self) -> SelectiveRewriteKind {
        self.kind
    }

    pub fn source_rows(&self) -> &[Range<usize>] {
        &self.source_rows
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn stage_occurrence(&self) -> Option<usize> {
        self.stage_occurrence
    }
}

/// Exact source-to-final slice binding for one fixed-point branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveSliceBinding {
    branch: R1csIvcBranch,
    requested_source_rows: Vec<usize>,
    closure_source_rows: Vec<usize>,
    additional_source_rows: Vec<usize>,
    retained_rows: Vec<SelectiveRetainedRowBinding>,
    rewrites: Vec<SelectiveRewriteBinding>,
    emitted_rows: Vec<usize>,
    final_rows: usize,
    final_columns: usize,
    final_public_input_count: usize,
    final_plan_digest: String,
    projected_slice_digest: String,
}

impl SelectiveSliceBinding {
    pub fn branch(&self) -> R1csIvcBranch {
        self.branch
    }

    pub fn requested_source_rows(&self) -> &[usize] {
        &self.requested_source_rows
    }

    /// Complete source closure after each touched rewrite is included.
    pub fn closure_source_rows(&self) -> &[usize] {
        &self.closure_source_rows
    }

    /// Rewrite-owned source rows outside the requested semantic family.
    pub fn additional_source_rows(&self) -> &[usize] {
        &self.additional_source_rows
    }

    pub fn retained_rows(&self) -> &[SelectiveRetainedRowBinding] {
        &self.retained_rows
    }

    pub fn rewrites(&self) -> &[SelectiveRewriteBinding] {
        &self.rewrites
    }

    pub fn emitted_rows(&self) -> &[usize] {
        &self.emitted_rows
    }

    pub fn final_rows(&self) -> usize {
        self.final_rows
    }

    pub fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub fn final_public_input_count(&self) -> usize {
        self.final_public_input_count
    }

    /// SHA-256 identity of the complete source arms and selective compiler
    /// plan. This is diagnostic and is not a protocol hash.
    pub fn final_plan_digest(&self) -> &str {
        &self.final_plan_digest
    }

    /// SHA-256 identity of this mapping and its exact projected row terms.
    /// This is diagnostic and is not a protocol hash.
    pub fn projected_slice_digest(&self) -> &str {
        &self.projected_slice_digest
    }
}

/// A cvc5 source problem with an exact binding to final selective rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedPointProblemExport {
    problem: Problem,
    binding: SelectiveSliceBinding,
}

impl FixedPointProblemExport {
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn binding(&self) -> &SelectiveSliceBinding {
        &self.binding
    }

    pub fn into_problem(self) -> Problem {
        self.problem
    }
}

pub(super) fn bind_fixed_point_problem(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
    problem: Problem,
) -> Result<FixedPointProblemExport, ExportError> {
    let source_rows = problem
        .rows
        .iter()
        .map(|row| row.source_index)
        .collect::<Vec<_>>();
    let arm = audit.arm(branch);
    if problem.source.total_rows != arm.n
        || problem.column_count != arm.m
        || problem.public_input_count != arm.m_in
        || source_rows.is_empty()
    {
        return Err(ExportError::new(
            "fixed-point problem geometry differs from its source arm",
        ));
    }

    let mapping = audit
        .fixed_point()
        .rows()
        .arms()
        .get(branch_index(branch))
        .ok_or_else(|| ExportError::new("fixed-point compiler audit omitted the requested arm"))?;
    validate_source_partition(mapping.source_runs(), arm.n)?;

    let mut retained_rows = Vec::new();
    let mut touched_rewrites = BTreeSet::new();
    let mut source_run_cursor = 0usize;
    for &source_row in &source_rows {
        while source_run_cursor < mapping.source_runs().len()
            && source_row >= mapping.source_runs()[source_run_cursor].source_rows().end
        {
            source_run_cursor += 1;
        }
        let run = mapping
            .source_runs()
            .get(source_run_cursor)
            .ok_or_else(|| ExportError::new(format!("source row {source_row} has no compiler disposition")))?;
        let run_rows = run.source_rows();
        if !run_rows.contains(&source_row) {
            return Err(ExportError::new(format!(
                "source row {source_row} has no compiler disposition"
            )));
        }
        match run.disposition() {
            SelectiveSourceRowDisposition::Retained => {
                let emitted_start = run.emitted_start().ok_or_else(|| {
                    ExportError::new(format!("retained source row {source_row} has no emitted start"))
                })?;
                let emitted_row = emitted_start + source_row - run_rows.start;
                validate_retained_owner(audit, branch, emitted_row)?;
                retained_rows.push(SelectiveRetainedRowBinding {
                    source_row,
                    emitted_row,
                    stage_occurrence: run.stage_occurrence(),
                });
            }
            disposition => {
                let rewrite_id = disposition
                    .rewrite_id()
                    .ok_or_else(|| ExportError::new("rewritten source row omitted its rewrite identifier"))?;
                touched_rewrites.insert(rewrite_id.index());
            }
        }
    }

    let requested = source_rows.iter().copied().collect::<BTreeSet<_>>();
    let mut closure = requested.clone();
    let mut rewrites = Vec::with_capacity(touched_rewrites.len());
    for rewrite_id in touched_rewrites {
        let records = audit
            .fixed_point()
            .rows()
            .rewrites()
            .iter()
            .filter(|rewrite| rewrite.id().index() == rewrite_id)
            .collect::<Vec<_>>();
        let [rewrite] = records.as_slice() else {
            return Err(ExportError::new(format!(
                "rewrite identifier {rewrite_id} does not name exactly one compiler record"
            )));
        };
        if rewrite.arm() != branch_index(branch) {
            return Err(ExportError::new(format!(
                "rewrite identifier {rewrite_id} belongs to arm {}, not {:?}",
                rewrite.arm(),
                branch
            )));
        }
        validate_rewrite_runs(mapping.source_runs(), rewrite_id, rewrite.kind(), rewrite.source_rows())?;
        for range in rewrite.source_rows() {
            closure.extend(range.clone());
        }
        validate_rewrite_owner(audit, rewrite_id, rewrite.kind(), rewrite.emitted_rows())?;
        rewrites.push(SelectiveRewriteBinding {
            rewrite_id,
            kind: rewrite.kind(),
            source_rows: rewrite.source_rows().to_vec(),
            emitted_rows: rewrite.emitted_rows(),
            stage_occurrence: rewrite.source_stage_occurrence(),
        });
    }

    let mut emitted = retained_rows
        .iter()
        .map(SelectiveRetainedRowBinding::emitted_row)
        .collect::<BTreeSet<_>>();
    for rewrite in &rewrites {
        for row in rewrite.emitted_rows() {
            if !emitted.insert(row) {
                return Err(ExportError::new(format!(
                    "final emitted row {row} has more than one source binding"
                )));
            }
        }
    }
    let emitted_rows = emitted.into_iter().collect::<Vec<_>>();
    let projected = audit
        .audit_selective_rows(&emitted_rows)
        .map_err(|error| ExportError::new(format!("cannot project exact selective rows: {error}")))?;
    if projected.rows() != audit.fixed_point().rows().total_rows()
        || projected.columns() != audit.fixed_point().layout().total_columns()
        || projected
            .row_artifacts()
            .iter()
            .map(SelectiveProjectedRowArtifact::emitted_row)
            .ne(emitted_rows.iter().copied())
    {
        return Err(ExportError::new(
            "projected selective rows differ from the fixed-point binding",
        ));
    }

    let closure_source_rows = closure.iter().copied().collect::<Vec<_>>();
    let additional_source_rows = closure.difference(&requested).copied().collect::<Vec<_>>();
    let final_plan_digest = hash_final_plan(audit, projected.compiler_audit())?;
    let projected_slice_digest = hash_projected_slice(
        branch,
        &problem.source.artifact_digest,
        &source_rows,
        &closure_source_rows,
        &retained_rows,
        &rewrites,
        projected.row_artifacts(),
        &final_plan_digest,
    )?;
    let binding = SelectiveSliceBinding {
        branch,
        requested_source_rows: source_rows,
        closure_source_rows,
        additional_source_rows,
        retained_rows,
        rewrites,
        emitted_rows,
        final_rows: projected.rows(),
        final_columns: projected.columns(),
        final_public_input_count: projected.compiler_audit().layout().public_input_len(),
        final_plan_digest,
        projected_slice_digest,
    };
    Ok(FixedPointProblemExport { problem, binding })
}

fn validate_source_partition(
    runs: &[neo_fold_clean::frontends::r1cs_f_prime::SelectiveSourceRowRunAudit],
    source_rows: usize,
) -> Result<(), ExportError> {
    let mut cursor = 0usize;
    for (index, run) in runs.iter().enumerate() {
        let rows = run.source_rows();
        if rows.start != cursor || rows.is_empty() {
            return Err(ExportError::new(format!(
                "compiler source run {index} does not continue a nonempty exact partition"
            )));
        }
        cursor = rows.end;
    }
    if cursor != source_rows {
        return Err(ExportError::new(format!(
            "compiler source partition ends at {cursor}, expected {source_rows}"
        )));
    }
    Ok(())
}

fn validate_rewrite_runs(
    runs: &[neo_fold_clean::frontends::r1cs_f_prime::SelectiveSourceRowRunAudit],
    rewrite_id: usize,
    kind: SelectiveRewriteKind,
    recorded_ranges: &[Range<usize>],
) -> Result<(), ExportError> {
    let recorded_rows = recorded_ranges
        .iter()
        .flat_map(|range| range.clone())
        .collect::<BTreeSet<_>>();
    let mut run_rows = BTreeSet::new();
    for run in runs {
        if run.disposition().rewrite_id().map(|id| id.index()) == Some(rewrite_id) {
            if disposition_kind(run.disposition()) != Some(kind) {
                return Err(ExportError::new(format!(
                    "rewrite identifier {rewrite_id} has inconsistent source dispositions"
                )));
            }
            run_rows.extend(run.source_rows());
        }
    }
    if run_rows != recorded_rows || run_rows.is_empty() {
        return Err(ExportError::new(format!(
            "rewrite identifier {rewrite_id} source runs differ from its rewrite record"
        )));
    }
    Ok(())
}

fn validate_retained_owner(
    audit: &R1csIvcConstraintSourceAudit,
    branch: R1csIvcBranch,
    emitted_row: usize,
) -> Result<(), ExportError> {
    let owners = audit
        .fixed_point()
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.emitted_rows().contains(&emitted_row))
        .collect::<Vec<_>>();
    let [owner] = owners.as_slice() else {
        return Err(ExportError::new(format!(
            "retained emitted row {emitted_row} does not have exactly one final owner"
        )));
    };
    if owner.family() != SelectiveEmittedRowFamily::Retained
        || owner.arm() != Some(branch_index(branch))
        || owner.rewrite_id().is_some()
    {
        return Err(ExportError::new(format!(
            "retained emitted row {emitted_row} has the wrong final owner"
        )));
    }
    Ok(())
}

fn validate_rewrite_owner(
    audit: &R1csIvcConstraintSourceAudit,
    rewrite_id: usize,
    kind: SelectiveRewriteKind,
    emitted_rows: Range<usize>,
) -> Result<(), ExportError> {
    let expected_family = rewrite_emitted_family(kind);
    if emitted_rows.is_empty() {
        if kind != SelectiveRewriteKind::LinearDefinition {
            return Err(ExportError::new(format!(
                "nonlinear rewrite identifier {rewrite_id} emits no final rows"
            )));
        }
        return Ok(());
    }
    let runs = audit
        .fixed_point()
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.rewrite_id().map(|id| id.index()) == Some(rewrite_id))
        .collect::<Vec<_>>();
    let [run] = runs.as_slice() else {
        return Err(ExportError::new(format!(
            "rewrite identifier {rewrite_id} does not have exactly one emitted run"
        )));
    };
    if run.emitted_rows() != emitted_rows || Some(run.family()) != expected_family {
        return Err(ExportError::new(format!(
            "rewrite identifier {rewrite_id} has inconsistent final ownership"
        )));
    }
    Ok(())
}

fn disposition_kind(disposition: SelectiveSourceRowDisposition) -> Option<SelectiveRewriteKind> {
    match disposition {
        SelectiveSourceRowDisposition::Retained => None,
        SelectiveSourceRowDisposition::Poseidon2(_) => Some(SelectiveRewriteKind::Poseidon2),
        SelectiveSourceRowDisposition::CenteredUnit(_) => Some(SelectiveRewriteKind::CenteredUnit),
        SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => {
            Some(SelectiveRewriteKind::ShiftedTernaryCanonical)
        }
        SelectiveSourceRowDisposition::PolynomialEvaluation(_) => Some(SelectiveRewriteKind::PolynomialEvaluation),
        SelectiveSourceRowDisposition::ProductSum(_) => Some(SelectiveRewriteKind::ProductSum),
        SelectiveSourceRowDisposition::LinearDefinition(_) => Some(SelectiveRewriteKind::LinearDefinition),
    }
}

fn rewrite_emitted_family(kind: SelectiveRewriteKind) -> Option<SelectiveEmittedRowFamily> {
    match kind {
        SelectiveRewriteKind::Poseidon2 => Some(SelectiveEmittedRowFamily::Poseidon2),
        SelectiveRewriteKind::CenteredUnit => Some(SelectiveEmittedRowFamily::CenteredUnit),
        SelectiveRewriteKind::ShiftedTernaryCanonical => Some(SelectiveEmittedRowFamily::ShiftedTernaryCanonical),
        SelectiveRewriteKind::PolynomialEvaluation => Some(SelectiveEmittedRowFamily::PolynomialEvaluation),
        SelectiveRewriteKind::ProductSum => Some(SelectiveEmittedRowFamily::ProductSum),
        SelectiveRewriteKind::LinearDefinition => None,
    }
}

fn hash_final_plan(
    audit: &R1csIvcConstraintSourceAudit,
    compiler: &SelectiveCompilerAudit,
) -> Result<String, ExportError> {
    let mut hasher = Sha256::new();
    hasher.update(PLAN_DIGEST_DOMAIN);
    hash_bytes(&mut hasher, R1CS_F_PRIME_COMPILER_ID.as_bytes())?;
    for branch in [
        R1csIvcBranch::Base,
        R1csIvcBranch::BootstrapRecursive,
        R1csIvcBranch::Recursive,
    ] {
        let arm = audit.arm(branch);
        hash_usize(&mut hasher, branch_index(branch))?;
        hash_usize(&mut hasher, arm.n)?;
        hash_usize(&mut hasher, arm.m)?;
        hash_usize(&mut hasher, arm.m_in)?;
        hash_physical_stages(&mut hasher, arm.physical_stage_ranges())?;
        hash_sparse_matrix(&mut hasher, 0, &arm.a)?;
        hash_sparse_matrix(&mut hasher, 1, &arm.b)?;
        hash_sparse_matrix(&mut hasher, 2, &arm.c)?;
    }
    hash_compiler_plan(&mut hasher, compiler)?;
    Ok(finish_digest(hasher))
}

fn hash_compiler_plan(hasher: &mut Sha256, compiler: &SelectiveCompilerAudit) -> Result<(), ExportError> {
    let layout = compiler.layout();
    for value in [
        layout.logical_public_input_len(),
        layout.public_input_len(),
        layout.total_columns(),
    ] {
        hash_usize(hasher, value)?;
    }
    hash_usize_slice(hasher, layout.public_padding_columns())?;
    hash_usize_slice(hasher, layout.selector_columns())?;
    hash_usize_slice(hasher, layout.private_alignment_padding_columns())?;
    hash_range(hasher, layout.shared_private_columns())?;
    hash_range(hasher, layout.branch_columns())?;
    hash_range(hasher, layout.ring_alignment_padding_columns())?;

    let rows = compiler.rows();
    hash_range(hasher, rows.prefix_rows())?;
    hash_usize(hasher, rows.arms().len())?;
    for arm in rows.arms() {
        hash_range(hasher, arm.retained_emitted_rows())?;
        hash_range(hasher, arm.emitted_rows())?;
        hash_optional_usize(hasher, arm.centered_domain_pair_row())?;
        hash_optional_usize(hasher, arm.centered_domain_tail_row())?;
        hash_usize(hasher, arm.source_runs().len())?;
        for run in arm.source_runs() {
            hash_range(hasher, run.source_rows())?;
            hash_usize(hasher, disposition_tag(run.disposition()))?;
            hash_optional_usize(hasher, run.disposition().rewrite_id().map(|id| id.index()))?;
            hash_optional_usize(hasher, run.stage_occurrence())?;
            hash_optional_usize(hasher, run.emitted_start())?;
        }
    }
    hash_range(hasher, rows.ring_padding_rows())?;
    hash_usize(hasher, rows.emitted_runs().len())?;
    for run in rows.emitted_runs() {
        hash_range(hasher, run.emitted_rows())?;
        hash_usize(hasher, emitted_family_tag(run.family()))?;
        hash_optional_usize(hasher, run.arm())?;
        hash_optional_usize(hasher, run.rewrite_id().map(|id| id.index()))?;
        hash_optional_usize(hasher, run.source_stage_occurrence())?;
    }
    hash_usize(hasher, rows.rewrites().len())?;
    for rewrite in rows.rewrites() {
        hash_usize(hasher, rewrite.id().index())?;
        hash_usize(hasher, rewrite.arm())?;
        hash_usize(hasher, rewrite_kind_tag(rewrite.kind()))?;
        hash_ranges(hasher, rewrite.source_rows())?;
        hash_range(hasher, rewrite.emitted_rows())?;
        hash_optional_usize(hasher, rewrite.source_stage_occurrence())?;
    }
    hash_usize(hasher, rows.total_rows())?;

    hash_usize(hasher, compiler.canonical_openings().len())?;
    for arm in compiler.canonical_openings() {
        hash_usize(hasher, arm.len())?;
        for opening in arm {
            hash_usize(hasher, opening.source_field())?;
            hash_usize_slice(hasher, opening.digit_coordinates())?;
            hash_usize_slice(hasher, opening.borrow_coordinates())?;
            hash_range(hasher, opening.emitted_rows())?;
        }
    }
    hash_usize(hasher, compiler.source_arm_physical_stages().len())?;
    for stages in compiler.source_arm_physical_stages() {
        hash_physical_stages(hasher, stages)?;
    }
    hash_usize(hasher, compiler.source_arm_linear_definitions().len())?;
    for definitions in compiler.source_arm_linear_definitions() {
        hash_usize(hasher, definitions.len())?;
        for definition in definitions {
            hash_optional_usize(hasher, definition.source_row())?;
            hash_usize(hasher, definition.target())?;
            hash_field(hasher, definition.constant());
            hash_usize(hasher, definition.terms().len())?;
            for term in definition.terms() {
                hash_usize(hasher, term.column())?;
                hash_field(hasher, term.coefficient());
            }
        }
    }
    hash_usize(hasher, compiler.first_accepted_selections().len())?;
    for selection in compiler.first_accepted_selections() {
        hash_usize(hasher, selection.arm())?;
        hash_usize(hasher, selection.rewrite_id().index())?;
        hash_usize(hasher, selection.stage_occurrence())?;
        hash_range(hasher, selection.source_rows())?;
        hash_range(hasher, selection.emitted_rows())?;
        hash_usize(hasher, selection.position())?;
        for values in [
            selection.selectors(),
            selection.accepts(),
            selection.prefixes(),
            selection.symbols(),
            selection.accepted_products(),
            selection.prefix_products(),
            selection.symbol_products(),
        ] {
            hash_usize_slice(hasher, values)?;
        }
        hash_usize(hasher, selection.output())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_projected_slice(
    branch: R1csIvcBranch,
    source_artifact_digest: &str,
    requested_source_rows: &[usize],
    closure_source_rows: &[usize],
    retained_rows: &[SelectiveRetainedRowBinding],
    rewrites: &[SelectiveRewriteBinding],
    projected_rows: &[SelectiveProjectedRowArtifact],
    final_plan_digest: &str,
) -> Result<String, ExportError> {
    let mut hasher = Sha256::new();
    hasher.update(SLICE_DIGEST_DOMAIN);
    hash_usize(&mut hasher, branch_index(branch))?;
    hash_bytes(&mut hasher, source_artifact_digest.as_bytes())?;
    hash_bytes(&mut hasher, final_plan_digest.as_bytes())?;
    hash_usize_slice(&mut hasher, requested_source_rows)?;
    hash_usize_slice(&mut hasher, closure_source_rows)?;
    hash_usize(&mut hasher, retained_rows.len())?;
    for retained in retained_rows {
        hash_usize(&mut hasher, retained.source_row)?;
        hash_usize(&mut hasher, retained.emitted_row)?;
        hash_optional_usize(&mut hasher, retained.stage_occurrence)?;
    }
    hash_usize(&mut hasher, rewrites.len())?;
    for rewrite in rewrites {
        hash_usize(&mut hasher, rewrite.rewrite_id)?;
        hash_usize(&mut hasher, rewrite_kind_tag(rewrite.kind))?;
        hash_ranges(&mut hasher, &rewrite.source_rows)?;
        hash_range(&mut hasher, rewrite.emitted_rows.clone())?;
        hash_optional_usize(&mut hasher, rewrite.stage_occurrence)?;
    }
    hash_usize(&mut hasher, projected_rows.len())?;
    for row in projected_rows {
        hash_usize(&mut hasher, row.schema_version() as usize)?;
        hash_usize(&mut hasher, row.rows())?;
        hash_usize(&mut hasher, row.columns())?;
        hash_usize(&mut hasher, row.emitted_row())?;
        hash_usize(&mut hasher, row.run_index())?;
        hash_usize(&mut hasher, emitted_family_tag(row.family()))?;
        hash_optional_usize(&mut hasher, row.arm())?;
        hash_usize(&mut hasher, row.ports().len())?;
        for port in row.ports() {
            hash_usize(&mut hasher, port.explicit().len())?;
            for term in port.explicit() {
                hash_usize(&mut hasher, term.column())?;
                hash_field(&mut hasher, term.coefficient());
            }
            hash_usize(&mut hasher, port.geometric_runs().len())?;
            for run in port.geometric_runs() {
                hash_usize(&mut hasher, run.column_start())?;
                hash_usize(&mut hasher, run.length())?;
                hash_field(&mut hasher, run.initial());
                hash_field(&mut hasher, run.ratio());
            }
        }
    }
    Ok(finish_digest(hasher))
}

fn hash_ranges(hasher: &mut Sha256, ranges: &[Range<usize>]) -> Result<(), ExportError> {
    hash_usize(hasher, ranges.len())?;
    for range in ranges {
        hash_range(hasher, range.clone())?;
    }
    Ok(())
}

fn hash_range(hasher: &mut Sha256, range: Range<usize>) -> Result<(), ExportError> {
    hash_usize(hasher, range.start)?;
    hash_usize(hasher, range.end)
}

fn hash_usize_slice(hasher: &mut Sha256, values: &[usize]) -> Result<(), ExportError> {
    hash_usize(hasher, values.len())?;
    for &value in values {
        hash_usize(hasher, value)?;
    }
    Ok(())
}

fn hash_optional_usize(hasher: &mut Sha256, value: Option<usize>) -> Result<(), ExportError> {
    match value {
        Some(value) => {
            hasher.update([1]);
            hash_usize(hasher, value)
        }
        None => {
            hasher.update([0]);
            Ok(())
        }
    }
}

fn hash_field(hasher: &mut Sha256, value: neo_math::F) {
    hasher.update(value.as_canonical_u64().to_le_bytes());
}

fn disposition_tag(disposition: SelectiveSourceRowDisposition) -> usize {
    match disposition {
        SelectiveSourceRowDisposition::Retained => 0,
        SelectiveSourceRowDisposition::Poseidon2(_) => 1,
        SelectiveSourceRowDisposition::CenteredUnit(_) => 2,
        SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => 3,
        SelectiveSourceRowDisposition::PolynomialEvaluation(_) => 4,
        SelectiveSourceRowDisposition::ProductSum(_) => 5,
        SelectiveSourceRowDisposition::LinearDefinition(_) => 6,
    }
}

fn rewrite_kind_tag(kind: SelectiveRewriteKind) -> usize {
    match kind {
        SelectiveRewriteKind::Poseidon2 => 0,
        SelectiveRewriteKind::CenteredUnit => 1,
        SelectiveRewriteKind::ShiftedTernaryCanonical => 2,
        SelectiveRewriteKind::PolynomialEvaluation => 3,
        SelectiveRewriteKind::ProductSum => 4,
        SelectiveRewriteKind::LinearDefinition => 5,
    }
}

fn emitted_family_tag(family: SelectiveEmittedRowFamily) -> usize {
    match family {
        SelectiveEmittedRowFamily::SelectorDomain => 0,
        SelectiveEmittedRowFamily::SharedDomain => 1,
        SelectiveEmittedRowFamily::ArmDomain => 2,
        SelectiveEmittedRowFamily::OneHot => 3,
        SelectiveEmittedRowFamily::PublicPadding => 4,
        SelectiveEmittedRowFamily::PrivatePadding => 5,
        SelectiveEmittedRowFamily::Retained => 6,
        SelectiveEmittedRowFamily::Poseidon2 => 7,
        SelectiveEmittedRowFamily::CenteredUnit => 8,
        SelectiveEmittedRowFamily::ShiftedTernaryCanonical => 9,
        SelectiveEmittedRowFamily::PolynomialEvaluation => 10,
        SelectiveEmittedRowFamily::ProductSum => 11,
        SelectiveEmittedRowFamily::RingPadding => 12,
    }
}

fn branch_index(branch: R1csIvcBranch) -> usize {
    match branch {
        R1csIvcBranch::Base => 0,
        R1csIvcBranch::BootstrapRecursive => 1,
        R1csIvcBranch::Recursive => 2,
    }
}
