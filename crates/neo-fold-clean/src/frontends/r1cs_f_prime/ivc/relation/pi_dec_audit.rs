//! Bounded exact source/selective projection for the active strict PiDEC slice.
//!
//! This module validates the concrete active carrier and its complete source
//! row ownership before asking the production selective emitter for exact row
//! and decoder provenance. It is diagnostic evidence, not acceptance
//! authority and not permission to remove a constraint family.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use neo_ccs::CcsMatrix;
use neo_math::{D, F, K};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

use super::{compress_source_rows, source_row_owner, R1csIvcBranch, R1csIvcRelation};
use crate::engine::r1cs_circuit::builder::{PiDecClaimAudit, PiDecCommitmentAudit, PiDecStrictAudit, RowFamilyRange};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::ivc::{R1csIvcError, R1csIvcFixedPointShapeAudit};
use crate::frontends::r1cs_f_prime::{
    R1csShape, SelectiveProjectedRowsAudit, SelectiveSourceRowDisposition, SparseR1cs,
};
use crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use crate::paper::params::Params;
use crate::paper::reductions::pi_dec_circuit::stage;
use crate::paper::relations::superneo_public_x_cols;

const ACTIVE_CHILDREN: usize = 16;
const ACTIVE_CLAIMS: usize = ACTIVE_CHILDREN + 1;
const ACTIVE_MATRICES: usize = 14;
const ACTIVE_ROW_POINT: usize = 24;
const ACTIVE_LOGICAL_X: usize = 270;
const ACTIVE_RING_DIMENSION: usize = 54;
const ACTIVE_NONCOMMITMENT_SOURCE_ROWS: usize = 12_790;
const ACTIVE_X_RECOMPOSITION_ROWS: usize = ACTIVE_LOGICAL_X;
const ACTIVE_X_CANONICALITY_ROWS: usize = ACTIVE_LOGICAL_X * (2 + ACTIVE_CHILDREN);
const ACTIVE_CANONICAL_X_SOURCE_ROWS: usize = ACTIVE_X_RECOMPOSITION_ROWS + ACTIVE_X_CANONICALITY_ROWS;
const ACTIVE_CANONICAL_X_SOURCE_COLUMNS: usize = 1 + ACTIVE_CLAIMS * ACTIVE_LOGICAL_X + 2 * ACTIVE_LOGICAL_X;

/// Exact bounded projection of the outer steady-recursive `nifs.pi_dec`
/// source rows. The distinct PiDEC invocation inside PiCCS running-parent
/// continuity is intentionally outside this ownership slice.
#[derive(Debug)]
pub struct R1csIvcPiDecSelectiveRowsAudit {
    source: R1csIvcPiDecSourceRowsAudit,
    projected_rows: SelectiveProjectedRowsAudit,
}

/// Exact selective projection of only the strict-PiDEC public-X
/// recomposition and uniform-sign canonicality leaves.
///
/// `semantic_source_ranges` names the 4,590 source equations whose exact
/// coefficients establish the public split. `source_rows` additionally
/// contains every complete compiler rewrite intersecting those equations.
/// Stage names locate the two leaves; the recovered A/B/C rows and compiler
/// provenance remain the authority.
#[derive(Debug)]
pub struct R1csIvcPiDecCanonicalXSelectiveRowsAudit {
    source: R1csIvcPiDecSourceRowsAudit,
    projected_rows: SelectiveProjectedRowsAudit,
    semantic_source_ranges: [RowFamilyRange; 2],
    semantic_source_columns: Vec<usize>,
}

/// Exact active source layout and rewrite expansion, without materializing
/// the expensive final selective row coefficients.
#[derive(Debug)]
pub struct R1csIvcPiDecSourceRowsAudit {
    fixed_point: R1csIvcFixedPointShapeAudit,
    strict: PiDecStrictAudit,
    leaf_source_ranges: Vec<RowFamilyRange>,
    source_rows: Vec<usize>,
    source_row_ranges: Vec<Range<usize>>,
    source_row_artifacts: Vec<R1csIvcPiDecSourceRowAudit>,
}

/// One exact normalized source-R1CS equation in the strict-PiDEC projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcPiDecSourceRowAudit {
    index: usize,
    ports: [Vec<(usize, F)>; 3],
}

impl R1csIvcPiDecSourceRowAudit {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn a(&self) -> &[(usize, F)] {
        &self.ports[0]
    }

    pub fn b(&self) -> &[(usize, F)] {
        &self.ports[1]
    }

    pub fn c(&self) -> &[(usize, F)] {
        &self.ports[2]
    }
}

impl R1csIvcPiDecSelectiveRowsAudit {
    pub fn source(&self) -> &R1csIvcPiDecSourceRowsAudit {
        &self.source
    }

    pub fn fixed_point(&self) -> &R1csIvcFixedPointShapeAudit {
        self.source.fixed_point()
    }

    pub fn projected_rows(&self) -> &SelectiveProjectedRowsAudit {
        &self.projected_rows
    }

    pub fn strict(&self) -> &PiDecStrictAudit {
        self.source.strict()
    }

    /// Exact source-row interval for each entry of `pi_dec_circuit::stage::LEAVES`.
    pub fn leaf_source_ranges(&self) -> &[RowFamilyRange] {
        self.source.leaf_source_ranges()
    }

    /// Sorted source rows after expanding every intersecting compiler rewrite.
    pub fn source_rows(&self) -> &[usize] {
        self.source.source_rows()
    }

    pub fn source_row_ranges(&self) -> &[Range<usize>] {
        self.source.source_row_ranges()
    }

    pub fn source_row_artifacts(&self) -> &[R1csIvcPiDecSourceRowAudit] {
        self.source.source_row_artifacts()
    }
}

impl R1csIvcPiDecCanonicalXSelectiveRowsAudit {
    pub fn fixed_point(&self) -> &R1csIvcFixedPointShapeAudit {
        self.source.fixed_point()
    }

    pub fn projected_rows(&self) -> &SelectiveProjectedRowsAudit {
        &self.projected_rows
    }

    pub fn strict(&self) -> &PiDecStrictAudit {
        self.source.strict()
    }

    /// In order: exact source ranges for `recomposition.x` and `alphabet`.
    pub fn semantic_source_ranges(&self) -> &[RowFamilyRange; 2] {
        &self.semantic_source_ranges
    }

    /// Constant one, all ordered parent/child active-X columns, and all 270
    /// `[sign, centered-product]` trace pairs. This is exactly 4,591 distinct
    /// columns for the active profile.
    pub fn semantic_source_columns(&self) -> &[usize] {
        &self.semantic_source_columns
    }

    /// Sorted source rows after complete expansion of every compiler rewrite
    /// intersecting the two semantic leaves.
    pub fn source_rows(&self) -> &[usize] {
        self.source.source_rows()
    }

    pub fn source_row_ranges(&self) -> &[Range<usize>] {
        self.source.source_row_ranges()
    }

    pub fn source_row_artifacts(&self) -> &[R1csIvcPiDecSourceRowAudit] {
        self.source.source_row_artifacts()
    }
}

impl R1csIvcPiDecSourceRowsAudit {
    pub fn fixed_point(&self) -> &R1csIvcFixedPointShapeAudit {
        &self.fixed_point
    }

    pub fn strict(&self) -> &PiDecStrictAudit {
        &self.strict
    }

    pub fn leaf_source_ranges(&self) -> &[RowFamilyRange] {
        &self.leaf_source_ranges
    }

    pub fn source_rows(&self) -> &[usize] {
        &self.source_rows
    }

    pub fn source_row_ranges(&self) -> &[Range<usize>] {
        &self.source_row_ranges
    }

    pub fn source_row_artifacts(&self) -> &[R1csIvcPiDecSourceRowAudit] {
        &self.source_row_artifacts
    }
}

struct PreparedPiDecSourceAudit {
    candidate: super::FixedPointCandidate,
    strict: PiDecStrictAudit,
    leaf_source_ranges: Vec<RowFamilyRange>,
    source_rows: Vec<usize>,
    source_row_ranges: Vec<Range<usize>>,
    source_row_artifacts: Vec<R1csIvcPiDecSourceRowAudit>,
    selected_rows: Vec<usize>,
    retained_row_pairs: Vec<(usize, usize)>,
    source_columns: Vec<usize>,
}

struct PreparedPiDecContext {
    candidate: super::FixedPointCandidate,
    strict: PiDecStrictAudit,
    leaf_source_ranges: Vec<RowFamilyRange>,
}

struct PreparedPiDecCanonicalXAudit {
    source: PreparedPiDecSourceAudit,
    semantic_source_ranges: [RowFamilyRange; 2],
    semantic_source_columns: Vec<usize>,
}

impl PreparedPiDecSourceAudit {
    fn into_source(self) -> R1csIvcPiDecSourceRowsAudit {
        let fixed_point = R1csIvcFixedPointShapeAudit::new(
            self.candidate.rounds,
            self.candidate.shape.compiler_audit,
            self.candidate.pi_ccs_output_digest,
        );
        R1csIvcPiDecSourceRowsAudit {
            fixed_point,
            strict: self.strict,
            leaf_source_ranges: self.leaf_source_ranges,
            source_rows: self.source_rows,
            source_row_ranges: self.source_row_ranges,
            source_row_artifacts: self.source_row_artifacts,
        }
    }
}

impl R1csIvcRelation {
    /// Validate the active outer PiDEC carrier, exact leaf partition, source
    /// equations, and complete compiler-rewrite expansion without invoking
    /// the expensive final selective term emitter.
    pub fn audit_fixed_point_pi_dec_source_rows(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcPiDecSourceRowsAudit, R1csIvcError> {
        Ok(prepare_pi_dec_source(params, app, plan)?.into_source())
    }

    /// Materialize the exact selective rows and complete decoder provenance
    /// for the source audit. This diagnostic is intentionally heavier than
    /// [`Self::audit_fixed_point_pi_dec_source_rows`].
    pub fn audit_fixed_point_pi_dec_rows(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcPiDecSelectiveRowsAudit, R1csIvcError> {
        let prepared = prepare_pi_dec_source(params, app, plan)?;
        let projected_rows = project_pi_dec_rows(&prepared)?;
        Ok(R1csIvcPiDecSelectiveRowsAudit {
            source: prepared.into_source(),
            projected_rows,
        })
    }

    /// Materialize only the final selectively emitted rows needed to recover
    /// the canonical 270-coordinate parent-to-fourteen-child public-X split.
    ///
    /// The source seed is exactly `recomposition.x` plus `alphabet`. Complete
    /// intersecting compiler rewrites are then added fail closed. The decoder
    /// request starts only from columns present in those exact rows; it does
    /// not pull the rest of the strict-PiDEC carrier into the projection.
    pub fn audit_fixed_point_pi_dec_canonical_x_rows(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcPiDecCanonicalXSelectiveRowsAudit, R1csIvcError> {
        let prepared = prepare_pi_dec_canonical_x(params, app, plan)?;
        let projected_rows = project_pi_dec_rows(&prepared.source)?;
        Ok(R1csIvcPiDecCanonicalXSelectiveRowsAudit {
            source: prepared.source.into_source(),
            projected_rows,
            semantic_source_ranges: prepared.semantic_source_ranges,
            semantic_source_columns: prepared.semantic_source_columns,
        })
    }
}

fn prepare_pi_dec_source(
    params: &Params,
    app: &R1csShape,
    plan: &RecursiveStepImagePlan,
) -> Result<PreparedPiDecSourceAudit, R1csIvcError> {
    let context = prepare_pi_dec_context(params, app, plan)?;
    let source_rows = (context.strict.row_start..context.strict.row_end).collect();
    prepare_pi_dec_projection(context, source_rows, true)
}

fn prepare_pi_dec_canonical_x(
    params: &Params,
    app: &R1csShape,
    plan: &RecursiveStepImagePlan,
) -> Result<PreparedPiDecCanonicalXAudit, R1csIvcError> {
    let context = prepare_pi_dec_context(params, app, plan)?;
    let semantic_source_ranges = canonical_x_source_ranges(&context.leaf_source_ranges)?;
    if (semantic_source_ranges[0].row_start..semantic_source_ranges[0].row_end) != context.strict.x_recomposition_rows
        || (semantic_source_ranges[1].row_start..semantic_source_ranges[1].row_end)
            != context.strict.x_canonicality_rows
    {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC canonical-X audit ranges differ from the emitter-owned schedule",
        ));
    }
    let semantic_source_rows = semantic_source_ranges
        .iter()
        .flat_map(|range| range.row_start..range.row_end)
        .collect::<BTreeSet<_>>();
    if semantic_source_rows.len() != ACTIVE_CANONICAL_X_SOURCE_ROWS {
        return Err(invalid_pi_dec_audit(format!(
            "strict PiDEC canonical-X leaves contain {} rows, expected {ACTIVE_CANONICAL_X_SOURCE_ROWS}",
            semantic_source_rows.len()
        )));
    }

    let steady_arm = &context.candidate.arms[R1csIvcBranch::Recursive.index()];
    let semantic_rows = semantic_source_rows.iter().copied().collect::<Vec<_>>();
    let semantic_row_artifacts = recover_source_rows(steady_arm, &semantic_rows)?;
    let semantic_source_columns = columns_in_source_rows(&semantic_row_artifacts);
    let expected_source_columns = canonical_x_source_columns(&context.strict)?;
    if semantic_source_columns != expected_source_columns {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC canonical-X source coefficients differ from the recorded parent/child/trace layout",
        ));
    }

    let source = prepare_pi_dec_projection(context, semantic_source_rows, false)?;
    Ok(PreparedPiDecCanonicalXAudit {
        source,
        semantic_source_ranges,
        semantic_source_columns: expected_source_columns.into_iter().collect(),
    })
}

fn prepare_pi_dec_context(
    params: &Params,
    app: &R1csShape,
    plan: &RecursiveStepImagePlan,
) -> Result<PreparedPiDecContext, R1csIvcError> {
    if D != ACTIVE_RING_DIMENSION || <K as BasedVectorSpace<F>>::DIMENSION != 2 {
        return Err(invalid_pi_dec_audit("active PiDEC ring or extension dimension drifted"));
    }

    let candidate = R1csIvcRelation::discover_fixed_point(params, app, plan)?;
    let steady_arm_index = R1csIvcBranch::Recursive.index();
    let steady_arm = &candidate.arms[steady_arm_index];
    let roots = steady_arm
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == stage::ROOT)
        .collect::<Vec<_>>();
    let [root] = roots.as_slice() else {
        return Err(invalid_pi_dec_audit(format!(
            "steady recursive arm has {} outer NIFS PiDEC ranges, expected exactly one",
            roots.len()
        )));
    };
    let strict_matches = steady_arm
        .pi_dec_strict_audits()
        .iter()
        .filter(|audit| audit.row_start == root.row_start && audit.row_end == root.row_end)
        .collect::<Vec<_>>();
    let [strict] = strict_matches.as_slice() else {
        return Err(invalid_pi_dec_audit(format!(
            "outer NIFS PiDEC range has {} matching strict audits, expected exactly one",
            strict_matches.len()
        )));
    };
    let strict = (*strict).clone();
    validate_active_strict(&strict, steady_arm, params)?;
    let leaf_source_ranges = reconcile_leaf_ranges(&strict, steady_arm)?;

    Ok(PreparedPiDecContext {
        candidate,
        strict,
        leaf_source_ranges,
    })
}

fn prepare_pi_dec_projection(
    context: PreparedPiDecContext,
    mut source_rows: BTreeSet<usize>,
    include_complete_strict_layout: bool,
) -> Result<PreparedPiDecSourceAudit, R1csIvcError> {
    let PreparedPiDecContext {
        candidate,
        strict,
        leaf_source_ranges,
    } = context;
    let steady_arm_index = R1csIvcBranch::Recursive.index();
    let steady_arm = &candidate.arms[steady_arm_index];

    let row_ledger = candidate.shape.compiler_audit.rows();
    let source_mapping = row_ledger
        .arms()
        .get(steady_arm_index)
        .ok_or_else(|| invalid_pi_dec_audit("selective compiler ledger omits the steady recursive arm"))?;
    let mut rewrite_indices = BTreeSet::new();
    for &source_row in &source_rows {
        let owner = source_row_owner(source_mapping, source_row).ok_or_else(|| {
            invalid_pi_dec_audit(format!("strict PiDEC source row {source_row} has no compiler owner"))
        })?;
        if let Some(rewrite) = owner.disposition().rewrite_id() {
            rewrite_indices.insert(rewrite.index());
        }
    }

    let mut selected_rows = BTreeSet::new();
    for &rewrite_index in &rewrite_indices {
        let rewrite = row_ledger
            .rewrites()
            .get(rewrite_index)
            .filter(|rewrite| rewrite.id().index() == rewrite_index && rewrite.arm() == steady_arm_index)
            .ok_or_else(|| {
                invalid_pi_dec_audit(format!(
                    "strict PiDEC source row references missing steady-arm rewrite {rewrite_index}"
                ))
            })?;
        for source_range in rewrite.source_rows() {
            if source_range.is_empty() || source_range.end > steady_arm.n {
                return Err(invalid_pi_dec_audit(format!(
                    "strict PiDEC rewrite {rewrite_index} has an empty or out-of-range source interval"
                )));
            }
            for source_row in source_range.clone() {
                let owner = source_row_owner(source_mapping, source_row).ok_or_else(|| {
                    invalid_pi_dec_audit(format!(
                        "strict PiDEC rewrite {rewrite_index} source row {source_row} has no compiler owner"
                    ))
                })?;
                if owner.disposition().rewrite_id().map(|id| id.index()) != Some(rewrite_index) {
                    return Err(invalid_pi_dec_audit(format!(
                        "strict PiDEC rewrite {rewrite_index} source row {source_row} has a different owner"
                    )));
                }
                source_rows.insert(source_row);
            }
        }
        for emitted_row in rewrite.emitted_rows() {
            if !selected_rows.insert(emitted_row) {
                return Err(invalid_pi_dec_audit(format!(
                    "strict PiDEC selective row {emitted_row} is owned by multiple rewrites"
                )));
            }
        }
    }

    let mut retained_row_pairs = Vec::new();
    for &source_row in &source_rows {
        let owner = source_row_owner(source_mapping, source_row).ok_or_else(|| {
            invalid_pi_dec_audit(format!("expanded PiDEC source row {source_row} has no compiler owner"))
        })?;
        match owner.disposition() {
            SelectiveSourceRowDisposition::Retained => {
                let emitted_start = owner.emitted_start().ok_or_else(|| {
                    invalid_pi_dec_audit(format!("retained PiDEC source row {source_row} has no emitted origin"))
                })?;
                let emitted_row = emitted_start + source_row - owner.source_rows().start;
                if !selected_rows.insert(emitted_row) {
                    return Err(invalid_pi_dec_audit(format!(
                        "strict PiDEC selective row {emitted_row} is multiply owned"
                    )));
                }
                retained_row_pairs.push((source_row, emitted_row));
            }
            disposition => {
                let rewrite_index = disposition
                    .rewrite_id()
                    .map(|id| id.index())
                    .ok_or_else(|| {
                        invalid_pi_dec_audit(format!(
                            "non-retained PiDEC source row {source_row} has no rewrite owner"
                        ))
                    })?;
                if !rewrite_indices.contains(&rewrite_index) {
                    return Err(invalid_pi_dec_audit(format!(
                        "expanded PiDEC source row {source_row} escaped selected rewrite {rewrite_index}"
                    )));
                }
            }
        }
    }
    if selected_rows.is_empty() {
        return Err(invalid_pi_dec_audit("strict PiDEC maps to no selectively emitted rows"));
    }

    let source_rows = source_rows.into_iter().collect::<Vec<_>>();
    let source_row_artifacts = recover_source_rows(steady_arm, &source_rows)?;
    let mut source_columns = columns_in_source_rows(&source_row_artifacts);
    if include_complete_strict_layout {
        insert_strict_columns(&strict, &mut source_columns);
    }
    if source_columns.iter().any(|&column| column >= steady_arm.m) {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC source-column closure escapes the steady recursive arm",
        ));
    }
    let source_row_ranges = compress_source_rows(&source_rows);
    Ok(PreparedPiDecSourceAudit {
        candidate,
        strict,
        leaf_source_ranges,
        source_rows,
        source_row_ranges,
        source_row_artifacts,
        selected_rows: selected_rows.into_iter().collect(),
        retained_row_pairs,
        source_columns: source_columns.into_iter().collect(),
    })
}

fn project_pi_dec_rows(prepared: &PreparedPiDecSourceAudit) -> Result<SelectiveProjectedRowsAudit, R1csIvcError> {
    let steady_arm_index = R1csIvcBranch::Recursive.index();
    let projected = super::super::super::selective::project_rows_with_complete_source_provenance_with_alignment(
        &prepared.candidate.arms,
        0,
        0,
        D,
        prepared.candidate.arms[0].m_in % D,
        &prepared.selected_rows,
        steady_arm_index,
        &prepared.source_columns,
        &prepared.retained_row_pairs,
    )?;
    validate_projection(
        &projected,
        &prepared.candidate.shape,
        steady_arm_index,
        &prepared.selected_rows,
        &prepared.retained_row_pairs,
    )?;
    Ok(projected)
}

fn canonical_x_source_ranges(ranges: &[RowFamilyRange]) -> Result<[RowFamilyRange; 2], R1csIvcError> {
    let unique = |name| {
        let matches = ranges
            .iter()
            .filter(|range| range.name == name)
            .copied()
            .collect::<Vec<_>>();
        let [range] = matches.as_slice() else {
            return Err(invalid_pi_dec_audit(format!(
                "strict PiDEC canonical-X leaf {name} has {} ranges, expected exactly one",
                matches.len()
            )));
        };
        Ok(*range)
    };
    let recomposition = unique(stage::RECOMPOSITION_X)?;
    let canonicality = unique(stage::ALPHABET)?;
    if recomposition.row_end - recomposition.row_start != ACTIVE_X_RECOMPOSITION_ROWS
        || canonicality.row_end - canonicality.row_start != ACTIVE_X_CANONICALITY_ROWS
        || recomposition.row_end > canonicality.row_start
    {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC canonical-X leaf ranges have the wrong size or ordering",
        ));
    }
    Ok([recomposition, canonicality])
}

fn canonical_x_source_columns(strict: &PiDecStrictAudit) -> Result<BTreeSet<usize>, R1csIvcError> {
    let mut columns = BTreeSet::from([0usize]);
    for (claim_index, claim) in std::iter::once(&strict.parent)
        .chain(&strict.children)
        .enumerate()
    {
        let active_columns = superneo_public_x_cols(claim.m_in);
        if claim.x_rows * active_columns != ACTIVE_LOGICAL_X || claim.x_cols.len() != claim.x_rows * claim.x_width {
            return Err(invalid_pi_dec_audit(format!(
                "strict PiDEC canonical-X claim {claim_index} has an invalid active-X layout"
            )));
        }
        for row in 0..claim.x_rows {
            for column in 0..active_columns {
                let index = row * claim.x_width + column;
                let source_column = claim.x_cols.get(index).copied().ok_or_else(|| {
                    invalid_pi_dec_audit(format!(
                        "strict PiDEC canonical-X claim {claim_index} omits active coordinate ({row}, {column})"
                    ))
                })?;
                columns.insert(source_column);
            }
        }
    }
    columns.extend(strict.x_sign_traces.iter().flatten().copied());
    if columns.len() != ACTIVE_CANONICAL_X_SOURCE_COLUMNS {
        return Err(invalid_pi_dec_audit(format!(
            "strict PiDEC canonical-X layout owns {} distinct source columns, expected {ACTIVE_CANONICAL_X_SOURCE_COLUMNS}",
            columns.len()
        )));
    }
    Ok(columns)
}

fn columns_in_source_rows(rows: &[R1csIvcPiDecSourceRowAudit]) -> BTreeSet<usize> {
    let mut columns = BTreeSet::from([0usize]);
    for row in rows {
        columns.extend(row.ports.iter().flatten().map(|term| term.0));
    }
    columns
}

fn validate_active_strict(strict: &PiDecStrictAudit, arm: &SparseR1cs, params: &Params) -> Result<(), R1csIvcError> {
    let expected_source_rows = ACTIVE_RING_DIMENSION * params.kappa() as usize + ACTIVE_NONCOMMITMENT_SOURCE_ROWS;
    let observed_source_rows = strict.row_end.saturating_sub(strict.row_start);
    if strict.radix != 2
        || strict.children.len() != ACTIVE_CHILDREN
        || strict.row_start >= strict.row_end
        || strict.row_end > arm.n
        || strict.x_recomposition_rows.start < strict.row_start
        || strict.x_recomposition_rows.end > strict.row_end
        || strict.x_recomposition_rows.len() != ACTIVE_X_RECOMPOSITION_ROWS
        || strict.x_canonicality_rows.start < strict.row_start
        || strict.x_canonicality_rows.end > strict.row_end
        || strict.x_canonicality_rows.len() != ACTIVE_X_CANONICALITY_ROWS
        || strict.first_allocated_column >= arm.m
        || strict.x_sign_traces.len() != ACTIVE_LOGICAL_X
        || observed_source_rows != expected_source_rows
    {
        return Err(invalid_pi_dec_audit(format!(
            "strict PiDEC header or source-row census is not the active radix-2 profile: \
             radix={}, children={}, rows={observed_source_rows}/{expected_source_rows}, \
             x_recomposition={}, x_canonicality={}, sign_traces={}",
            strict.radix,
            strict.children.len(),
            strict.x_recomposition_rows.len(),
            strict.x_canonicality_rows.len(),
            strict.x_sign_traces.len(),
        )));
    }

    let sign_columns = strict
        .x_sign_traces
        .iter()
        .flat_map(|pair| pair.iter().copied())
        .collect::<BTreeSet<_>>();
    if sign_columns.len() != 2 * ACTIVE_LOGICAL_X || sign_columns.iter().any(|&column| column >= arm.m) {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC sign/product traces are not 270 disjoint in-range pairs",
        ));
    }

    let claims = std::iter::once(&strict.parent)
        .chain(&strict.children)
        .collect::<Vec<_>>();
    if claims.len() != ACTIVE_CLAIMS {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC must carry one parent and sixteen children",
        ));
    }
    for (index, claim) in claims.into_iter().enumerate() {
        validate_active_claim(claim, index, params.kappa() as usize)?;
    }
    Ok(())
}

fn validate_active_claim(claim: &PiDecClaimAudit, index: usize, kappa: usize) -> Result<(), R1csIvcError> {
    let active_columns = superneo_public_x_cols(claim.m_in);
    let padded_ring_lanes = D.next_power_of_two();
    let extension_limbs = <K as BasedVectorSpace<F>>::DIMENSION;
    if claim.commitment.data_cols.len() != D * kappa
        || claim.x_rows != D
        || claim.m_in % D != 0
        || claim.m_in != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN
        || claim.x_width != active_columns
        || claim.x_cols.len() != claim.x_rows * active_columns
        || claim.x_rows * active_columns != ACTIVE_LOGICAL_X
        || claim.y_ring_cols.len() != ACTIVE_MATRICES
        || claim.ct_cols.len() != ACTIVE_MATRICES
        || claim
            .y_ring_cols
            .iter()
            .any(|row| row.len() != padded_ring_lanes * extension_limbs)
        || claim.r_cols.len() != ACTIVE_ROW_POINT
    {
        return Err(invalid_pi_dec_audit(format!(
            "strict PiDEC claim {index} is not the active 54x5 compact-X, 14-matrix identity-first profile"
        )));
    }
    if claim.adv.is_some() {
        return Err(invalid_pi_dec_audit(format!(
            "strict PiDEC claim {index} carries advice outside the exact paper carrier"
        )));
    }
    Ok(())
}

fn reconcile_leaf_ranges(strict: &PiDecStrictAudit, arm: &SparseR1cs) -> Result<Vec<RowFamilyRange>, R1csIvcError> {
    let mut ranges = Vec::with_capacity(stage::LEAVES.len());
    for &name in stage::LEAVES {
        let matches = arm
            .row_family_ranges()
            .iter()
            .filter(|range| {
                range.name == name && range.row_start >= strict.row_start && range.row_end <= strict.row_end
            })
            .copied()
            .collect::<Vec<_>>();
        let [range] = matches.as_slice() else {
            return Err(invalid_pi_dec_audit(format!(
                "strict PiDEC leaf {name} has {} source ranges, expected exactly one",
                matches.len()
            )));
        };
        ranges.push(*range);
    }
    let mut cursor = strict.row_start;
    for (&name, range) in stage::LEAVES.iter().zip(&ranges) {
        if range.name != name || range.row_start != cursor || range.row_end < range.row_start {
            return Err(invalid_pi_dec_audit(format!(
                "strict PiDEC leaf {name} does not continue the exact source-row partition"
            )));
        }
        cursor = range.row_end;
    }
    if cursor != strict.row_end {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC leaf ranges do not exactly cover its recorded source interval",
        ));
    }
    Ok(ranges)
}

fn insert_commitment_columns(commitment: &PiDecCommitmentAudit, columns: &mut BTreeSet<usize>) {
    columns.insert(commitment.d_col);
    columns.insert(commitment.kappa_col);
    columns.extend(commitment.data_cols.iter().copied());
}

fn insert_claim_columns(claim: &PiDecClaimAudit, columns: &mut BTreeSet<usize>) {
    insert_commitment_columns(&claim.commitment, columns);
    if let Some(adv) = &claim.adv {
        insert_commitment_columns(&adv.ops, columns);
        insert_commitment_columns(&adv.is, columns);
        insert_commitment_columns(&adv.fs, columns);
    }
    columns.extend(claim.x_cols.iter().copied());
    columns.extend(claim.y_ring_cols.iter().flatten().copied());
    columns.extend(claim.ct_cols.iter().flatten().copied());
    columns.extend(claim.r_cols.iter().flatten().copied());
    columns.extend(claim.fold_digest_cols);
    columns.extend([claim.x_rows_col, claim.x_width_col, claim.m_in_col]);
}

fn insert_strict_columns(strict: &PiDecStrictAudit, columns: &mut BTreeSet<usize>) {
    insert_claim_columns(&strict.parent, columns);
    for child in &strict.children {
        insert_claim_columns(child, columns);
    }
    columns.extend(strict.x_sign_traces.iter().flatten().copied());
}

fn validate_projection(
    projected: &SelectiveProjectedRowsAudit,
    shape: &crate::frontends::r1cs_f_prime::SelectiveLowNormShape,
    arm: usize,
    selected_rows: &[usize],
    retained_pairs: &[(usize, usize)],
) -> Result<(), R1csIvcError> {
    if projected.rows() != shape.rows
        || projected.columns() != shape.columns
        || projected.compiler_audit() != &shape.compiler_audit
        || projected.row_artifacts().len() != selected_rows.len()
        || projected
            .row_artifacts()
            .iter()
            .map(|row| row.emitted_row())
            .ne(selected_rows.iter().copied())
    {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC projected rows differ from the stabilized selective emitter",
        ));
    }
    let provenance = projected
        .source_provenance()
        .ok_or_else(|| invalid_pi_dec_audit("strict PiDEC projection omitted exact source-row provenance"))?;
    if provenance.arm() != arm
        || provenance.retained_steps().len() != retained_pairs.len()
        || provenance
            .retained_steps()
            .iter()
            .map(|step| (step.source_row(), step.emitted_row()))
            .ne(retained_pairs.iter().copied())
    {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC retained source/emitted pairing drifted during projection",
        ));
    }
    let decoder = projected
        .decoder_provenance()
        .ok_or_else(|| invalid_pi_dec_audit("strict PiDEC projection omitted decoder provenance"))?;
    let source_columns = provenance.source_columns();
    if decoder.arm() != arm
        || decoder.decoders().len() != source_columns.len()
        || decoder
            .decoders()
            .iter()
            .map(|decoder| decoder.column())
            .ne(source_columns.iter().copied())
    {
        return Err(invalid_pi_dec_audit(
            "strict PiDEC source-column decoder differs from the requested closure",
        ));
    }
    Ok(())
}

fn recover_source_rows(arm: &SparseR1cs, rows: &[usize]) -> Result<Vec<R1csIvcPiDecSourceRowAudit>, R1csIvcError> {
    let indices = rows
        .iter()
        .enumerate()
        .map(|(index, &row)| (row, index))
        .collect::<BTreeMap<_, _>>();
    let mut artifacts = rows
        .iter()
        .map(|&index| R1csIvcPiDecSourceRowAudit {
            index,
            ports: std::array::from_fn(|_| Vec::new()),
        })
        .collect::<Vec<_>>();
    for (port, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
        if matrix.seeded_phi81_blocks().iter().any(|block| {
            rows.iter()
                .any(|&row| (block.row_start()..block.row_end()).contains(&row))
        }) || matrix
            .geometric_runs()
            .iter()
            .any(|run| indices.contains_key(&run.row()))
        {
            return Err(invalid_pi_dec_audit(
                "strict PiDEC source rows overlap a compact matrix encoding",
            ));
        }
        match matrix {
            CcsMatrix::Identity { n } => {
                for (&row, &index) in &indices {
                    if row >= *n {
                        return Err(invalid_pi_dec_audit(
                            "strict PiDEC source row exceeds an identity matrix",
                        ));
                    }
                    artifacts[index].ports[port].push((row, F::ONE));
                }
            }
            CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
                for column in 0..csc.ncols {
                    for entry in csc.column_range(column) {
                        if let Some(&index) = indices.get(&csc.row_index(entry)) {
                            let coefficient = csc.vals[entry];
                            if coefficient != F::ZERO {
                                artifacts[index].ports[port].push((column, coefficient));
                            }
                        }
                    }
                }
            }
            CcsMatrix::VerifierArtifact { .. } => {
                return Err(invalid_pi_dec_audit(
                    "strict PiDEC source-row audit requires materialized matrix content",
                ));
            }
        }
    }
    for artifact in &mut artifacts {
        for terms in &mut artifact.ports {
            let mut normalized = BTreeMap::<usize, F>::new();
            for (column, coefficient) in std::mem::take(terms) {
                *normalized.entry(column).or_insert(F::ZERO) += coefficient;
            }
            *terms = normalized
                .into_iter()
                .filter(|(_, coefficient)| *coefficient != F::ZERO)
                .collect();
        }
    }
    Ok(artifacts)
}

fn invalid_pi_dec_audit(message: impl Into<String>) -> R1csIvcError {
    R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(message.into()))
}
