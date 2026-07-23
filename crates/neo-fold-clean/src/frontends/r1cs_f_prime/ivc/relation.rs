//! Fixed, selectively lowered relation used by the generic R1CS IVC path.

mod pi_dec_audit;
mod source_audit;

pub use pi_dec_audit::{
    R1csIvcPiDecCanonicalXSelectiveRowsAudit, R1csIvcPiDecSelectiveRowsAudit, R1csIvcPiDecSourceRowAudit,
    R1csIvcPiDecSourceRowsAudit,
};
pub use source_audit::{
    R1csIvcBlockLaneNcSourceRowAudit, R1csIvcRawRunningAssignmentAudit, R1csIvcRawRunningEncodingAudit,
};

use source_audit::{FreshSourceAssignmentAudit, FreshSourceResolutionAudit};

use neo_ccs::CcsMatrix;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::compilation_audit::{
    block_lane_nc_schedule, ArmShapeAudit, FixedPointRoundAudit, R1csIvcCompilationAudit, R1csIvcFixedPointShapeAudit,
    RelationHeaderAudit,
};
use super::pi_ccs_output_digest_audit;
use super::{shape, PiRlcYZcolProjectionLoweringDisposition, R1csIvcError};
use crate::engine::r1cs_circuit::builder::{BlockLaneNcBoundaryAudit, SumcheckRoundAudit};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::lowering::MultiBranchLowNormR1cs;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, R1csShape, SelectiveLowNormShape,
    SelectiveProjectedRowsAudit, SelectiveSourceRowDisposition, SparseR1cs,
};
use crate::lifecycle::Preprocessing;
use crate::paper::digest::PENDING_ACCUMULATOR_FAMILY_ROW_POINT;
use crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcVerifierRelation;
use crate::paper::relations::{CcsInstance, Structure};

/// Hard ceiling shared with the production Road-A relation.
pub const R1CS_IVC_COMMITTED_COORDINATE_BUDGET: usize = 16_000_000;

struct FixedPointCandidate {
    arms: [SparseR1cs; 3],
    shape: SelectiveLowNormShape,
    rounds: Vec<FixedPointRoundAudit>,
    pi_ccs_output_digest: super::PiCcsOutputDigestAudit,
    raw_running_source_columns: Vec<shape::RawRunningSourceColumn>,
    fresh_source_columns: Vec<shape::FreshSourceColumn>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcBranch {
    Base,
    BootstrapRecursive,
    Recursive,
}

impl R1csIvcBranch {
    pub(super) const fn index(self) -> usize {
        match self {
            Self::Base => 0,
            Self::BootstrapRecursive => 1,
            Self::Recursive => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ArmShape {
    pub rows: usize,
    pub columns: usize,
    pub public_columns: usize,
}

/// One verifier-owned relation for the base, bootstrap-recursive, and steady
/// recursive branches of an arbitrary R1CS application.
pub struct R1csIvcRelation {
    relation: MultiBranchLowNormR1cs,
    arm_shapes: [ArmShape; 3],
    compilation_audit: R1csIvcCompilationAudit,
    preprocessing_digest: Option<[F; 4]>,
}

/// Exact bounded row projection from a stabilized fixed-point emitter run.
///
/// This keeps the shape-only fixed-point evidence and selected materialized
/// rows together, while deliberately avoiding allocation of the complete
/// column-sized CSC matrices.
#[derive(Debug)]
pub struct R1csIvcYZcolSelectiveRowsAudit {
    fixed_point: R1csIvcFixedPointShapeAudit,
    projected_rows: SelectiveProjectedRowsAudit,
    raw_running_assignments: Vec<R1csIvcRawRunningAssignmentAudit>,
    fresh_source_assignments: Vec<FreshSourceAssignmentAudit>,
}

/// Exact bounded projection of the production steady-recursive combined-NC
/// verifier rows.
///
/// The source boundary and round schedules are checked against their exact
/// sparse A/B/C rows and the selective compiler ledger. Stage paths are used
/// only to select every output-padding occurrence; they are not semantic
/// authority.
#[derive(Debug)]
pub struct R1csIvcBlockLaneNcSelectiveRowsAudit {
    fixed_point: R1csIvcFixedPointShapeAudit,
    projected_rows: SelectiveProjectedRowsAudit,
    boundary: BlockLaneNcBoundaryAudit,
    rounds: Vec<SumcheckRoundAudit>,
    round_column_maps: Vec<[usize; 43]>,
    output_padding_source_ranges: Vec<std::ops::Range<usize>>,
    source_rows: Vec<usize>,
    source_row_ranges: Vec<std::ops::Range<usize>>,
    source_row_artifacts: Vec<R1csIvcBlockLaneNcSourceRowAudit>,
}

impl R1csIvcYZcolSelectiveRowsAudit {
    pub fn fixed_point(&self) -> &R1csIvcFixedPointShapeAudit {
        &self.fixed_point
    }

    pub fn projected_rows(&self) -> &SelectiveProjectedRowsAudit {
        &self.projected_rows
    }

    pub fn raw_running_assignments(&self) -> &[R1csIvcRawRunningAssignmentAudit] {
        &self.raw_running_assignments
    }

    /// Exact decoder disposition of `prior_link.fresh_public_inputs[0]`.
    ///
    /// Tuple fields are, in order: logical coordinate, normalized source-arm
    /// column, resolution tag, alias source, decomposition digit, final start,
    /// width, and centeredness. Tags are `0=constant-one`, `1=direct`,
    /// `2=decomposition-alias`, `3=equality-alias`, `4=linear-definition`, and
    /// `5=trace-eliminated`. Optional fields are populated only when owned by
    /// that exact decoder variant.
    #[allow(clippy::type_complexity)]
    pub fn fresh_source_assignments(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            usize,
            usize,
            u8,
            Option<usize>,
            Option<usize>,
            Option<usize>,
            Option<usize>,
            Option<bool>,
        ),
    > + '_ {
        self.fresh_source_assignments.iter().map(|entry| {
            let (tag, source, digit, start, width, centered) = match entry.resolution {
                FreshSourceResolutionAudit::ConstantOne => (0, None, None, None, None, None),
                FreshSourceResolutionAudit::Direct { start, width, centered } => {
                    (1, None, None, Some(start), Some(width), Some(centered))
                }
                FreshSourceResolutionAudit::DecompositionAlias {
                    source,
                    digit,
                    start,
                    centered,
                } => (2, Some(source), Some(digit), Some(start), Some(1), Some(centered)),
                FreshSourceResolutionAudit::EqualityAlias {
                    source,
                    start,
                    width,
                    centered,
                } => (3, Some(source), None, Some(start), Some(width), Some(centered)),
                FreshSourceResolutionAudit::LinearDefinition => (4, None, None, None, None, None),
                FreshSourceResolutionAudit::TraceEliminated => (5, None, None, None, None, None),
            };
            (
                entry.logical_column,
                entry.source_column,
                tag,
                source,
                digit,
                start,
                width,
                centered,
            )
        })
    }
}

impl R1csIvcBlockLaneNcSelectiveRowsAudit {
    pub fn fixed_point(&self) -> &R1csIvcFixedPointShapeAudit {
        &self.fixed_point
    }

    pub fn projected_rows(&self) -> &SelectiveProjectedRowsAudit {
        &self.projected_rows
    }

    pub fn boundary(&self) -> &BlockLaneNcBoundaryAudit {
        &self.boundary
    }

    pub fn rounds(&self) -> &[SumcheckRoundAudit] {
        &self.rounds
    }

    /// Dense local-to-source maps for the isolated 30-row production quartic
    /// round artifact. Five extension coefficients are allocated and absorbed
    /// exactly as serialized by the native block-lane proof.
    pub fn round_column_maps(&self) -> &[[usize; 43]] {
        &self.round_column_maps
    }

    /// The fifteen exact source-stage occurrences enforcing lanes 54..64 of
    /// each output `y_zcol` to zero.
    pub fn output_padding_source_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.output_padding_source_ranges
    }

    /// Sorted exact source-arm row indices covered by this projection.
    pub fn source_rows(&self) -> &[usize] {
        &self.source_rows
    }

    /// Maximal half-open ranges compressing [`Self::source_rows`].
    pub fn source_row_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.source_row_ranges
    }

    /// Exact normalized A/B/C coefficients at every selected source row.
    pub fn source_row_artifacts(&self) -> &[R1csIvcBlockLaneNcSourceRowAudit] {
        &self.source_row_artifacts
    }
}

impl R1csIvcRelation {
    pub fn compile_fixed_point(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<Self, R1csIvcError> {
        let candidate = Self::discover_fixed_point(params, app, plan)?;
        Self::compile_arms_selected(
            candidate.arms,
            candidate.shape,
            candidate.rounds,
            candidate.pi_ccs_output_digest,
        )
    }

    /// Discover the exact stabilized field-R1CS arms and selective layout
    /// without materializing the final sparse CCS matrices.
    ///
    /// The returned values are diagnostic shape evidence. They do not prove
    /// that the planned relation was emitted or that any obligation is sound.
    pub fn audit_fixed_point_shape(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcFixedPointShapeAudit, R1csIvcError> {
        let candidate = Self::discover_fixed_point(params, app, plan)?;
        Ok(R1csIvcFixedPointShapeAudit::new(
            candidate.rounds,
            candidate.shape.compiler_audit,
            candidate.pi_ccs_output_digest,
        ))
    }

    /// Stabilize the real source arms and run the exact selective emitter, but
    /// canonicalize only the rows owned by the PiRLC `y_zcol` slice.
    ///
    /// Unlike [`Self::audit_fixed_point_shape`], this produces coefficient
    /// data. Unlike [`Self::compile_fixed_point`], it never allocates arrays
    /// over the complete final column domain and does not bypass the
    /// production committed-coordinate budget.
    pub fn audit_fixed_point_y_zcol_rows(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcYZcolSelectiveRowsAudit, R1csIvcError> {
        let candidate = Self::discover_fixed_point(params, app, plan)?;
        let projection = candidate.pi_ccs_output_digest.y_zcol_projection();
        let mut selected_rows = projection
            .selective_rows()
            .leaves()
            .iter()
            .flat_map(|leaf| leaf.fragments())
            .flat_map(|fragment| fragment.emitted_rows())
            .collect::<Vec<_>>();
        selected_rows.sort_unstable();
        if selected_rows.len() != projection.selective_rows().emitted_row_count()
            || selected_rows.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "PiRLC y_zcol selected-row projection is not one-to-one".into(),
            )));
        }
        let source_columns = projection
            .identity()
            .source_rows()
            .iter()
            .flat_map(|row| [row.a(), row.b(), row.c()])
            .flatten()
            .map(|&(column, _)| column)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        const CHILD_COUNT: usize = 14;
        const LOGICAL_COLUMN_COUNT: usize = 270;
        let expected_raw_running_count = CHILD_COUNT * LOGICAL_COLUMN_COUNT;
        if candidate.raw_running_source_columns.len() != expected_raw_running_count
            || candidate
                .raw_running_source_columns
                .iter()
                .enumerate()
                .any(|(index, entry)| {
                    entry.child != index / LOGICAL_COLUMN_COUNT || entry.logical_column != index % LOGICAL_COLUMN_COUNT
                })
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                format!(
                    "steady raw running-assignment map is not the exact child-major {CHILD_COUNT}x{LOGICAL_COLUMN_COUNT} profile"
                ),
            )));
        }
        let raw_running_source_columns = candidate
            .raw_running_source_columns
            .iter()
            .map(|entry| entry.source_column)
            .collect::<std::collections::BTreeSet<_>>();
        if raw_running_source_columns.len() != expected_raw_running_count {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "steady raw running-assignment coordinates do not have unique source-arm columns".into(),
            )));
        }
        if candidate.fresh_source_columns.len() != LOGICAL_COLUMN_COUNT
            || candidate
                .fresh_source_columns
                .iter()
                .enumerate()
                .any(|(index, entry)| entry.logical_column != index)
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "steady fresh public-X map is not the exact coordinate range 0..270".into(),
            )));
        }
        let fresh_source_columns = candidate
            .fresh_source_columns
            .iter()
            .map(|entry| entry.source_column)
            .collect::<std::collections::BTreeSet<_>>();
        if fresh_source_columns.len() != LOGICAL_COLUMN_COUNT {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "steady fresh public-X coordinates do not have unique source-arm columns".into(),
            )));
        }
        let mut decoder_source_columns = raw_running_source_columns.clone();
        for source_column in fresh_source_columns {
            if !decoder_source_columns.insert(source_column) {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    format!("fresh public-X source column {source_column} overlaps a raw running-X source column"),
                )));
            }
        }
        let decoder_source_columns = decoder_source_columns.into_iter().collect::<Vec<_>>();
        let mut retained_row_pairs = Vec::new();
        for fragment in projection
            .selective_rows()
            .leaves()
            .iter()
            .flat_map(|leaf| leaf.fragments())
        {
            if fragment.disposition() != PiRlcYZcolProjectionLoweringDisposition::Retained {
                continue;
            }
            let source_rows = fragment
                .source_rows()
                .iter()
                .flat_map(|rows| rows.clone())
                .collect::<Vec<_>>();
            let emitted_rows = fragment.emitted_rows().collect::<Vec<_>>();
            if source_rows.len() != emitted_rows.len() {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    "PiRLC y_zcol retained source/emitted row mapping is not one-to-one".into(),
                )));
            }
            retained_row_pairs.extend(source_rows.into_iter().zip(emitted_rows));
        }
        let steady_arm = R1csIvcBranch::Recursive.index();
        let private_source_range = candidate.arms[steady_arm].m_in..candidate.arms[steady_arm].m;
        let projected_rows =
            super::super::selective::project_rows_with_source_provenance_and_decoder_runs_with_alignment(
                &candidate.arms,
                0,
                0,
                D,
                candidate.arms[0].m_in % D,
                &selected_rows,
                steady_arm,
                &source_columns,
                &retained_row_pairs,
                &decoder_source_columns,
                private_source_range.clone(),
            )?;
        if projected_rows.rows() != candidate.shape.rows
            || projected_rows.columns() != candidate.shape.columns
            || projected_rows.compiler_audit() != &candidate.shape.compiler_audit
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "projected selective emitter differs from stabilized fixed-point shape".into(),
            )));
        }
        let source_decoder = projected_rows.decoder_provenance().ok_or_else(|| {
            R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "projected selective emitter omitted requested source-column decoder provenance".into(),
            ))
        })?;
        if source_decoder.arm() != steady_arm {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "requested source-column decoder provenance belongs to the wrong source arm".into(),
            )));
        }
        let private_decoder = projected_rows.decoder_run_provenance().ok_or_else(|| {
            R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "projected selective emitter omitted complete private decoder provenance".into(),
            ))
        })?;
        if private_decoder.arm() != steady_arm || private_decoder.source_range() != private_source_range {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "complete private decoder provenance differs from the steady source arm".into(),
            )));
        }
        let decoder_by_source = source_decoder
            .decoders()
            .iter()
            .copied()
            .map(|decoder| (decoder.column(), decoder.resolution()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if decoder_by_source.len() != source_decoder.decoders().len() {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "requested source-column decoder provenance repeats a source column".into(),
            )));
        }
        let mut raw_running_assignments = Vec::with_capacity(expected_raw_running_count);
        let mut final_columns = std::collections::BTreeSet::new();
        for entry in &candidate.raw_running_source_columns {
            let resolution = decoder_by_source
                .get(&entry.source_column)
                .copied()
                .ok_or_else(|| {
                    R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                        "raw running source column {} has no selective decoder",
                        entry.source_column
                    )))
                })?;
            let super::super::selective::SelectiveProjectedSourceResolution::Direct { start, width, centered } =
                resolution
            else {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    format!(
                        "raw running source column {} is not a direct final slot: {resolution:?}",
                        entry.source_column
                    ),
                )));
            };
            let encoding = if centered {
                if width != 1 {
                    return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                        format!(
                            "raw running source column {} has centered width {width}, expected one",
                            entry.source_column
                        ),
                    )));
                }
                R1csIvcRawRunningEncodingAudit::CenteredScalar
            } else if width == super::super::ternary_encoding::BALANCED_TERNARY_FIELD_WIDTH {
                R1csIvcRawRunningEncodingAudit::BalancedTernary
            } else {
                R1csIvcRawRunningEncodingAudit::Binary
            };
            let end = start
                .checked_add(width)
                .filter(|&end| end <= projected_rows.columns())
                .ok_or_else(|| {
                    R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                        "raw running source column {} final interval {start}..{} exceeds selective width {}",
                        entry.source_column,
                        start.saturating_add(width),
                        projected_rows.columns()
                    )))
                })?;
            if width == 0 || (start..end).any(|column| !final_columns.insert(column)) {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    format!(
                        "raw running source column {} has an empty or multiply-owned final interval {start}..{end}",
                        entry.source_column
                    ),
                )));
            }
            raw_running_assignments.push(R1csIvcRawRunningAssignmentAudit {
                child: entry.child,
                logical_column: entry.logical_column,
                source_column: entry.source_column,
                final_start: start,
                width,
                encoding,
            });
        }
        let mut fresh_source_assignments = Vec::with_capacity(LOGICAL_COLUMN_COUNT);
        for entry in &candidate.fresh_source_columns {
            let resolution = decoder_by_source
                .get(&entry.source_column)
                .copied()
                .ok_or_else(|| {
                    R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                        "fresh public-X source column {} has no selective decoder",
                        entry.source_column
                    )))
                })?;
            let resolution = match resolution {
                super::super::selective::SelectiveProjectedSourceResolution::ConstantOne => {
                    FreshSourceResolutionAudit::ConstantOne
                }
                super::super::selective::SelectiveProjectedSourceResolution::Direct { start, width, centered } => {
                    FreshSourceResolutionAudit::Direct { start, width, centered }
                }
                super::super::selective::SelectiveProjectedSourceResolution::DecompositionAlias {
                    source,
                    digit,
                    start,
                    centered,
                } => FreshSourceResolutionAudit::DecompositionAlias {
                    source,
                    digit,
                    start,
                    centered,
                },
                super::super::selective::SelectiveProjectedSourceResolution::EqualityAlias {
                    source,
                    start,
                    width,
                    centered,
                } => FreshSourceResolutionAudit::EqualityAlias {
                    source,
                    start,
                    width,
                    centered,
                },
                super::super::selective::SelectiveProjectedSourceResolution::LinearDefinition => {
                    FreshSourceResolutionAudit::LinearDefinition
                }
                super::super::selective::SelectiveProjectedSourceResolution::TraceEliminated => {
                    FreshSourceResolutionAudit::TraceEliminated
                }
            };
            fresh_source_assignments.push(FreshSourceAssignmentAudit {
                logical_column: entry.logical_column,
                source_column: entry.source_column,
                resolution,
            });
        }
        let fixed_point = R1csIvcFixedPointShapeAudit::new(
            candidate.rounds,
            candidate.shape.compiler_audit,
            candidate.pi_ccs_output_digest,
        );
        Ok(R1csIvcYZcolSelectiveRowsAudit {
            fixed_point,
            projected_rows,
            raw_running_assignments,
            fresh_source_assignments,
        })
    }

    /// Stabilize the production recursive arm and project the exact source
    /// rows implementing its combined block-by-lane NC check.
    ///
    /// Selection starts from verifier-owned boundary schedules and every
    /// physical output-`y_zcol` padding occurrence. The compiler ledger then
    /// expands every intersecting rewrite to its complete source and emitted
    /// intervals before exact sparse rows are projected.
    pub fn audit_fixed_point_block_lane_nc_rows(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcBlockLaneNcSelectiveRowsAudit, R1csIvcError> {
        const ACTIVE_LANES: usize = 54;
        const LANE_BITS: usize = 6;
        const PADDED_LANES: usize = 1 << LANE_BITS;
        const BLOCK_BITS: usize = 19;
        const ROUND_COUNT: usize = BLOCK_BITS + LANE_BITS;
        const ROUND_COEFFICIENTS: usize = 5;
        const ROUND_ROWS: usize = 30;
        const ROUND_ALLOCATED_COLUMNS: usize = 28;
        const ROUND_LOCAL_COLUMNS: usize = 43;
        const PENDING_PARENT_LANES: usize = ACTIVE_LANES;
        const OUTPUT_COUNT: usize = 15;

        if D != ACTIVE_LANES {
            return Err(invalid_block_lane_audit(format!(
                "production active-lane constant is {D}, expected {ACTIVE_LANES}"
            )));
        }

        let candidate = Self::discover_fixed_point(params, app, plan)?;
        let steady_arm_index = R1csIvcBranch::Recursive.index();
        let steady_arm = &candidate.arms[steady_arm_index];
        let [boundary] = steady_arm.block_lane_nc_boundary_audits() else {
            return Err(invalid_block_lane_audit(format!(
                "steady recursive arm has {} block/lane NC boundaries, expected exactly one",
                steady_arm.block_lane_nc_boundary_audits().len()
            )));
        };
        let boundary = boundary.clone();
        let pending_old_block = boundary
            .pending_old_block_cols
            .as_deref()
            .ok_or_else(|| invalid_block_lane_audit("steady block/lane NC boundary omits pending old-block wires"))?;
        let pending_parent = boundary
            .pending_parent_y_zcol_cols
            .as_deref()
            .ok_or_else(|| {
                invalid_block_lane_audit("steady block/lane NC boundary omits pending parent-y_zcol wires")
            })?;
        if pending_old_block.len() != BLOCK_BITS
            || pending_parent.len() != PENDING_PARENT_LANES
            || boundary.output_y_zcol_cols.len() != OUTPUT_COUNT
            || boundary
                .output_y_zcol_cols
                .iter()
                .any(|output| output.len() != PADDED_LANES)
            || boundary.beta_lane_cols.len() != LANE_BITS
            || boundary.beta_block_cols.len() != BLOCK_BITS
            || boundary.block_point_cols.len() != BLOCK_BITS
            || boundary.lane_point_cols.len() != LANE_BITS
        {
            return Err(invalid_block_lane_audit(format!(
                "steady block/lane NC boundary shape drift: old_block={} parent={} outputs={:?} beta_lane={} beta_block={} block_point={} lane_point={}",
                pending_old_block.len(),
                pending_parent.len(),
                boundary
                    .output_y_zcol_cols
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<_>>(),
                boundary.beta_lane_cols.len(),
                boundary.beta_block_cols.len(),
                boundary.block_point_cols.len(),
                boundary.lane_point_cols.len(),
            )));
        }

        let all_rounds = steady_arm.sumcheck_round_audits();
        if boundary.round_audit_indices.end > all_rounds.len()
            || boundary.round_audit_indices.start > boundary.round_audit_indices.end
        {
            return Err(invalid_block_lane_audit(
                "block/lane NC round-audit interval escapes the steady recursive arm",
            ));
        }
        let rounds = all_rounds[boundary.round_audit_indices.clone()].to_vec();
        if rounds.len() != ROUND_COUNT
            || rounds
                .iter()
                .any(|round| round.coefficient_cols.len() != ROUND_COEFFICIENTS)
        {
            return Err(invalid_block_lane_audit(format!(
                "block/lane NC requires {ROUND_COUNT} quartic rounds with {ROUND_COEFFICIENTS} coefficient pairs; got {} rounds with coefficient counts {:?} from audit interval {:?} of {} total rounds",
                rounds.len(),
                rounds
                    .iter()
                    .map(|round| round.coefficient_cols.len())
                    .collect::<Vec<_>>(),
                boundary.round_audit_indices,
                all_rounds.len(),
            )));
        }
        if rounds.iter().any(|round| {
            round.row_end.checked_sub(round.row_start) != Some(ROUND_ROWS)
                || round.row_end > steady_arm.n
                || round.allocated_cols.len() != ROUND_ALLOCATED_COLUMNS
                || round
                    .allocated_cols
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || round
                    .coefficient_cols
                    .iter()
                    .flatten()
                    .chain(round.challenge_cols.iter())
                    .chain(round.claim_in_cols.iter())
                    .chain(round.claim_out_cols.iter())
                    .any(|&column| column >= steady_arm.m)
        }) {
            return Err(invalid_block_lane_audit(
                "block/lane NC round rows or columns are malformed",
            ));
        }
        let round_column_maps = rounds
            .iter()
            .map(|round| {
                let mut map = [usize::MAX; ROUND_LOCAL_COLUMNS];
                map[0] = 0;
                map[1] = round.claim_in_cols[0];
                map[ROUND_COEFFICIENTS + 2] = round.claim_in_cols[1];
                for (index, pair) in round.coefficient_cols.iter().enumerate() {
                    map[2 + index] = pair[0];
                    map[ROUND_COEFFICIENTS + 3 + index] = pair[1];
                }
                map[2 * ROUND_COEFFICIENTS + 3] = round.challenge_cols[0];
                map[2 * ROUND_COEFFICIENTS + 5] = round.challenge_cols[1];
                map[2 * ROUND_COEFFICIENTS + 4] = round.allocated_cols[0];
                for (local, &source) in
                    ((2 * ROUND_COEFFICIENTS + 6)..ROUND_LOCAL_COLUMNS).zip(&round.allocated_cols[1..])
                {
                    map[local] = source;
                }
                if round.claim_out_cols != [map[ROUND_LOCAL_COLUMNS - 2], map[ROUND_LOCAL_COLUMNS - 1]]
                    || map.contains(&usize::MAX)
                    || map
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != map.len()
                {
                    return Err(invalid_block_lane_audit(
                        "block/lane NC round does not define the injective 43-column isolated-round map",
                    ));
                }
                Ok(map)
            })
            .collect::<Result<Vec<_>, R1csIvcError>>()?;
        let expected_challenges = boundary
            .block_point_cols
            .iter()
            .chain(&boundary.lane_point_cols)
            .copied()
            .collect::<Vec<_>>();
        if rounds[0].claim_in_cols != boundary.claimed_initial_cols
            || rounds[ROUND_COUNT - 1].claim_out_cols != boundary.final_sum_cols
            || rounds
                .iter()
                .zip(expected_challenges)
                .any(|(round, challenge)| round.challenge_cols != challenge)
            || rounds
                .windows(2)
                .any(|pair| pair[0].claim_out_cols != pair[1].claim_in_cols || pair[0].row_end > pair[1].row_start)
        {
            return Err(invalid_block_lane_audit(
                "block/lane NC claimed-chain or challenge continuity is broken",
            ));
        }
        if boundary.claimed_initial_rows.is_empty()
            || boundary.claimed_initial_rows.end > rounds[0].row_start
            || boundary.terminal_identity_rows.is_empty()
            || rounds[ROUND_COUNT - 1].row_end > boundary.terminal_identity_rows.start
            || boundary.terminal_identity_rows.end > boundary.terminal_final_equality_rows.start
            || boundary.terminal_final_equality_rows.len() != 2
            || boundary.terminal_final_equality_rows.end > steady_arm.n
        {
            return Err(invalid_block_lane_audit(
                "block/lane NC boundary row intervals are empty, overlapping, or out of order",
            ));
        }

        let padding_occurrences = steady_arm
            .physical_stage_ranges()
            .iter()
            .filter(|range| {
                range.path()
                    == crate::paper::reductions::pi_ccs_split_nc_circuit::stage::RUNNING_AUTHORITY_OUTPUT_Y_ZCOL_PADDING
            })
            .map(|range| range.rows())
            .collect::<Vec<_>>();
        let expected_padding_rows = 2 * (PADDED_LANES - ACTIVE_LANES);
        if padding_occurrences.len() != OUTPUT_COUNT
            || padding_occurrences
                .iter()
                .any(|range| range.len() != expected_padding_rows || range.end > steady_arm.n)
        {
            return Err(invalid_block_lane_audit(format!(
                "steady output-y_zcol padding must have {OUTPUT_COUNT} physical occurrences of {expected_padding_rows} rows"
            )));
        }

        let mut initial_source_rows = std::collections::BTreeSet::new();
        insert_disjoint_source_range(
            &mut initial_source_rows,
            boundary.claimed_initial_rows.clone(),
            steady_arm.n,
            "claimed-initial",
        )?;
        for round in &rounds {
            insert_disjoint_source_range(
                &mut initial_source_rows,
                round.row_start..round.row_end,
                steady_arm.n,
                "sumcheck-round",
            )?;
        }
        insert_disjoint_source_range(
            &mut initial_source_rows,
            boundary.terminal_identity_rows.clone(),
            steady_arm.n,
            "terminal-identity",
        )?;
        insert_disjoint_source_range(
            &mut initial_source_rows,
            boundary.terminal_final_equality_rows.clone(),
            steady_arm.n,
            "terminal-final-equality",
        )?;
        for range in padding_occurrences.iter().cloned() {
            insert_disjoint_source_range(&mut initial_source_rows, range, steady_arm.n, "output-y_zcol-padding")?;
        }

        let row_ledger = candidate.shape.compiler_audit.rows();
        let source_mapping = row_ledger
            .arms()
            .get(steady_arm_index)
            .ok_or_else(|| invalid_block_lane_audit("selective compiler ledger omits the steady recursive arm"))?;
        let mut rewrite_indices = std::collections::BTreeSet::new();
        for &source_row in &initial_source_rows {
            let owner = source_row_owner(source_mapping, source_row).ok_or_else(|| {
                invalid_block_lane_audit(format!(
                    "target source row {source_row} has no selective compiler owner"
                ))
            })?;
            if let Some(rewrite) = owner.disposition().rewrite_id() {
                rewrite_indices.insert(rewrite.index());
            }
        }

        let mut source_rows = initial_source_rows;
        let mut selected_rows = std::collections::BTreeSet::new();
        for rewrite_index in &rewrite_indices {
            let rewrite = row_ledger
                .rewrites()
                .get(*rewrite_index)
                .filter(|rewrite| rewrite.id().index() == *rewrite_index && rewrite.arm() == steady_arm_index)
                .ok_or_else(|| {
                    invalid_block_lane_audit(format!(
                        "target source row references missing steady-arm rewrite {rewrite_index}"
                    ))
                })?;
            for source_range in rewrite.source_rows() {
                if source_range.is_empty() || source_range.end > steady_arm.n {
                    return Err(invalid_block_lane_audit(format!(
                        "rewrite {rewrite_index} has an empty or out-of-range source interval"
                    )));
                }
                for source_row in source_range.clone() {
                    let owner = source_row_owner(source_mapping, source_row).ok_or_else(|| {
                        invalid_block_lane_audit(format!(
                            "rewrite {rewrite_index} source row {source_row} has no compiler owner"
                        ))
                    })?;
                    if owner.disposition().rewrite_id().map(|id| id.index()) != Some(*rewrite_index) {
                        return Err(invalid_block_lane_audit(format!(
                            "rewrite {rewrite_index} source row {source_row} has a different compiler owner"
                        )));
                    }
                    source_rows.insert(source_row);
                }
            }
            for emitted_row in rewrite.emitted_rows() {
                if !selected_rows.insert(emitted_row) {
                    return Err(invalid_block_lane_audit(format!(
                        "selective emitted row {emitted_row} is owned by multiple selected rewrites"
                    )));
                }
            }
        }

        let mut retained_row_pairs = Vec::new();
        for &source_row in &source_rows {
            let owner = source_row_owner(source_mapping, source_row).ok_or_else(|| {
                invalid_block_lane_audit(format!(
                    "expanded source row {source_row} has no selective compiler owner"
                ))
            })?;
            match owner.disposition() {
                SelectiveSourceRowDisposition::Retained => {
                    let emitted_start = owner.emitted_start().ok_or_else(|| {
                        invalid_block_lane_audit(format!("retained source row {source_row} has no emitted-row origin"))
                    })?;
                    let emitted_row = emitted_start + (source_row - owner.source_rows().start);
                    if !selected_rows.insert(emitted_row) {
                        return Err(invalid_block_lane_audit(format!(
                            "selective emitted row {emitted_row} is multiply owned"
                        )));
                    }
                    retained_row_pairs.push((source_row, emitted_row));
                }
                disposition => {
                    let rewrite_index = disposition
                        .rewrite_id()
                        .map(|id| id.index())
                        .ok_or_else(|| {
                            invalid_block_lane_audit(format!(
                                "non-retained source row {source_row} has no rewrite owner"
                            ))
                        })?;
                    if !rewrite_indices.contains(&rewrite_index) {
                        return Err(invalid_block_lane_audit(format!(
                            "expanded source row {source_row} escaped selected rewrite {rewrite_index}"
                        )));
                    }
                }
            }
        }
        if selected_rows.is_empty() {
            return Err(invalid_block_lane_audit(
                "combined-NC source target maps to no selectively emitted rows",
            ));
        }

        let source_rows = source_rows.into_iter().collect::<Vec<_>>();
        let source_row_artifacts = recover_source_row_artifacts(steady_arm, &source_rows)?;
        let mut source_columns = std::collections::BTreeSet::from([0usize]);
        for row in &source_row_artifacts {
            source_columns.extend(row.ports.iter().flatten().map(|term| term.0));
        }
        source_columns.extend(boundary.gamma_cols);
        source_columns.extend(boundary.producer_beta_cols);
        source_columns.extend(boundary.batch_weight_cols);
        source_columns.extend(boundary.claimed_initial_cols);
        source_columns.extend(boundary.final_sum_cols);
        source_columns.extend(boundary.terminal_rhs_cols);
        for pair in boundary
            .beta_lane_cols
            .iter()
            .chain(&boundary.beta_block_cols)
            .chain(pending_old_block)
            .chain(pending_parent)
            .chain(&boundary.block_point_cols)
            .chain(&boundary.lane_point_cols)
            .chain(boundary.output_y_zcol_cols.iter().flatten())
        {
            source_columns.extend(*pair);
        }
        for round in &rounds {
            source_columns.extend(round.allocated_cols.iter().copied());
            for pair in round
                .coefficient_cols
                .iter()
                .chain(std::slice::from_ref(&round.challenge_cols))
                .chain(std::slice::from_ref(&round.claim_in_cols))
                .chain(std::slice::from_ref(&round.claim_out_cols))
            {
                source_columns.extend(*pair);
            }
        }
        if source_columns.iter().any(|&column| column >= steady_arm.m) {
            return Err(invalid_block_lane_audit(
                "combined-NC source-column closure escapes the steady recursive arm",
            ));
        }

        let selected_rows = selected_rows.into_iter().collect::<Vec<_>>();
        let source_columns = source_columns.into_iter().collect::<Vec<_>>();
        // The compiler provenance expands the requested row columns through
        // reachable linear definitions. Decode that exact transitive closure,
        // not merely the seed columns collected from the physical rows.
        let seed_projection = super::super::selective::project_rows_with_source_provenance_with_alignment(
            &candidate.arms,
            0,
            0,
            D,
            candidate.arms[0].m_in % D,
            &selected_rows,
            steady_arm_index,
            &source_columns,
            &retained_row_pairs,
            &source_columns,
        )?;
        let decoder_source_columns = seed_projection
            .source_provenance()
            .ok_or_else(|| invalid_block_lane_audit("combined-NC projection omitted source provenance"))?
            .source_columns()
            .to_vec();
        let projected_rows = super::super::selective::project_rows_with_source_provenance_with_alignment(
            &candidate.arms,
            0,
            0,
            D,
            candidate.arms[0].m_in % D,
            &selected_rows,
            steady_arm_index,
            &source_columns,
            &retained_row_pairs,
            &decoder_source_columns,
        )?;
        if projected_rows.rows() != candidate.shape.rows
            || projected_rows.columns() != candidate.shape.columns
            || projected_rows.compiler_audit() != &candidate.shape.compiler_audit
            || projected_rows.row_artifacts().len() != selected_rows.len()
            || projected_rows
                .row_artifacts()
                .iter()
                .map(|row| row.emitted_row())
                .ne(selected_rows.iter().copied())
        {
            return Err(invalid_block_lane_audit(
                "combined-NC projected rows differ from the stabilized selective emitter",
            ));
        }
        let provenance = projected_rows
            .source_provenance()
            .ok_or_else(|| invalid_block_lane_audit("combined-NC projection omitted exact source-row provenance"))?;
        if provenance.arm() != steady_arm_index
            || provenance.retained_steps().len() != retained_row_pairs.len()
            || provenance
                .retained_steps()
                .iter()
                .map(|step| (step.source_row(), step.emitted_row()))
                .ne(retained_row_pairs.iter().copied())
        {
            return Err(invalid_block_lane_audit(
                "combined-NC retained source/emitted pairing drifted during projection",
            ));
        }
        if projected_rows.decoder_provenance().is_none() {
            return Err(invalid_block_lane_audit(
                "combined-NC projection omitted requested source-column decoder provenance",
            ));
        }

        let source_row_ranges = compress_source_rows(&source_rows);
        let fixed_point = R1csIvcFixedPointShapeAudit::new(
            candidate.rounds,
            candidate.shape.compiler_audit,
            candidate.pi_ccs_output_digest,
        );
        Ok(R1csIvcBlockLaneNcSelectiveRowsAudit {
            fixed_point,
            projected_rows,
            boundary,
            rounds,
            round_column_maps,
            output_padding_source_ranges: padding_occurrences,
            source_rows,
            source_row_ranges,
            source_row_artifacts,
        })
    }

    fn discover_fixed_point(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<FixedPointCandidate, R1csIvcError> {
        const MAX_ROUNDS: usize = 8;

        app.validate_shape()?;
        super::super::validate_plan(plan, app)?;
        // The first guess only supplies NIFS dimensions. It must already be
        // wide enough to host F's fixed `[1 || enc_inst(x_out)]` public
        // input even when the user app itself is narrower (a tiny app may
        // have fewer than 257 columns). Every later round uses the compiled
        // relation itself, and acceptance still requires exact stabilization.
        let variant = neo_reductions::optimized_engine::PiCcsProofVariant::BlockLaneNcDelayedV1;
        // The delayed production codec has a fixed 25-element row point. The
        // matrix-independent seed must therefore start in that row domain;
        // otherwise the first synthesis round cannot encode its outgoing
        // pending family. This is only a verifier header -- no 2^24-row
        // matrix is allocated here.
        let seed_rows = (1usize << (PENDING_ACCUMULATOR_FAMILY_ROW_POINT - 1)) + 1;
        let mut verifier_relation = SplitNcVerifierRelation::from_parts_with_variant(
            seed_rows,
            F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
            super::super::selective::selective_polynomial(),
            variant,
        );
        let mut folded_public_input_len = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
        let mut last_output = (verifier_relation.n(), verifier_relation.m());
        let mut rounds = Vec::new();
        for round in 0..MAX_ROUNDS {
            let input_signature = verifier_relation_signature(&verifier_relation, folded_public_input_len);
            let synthesized =
                shape::synthesize_arm_shapes(params, &verifier_relation, folded_public_input_len, app, plan)?;
            let arms = synthesized.arms;
            let next_shape = audit_multi_branch_selective_low_norm_shape_with_alignment(&arms, 0, D, arms[0].m_in % D)?;
            last_output = (next_shape.rows, next_shape.columns);
            rounds.push(FixedPointRoundAudit {
                round,
                input: RelationHeaderAudit {
                    rows: verifier_relation.n(),
                    columns: verifier_relation.m(),
                    public_input_len: folded_public_input_len,
                    polynomial: verifier_relation.polynomial().clone(),
                },
                arms: std::array::from_fn(|index| ArmShapeAudit {
                    rows: arms[index].n,
                    columns: arms[index].m,
                    public_columns: arms[index].m_in,
                }),
                output: RelationHeaderAudit {
                    rows: next_shape.rows,
                    columns: next_shape.columns,
                    public_input_len: next_shape.public_input_len,
                    polynomial: next_shape.polynomial.clone(),
                },
            });
            if round > 0
                && input_signature == shape_signature(&next_shape)
                && same_polynomial(verifier_relation.polynomial(), &next_shape.polynomial)
            {
                // The bootstrap arm has no running claims. Only the stabilized
                // recursive arm carries the complete fresh-plus-running PiCCS
                // output-message profile.
                let recursive_arm = R1csIvcBranch::Recursive.index();
                let pi_ccs_output_digest = pi_ccs_output_digest_audit::recover(
                    &arms[recursive_arm],
                    next_shape.compiler_audit.rows(),
                    recursive_arm,
                )?;
                return Ok(FixedPointCandidate {
                    arms,
                    shape: next_shape,
                    rounds,
                    pi_ccs_output_digest,
                    raw_running_source_columns: synthesized.raw_running_source_columns,
                    fresh_source_columns: synthesized.fresh_source_columns,
                });
            }
            folded_public_input_len = next_shape.public_input_len;
            verifier_relation = SplitNcVerifierRelation::from_parts_with_variant(
                next_shape.rows,
                next_shape.columns,
                next_shape.polynomial,
                variant,
            );
        }
        Err(R1csIvcError::NoFixedPoint {
            rounds: MAX_ROUNDS,
            input_rows: verifier_relation.n(),
            input_columns: verifier_relation.m(),
            output_rows: last_output.0,
            output_columns: last_output.1,
        })
    }

    fn compile_arms_selected(
        arms: [SparseR1cs; 3],
        shape: SelectiveLowNormShape,
        rounds: Vec<FixedPointRoundAudit>,
        pi_ccs_output_digest: super::PiCcsOutputDigestAudit,
    ) -> Result<Self, R1csIvcError> {
        let steady_arm = &arms[R1csIvcBranch::Recursive.index()];
        let (block_lane_nc_boundary, block_lane_nc_rounds) =
            block_lane_nc_schedule(steady_arm).map_err(invalid_block_lane_audit)?;
        let arm_shapes = std::array::from_fn(|index| ArmShape {
            rows: arms[index].n,
            columns: arms[index].m,
            public_columns: arms[index].m_in,
        });
        let public_fields = arms[0].m_in;
        if shape.compiler_audit.width().total_coordinates > R1CS_IVC_COMMITTED_COORDINATE_BUDGET {
            return Err(R1csIvcError::BudgetExceeded {
                required: shape.compiler_audit.width().total_coordinates,
                budget: R1CS_IVC_COMMITTED_COORDINATE_BUDGET,
            });
        }
        let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&arms, 0, D, public_fields % D)?;
        let emitted_audit = relation
            .selective_compiler_audit()
            .cloned()
            .ok_or_else(|| {
                R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    "selective relation omitted its exact compiler audit".into(),
                ))
            })?;
        let emitted_layout = emitted_audit.layout();
        if relation_signature(relation.structure(), relation.public_input_len()) != shape_signature(&shape)
            || !same_polynomial(&relation.structure().f, &shape.polynomial)
            || emitted_audit != shape.compiler_audit
            || emitted_layout.total_columns() != relation.structure().m
            || emitted_layout.public_input_len() != relation.public_input_len()
            || emitted_layout.selector_columns() != relation.selector_cols()
            || emitted_audit.width().total_coordinates != emitted_layout.ring_alignment_padding_columns().start
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "shape-only or exact selective audit differs from emitted relation".into(),
            )));
        }
        if relation.structure().m > R1CS_IVC_COMMITTED_COORDINATE_BUDGET {
            return Err(R1csIvcError::BudgetExceeded {
                required: relation.structure().m,
                budget: R1CS_IVC_COMMITTED_COORDINATE_BUDGET,
            });
        }
        let compilation_audit = R1csIvcCompilationAudit::new(
            rounds,
            emitted_audit,
            pi_ccs_output_digest,
            block_lane_nc_boundary,
            block_lane_nc_rounds,
        );
        Ok(Self {
            relation,
            arm_shapes,
            compilation_audit,
            preprocessing_digest: None,
        })
    }

    pub fn structure(&self) -> &Structure {
        self.relation.structure()
    }

    pub fn public_input_len(&self) -> usize {
        self.relation.public_input_len()
    }

    pub fn compilation_audit(&self) -> &R1csIvcCompilationAudit {
        &self.compilation_audit
    }

    pub(super) fn arm_shape(&self, branch: R1csIvcBranch) -> ArmShape {
        self.arm_shapes[branch.index()]
    }

    pub(super) fn bind_preprocessing(&mut self, prep: &Preprocessing) -> Result<(), R1csIvcError> {
        let structure = self.structure();
        let prep_structure = prep.structure();
        if (structure.n, structure.m, structure.t(), structure.max_degree())
            != (
                prep_structure.n,
                prep_structure.m,
                prep_structure.t(),
                prep_structure.max_degree(),
            )
            || prep.public_input_len != Some(self.public_input_len())
        {
            return Err(R1csIvcError::PreprocessingMismatch);
        }
        self.preprocessing_digest = Some(*prep.structure_digest());
        Ok(())
    }

    pub(super) fn encode(&self, branch: R1csIvcBranch, field_assignment: &[F]) -> Result<Vec<F>, R1csIvcError> {
        // The field builder was checked immediately before this deterministic
        // lowering. NIFS and terminal latest-instance verification enforce the
        // lowered relation and reject consistently recommitted invalid witnesses.
        Ok(self.relation.encode(branch.index(), field_assignment)?)
    }

    pub(super) fn build_instance(
        &self,
        prep: &Preprocessing,
        branch: R1csIvcBranch,
        field_assignment: &[F],
    ) -> Result<CcsInstance, R1csIvcError> {
        if self.preprocessing_digest != Some(*prep.structure_digest())
            || prep.public_input_len != Some(self.public_input_len())
        {
            return Err(R1csIvcError::PreprocessingMismatch);
        }
        #[cfg(feature = "perf-timers")]
        let encode_start = std::time::Instant::now();
        let assignment = self.encode(branch, field_assignment)?;
        #[cfg(feature = "perf-timers")]
        let encode_elapsed = encode_start.elapsed();
        #[cfg(feature = "perf-timers")]
        let instance_start = std::time::Instant::now();
        let instance = CcsInstance::from_low_norm_assignment(
            &prep.params,
            &prep.log,
            prep.structure(),
            &assignment,
            self.public_input_len(),
        )?;
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[r1cs-ivc-instance] encode+relation-check {:>7.2}s pack+norm+commit {:>7.2}s",
            encode_elapsed.as_secs_f64(),
            instance_start.elapsed().as_secs_f64(),
        );
        Ok(instance)
    }
}

fn relation_signature(structure: &Structure, public_input_len: usize) -> (usize, usize, usize, u32, usize) {
    (
        structure.n,
        structure.m,
        structure.t(),
        structure.max_degree(),
        public_input_len,
    )
}

fn verifier_relation_signature(
    relation: &SplitNcVerifierRelation,
    public_input_len: usize,
) -> (usize, usize, usize, u32, usize) {
    (
        relation.n(),
        relation.m(),
        relation.t(),
        relation.max_degree(),
        public_input_len,
    )
}

fn shape_signature(shape: &SelectiveLowNormShape) -> (usize, usize, usize, u32, usize) {
    (
        shape.rows,
        shape.columns,
        shape.polynomial.arity(),
        shape.polynomial.max_degree(),
        shape.public_input_len,
    )
}

fn same_polynomial(left: &neo_ccs::SparsePoly<F>, right: &neo_ccs::SparsePoly<F>) -> bool {
    left.arity() == right.arity()
        && left.terms().len() == right.terms().len()
        && left
            .terms()
            .iter()
            .zip(right.terms())
            .all(|(left, right)| left.coeff == right.coeff && left.exps == right.exps)
}

fn invalid_block_lane_audit(message: impl Into<String>) -> R1csIvcError {
    R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(message.into()))
}

fn insert_disjoint_source_range(
    rows: &mut std::collections::BTreeSet<usize>,
    range: std::ops::Range<usize>,
    source_row_count: usize,
    owner: &str,
) -> Result<(), R1csIvcError> {
    if range.is_empty() || range.end > source_row_count {
        return Err(invalid_block_lane_audit(format!(
            "{owner} source-row interval is empty or out of range"
        )));
    }
    for row in range {
        if !rows.insert(row) {
            return Err(invalid_block_lane_audit(format!(
                "source row {row} is selected by multiple combined-NC owners"
            )));
        }
    }
    Ok(())
}

fn source_row_owner(
    mapping: &crate::frontends::r1cs_f_prime::SelectiveArmRowMappingAudit,
    row: usize,
) -> Option<&crate::frontends::r1cs_f_prime::SelectiveSourceRowRunAudit> {
    let runs = mapping.source_runs();
    let index = runs.partition_point(|run| run.source_rows().end <= row);
    runs.get(index)
        .filter(|run| run.source_rows().contains(&row))
}

fn recover_source_row_artifacts(
    arm: &SparseR1cs,
    rows: &[usize],
) -> Result<Vec<R1csIvcBlockLaneNcSourceRowAudit>, R1csIvcError> {
    let indices = rows
        .iter()
        .enumerate()
        .map(|(index, &row)| (row, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut artifacts = rows
        .iter()
        .map(|&index| R1csIvcBlockLaneNcSourceRowAudit {
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
            return Err(invalid_block_lane_audit(
                "combined-NC source rows overlap a compact matrix encoding",
            ));
        }
        match matrix {
            CcsMatrix::Identity { n } => {
                for (&row, &index) in &indices {
                    if row >= *n {
                        return Err(invalid_block_lane_audit(
                            "combined-NC source row exceeds an identity matrix",
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
        }
    }
    for artifact in &mut artifacts {
        for terms in &mut artifact.ports {
            let mut normalized = std::collections::BTreeMap::<usize, F>::new();
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

fn compress_source_rows(rows: &[usize]) -> Vec<std::ops::Range<usize>> {
    let Some((&first, rest)) = rows.split_first() else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    let mut start = first;
    let mut prior = first;
    for &row in rest {
        if row != prior + 1 {
            ranges.push(start..prior + 1);
            start = row;
        }
        prior = row;
    }
    ranges.push(start..prior + 1);
    ranges
}
