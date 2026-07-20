//! Fixed, selectively lowered relation used by the generic R1CS IVC path.

use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::compilation_audit::{
    ArmShapeAudit, FixedPointRoundAudit, R1csIvcCompilationAudit, R1csIvcFixedPointShapeAudit, RelationHeaderAudit,
};
use super::pi_ccs_output_digest_audit;
use super::{shape, PiRlcYZcolProjectionLoweringDisposition, R1csIvcError};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::lowering::MultiBranchLowNormR1cs;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, R1csShape, SelectiveLowNormShape,
    SelectiveProjectedRowsAudit, SparseR1cs,
};
use crate::lifecycle::Preprocessing;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreshSourceAssignmentAudit {
    logical_column: usize,
    source_column: usize,
    resolution: FreshSourceResolutionAudit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreshSourceResolutionAudit {
    ConstantOne,
    Direct {
        start: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        digit: usize,
        start: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        start: usize,
        width: usize,
        centered: bool,
    },
    LinearDefinition,
    TraceEliminated,
}

/// Exact fixed-profile path from one authoritative incoming running-X
/// coordinate through the normalized source arm to its one-coordinate final
/// selective assignment slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcRawRunningAssignmentAudit {
    child: usize,
    logical_column: usize,
    source_column: usize,
    final_column: usize,
}

impl R1csIvcRawRunningAssignmentAudit {
    pub fn child(self) -> usize {
        self.child
    }

    pub fn logical_column(self) -> usize {
        self.logical_column
    }

    pub fn source_column(self) -> usize {
        self.source_column
    }

    pub fn final_column(self) -> usize {
        self.final_column
    }
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
        let projected_rows = super::super::selective::project_rows_with_source_provenance_with_alignment(
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
            let super::super::selective::SelectiveProjectedSourceResolution::Direct {
                start,
                width: 1,
                centered: true,
            } = resolution
            else {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    format!(
                        "raw running source column {} is not a direct centered width-1 final slot: {resolution:?}",
                        entry.source_column
                    ),
                )));
            };
            if !final_columns.insert(start) {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    format!("raw running selective final column {start} is multiply owned"),
                )));
            }
            raw_running_assignments.push(R1csIvcRawRunningAssignmentAudit {
                child: entry.child,
                logical_column: entry.logical_column,
                source_column: entry.source_column,
                final_column: start,
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
        let seed = fixed_point_seed_structure();
        let mut verifier_relation = SplitNcVerifierRelation::from_structure(&seed);
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
            verifier_relation =
                SplitNcVerifierRelation::from_parts(next_shape.rows, next_shape.columns, next_shape.polynomial);
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
        let compilation_audit = R1csIvcCompilationAudit::new(rounds, emitted_audit, pi_ccs_output_digest);
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

fn fixed_point_seed_structure() -> Structure {
    let columns = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
    let a = Mat::zero(1, columns, F::ZERO);
    let b = Mat::zero(1, columns, F::ZERO);
    let c = Mat::zero(1, columns, F::ZERO);
    neo_ccs::r1cs_to_ccs(a, b, c)
}
