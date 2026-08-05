//! Fixed, selectively lowered relation used by the generic R1CS IVC path.

mod pi_dec_audit;
mod source_audit;

pub use pi_dec_audit::{
    R1csIvcPiDecCanonicalXSelectiveRowsAudit, R1csIvcPiDecSelectiveRowsAudit, R1csIvcPiDecSourceRowAudit,
    R1csIvcPiDecSourceRowsAudit,
};
pub use source_audit::{R1csIvcRawRunningAssignmentAudit, R1csIvcRawRunningEncodingAudit};

use neo_math::{D, F};

use super::compilation_audit::{
    ArmShapeAudit, FixedPointRoundAudit, R1csIvcCompilationAudit, R1csIvcFixedPointShapeAudit, RelationHeaderAudit,
};
use super::pi_ccs_output_digest_audit;
use super::{shape, R1csIvcError};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::lowering::MultiBranchLowNormR1cs;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, R1csShape, SelectiveLowNormShape, SparseR1cs,
};
use crate::lifecycle::Preprocessing;
use crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_circuit::PiCcsVerifierRelation;
use crate::paper::relations::{CcsInstance, Structure};

/// Hard ceiling shared with the production Road-A relation.
pub const R1CS_IVC_COMMITTED_COORDINATE_BUDGET: usize = 16_000_000;

struct FixedPointCandidate {
    arms: [SparseR1cs; 3],
    shape: SelectiveLowNormShape,
    rounds: Vec<FixedPointRoundAudit>,
    pi_ccs_output_digest: super::PiCcsOutputDigestAudit,
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
        let seed_rows = 2;
        let mut verifier_relation = PiCcsVerifierRelation::from_parts(
            seed_rows,
            F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
            super::super::selective::selective_polynomial(),
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
                });
            }
            folded_public_input_len = next_shape.public_input_len;
            verifier_relation =
                PiCcsVerifierRelation::from_parts(next_shape.rows, next_shape.columns, next_shape.polynomial);
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
        if let Some(row) = self.relation.first_unsatisfied_row(&assignment) {
            let owner = self
                .compilation_audit
                .rows()
                .emitted_runs()
                .iter()
                .find(|run| run.emitted_rows().contains(&row))
                .map(|run| {
                    let stage = run.arm().and_then(|arm| {
                        run.source_stage_occurrence().and_then(|occurrence| {
                            self.compilation_audit
                                .source_arm_physical_stages()
                                .get(arm)?
                                .get(occurrence)
                                .map(|stage| stage.path())
                        })
                    });
                    format!(
                        "family={:?}, arm={:?}, rewrite={:?}, source_stage={stage:?}",
                        run.family(),
                        run.arm(),
                        run.rewrite_id()
                    )
                })
                .unwrap_or_else(|| "no compiler owner".into());
            return Err(R1csIvcError::UnsatisfiedEncodedRelation { branch, row, owner });
        }
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
    relation: &PiCcsVerifierRelation,
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

fn source_row_owner(
    mapping: &crate::frontends::r1cs_f_prime::SelectiveArmRowMappingAudit,
    row: usize,
) -> Option<&crate::frontends::r1cs_f_prime::SelectiveSourceRowRunAudit> {
    let runs = mapping.source_runs();
    let index = runs.partition_point(|run| run.source_rows().end <= row);
    runs.get(index)
        .filter(|run| run.source_rows().contains(&row))
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
