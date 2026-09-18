//! Fixed, selectively lowered relation used by the generic R1CS IVC path.

use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::{shape, R1csIvcError};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::lowering::MultiBranchLowNormR1cs;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_low_norm_shape_with_alignment,
    build_multi_branch_selective_low_norm_r1cs_with_alignment, R1csShape, SelectiveLowNormShape, SparseR1cs,
};
use crate::lifecycle::Preprocessing;
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcVerifierRelation;
use crate::paper::relations::{CcsInstance, Structure};

/// Hard ceiling shared with the production Road-A relation.
pub const R1CS_IVC_COMMITTED_COORDINATE_BUDGET: usize = 16_000_000;

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
    preprocessing_digest: Option<[F; 4]>,
}

impl R1csIvcRelation {
    pub fn compile_fixed_point(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<Self, R1csIvcError> {
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
        let mut last_output = (verifier_relation.n(), verifier_relation.m());
        for round in 0..MAX_ROUNDS {
            let input_signature = verifier_relation_signature(&verifier_relation);
            let arms = shape::synthesize_arm_shapes(params, &verifier_relation, app, plan)?;
            let next_shape = audit_multi_branch_selective_low_norm_shape_with_alignment(&arms, 0, D, arms[0].m_in % D)?;
            last_output = (next_shape.rows, next_shape.columns);
            if round > 0 && input_signature == shape_signature(&next_shape) {
                return Self::compile_arms_selected(arms, next_shape);
            }
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

    fn compile_arms_selected(arms: [SparseR1cs; 3], shape: SelectiveLowNormShape) -> Result<Self, R1csIvcError> {
        let arm_shapes = std::array::from_fn(|index| ArmShape {
            rows: arms[index].n,
            columns: arms[index].m,
            public_columns: arms[index].m_in,
        });
        let public_fields = arms[0].m_in;
        if shape.audit.total_coordinates > R1CS_IVC_COMMITTED_COORDINATE_BUDGET {
            return Err(R1csIvcError::BudgetExceeded {
                required: shape.audit.total_coordinates,
                budget: R1CS_IVC_COMMITTED_COORDINATE_BUDGET,
            });
        }
        let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&arms, 0, D, public_fields % D)?;
        if relation_signature(relation.structure()) != shape_signature(&shape) {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "shape-only selective audit differs from emitted relation".into(),
            )));
        }
        if relation.structure().m > R1CS_IVC_COMMITTED_COORDINATE_BUDGET {
            return Err(R1csIvcError::BudgetExceeded {
                required: relation.structure().m,
                budget: R1CS_IVC_COMMITTED_COORDINATE_BUDGET,
            });
        }
        Ok(Self {
            relation,
            arm_shapes,
            preprocessing_digest: None,
        })
    }

    pub fn structure(&self) -> &Structure {
        self.relation.structure()
    }

    pub fn public_input_len(&self) -> usize {
        self.relation.public_input_len()
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

fn relation_signature(structure: &Structure) -> (usize, usize, usize, u32) {
    (structure.n, structure.m, structure.t(), structure.max_degree())
}

fn verifier_relation_signature(relation: &SplitNcVerifierRelation) -> (usize, usize, usize, u32) {
    (relation.n(), relation.m(), relation.t(), relation.max_degree())
}

fn shape_signature(shape: &SelectiveLowNormShape) -> (usize, usize, usize, u32) {
    (
        shape.rows,
        shape.columns,
        shape.polynomial.arity(),
        shape.polynomial.max_degree(),
    )
}

fn fixed_point_seed_structure() -> Structure {
    let columns = crate::paper::f_prime::r1cs::F_PRIME_PUBLIC_INPUT_LEN;
    let a = Mat::zero(1, columns, F::ZERO);
    let b = Mat::zero(1, columns, F::ZERO);
    let c = Mat::zero(1, columns, F::ZERO);
    neo_ccs::r1cs_to_ccs(a, b, c)
}
