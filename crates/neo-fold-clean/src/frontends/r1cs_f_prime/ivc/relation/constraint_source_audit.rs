//! Exact source-arm view for recursive constraint analysis.
//!
//! Owns: the stabilized base, bootstrap-recursive, and recursive field-R1CS
//! arms together with the fixed-point compiler audit produced in the same
//! discovery run.
//!
//! Does not own: full selective CCS materialization, semantic family
//! validation, row-removal authority, or proof authority.

use super::{FixedPointCandidate, R1csIvcBranch, R1csIvcRelation};
use crate::frontends::f_prime::recursive_plan::RecursiveStepImagePlan;
use crate::frontends::r1cs_f_prime::ivc::{R1csIvcError, R1csIvcFixedPointShapeAudit};
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_rows_with_alignment, R1csShape, SelectiveProjectedRowsAudit, SparseR1cs,
};
use crate::paper::params::Params;
use neo_math::D;

/// Read-only source relations from one successful fixed-point discovery.
pub struct R1csIvcConstraintSourceAudit {
    fixed_point: R1csIvcFixedPointShapeAudit,
    arms: [SparseR1cs; 3],
}

impl R1csIvcConstraintSourceAudit {
    /// The compiler ledger derived from these exact source arms.
    pub fn fixed_point(&self) -> &R1csIvcFixedPointShapeAudit {
        &self.fixed_point
    }

    /// One exact stabilized source arm.
    pub fn arm(&self, branch: R1csIvcBranch) -> &SparseR1cs {
        &self.arms[branch.index()]
    }

    /// Project exact final rows from the same prepared emitter plan that
    /// produced this fixed-point audit.
    pub fn audit_selective_rows(&self, selected_rows: &[usize]) -> Result<SelectiveProjectedRowsAudit, R1csIvcError> {
        let projected =
            audit_multi_branch_selective_rows_with_alignment(&self.arms, 0, D, self.arms[0].m_in % D, selected_rows)?;
        if projected.compiler_audit() != self.fixed_point.selective_compiler_audit() {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                "bounded selective projection differs from the fixed-point compiler audit".into(),
            )));
        }
        Ok(projected)
    }
}

impl R1csIvcRelation {
    /// Discover the fixed point once and retain its exact source arms for
    /// diagnostic constraint analysis.
    pub fn audit_fixed_point_constraint_sources(
        params: &Params,
        app: &R1csShape,
        plan: &RecursiveStepImagePlan,
    ) -> Result<R1csIvcConstraintSourceAudit, R1csIvcError> {
        let FixedPointCandidate {
            arms,
            shape,
            rounds,
            pi_ccs_output_digest,
        } = Self::discover_fixed_point(params, app, plan)?;
        let fixed_point = R1csIvcFixedPointShapeAudit::new(rounds, shape.compiler_audit, pi_ccs_output_digest);
        Ok(R1csIvcConstraintSourceAudit { fixed_point, arms })
    }
}
