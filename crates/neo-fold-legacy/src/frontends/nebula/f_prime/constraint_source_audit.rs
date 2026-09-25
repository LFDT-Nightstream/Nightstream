//! Exact source-arm view for Nebula F-prime constraint analysis.
//!
//! Owns the two stabilized field-R1CS arms and the selective compiler audit
//! completed from the same prepared fixed-point plan. It does not authorize
//! row removal or prove lifecycle reachability.

use neo_math::{D, F};

use super::{NebulaFPrimeBranch, NebulaFPrimeRelation, NebulaFPrimeRelationError};
use crate::frontends::nebula::application::NebulaApplication;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_rows_with_alignment, SelectiveCompilerAudit, SelectiveProjectedRowsAudit, SparseR1cs,
};
use crate::paper::params::Params;

/// Read-only source relations from one successful Nebula fixed-point run.
pub struct NebulaFPrimeConstraintSourceAudit {
    arms: [SparseR1cs; 2],
    compiler: SelectiveCompilerAudit,
    fixed_point_rounds: usize,
    verifier_rows: usize,
    verifier_columns: usize,
    shared_private_fields: usize,
    public_alignment_residue: usize,
    plan_digest: [F; 4],
    application_bound: bool,
}

impl NebulaFPrimeConstraintSourceAudit {
    /// One exact physical source arm. Bootstrap and steady recursion share
    /// the same recursive relation arm.
    pub fn arm(&self, branch: NebulaFPrimeBranch) -> &SparseR1cs {
        &self.arms[branch.relation_arm_index()]
    }

    pub fn physical_arms(&self) -> &[SparseR1cs; 2] {
        &self.arms
    }

    pub fn compiler_audit(&self) -> &SelectiveCompilerAudit {
        &self.compiler
    }

    pub fn fixed_point_rounds(&self) -> usize {
        self.fixed_point_rounds
    }

    pub fn verifier_rows(&self) -> usize {
        self.verifier_rows
    }

    pub fn verifier_columns(&self) -> usize {
        self.verifier_columns
    }

    pub fn plan_digest(&self) -> [F; 4] {
        self.plan_digest
    }

    pub fn application_bound(&self) -> bool {
        self.application_bound
    }

    /// Project exact final rows from the same two source arms and require the
    /// resulting compiler plan to equal the retained fixed-point audit.
    pub fn audit_selective_rows(
        &self,
        selected_rows: &[usize],
    ) -> Result<SelectiveProjectedRowsAudit, NebulaFPrimeRelationError> {
        let projected = audit_multi_branch_selective_rows_with_alignment(
            &self.arms,
            self.shared_private_fields,
            D,
            self.public_alignment_residue,
            selected_rows,
        )?;
        if projected.compiler_audit() != &self.compiler {
            return Err(NebulaFPrimeRelationError::Geometry(
                "projected Nebula rows differ from the fixed-point compiler audit".into(),
            ));
        }
        Ok(projected)
    }
}

impl NebulaFPrimeRelation {
    /// Discover the norm-base-two Nebula fixed point and retain its exact
    /// source arms without emitting the final sparse CCS matrices.
    pub fn audit_fixed_point_constraint_sources(
        params: &Params,
        plan: &NebulaPlan,
    ) -> Result<NebulaFPrimeConstraintSourceAudit, NebulaFPrimeRelationError> {
        Self::audit_constraint_sources_inner(params, plan, None)
    }

    /// The application-bound form of [`Self::audit_fixed_point_constraint_sources`].
    pub fn audit_application_fixed_point_constraint_sources(
        params: &Params,
        plan: &NebulaPlan,
        application: &NebulaApplication,
    ) -> Result<NebulaFPrimeConstraintSourceAudit, NebulaFPrimeRelationError> {
        application.validate_for(plan)?;
        Self::audit_constraint_sources_inner(params, plan, Some(application))
    }

    fn audit_constraint_sources_inner(
        params: &Params,
        plan: &NebulaPlan,
        application: Option<&NebulaApplication>,
    ) -> Result<NebulaFPrimeConstraintSourceAudit, NebulaFPrimeRelationError> {
        if params.b() != 2 {
            return Err(NebulaFPrimeRelationError::Geometry(
                "constraint-source audit requires the fixed norm-base-two profile".into(),
            ));
        }
        let candidate = Self::discover_fixed_point(params, plan, application)?;
        let (arms, compiler) = candidate.prepared.into_source_audit_parts()?;
        let arms: [SparseR1cs; 2] = arms.try_into().map_err(|arms: Vec<SparseR1cs>| {
            NebulaFPrimeRelationError::Geometry(format!(
                "constraint-source audit expected two physical arms, got {}",
                arms.len()
            ))
        })?;
        if candidate.rounds == 0
            || compiler.rows().arms().len() != arms.len()
            || compiler.source_arm_physical_stages().len() != arms.len()
            || compiler.rows().total_rows() != candidate.verifier_rows
            || compiler.layout().total_columns() != candidate.verifier_columns
        {
            return Err(NebulaFPrimeRelationError::Geometry(
                "constraint-source audit differs from its stabilized verifier relation".into(),
            ));
        }
        for (index, arm) in arms.iter().enumerate() {
            if compiler.source_arm_physical_stages()[index] != arm.physical_stage_ranges() {
                return Err(NebulaFPrimeRelationError::Geometry(format!(
                    "constraint-source arm {index} physical stages differ from the compiler audit"
                )));
            }
        }
        let circuit = plan.circuit();
        let logical_public_fields = circuit.logical_public_input_len();
        Ok(NebulaFPrimeConstraintSourceAudit {
            arms,
            compiler,
            fixed_point_rounds: candidate.rounds,
            verifier_rows: candidate.verifier_rows,
            verifier_columns: candidate.verifier_columns,
            shared_private_fields: circuit.cols() - logical_public_fields,
            public_alignment_residue: logical_public_fields % D,
            plan_digest: plan.plan_digest(),
            application_bound: application.is_some(),
        })
    }
}
