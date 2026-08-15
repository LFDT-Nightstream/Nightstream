//! Accepted source assignments for external constraint replay.

use neo_math::F;

use super::NebulaFPrimeBranch;

/// Exact normalized source assignment from one accepted Nebula F-prime step.
///
/// This diagnostic value is not a proof. A consumer must replay it against
/// the exact exported source arm before it is used as a solver background.
#[derive(Debug)]
pub struct NebulaFPrimeConstraintWitnessAudit {
    branch: NebulaFPrimeBranch,
    source_assignment: Vec<F>,
}

impl NebulaFPrimeConstraintWitnessAudit {
    pub fn branch(&self) -> NebulaFPrimeBranch {
        self.branch
    }

    pub fn source_assignment(&self) -> &[F] {
        &self.source_assignment
    }
}

pub(super) fn retain_first_constraint_witness(
    witnesses: Option<&mut Vec<NebulaFPrimeConstraintWitnessAudit>>,
    branch: NebulaFPrimeBranch,
    source_assignment: Vec<F>,
) {
    let Some(witnesses) = witnesses else {
        return;
    };
    if witnesses.iter().any(|witness| witness.branch == branch) {
        return;
    }
    witnesses.push(NebulaFPrimeConstraintWitnessAudit {
        branch,
        source_assignment,
    });
}
