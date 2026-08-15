//! Checked saved assignments for exact Nebula source relations.
//!
//! A saved assignment is untrusted diagnostic input. This module checks its
//! identity and replays every source row before it can become solver input.

use neo_fold_clean::engine::r1cs_circuit::Var;
use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeConstraintSourceAudit};
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use recursive_constraint_minimizer::GOLDILOCKS_MODULUS;
use serde::{Deserialize, Serialize};

use super::{validate_nebula_stage_vocabulary, ExportError, SparseProblemExporter};

pub const NEBULA_SOURCE_ASSIGNMENT_SCHEMA: &str = "nightstream/nebula-source-assignment/v1";

/// One of the two physical Nebula source relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NebulaPhysicalSourceArm {
    Base,
    Recursive,
}

impl NebulaPhysicalSourceArm {
    fn for_branch(branch: NebulaFPrimeBranch) -> Self {
        match branch {
            NebulaFPrimeBranch::Base => Self::Base,
            NebulaFPrimeBranch::BootstrapRecursive | NebulaFPrimeBranch::Recursive => Self::Recursive,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceAssignmentArtifact {
    schema: String,
    profile: String,
    source_arm: NebulaPhysicalSourceArm,
    source_artifact_digest: String,
    field_modulus: String,
    source_rows: usize,
    source_columns: usize,
    public_input_count: usize,
    constant_one_column: usize,
    values: Vec<String>,
}

/// A saved assignment that passed exact identity checks and full Rust replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckedNebulaSourceAssignment {
    artifact: SourceAssignmentArtifact,
    values: Vec<F>,
}

impl CheckedNebulaSourceAssignment {
    pub fn profile(&self) -> &str {
        &self.artifact.profile
    }

    pub fn source_arm(&self) -> NebulaPhysicalSourceArm {
        self.artifact.source_arm
    }

    pub fn source_artifact_digest(&self) -> &str {
        &self.artifact.source_artifact_digest
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Serialize the checked artifact with canonical decimal field values.
    pub fn to_json_vec(&self) -> Result<Vec<u8>, ExportError> {
        serde_json::to_vec_pretty(&self.artifact)
            .map_err(|error| ExportError::new(format!("cannot serialize checked source assignment: {error}")))
    }
}

/// Check and bind an in-memory assignment to one exact Nebula source arm.
pub fn bind_nebula_source_assignment(
    audit: &NebulaFPrimeConstraintSourceAudit,
    branch: NebulaFPrimeBranch,
    profile: &str,
    values: &[F],
) -> Result<CheckedNebulaSourceAssignment, ExportError> {
    validate_profile(profile)?;
    validate_nebula_stage_vocabulary(audit, branch)?;
    let arm = audit.arm(branch);
    let exporter = SparseProblemExporter::new(arm)?;
    let artifact = SourceAssignmentArtifact {
        schema: NEBULA_SOURCE_ASSIGNMENT_SCHEMA.to_owned(),
        profile: profile.to_owned(),
        source_arm: NebulaPhysicalSourceArm::for_branch(branch),
        source_artifact_digest: exporter.artifact_digest().to_owned(),
        field_modulus: GOLDILOCKS_MODULUS.to_owned(),
        source_rows: arm.n,
        source_columns: arm.m,
        public_input_count: arm.m_in,
        constant_one_column: Var::ONE.col(),
        values: values
            .iter()
            .map(|value| value.as_canonical_u64().to_string())
            .collect(),
    };
    validate_artifact(audit, branch, profile, artifact)
}

/// Parse an untrusted saved assignment and replay it against the exact arm.
///
/// Bootstrap and steady-recursive branches both expect the recursive physical
/// arm, so one checked bootstrap assignment can be loaded for either branch.
pub fn load_nebula_source_assignment(
    audit: &NebulaFPrimeConstraintSourceAudit,
    branch: NebulaFPrimeBranch,
    expected_profile: &str,
    json: &[u8],
) -> Result<CheckedNebulaSourceAssignment, ExportError> {
    validate_profile(expected_profile)?;
    let artifact = serde_json::from_slice(json)
        .map_err(|error| ExportError::new(format!("cannot parse source assignment: {error}")))?;
    validate_artifact(audit, branch, expected_profile, artifact)
}

fn validate_artifact(
    audit: &NebulaFPrimeConstraintSourceAudit,
    branch: NebulaFPrimeBranch,
    expected_profile: &str,
    artifact: SourceAssignmentArtifact,
) -> Result<CheckedNebulaSourceAssignment, ExportError> {
    validate_nebula_stage_vocabulary(audit, branch)?;
    let arm = audit.arm(branch);
    let expected_source_arm = NebulaPhysicalSourceArm::for_branch(branch);
    let exporter = SparseProblemExporter::new(arm)?;

    if artifact.schema != NEBULA_SOURCE_ASSIGNMENT_SCHEMA {
        return Err(ExportError::new(format!(
            "unsupported source-assignment schema {:?}; expected {NEBULA_SOURCE_ASSIGNMENT_SCHEMA:?}",
            artifact.schema
        )));
    }
    if artifact.profile != expected_profile {
        return Err(ExportError::new(format!(
            "source-assignment profile {:?} differs from expected profile {expected_profile:?}",
            artifact.profile
        )));
    }
    if artifact.source_arm != expected_source_arm {
        return Err(ExportError::new(format!(
            "source-assignment arm {:?} differs from expected arm {expected_source_arm:?}",
            artifact.source_arm
        )));
    }
    if artifact.source_artifact_digest != exporter.artifact_digest() {
        return Err(ExportError::new(
            "source-assignment relation identity does not match the exact Rust arm",
        ));
    }
    if artifact.field_modulus != GOLDILOCKS_MODULUS {
        return Err(ExportError::new(format!(
            "source-assignment field modulus must be {GOLDILOCKS_MODULUS}"
        )));
    }
    if artifact.source_rows != arm.n
        || artifact.source_columns != arm.m
        || artifact.public_input_count != arm.m_in
        || artifact.constant_one_column != Var::ONE.col()
    {
        return Err(ExportError::new(
            "source-assignment dimensions do not match the exact Rust arm",
        ));
    }
    if artifact.values.len() != arm.m {
        return Err(ExportError::new(format!(
            "source assignment has {} values; expected {}",
            artifact.values.len(),
            arm.m
        )));
    }

    let values = artifact
        .values
        .iter()
        .enumerate()
        .map(|(column, value)| parse_canonical_value(column, value))
        .collect::<Result<Vec<_>, _>>()?;
    if values[artifact.constant_one_column] != F::ONE {
        return Err(ExportError::new(
            "source assignment must set the constant-one column to one",
        ));
    }
    replay(arm, &values)?;
    Ok(CheckedNebulaSourceAssignment { artifact, values })
}

fn validate_profile(profile: &str) -> Result<(), ExportError> {
    if profile.trim().is_empty() {
        return Err(ExportError::new("source-assignment profile must not be empty"));
    }
    Ok(())
}

fn parse_canonical_value(column: usize, value: &str) -> Result<F, ExportError> {
    let modulus = GOLDILOCKS_MODULUS
        .parse::<u64>()
        .expect("the fixed Goldilocks modulus is a u64");
    let residue = value.parse::<u64>().map_err(|_| {
        ExportError::new(format!(
            "source-assignment column {column} has a non-decimal value {value:?}"
        ))
    })?;
    if residue >= modulus || value != residue.to_string() {
        return Err(ExportError::new(format!(
            "source-assignment column {column} has a noncanonical value {value:?}"
        )));
    }
    Ok(F::from_u64(residue))
}

fn replay(arm: &SparseR1cs, values: &[F]) -> Result<(), ExportError> {
    arm.is_satisfied_by(values)
        .map_err(|error| ExportError::new(format!("source assignment failed complete Rust replay: {error}")))
}
