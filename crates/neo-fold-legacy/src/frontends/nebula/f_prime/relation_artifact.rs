//! Exact relation-artifact boundary for authoritative Nebula F′.
//!
//! The relation artifact owns the verifier-selected CCS matrices. The live
//! [`NebulaFPrimePreprocessing`] still owns the program plan and application;
//! a matrix artifact must not replace that authority.

use std::io::Write;

use super::{relation_config, NebulaFPrimePreprocessing};
use crate::paper::construction2::NebulaConfig;
use crate::relation_artifact::{
    same_structure, RelationArtifactError, RelationArtifactReceipt, VerifierKeyRelationArtifact,
};

impl NebulaFPrimePreprocessing {
    /// Write the exact verifier-owned recursive CCS relation.
    ///
    /// This artifact does not authorize a Nebula program. Validation also
    /// requires this live preprocessing object, which owns the plan and the
    /// application binding.
    pub fn write_relation_artifact(
        &self,
        writer: impl Write,
    ) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        validate_live_context(self)?;
        VerifierKeyRelationArtifact::write(self.preprocessing(), writer)
    }

    /// Return the exact recursive relation artifact as canonical JSON.
    pub fn relation_artifact_json(&self) -> Result<Vec<u8>, RelationArtifactError> {
        let mut bytes = Vec::new();
        self.write_relation_artifact(&mut bytes)?;
        Ok(bytes)
    }

    /// Validate canonical relation bytes against the live verifier context.
    pub fn validate_relation_artifact_json(
        &self,
        bytes: &[u8],
    ) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        validate_live_context(self)?;
        VerifierKeyRelationArtifact::validate_json(self.preprocessing(), bytes)
    }

    /// Return the exact recursive relation census without serializing it.
    pub fn relation_artifact_receipt(&self) -> Result<RelationArtifactReceipt, RelationArtifactError> {
        validate_live_context(self)?;
        VerifierKeyRelationArtifact::receipt(self.preprocessing())
    }
}

fn validate_live_context(prep: &NebulaFPrimePreprocessing) -> Result<(), RelationArtifactError> {
    prep.preprocessing()
        .validate_cached_structure()
        .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    prep.preprocessing()
        .validate_verifier_key_binding()
        .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;

    let relation = prep.relation();
    if !same_structure(prep.preprocessing().structure(), relation.structure()) {
        return Err(RelationArtifactError::Profile(
            "compiled relation differs from verifier-owned preprocessing".to_owned(),
        ));
    }
    if prep.preprocessing().public_input_len != Some(relation.public_input_len()) {
        return Err(RelationArtifactError::Profile(
            "compiled and verifier-owned public layouts differ".to_owned(),
        ));
    }
    if !prep.preprocessing().enforces_f_prime_recursive_link() || !prep.preprocessing().enforces_terminal_induction() {
        return Err(RelationArtifactError::Profile(
            "verifier context does not own the complete recursive F-prime induction".to_owned(),
        ));
    }

    if let Some(application) = relation.application() {
        application
            .validate_for(prep.plan())
            .map_err(|error| RelationArtifactError::Profile(error.to_string()))?;
    }

    let compiled_config = relation.nebula_config();
    let verifier_config = prep
        .preprocessing()
        .nebula()
        .ok_or_else(|| RelationArtifactError::Profile("verifier context has no Nebula plan".to_owned()))?;
    if !same_exact_config(compiled_config, verifier_config) {
        return Err(RelationArtifactError::Profile(
            "compiled relation and verifier-owned Nebula contexts differ".to_owned(),
        ));
    }

    let source_config = relation_config(prep.plan(), relation.application());
    if !same_plan_profile(&source_config, compiled_config) {
        return Err(RelationArtifactError::Profile(
            "compiled relation differs from its authoritative plan profile".to_owned(),
        ));
    }
    Ok(())
}

fn same_exact_config(left: &NebulaConfig, right: &NebulaConfig) -> bool {
    same_scalar_config(left, right)
        && left.scheme.seeded_setup() == right.scheme.seeded_setup()
        && left.scheme.lane_ranges() == right.scheme.lane_ranges()
}

fn same_plan_profile(source: &NebulaConfig, compiled: &NebulaConfig) -> bool {
    let source_ranges = source.scheme.lane_ranges();
    let compiled_ranges = compiled.scheme.lane_ranges();
    same_scalar_config(source, compiled)
        && source.scheme.seeded_setup() == compiled.scheme.seeded_setup()
        && (source_ranges.ops.len(), source_ranges.is.len(), source_ranges.fs.len())
            == (
                compiled_ranges.ops.len(),
                compiled_ranges.is.len(),
                compiled_ranges.fs.len(),
            )
}

fn same_scalar_config(left: &NebulaConfig, right: &NebulaConfig) -> bool {
    left.steps_per_segment == right.steps_per_segment
        && left.seg_max == right.seg_max
        && left.stacks == right.stacks
        && left.initial_semantic_state_digest == right.initial_semantic_state_digest
        && left.plan_digest == right.plan_digest
        && left.d_init == right.d_init
}
