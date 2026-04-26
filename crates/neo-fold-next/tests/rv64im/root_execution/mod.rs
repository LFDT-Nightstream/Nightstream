use crate::common::proof_cases::{
    accepted_branch, accepted_test_guard, expect_accepted_audit_failure, refresh_step_composition_surface_digest,
};
use neo_fold_next::rv64im::Rv64imAcceptedProofArtifact;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn refresh_root_execution_bundle_digest(artifact: &mut Rv64imAcceptedProofArtifact) {
    artifact.root_execution.digest = artifact.root_execution.expected_digest();
    artifact.step_composition.root_execution_digest = artifact.root_execution.digest;
    refresh_step_composition_surface_digest(artifact);
}

fn refresh_row_local_ccs_acceptance_summary(artifact: &mut Rv64imAcceptedProofArtifact) {
    let summary = &mut artifact.root_execution.row_local_ccs_acceptance;
    summary.acceptance_count = summary.acceptances.len() as u64;
    summary.first_acceptance_digest = summary
        .acceptances
        .first()
        .map(|acceptance| acceptance.digest);
    summary.last_acceptance_digest = summary
        .acceptances
        .last()
        .map(|acceptance| acceptance.digest);
    summary.digest = summary.expected_digest();
    refresh_root_execution_bundle_digest(artifact);
}

fn refresh_execution_semantics_refinement_summary(artifact: &mut Rv64imAcceptedProofArtifact) {
    let summary = &mut artifact.root_execution.execution_semantics_refinement;
    summary.refinement_count = summary.refinements.len() as u64;
    summary.first_refinement_digest = summary
        .refinements
        .first()
        .map(|refinement| refinement.digest);
    summary.last_refinement_digest = summary
        .refinements
        .last()
        .map(|refinement| refinement.digest);
    summary.digest = summary.expected_digest();
    refresh_root_execution_bundle_digest(artifact);
}

#[test]
fn accepted_root_execution_rejects_tampered_semantic_row_values() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.root_execution.semantic_rows[0].values[0] += F::ONE;

    expect_accepted_audit_failure(&artifact, "root execution semantic-row digest mismatch");
}

#[test]
fn accepted_root_execution_rejects_tampered_binding_row_opening_digest() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.root_execution.prepared_step_bindings.bindings[0].row_opening_digest[0] ^= 1;

    expect_accepted_audit_failure(&artifact, "root execution prepared-step bindings mismatch");
}

#[test]
fn accepted_root_execution_rejects_tampered_acceptance_public_step_digest_after_rebinding() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    let acceptance = &mut artifact.root_execution.row_local_ccs_acceptance.acceptances[0];
    acceptance.public_step_digest[0] ^= 1;
    acceptance.digest = acceptance.expected_digest();
    refresh_row_local_ccs_acceptance_summary(&mut artifact);

    expect_accepted_audit_failure(&artifact, "root execution row-local CCS acceptance mismatch");
}

#[test]
fn accepted_root_execution_rejects_tampered_refinement_semantic_row_digest_after_rebinding() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    let refinement = &mut artifact
        .root_execution
        .execution_semantics_refinement
        .refinements[0];
    refinement.semantic_row_digest[0] ^= 1;
    refinement.digest = refinement.expected_digest();
    refresh_execution_semantics_refinement_summary(&mut artifact);

    expect_accepted_audit_failure(&artifact, "root execution semantics refinement mismatch");
}
