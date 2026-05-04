use crate::common::proof_cases::{
    accepted_alu, accepted_test_guard, alu_input, expect_accepted_audit_failure, refresh_accepted_artifact_digest,
    refresh_step_composition_surface_digest,
};
use neo_fold_next::rv32im::audit::{
    audit_rv32im_accepted_proof, audit_rv32im_accepted_proof_against_input, audit_rv32im_accepted_proof_with_perf,
};

#[test]
fn accepted_artifact_audit_replays_transcript_without_input_audit() {
    let _serial = accepted_test_guard();
    let (artifact, _) = accepted_alu();
    let perf = audit_rv32im_accepted_proof_with_perf(&artifact).expect("accepted audit");
    assert!(perf.public_claim_digests_ms >= 0.0);
    assert!(perf.summary_consistency_ms >= 0.0);
}

#[test]
fn audit_path_checks_public_input_only_when_requested() {
    let _serial = accepted_test_guard();
    let input = alu_input();
    let (artifact, audit) = accepted_alu();
    audit_rv32im_accepted_proof(&artifact).expect("accepted audit");
    audit_rv32im_accepted_proof_against_input(&input, &artifact, &audit).expect("input audit");
}

#[test]
fn accepted_artifact_audit_rejects_tampered_transcript() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.transcript.events[0].message.push(0xA5);
    expect_accepted_audit_failure(&artifact, "accepted proof artifact digest mismatch");
}

#[test]
fn accepted_artifact_digest_binds_transcript_contents() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    let original_digest = artifact.digest;
    artifact.transcript.events[0].message.push(0xA5);
    refresh_accepted_artifact_digest(&mut artifact);
    assert_ne!(artifact.digest, original_digest);
}

#[test]
fn accepted_artifact_audit_rejects_tampered_transcript_even_if_digest_is_refreshed() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.transcript.events[0].message.push(0xA5);
    refresh_accepted_artifact_digest(&mut artifact);
    expect_accepted_audit_failure(&artifact, "transcript replay");
}

#[test]
fn accepted_artifact_audit_rejects_tampered_step_composition_surface() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.step_composition.last_real_step_index ^= 1;
    refresh_step_composition_surface_digest(&mut artifact);
    expect_accepted_audit_failure(&artifact, "step composition surface mismatch");
}
