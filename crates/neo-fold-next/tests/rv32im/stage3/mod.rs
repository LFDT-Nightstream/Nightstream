use crate::common::proof_cases::{
    accepted_branch, accepted_test_guard, expect_accepted_audit_failure, refresh_stage3_semantic_digests,
};
use neo_fold_next::rv32im::audit::audit_rv32im_accepted_proof;

#[test]
fn accepted_stage3_bundle_tracks_pc_adjacent_bridge() {
    let _serial = accepted_test_guard();
    let (artifact, _) = accepted_branch();
    audit_rv32im_accepted_proof(&artifact).expect("accepted proof audits");
    assert!(!artifact.stage3.bridge.continuity.is_empty());
}

#[test]
fn accepted_stage3_rejects_tampered_continuity() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.stage3.bridge.continuity[0].successor_pc = Some(artifact.stage3.bridge.continuity[0].next_pc + 8);
    expect_accepted_audit_failure(&artifact, "stage3 continuity surface mismatch");
}

#[test]
fn accepted_stage3_rejects_tampered_start_boundary_bridge() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.stage3.semantics.initial_pc = artifact.stage3.semantics.initial_pc.wrapping_add(4);
    refresh_stage3_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage3 semantic bridge mismatch");
}

#[test]
fn accepted_stage3_rejects_tampered_stage2_temporal_bridge() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.stage3.semantics.stage2_temporal_digest[0] ^= 1;
    refresh_stage3_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage3 semantic bridge mismatch");
}
