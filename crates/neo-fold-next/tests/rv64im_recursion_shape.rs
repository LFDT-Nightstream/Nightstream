use neo_fold_next::rv64im::main_recursion::build_rv64im_main_recursion_verifier_key_fs;
use neo_fold_next::rv64im::{build_rv64im_recursion_shape, FamilyEvalSchemaId, ProtocolVersion, ShapeError};

#[test]
fn rv64im_recursion_shape_builder_is_deterministic() {
    let left = build_rv64im_recursion_shape().expect("build left recursion shape");
    let right = build_rv64im_recursion_shape().expect("build right recursion shape");

    assert_eq!(left, right);
    assert_eq!(left.canonical_digest(), right.canonical_digest());
}

#[test]
fn rv64im_recursion_shape_matches_current_specialization() {
    let shape = build_rv64im_recursion_shape().expect("build recursion shape");

    assert_eq!(shape.k, 14);
    assert_eq!(shape.big_k, 1);
    assert_eq!(shape.b, 2);
    assert_eq!(shape.k_decomp, 14);
    assert_eq!(shape.version, ProtocolVersion { major: 1, minor: 0 });
    assert_eq!(shape.side_families_active.len(), 6);
    assert_eq!(shape.side_slot_count(FamilyEvalSchemaId::Stage1Rows), Some(4));
    assert_eq!(shape.side_slot_count(FamilyEvalSchemaId::Stage2RegisterReads), Some(1));
    assert_eq!(shape.side_slot_count(FamilyEvalSchemaId::Stage3Continuity), Some(1));
    shape
        .validate_soundness()
        .expect("current recursion specialization must satisfy Def 14");
}

#[test]
fn rv64im_recursion_shape_digest_tracks_shape_fields() {
    let base = build_rv64im_recursion_shape().expect("build base recursion shape");
    let mut changed = base.clone();
    changed.big_k += 1;

    assert_ne!(base.canonical_digest(), changed.canonical_digest());
}

#[test]
fn rv64im_verifier_key_fs_uses_recursion_shape_digest() {
    let shape = build_rv64im_recursion_shape().expect("build recursion shape");
    let vk_fs = build_rv64im_main_recursion_verifier_key_fs().expect("build recursion verifier key fs");

    assert_eq!(vk_fs.main_lane_shape_digest, shape.canonical_digest());
}

#[test]
fn rv64im_recursion_shape_rejects_invalid_versions_and_soundness_violations() {
    let mut invalid_version = build_rv64im_recursion_shape().expect("build recursion shape");
    invalid_version.version = ProtocolVersion { major: 9, minor: 9 };
    assert!(matches!(
        invalid_version.validate_soundness(),
        Err(ShapeError::UnsupportedVersion { major: 9, minor: 9 })
    ));

    let mut invalid_soundness = build_rv64im_recursion_shape().expect("build recursion shape");
    invalid_soundness.big_k = 62;
    assert!(matches!(
        invalid_soundness.validate_soundness(),
        Err(ShapeError::SoundnessViolation { .. })
    ));
}
