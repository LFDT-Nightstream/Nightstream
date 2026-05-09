use neo_fold_next::rv32im::f_prime::{
    build_rv32im_main_recursion_verifier_key_fs, build_rv32im_main_recursion_verifier_key_fs_for_step_cap,
};
use neo_fold_next::rv32im::kernel::FamilyEvalSchemaId;
use neo_fold_next::rv32im::recursion_shape::RV32IM_RECURSION_SOUNDNESS_T;
use neo_fold_next::rv32im::{
    build_rv32im_recursion_shape, build_rv32im_recursion_shape_for_step_cap, ProtocolVersion, ShapeError,
};
use neo_fold_next::rv32im::{rv32im_simple_root_params, rv32im_simple_root_params_for_step_cap};

#[test]
fn rv32im_recursion_shape_builder_is_deterministic() {
    let left = build_rv32im_recursion_shape().expect("build left recursion shape");
    let right = build_rv32im_recursion_shape().expect("build right recursion shape");

    assert_eq!(left, right);
    assert_eq!(left.canonical_digest(), right.canonical_digest());
}

#[test]
fn rv32im_recursion_shape_matches_current_specialization() {
    let shape = build_rv32im_recursion_shape().expect("build recursion shape");
    let params = rv32im_simple_root_params();

    assert_eq!(shape.soundness_k, params.k_rho);
    assert_eq!(shape.soundness_big_k, 1);
    assert_eq!(shape.b, params.b as u8);
    assert_eq!(shape.decomposition_k, params.k_rho as u8);
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
fn rv32im_recursion_shape_digest_tracks_shape_fields() {
    let base = build_rv32im_recursion_shape().expect("build base recursion shape");
    let mut changed = base.clone();
    changed.soundness_big_k += 1;

    assert_ne!(base.canonical_digest(), changed.canonical_digest());
}

#[test]
fn rv32im_recursion_shape_digest_tracks_step_cap() {
    let single = build_rv32im_recursion_shape_for_step_cap(1).expect("build single-step recursion shape");
    let multi = build_rv32im_recursion_shape_for_step_cap(5).expect("build multi-step recursion shape");
    let multi_params = rv32im_simple_root_params_for_step_cap(5);

    assert_ne!(single.canonical_digest(), multi.canonical_digest());
    assert_eq!(single.step_cap, 1);
    assert_eq!(multi.step_cap, 5);
    assert_eq!(multi.soundness_k, multi_params.k_rho);
    assert_eq!(multi.decomposition_k, multi_params.k_rho as u8);
}

#[test]
fn rv32im_verifier_key_fs_uses_recursion_shape_digest() {
    let shape = build_rv32im_recursion_shape().expect("build recursion shape");
    let vk_fs = build_rv32im_main_recursion_verifier_key_fs().expect("build recursion verifier key fs");

    assert_eq!(vk_fs.main_lane_shape_digest, shape.canonical_digest());
}

#[test]
fn rv32im_verifier_key_fs_tracks_step_cap() {
    let multi_shape = build_rv32im_recursion_shape_for_step_cap(5).expect("build multi-step recursion shape");
    let multi_vk_fs =
        build_rv32im_main_recursion_verifier_key_fs_for_step_cap(5).expect("build multi-step recursion verifier key");

    assert_eq!(multi_vk_fs.main_lane_shape_digest, multi_shape.canonical_digest());
    assert_eq!(multi_vk_fs.step_cap, 5);
}

#[test]
fn rv32im_recursion_shape_rejects_invalid_versions_and_soundness_violations() {
    let mut invalid_version = build_rv32im_recursion_shape().expect("build recursion shape");
    invalid_version.version = ProtocolVersion { major: 9, minor: 9 };
    assert!(matches!(
        invalid_version.validate_soundness(),
        Err(ShapeError::UnsupportedVersion { major: 9, minor: 9 })
    ));

    let mut invalid_soundness = build_rv32im_recursion_shape().expect("build recursion shape");
    let big_b = (invalid_soundness.b as u64).pow(invalid_soundness.decomposition_k as u32);
    let per_claim_expansion = (RV32IM_RECURSION_SOUNDNESS_T as u64) * ((invalid_soundness.b as u64) - 1);
    let max_valid_k_plus_big_k = (big_b - 1) / per_claim_expansion;
    invalid_soundness.soundness_big_k = (max_valid_k_plus_big_k + 1)
        .saturating_sub(invalid_soundness.soundness_k as u64)
        .try_into()
        .expect("soundness_big_k fits in u32");

    assert!(matches!(
        invalid_soundness.validate_soundness(),
        Err(ShapeError::SoundnessViolation { .. })
    ));
}
