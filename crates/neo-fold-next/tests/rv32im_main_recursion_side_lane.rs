#![allow(dead_code)]

#[path = "support/rv32im_n2.rs"]
mod rv32im_n2_support;

use neo_fold_next::rv32im::build_rv32im_main_recursion_side_lane_from_side_opening_public;
use neo_fold_next::rv32im::{
    RV32IM_MAIN_RECURSION_PHI_SIDE_ACTIVE, RV32IM_MAIN_RECURSION_SIDE_LANE_ACTIVE,
    RV32IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE,
};

#[test]
fn rv32im_recursion_side_lane_activity_flag_tracks_active_phi_side() {
    assert!(
        RV32IM_MAIN_RECURSION_PHI_SIDE_ACTIVE,
        "expected the authoritative phi_side lane to be active on the current RV32IM recursion path"
    );
    assert!(
        !RV32IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE,
        "side_witness should remain inactive until the explicit witness lane is wired"
    );
    assert!(
        RV32IM_MAIN_RECURSION_SIDE_LANE_ACTIVE,
        "exported side-lane activity flag must reflect the active authoritative phi_side lane"
    );
}

#[test]
fn rv32im_authoritative_side_public_maps_to_nonzero_recursion_side_lane() {
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
    let public = fixture.side_proof.opening_public().clone();

    let (side_witness, phi_side) = build_rv32im_main_recursion_side_lane_from_side_opening_public(&public)
        .expect("map authoritative side public into recursion side lane");

    assert!(
        !side_witness.is_zero(),
        "authoritative side public should map to a non-zero recursion side lane"
    );
    assert!(
        !phi_side.is_zero(),
        "authoritative side public should map to a non-zero phi_side image"
    );
    assert_eq!(
        side_witness.claim_count() as usize,
        public.evals.len(),
        "side lane must carry every authoritative eval claim"
    );
    assert_eq!(
        phi_side.commitment_count() as usize,
        public.opened_objects.len(),
        "phi_side must carry every authoritative opened-object public"
    );

    for (mapped, eval) in side_witness.claims().iter().zip(&public.evals) {
        assert_eq!(mapped.schema, eval.claim.payload.schema);
        assert_eq!(mapped.slot, eval.claim.id.slot);
        assert!(
            !mapped.point_words.is_empty(),
            "mapped side claims must carry explicit point words"
        );
        assert!(
            !mapped.payload_words.is_empty(),
            "mapped side claims must carry explicit payload words"
        );
    }

    let mut tampered = public.clone();
    tampered.opened_objects[0].commitment_context.pp_seed_digest[0] ^= 1;
    tampered.opened_objects[0].digest = tampered.opened_objects[0].expected_digest();
    tampered.digest = tampered.expected_digest();
    let err = build_rv32im_main_recursion_side_lane_from_side_opening_public(&tampered)
        .expect_err("mismatched opened-object public must be rejected");
    let err = err.to_string();
    assert!(err.contains("opened-object public"), "unexpected adapter error: {err}");
}
