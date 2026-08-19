//! `adv` lane-commitment tuple on the claims — spec §5.1/§5.2 R1, M1a
//! (inert) slice: the type, its serde compatibility, and the leaf-digest
//! absorb rule. Folding mirrors and decider openings are M1b.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_fold_clean::paper::digest::{ccs_claim_digest, nebula_lane_leaf_digests};
use neo_fold_clean::paper::relations::CcsClaim;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// A distinct, deterministic commitment per seed — shape matches the
/// Goldilocks preset (κ = 18, d = 54) but contents are arbitrary: leaves
/// bind bytes, not validity.
fn commitment(seed: u64) -> Commitment {
    let (d, kappa) = (54, 18);
    Commitment {
        d,
        kappa,
        data: (0..(d * kappa) as u64)
            .map(|i| F::from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i)))
            .collect(),
    }
}

fn lanes(ops_seed: u64, is_seed: u64, fs_seed: u64) -> LaneCommitments<Commitment> {
    LaneCommitments {
        ops: commitment(ops_seed),
        is: commitment(is_seed),
        fs: commitment(fs_seed),
    }
}

fn claim(adv: Option<LaneCommitments<Commitment>>) -> CcsClaim {
    CcsClaim {
        c: commitment(0),
        x: vec![F::ONE; 8],
        m_in: 8,
        adv,
    }
}

/// Spec §6.1 tag discipline: `is` and `fs` share the lane-neutral mem
/// tag, so equal commitments give equal leaves — the formula identity the
/// cross-segment boundary equality consumes (security-note Cor. 1.1).
/// `ops` has its own domain: the same commitment leafs differently there.
#[test]
fn mem_leaves_are_lane_neutral_and_ops_is_domain_separated() {
    let same = lanes(7, 7, 7);
    let [ops, is, fs] = nebula_lane_leaf_digests(&same);
    assert_eq!(is, fs, "identical is/fs commitments must produce identical leaves");
    assert_ne!(ops, is, "ops leaf must be domain-separated from the mem leaves");
}

#[test]
fn leaves_bind_each_lane() {
    let base = nebula_lane_leaf_digests(&lanes(1, 2, 3));
    let ops_changed = nebula_lane_leaf_digests(&lanes(9, 2, 3));
    let is_changed = nebula_lane_leaf_digests(&lanes(1, 9, 3));
    let fs_changed = nebula_lane_leaf_digests(&lanes(1, 2, 9));
    assert_ne!(base[0], ops_changed[0]);
    assert_eq!(base[1], ops_changed[1], "ops change must not touch the is leaf");
    assert_ne!(base[1], is_changed[1]);
    assert_ne!(base[2], fs_changed[2]);
}

/// Absorb rule R1: a present tuple changes the claim digest, and each
/// lane's content reaches it (through its leaf). A `None` claim keeps the
/// pre-Nebula preimage — pinned here by asserting Some ≠ None so the
/// present-only extension is observable.
#[test]
fn claim_digest_binds_adv_presence_and_content() {
    let without = ccs_claim_digest(&claim(None));
    let with = ccs_claim_digest(&claim(Some(lanes(1, 2, 3))));
    let with_other_fs = ccs_claim_digest(&claim(Some(lanes(1, 2, 4))));
    assert_ne!(without, with, "present adv must extend the claim digest");
    assert_ne!(with, with_other_fs, "an fs lane change must reach the claim digest");
}

/// Serde compatibility: pre-Nebula serialized claims (no `adv` field)
/// deserialize to `None`; a present tuple round-trips exactly. This is
/// the "existing artifacts untouched" acceptance of §13 step 3.
#[test]
fn serde_default_and_round_trip() {
    let legacy = serde_json::to_value(claim(None)).map(|mut v| {
        v.as_object_mut()
            .expect("claim serializes as an object")
            .remove("adv");
        v
    });
    let legacy: CcsClaim = serde_json::from_value(legacy.unwrap()).expect("legacy format must deserialize");
    assert_eq!(legacy.adv, None);

    let full = claim(Some(lanes(1, 2, 3)));
    let round: CcsClaim = serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
    assert_eq!(round.adv, full.adv);
}
