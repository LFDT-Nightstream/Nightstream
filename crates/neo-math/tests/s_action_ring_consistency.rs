use neo_math::{
    ring::{cf, Rq},
    s_action::SAction,
    Fq, D,
};
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn s_action_matches_ring_mul() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..256 {
        let mut ac = [Fq::ZERO; D];
        let mut bc = [Fq::ZERO; D];
        for i in 0..D {
            ac[i] = Fq::from_u64(rng.random());
            bc[i] = Fq::from_u64(rng.random());
        }
        let a = Rq(ac);
        let b = Rq(bc);
        let left = SAction::from_ring(a).apply_vec(&cf(b));
        let right = cf(a.mul(&b));
        assert_eq!(left, right, "S-action identity failed for iteration");
    }
}

#[test]
fn s_action_distributive() {
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..64 {
        // Test that SAction(a).apply(b + c) = SAction(a).apply(b) + SAction(a).apply(c)
        let mut ac = [Fq::ZERO; D];
        let mut bc = [Fq::ZERO; D];
        let mut cc = [Fq::ZERO; D];

        for i in 0..D {
            ac[i] = Fq::from_u64(rng.random());
            bc[i] = Fq::from_u64(rng.random());
            cc[i] = Fq::from_u64(rng.random());
        }

        let a = Rq(ac);
        let b_vec = bc;
        let c_vec = cc;

        // Compute b + c in coefficient space
        let mut bc_sum = [Fq::ZERO; D];
        for i in 0..D {
            bc_sum[i] = b_vec[i] + c_vec[i];
        }

        let s_action = SAction::from_ring(a);

        // Left side: SAction(a).apply(b + c)
        let left = s_action.apply_vec(&bc_sum);

        // Right side: SAction(a).apply(b) + SAction(a).apply(c)
        let sb = s_action.apply_vec(&b_vec);
        let sc = s_action.apply_vec(&c_vec);
        let mut right = [Fq::ZERO; D];
        for i in 0..D {
            right[i] = sb[i] + sc[i];
        }

        assert_eq!(left, right, "S-action distributivity failed");
    }
}

#[test]
fn s_action_identity_element() {
    // Test that SAction(1).apply(v) = v for the ring identity
    let one = Rq::one();
    let s_action = SAction::from_ring(one);

    let mut rng = StdRng::seed_from_u64(123);
    for _ in 0..32 {
        let mut v = [Fq::ZERO; D];
        for i in 0..D {
            v[i] = Fq::from_u64(rng.random());
        }

        let result = s_action.apply_vec(&v);
        assert_eq!(result, v, "S-action with identity should preserve vector");
    }
}
