#![cfg(feature = "testing")]

use neo_ajtai::rot_step;
use neo_math::{cf, Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn rot_step_phi81_matches_ring_mul_by_x() {
    let mut rng = StdRng::seed_from_u64(1234);

    for _ in 0..50 {
        let a = Rq::random_uniform(&mut rng);
        let mut col = cf(a);
        let mut nxt = [Fq::ZERO; D];

        let a_times_x = a.mul_by_monomial(1);
        let expected = cf(a_times_x);

        rot_step(&col, &mut nxt);

        assert_eq!(
            nxt, expected,
            "rot_step should produce cf(a * X), but got different result"
        );

        col = nxt;
        let a_times_x2 = a.mul_by_monomial(2);
        let expected2 = cf(a_times_x2);

        rot_step(&col, &mut nxt);
        assert_eq!(nxt, expected2, "Second rot_step should produce cf(a * X^2)");
    }
}

#[test]
fn rot_step_cyclotomic_identity() {
    let x = {
        let mut coeffs = vec![Fq::ZERO; D];
        coeffs[1] = Fq::ONE;
        Rq::from_field_coeffs(&coeffs)
    };

    let mut col = cf(x);
    let mut nxt = [Fq::ZERO; D];

    for _ in 0..53 {
        rot_step(&col, &mut nxt);
        core::mem::swap(&mut col, &mut nxt);
    }

    let x27 = {
        let mut coeffs = vec![Fq::ZERO; D];
        coeffs[27] = Fq::ONE;
        Rq::from_field_coeffs(&coeffs)
    };
    let neg_x27_minus_1 = cf(Rq::zero() - x27 - Rq::one());

    assert_eq!(col, neg_x27_minus_1, "X^54 should equal -X^27 - 1 under Φ₈₁ reduction");
}
