#![cfg(feature = "testing")]

use neo_ajtai::mul_by_x;
use neo_math::{cf, Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn mul_by_x_matches_ring_monomial_mul() {
    let mut rng = StdRng::seed_from_u64(1234);

    for _ in 0..50 {
        let a = Rq::random_uniform(&mut rng);
        let mut col = cf(a);
        let mut nxt = [Fq::ZERO; D];

        let a_times_x = a.mul_by_monomial(1);
        let expected = cf(a_times_x);

        mul_by_x(&col, &mut nxt);

        assert_eq!(
            nxt, expected,
            "mul_by_x should produce cf(a * X), but got different result"
        );

        col = nxt;
        let a_times_x2 = a.mul_by_monomial(2);
        let expected2 = cf(a_times_x2);

        mul_by_x(&col, &mut nxt);
        assert_eq!(nxt, expected2, "Second mul_by_x should produce cf(a * X^2)");
    }
}

#[test]
fn mul_by_x_cyclotomic_identity() {
    let x = {
        let mut coeffs = vec![Fq::ZERO; D];
        coeffs[1] = Fq::ONE;
        Rq::from_field_coeffs(&coeffs)
    };

    let mut col = cf(x);
    let mut nxt = [Fq::ZERO; D];

    for _ in 1..D {
        mul_by_x(&col, &mut nxt);
        core::mem::swap(&mut col, &mut nxt);
    }

    assert_eq!(col, cf(Rq::zero() - Rq::one()), "X^D must equal -1");
}
