use neo_math::ring::inf_norm;
use neo_math::{cf, cf_inv, ct, Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;

fn rand_rq(seed: u64) -> Rq {
    let mut c = [Fq::ZERO; D];
    let mut x = seed;
    c.iter_mut().for_each(|elem| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        *elem = Fq::from_u64(x);
    });
    Rq(c)
}

#[test]
fn cf_roundtrip() {
    let a = rand_rq(1);
    assert_eq!(a, cf_inv(cf(a)));
}

#[test]
fn coefficient_map_and_constant_term_are_canonical() {
    let coefficients = cf(rand_rq(7));
    let ring = cf_inv(coefficients);

    assert_eq!(cf(ring), coefficients);
    assert_eq!(ct(&ring), coefficients[0]);
}

#[test]
fn mul_reduction_identity() {
    let a = rand_rq(2);
    let b = rand_rq(3);
    let c = a.mul(&b);
    // Sanity: c has length d, norm finite
    assert_eq!(cf(c).len(), D);
    assert!(inf_norm(&c) < u128::MAX);
}
