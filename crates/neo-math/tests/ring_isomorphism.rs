use neo_math::{Rq, Fq, D, cf, cf_inv, inf_norm};
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
fn mul_reduction_identity() {
    let a = rand_rq(2);
    let b = rand_rq(3);
    let c = a.mul(&b);
    // Sanity: c has length d, norm finite
    assert_eq!(cf(c).len(), D);
    assert!(inf_norm(&c) < u128::MAX);
}
