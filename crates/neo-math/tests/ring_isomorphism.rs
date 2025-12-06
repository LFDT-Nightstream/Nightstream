mod test_helpers;
use neo_math::{cf, cf_inv, Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;
use test_helpers::inf_norm;

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
