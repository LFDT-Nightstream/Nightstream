use neo_math::{cf, cf_inv, Fq, Rq, SAction, D};
use p3_field::PrimeCharacteristicRing;

#[test]
fn rot_matches_ring_multiplication_on_vectors() {
    // random-looking but deterministic vectors
    let mut v = [Fq::ZERO; D];
    v.iter_mut().enumerate().for_each(|(i, elem)| {
        *elem = Fq::from_u64((i as u64).wrapping_mul(7919));
    });
    let mut a_coeffs = [Fq::ZERO; D];
    a_coeffs.iter_mut().enumerate().for_each(|(i, elem)| {
        *elem = Fq::from_u64((i as u64).wrapping_mul(104729));
    });
    let a = Rq(a_coeffs);

    let rot = SAction::from_ring(a);
    let lhs = rot.apply_vec(&v);

    let rhs = cf(a.mul(&cf_inv(v)));
    assert_eq!(lhs, rhs);
}

#[test]
fn s_action_identity() {
    let id = SAction::from_ring(Rq::one());
    let v = [Fq::ONE; D];
    assert_eq!(id.apply_vec(&v), v);
}
