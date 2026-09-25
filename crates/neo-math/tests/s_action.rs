use neo_math::{cf, cf_inv, Fq, Rq, SAction, D, K};
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

#[test]
fn s_action_composition_matches_sequential_application() {
    let a = SAction::from_ring(Rq::from_field_scalar(Fq::from_u64(7)));
    let b = SAction::from_ring(Rq::from_field_scalar(Fq::from_u64(11)));
    let v = std::array::from_fn(|i| Fq::from_u64((i + 1) as u64));

    assert_eq!(a.compose(&b).apply_vec(&v), a.apply_vec(&b.apply_vec(&v)));
}

#[test]
fn scalar_action_is_coefficient_scaling() {
    let scalar = Fq::from_u64(13);
    let v = std::array::from_fn(|i| Fq::from_u64((2 * i + 1) as u64));

    assert_eq!(
        SAction::scalar(scalar).apply_vec(&v),
        std::array::from_fn(|i| scalar * v[i])
    );
}

#[test]
fn k_action_preserves_canonical_zero_padding() {
    let action = SAction::scalar(Fq::from_u64(3));
    let input = vec![K::ZERO; D + 4];
    let output = action
        .apply_k_vec(&input)
        .expect("zero padding is canonical");

    assert_eq!(output.len(), input.len());
    assert!(output[D..].iter().all(|value| *value == K::ZERO));
}
