// neo-commit/tests/random_linear_combo_edge.rs
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::F;
use neo_modint::ModInt;
use neo_ring::RingElement;
use p3_field::PrimeCharacteristicRing;

fn zero(n: usize) -> RingElement<ModInt> { RingElement::from_scalar(ModInt::from_u64(0), n) }
fn one(n: usize) -> RingElement<ModInt> { RingElement::from_scalar(ModInt::from_u64(1), n) }

#[test]
fn random_linear_combo_empty_left_passes_through_scaled_right() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);
    let n = params.n;

    let c1: Vec<_> = vec![];                     // empty (degenerate)
    let c2: Vec<_> = vec![one(n); params.k];     // non-empty
    let rho = F::from_u64(7);

    let out = comm.random_linear_combo(&c1, &c2, rho);
    assert_eq!(out.len(), c2.len());
    // out[i] == 0 + c2[i] * rho
    for (o, r) in out.iter().zip(c2.iter()) {
        let expected = zero(n) + r.clone() * RingElement::from_scalar(ModInt::from_u64(7), n);
        assert_eq!(o, &expected);
    }
}

#[test]
fn random_linear_combo_empty_right_is_identity() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);
    let n = params.n;

    let c1: Vec<_> = vec![one(n); params.k];
    let c2: Vec<_> = vec![];                     // empty
    let rho = F::from_u64(42);

    let out = comm.random_linear_combo(&c1, &c2, rho);
    assert_eq!(out, c1);
}

#[test]
fn random_linear_combo_mismatched_lengths_zero_broadcast() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);
    let n = params.n;

    let c1: Vec<_> = vec![one(n); params.k];
    let mut c2: Vec<_> = vec![one(n); params.k.saturating_sub(1)]; // shorter by one
    if c2.is_empty() { c2.push(one(n)); } // ensure mismatch not both empty
    let rho = F::from_u64(3);

    let out = comm.random_linear_combo(&c1, &c2, rho);
    assert_eq!(out.len(), c1.len().max(c2.len()));
    // last element uses zero for missing c2
    let last = out.last().unwrap().clone();
    let expected_last = c1.last().unwrap().clone() /* + 0 * rho */;
    assert_eq!(last, expected_last);
}

#[test]
fn random_linear_combo_rotation_empty_sides() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);
    let n = params.n;

    let c1: Vec<_> = vec![];                     // empty
    let c2: Vec<_> = vec![one(n); params.k];
    let rho_rot = RingElement::from_scalar(ModInt::from_u64(5), n);

    let out = comm.random_linear_combo_rotation(&c1, &c2, &rho_rot);
    assert_eq!(out.len(), c2.len());
    for (o, r) in out.iter().zip(c2.iter()) {
        assert_eq!(o, &(zero(n) + rho_rot.clone() * r.clone()));
    }

    let out2 = comm.random_linear_combo_rotation(&c2, &[], &rho_rot);
    assert_eq!(out2, c2); // identity when right side empty
}
