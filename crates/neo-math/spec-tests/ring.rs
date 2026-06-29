//! Spec-derived invariant tests for Ring.spec.md
//!
//! Each test corresponds to a row in the Invariant Obligations table.

#[path = "common/mod.rs"]
mod common;

use common::{random_fq_array, seeded_rng};
use neo_math::ring::{cf, cf_inv, ct, test_reduce_mod_phi_81, Rq, D};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

/// Ring.spec.md: cf(cf_inv(v)) = v for all v in F_q^d
#[test]
fn cf_cf_inv_roundtrip() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    for _ in 0..20 {
        let v = random_fq_array(&mut rng);
        assert_eq!(cf(cf_inv(v)), v);
    }
}

/// Ring.spec.md: cf_inv(cf(a)) = a for all a in R_q
#[test]
fn cf_inv_cf_roundtrip() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    for _ in 0..20 {
        let a = Rq::random_uniform(&mut rng);
        assert_eq!(cf_inv(cf(a)), a);
    }
}

/// Ring.spec.md: ct(a) = cf(a)[0]
#[test]
fn ct_equals_first_coefficient() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    for _ in 0..20 {
        let a = Rq::random_uniform(&mut rng);
        assert_eq!(ct(&a), cf(a)[0]);
    }
}

/// Ring.spec.md: Phi_81 reduction — X^54 = -X^27 - 1
#[test]
fn phi_81_reduction_identity() {
    // Build X^54 as a polynomial and reduce it
    let mut coeffs = [Fq::ZERO; 2 * D - 1];
    coeffs[54] = Fq::ONE;
    test_reduce_mod_phi_81(&mut coeffs);

    // After reduction, X^54 should become -X^27 - 1
    // i.e. coeffs[0] = -1, coeffs[27] = -1, rest zero
    for i in 0..D {
        let expected = match i {
            0 => -Fq::ONE,
            27 => -Fq::ONE,
            _ => Fq::ZERO,
        };
        assert_eq!(coeffs[i], expected, "mismatch at degree {i}");
    }
}

/// Ring.spec.md: Ring mul is correct — a*b mod Phi_81
/// Verified by checking (a * 1 = a) and (a * 0 = 0)
#[test]
fn ring_mul_identity_and_zero() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    let a = Rq::random_uniform(&mut rng);
    assert_eq!(a.mul(&Rq::one()), a);
    assert_eq!(a.mul(&Rq::zero()), Rq::zero());
}

/// Ring.spec.md: Ring mul associativity — (a*b)*c = a*(b*c)
#[test]
fn ring_mul_associative() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    let a = Rq::random_small(&mut rng, 100);
    let b = Rq::random_small(&mut rng, 100);
    let c = Rq::random_small(&mut rng, 100);
    assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
}

/// Ring.spec.md: Ring mul commutativity — a*b = b*a
#[test]
fn ring_mul_commutative() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    let a = Rq::random_uniform(&mut rng);
    let b = Rq::random_uniform(&mut rng);
    assert_eq!(a.mul(&b), b.mul(&a));
}

/// Ring.spec.md: Ring mul distributive — a*(b+c) = a*b + a*c
#[test]
fn ring_mul_distributive() {
    let mut rng = seeded_rng(0xDEAD_BEEF);
    let a = Rq::random_small(&mut rng, 100);
    let b = Rq::random_small(&mut rng, 100);
    let c = Rq::random_small(&mut rng, 100);
    assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
}

/// Ring.spec.md: inf_norm uses centered representatives
/// Verify that inf_norm(Rq::from_field_scalar(1)) = 1
/// and inf_norm(Rq::zero()) = 0
#[test]
fn inf_norm_basic() {
    use neo_math::ring::inf_norm;
    assert_eq!(inf_norm(&Rq::zero()), 0);
    assert_eq!(inf_norm(&Rq::one()), 1);

    // Element with known norm
    let mut coeffs = [Fq::ZERO; D];
    coeffs[0] = Fq::from_u64(42);
    assert_eq!(inf_norm(&Rq(coeffs)), 42);
}
