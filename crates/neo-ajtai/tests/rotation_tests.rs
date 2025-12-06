//! Tests for rotation step functionality in Ajtai commitments
//!
//! These tests verify the mathematical correctness of the rotation step implementation
//! used in constant-time commit algorithms. They ensure that the hand-rolled rotation
//! steps match the underlying ring arithmetic.

// Enable testing feature to access rot_step function
#![cfg(feature = "testing")]

use neo_ajtai::rot_step;
use neo_math::{cf, Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

/// Test that rot_step_phi_81 matches ring multiplication by X
/// This directly tests the hand-rolled rotation step against the ring arithmetic
#[test]
fn rot_step_phi81_matches_ring_mul_by_x() {
    let mut rng = ChaCha20Rng::seed_from_u64(1234);

    for _ in 0..50 {
        let a = Rq::random_uniform(&mut rng);
        let mut col = cf(a); // column 0 = cf(a)
        let mut nxt = [Fq::ZERO; D];

        // Test: rot_step should turn 'col' into cf(a * X)
        let a_times_x = a.mul_by_monomial(1); // ring multiply by X
        let expected = cf(a_times_x); // cf(a * X)

        rot_step(&col, &mut nxt); // our rotation step

        assert_eq!(
            nxt, expected,
            "rot_step should produce cf(a * X), but got different result"
        );

        // Verify we can continue the sequence correctly
        col = nxt; // advance to next column
        let a_times_x2 = a.mul_by_monomial(2);
        let expected2 = cf(a_times_x2);

        rot_step(&col, &mut nxt);
        assert_eq!(nxt, expected2, "Second rot_step should produce cf(a * X^2)");
    }
}

/// Test the cyclotomic identity X^54 ≡ -X^27 - 1 through rotation steps
#[test]
fn rot_step_cyclotomic_identity() {
    // Start with X (monomial degree 1)
    let x = {
        let mut coeffs = vec![Fq::ZERO; D];
        coeffs[1] = Fq::ONE; // X = 0 + 1*X + 0*X^2 + ...
        Rq::from_field_coeffs(coeffs)
    };

    let mut col = cf(x);
    let mut nxt = [Fq::ZERO; D];

    // Apply rot_step 53 times to get from cf(X) to cf(X^54)
    for _ in 0..53 {
        rot_step(&col, &mut nxt);
        core::mem::swap(&mut col, &mut nxt);
    }

    // col now represents cf(X^54)
    // According to Φ₈₁(X) = X^54 + X^27 + 1, we have X^54 ≡ -X^27 - 1

    // Compute expected value: cf(-X^27 - 1)
    let x27 = {
        let mut coeffs = vec![Fq::ZERO; D];
        coeffs[27] = Fq::ONE;
        Rq::from_field_coeffs(coeffs)
    };
    let neg_x27_minus_1 = cf(Rq::zero() - x27 - Rq::one());

    assert_eq!(col, neg_x27_minus_1, "X^54 should equal -X^27 - 1 under Φ₈₁ reduction");
}
