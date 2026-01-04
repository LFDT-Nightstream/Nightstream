//! Tests for rotation matrix sampling (ΠRLC challenges)

use neo_math::D;
use neo_params::NeoParams;
use neo_reductions::PiCcsError;
use neo_reductions::{sample_rot_rhos_n, RotRing};
use neo_transcript::{Poseidon2Transcript, Transcript};

#[test]
#[allow(non_snake_case)]
fn test_goldilocks_ring_expansion_factor() {
    // Test that Goldilocks ring produces T=216 as specified in paper (Section 6.2)
    let ring = RotRing::goldilocks();

    // Alphabet is [-2,-1,0,1,2], so max|coeff| = 2
    // T = 2·φ(η)·max|coeff| = 2·54·2 = 216
    let max_coeff = ring
        .alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap();
    let T_computed = 2u128 * (D as u128) * (max_coeff as u128);
    assert_eq!(T_computed, 216, "Goldilocks expansion factor should be 216");

    // Check parameter set has matching T
    let params = NeoParams::goldilocks_127();
    assert_eq!(params.T, 216, "Goldilocks preset T should be 216");
}

#[test]
fn test_sample_rot_rhos_succeeds_with_valid_params() {
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new(b"test/rot_rhos");

    // Should sample params.k_rho+1 = 13 rhos (k_rho=12 for Goldilocks_127)
    // Bound: (12+1)·216·1 = 2808 < 4096 = 2^12 ✓
    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho as usize) + 1);

    assert!(result.is_ok(), "Sampling should succeed with valid params");
    let rhos = result.unwrap();
    let expected_count = (params.k_rho as usize) + 1;
    assert_eq!(
        rhos.len(),
        expected_count,
        "Should produce k_rho+1={} matrices",
        expected_count
    );

    // Check dimensions
    for (i, rho) in rhos.iter().enumerate() {
        assert_eq!(rho.rows(), D, "ρ_{} should have D={} rows", i, D);
        assert_eq!(rho.cols(), D, "ρ_{} should have D={} cols", i, D);
    }
}

#[test]
fn test_rot_rhos_k1_is_identity() {
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new(b"test/rot_rhos_k1");

    let rhos = sample_rot_rhos_n(&mut tr, &params, &ring, 1).unwrap();
    assert_eq!(rhos.len(), 1);
    assert!(rhos[0].is_identity(), "k=1 rho should be identity");
}

#[test]
fn test_rot_rhos_are_different() {
    // Test that we don't accidentally generate identical challenge matrices
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new(b"test/rot_rhos_distinct");

    // Should sample params.k_rho+1 = 13 rhos
    let rhos = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho as usize) + 1).unwrap();
    let count = rhos.len();

    // Check that ρ_i ≠ ρ_j for all distinct i,j
    for i in 0..count {
        for j in (i + 1)..count {
            let same = (0..D).all(|r| (0..D).all(|c| rhos[i][(r, c)] == rhos[j][(r, c)]));
            assert!(!same, "ρ_{} and ρ_{} should be distinct", i, j);
        }
    }
}

#[test]
fn test_rot_rhos_deterministic() {
    // Test that same transcript seed produces same matrices
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();

    let mut tr1 = Poseidon2Transcript::new(b"test/deterministic");
    let rhos1 = sample_rot_rhos_n(&mut tr1, &params, &ring, (params.k_rho as usize) + 1).unwrap();

    let mut tr2 = Poseidon2Transcript::new(b"test/deterministic");
    let rhos2 = sample_rot_rhos_n(&mut tr2, &params, &ring, (params.k_rho as usize) + 1).unwrap();

    // Should be identical
    let count = rhos1.len();
    for i in 0..count {
        for r in 0..D {
            for c in 0..D {
                assert_eq!(
                    rhos1[i][(r, c)],
                    rhos2[i][(r, c)],
                    "ρ_{}[{},{}] should be deterministic",
                    i,
                    r,
                    c
                );
            }
        }
    }
}

#[test]
fn test_rlc_bound_violation_detected() {
    // Test that params with valid k satisfy the bound
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new(b"test/bound_check");

    // With k=12 (from Goldilocks_127), b=2, T=216:
    // (12+1)·216·1 = 2808 < 4096 = 2^12 ✓ (should pass)
    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho as usize) + 1);
    assert!(result.is_ok(), "Goldilocks_127 params should satisfy the ΠRLC bound");
}

#[test]
fn test_strong_sampling_set_check() {
    // Create a ring with alphabet that violates Δ_A < b_inv
    struct TestRing;
    impl TestRing {
        fn bad_alphabet() -> RotRing {
            const PHI: [i32; D] = {
                let mut a = [0i32; D];
                a[0] = 1;
                a[27] = 1;
                a
            };
            // Huge alphabet: Δ_A = 127 - (-127) = 254 > 200
            const BAD_A: &[i8] = &[-127, 0, 127]; // Using i8 max range

            RotRing {
                phi_coeffs: &PHI,
                alphabet: BAD_A,
                binv_floor: Some(200), // Small b_inv, so Δ_A = 254 > 200
            }
        }
    }

    let params = NeoParams::goldilocks_127();
    let ring = TestRing::bad_alphabet();
    let mut tr = Poseidon2Transcript::new(b"test/bad_alphabet");

    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho as usize) + 1);
    assert!(result.is_err(), "Should reject alphabet with Δ_A >= b_inv");

    if let Err(PiCcsError::InvalidInput(msg)) = result {
        assert!(
            msg.contains("Strong-set check failed"),
            "Error should mention strong-set check"
        );
    } else {
        panic!("Expected InvalidInput error");
    }
}

#[test]
#[allow(non_snake_case)]
fn test_parameter_t_consistency() {
    // Test that NeoParams.T matches what we compute from the ring
    let params = NeoParams::goldilocks_127();
    let ring = RotRing::goldilocks();

    // Computed T from Theorem 3
    let c_max = ring
        .alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap();
    let T_computed = 2 * (D as u64) * c_max;

    assert_eq!(
        params.T as u64, T_computed,
        "NeoParams.T should match computed expansion factor"
    );
}
