//! Tests for rotation matrix sampling (ΠRLC challenges)

use neo_math::D;
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::PiCcsError;
use neo_reductions::{sample_rot_rhos_n, RotRing};
use neo_transcript::Poseidon2Transcript;

#[test]
#[allow(non_snake_case)]
fn test_goldilocks_ring_expansion_factor() {
    // Test that Goldilocks ring produces the Appendix B.2 expansion factor.
    let ring = RotRing::goldilocks();

    let max_coeff = ring
        .alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap();
    let T_computed = 2u128 * (D as u128) * (max_coeff as u128);
    assert_eq!(
        T_computed,
        goldilocks_paper_b2::T as u128,
        "Goldilocks expansion factor should match Appendix B.2"
    );

    // Check parameter set has matching T
    let params = NeoParams::goldilocks_paper_b2();
    assert_eq!(
        params.T,
        goldilocks_paper_b2::T,
        "Goldilocks preset T should match Appendix B.2"
    );
}

#[test]
fn test_sample_rot_rhos_succeeds_with_valid_params() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new_v1_1();

    // Should sample params.k_rho+1 rhos under the Appendix B.2 norm budget.
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
fn test_sample_rot_rhos_rejects_nonzero_absorb_cursor() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let state = Poseidon2Transcript::new_v1_1().state();
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state, 1);

    let error = sample_rot_rhos_n(&mut transcript, &params, &ring, 1)
        .expect_err("v1_1 sampling must reject a nonzero absorb cursor");
    assert!(error.to_string().contains("zero transcript absorb cursor"));
}

#[test]
fn test_rot_rhos_k1_matches_first_sampled_rho() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr_single = Poseidon2Transcript::new_v1_1();
    let mut tr_pair = Poseidon2Transcript::new_v1_1();

    let rho_single = sample_rot_rhos_n(&mut tr_single, &params, &ring, 1).unwrap();
    let rho_pair = sample_rot_rhos_n(&mut tr_pair, &params, &ring, 2).unwrap();

    assert_eq!(rho_single.len(), 1);
    assert_eq!(rho_pair.len(), 2);
    assert_eq!(
        rho_single[0], rho_pair[0],
        "count=1 must use the same sampled rho as the first rho in count=2"
    );
}

#[test]
fn test_rot_rhos_are_different() {
    // Test that we don't accidentally generate identical challenge matrices
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new_v1_1();

    // Should sample params.k_rho+1 rhos.
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
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();

    let mut tr1 = Poseidon2Transcript::new_v1_1();
    let rhos1 = sample_rot_rhos_n(&mut tr1, &params, &ring, (params.k_rho as usize) + 1).unwrap();

    let mut tr2 = Poseidon2Transcript::new_v1_1();
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
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new_v1_1();

    // Appendix B.2 Goldilocks should satisfy the ΠRLC norm budget.
    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho as usize) + 1);
    assert!(
        result.is_ok(),
        "Appendix B.2 Goldilocks params should satisfy the ΠRLC bound"
    );
}

#[test]
fn test_strong_sampling_set_check() {
    // Create a ring with alphabet that violates Δ_A < b_inv
    struct TestRing;
    impl TestRing {
        fn bad_alphabet() -> RotRing {
            // Huge alphabet: Δ_A = 127 - (-127) = 254 > 200
            const BAD_A: &[i8] = &[-127, 0, 127]; // Using i8 max range

            RotRing {
                phi_coeffs: &goldilocks_paper_b2::PHI_COEFFS,
                alphabet: BAD_A,
                binv_floor: Some(200), // Small b_inv, so Δ_A = 254 > 200
            }
        }
    }

    let params = NeoParams::goldilocks_paper_b2();
    let ring = TestRing::bad_alphabet();
    let mut tr = Poseidon2Transcript::new_v1_1();

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
    let params = NeoParams::goldilocks_paper_b2();
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
