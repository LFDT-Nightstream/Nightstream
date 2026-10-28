use neo_ccs::Mat;
use neo_math::D;
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::PiCcsError;
use neo_reductions::{rot_rhos_to_mats, sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;

fn sample_rot_rhos_n(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    ring: &RotRing,
    count: usize,
) -> Result<Vec<Mat<F>>, PiCcsError> {
    sample_rot_rhos_n_typed(transcript, params, ring, count).map(|rhos| rot_rhos_to_mats(&rhos))
}

#[test]
#[allow(non_snake_case)]
fn test_goldilocks_ring_expansion_factor() {
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
        "Goldilocks expansion factor should match Section 8.2"
    );

    let params = NeoParams::goldilocks_paper_b2();
    assert_eq!(
        params.T(),
        goldilocks_paper_b2::T,
        "Goldilocks preset T should match Section 8.2"
    );
}

#[test]
fn test_sample_rot_rhos_succeeds_with_valid_params() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new_v1_2();

    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho() as usize) + 1);

    assert!(result.is_ok(), "Sampling should succeed with valid params");
    let rhos = result.unwrap();
    let expected_count = (params.k_rho() as usize) + 1;
    assert_eq!(
        rhos.len(),
        expected_count,
        "Should produce k_rho+1={} matrices",
        expected_count
    );

    for (i, rho) in rhos.iter().enumerate() {
        assert_eq!(rho.rows(), D, "ρ_{} should have D={} rows", i, D);
        assert_eq!(rho.cols(), D, "ρ_{} should have D={} cols", i, D);
    }
}

#[test]
fn test_sample_rot_rhos_rejects_nonzero_absorb_cursor() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let state = Poseidon2Transcript::new_v1_2().state();
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state, 1);

    let error = sample_rot_rhos_n(&mut transcript, &params, &ring, 1)
        .expect_err("v1_2 sampling must reject a nonzero absorb cursor");
    assert!(error.to_string().contains("zero transcript absorb cursor"));
}

#[test]
fn test_rot_rhos_k1_matches_first_sampled_rho() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr_single = Poseidon2Transcript::new_v1_2();
    let mut tr_pair = Poseidon2Transcript::new_v1_2();

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
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new_v1_2();

    let rhos = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho() as usize) + 1).unwrap();
    let count = rhos.len();

    for i in 0..count {
        for j in (i + 1)..count {
            let same = (0..D).all(|r| (0..D).all(|c| rhos[i][(r, c)] == rhos[j][(r, c)]));
            assert!(!same, "ρ_{} and ρ_{} should be distinct", i, j);
        }
    }
}

#[test]
fn test_rot_rhos_deterministic() {
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();

    let mut tr1 = Poseidon2Transcript::new_v1_2();
    let rhos1 = sample_rot_rhos_n(&mut tr1, &params, &ring, (params.k_rho() as usize) + 1).unwrap();

    let mut tr2 = Poseidon2Transcript::new_v1_2();
    let rhos2 = sample_rot_rhos_n(&mut tr2, &params, &ring, (params.k_rho() as usize) + 1).unwrap();

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
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();
    let mut tr = Poseidon2Transcript::new_v1_2();

    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho() as usize) + 1);
    assert!(
        result.is_ok(),
        "Section 8.2 Goldilocks params should satisfy the ΠRLC bound"
    );
}

#[test]
fn test_strong_sampling_set_check() {
    struct TestRing;
    impl TestRing {
        fn bad_alphabet() -> RotRing {
            const BAD_A: &[i8] = &[-127, 0, 127];

            RotRing {
                phi_low_coeffs: &goldilocks_paper_b2::PHI_COEFFS,
                alphabet: BAD_A,
                binv_floor: Some(200),
            }
        }
    }

    let params = NeoParams::goldilocks_paper_b2();
    let ring = TestRing::bad_alphabet();
    let mut tr = Poseidon2Transcript::new_v1_2();

    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho() as usize) + 1);
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
    let params = NeoParams::goldilocks_paper_b2();
    let ring = RotRing::goldilocks();

    let c_max = ring
        .alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap();
    let T_computed = 2 * (D as u64) * c_max;

    assert_eq!(
        params.T() as u64,
        T_computed,
        "NeoParams.T should match computed expansion factor"
    );
}

#[test]
fn public_sampler_rejects_a_ring_without_the_strong_set_property() {
    const NILPOTENT_PHI: [i32; D] = [0; D];
    let ring = RotRing {
        phi_low_coeffs: &NILPOTENT_PHI,
        alphabet: &goldilocks_paper_b2::CHALLENGE_ALPHABET,
        binv_floor: Some(goldilocks_paper_b2::B_INV_FLOOR),
    };
    let params = NeoParams::goldilocks_paper_b2();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    assert!(
        sample_rot_rhos_n(&mut transcript, &params, &ring, 6).is_err(),
        "the only public sampler must reject a quotient ring other than Phi81"
    );
}
