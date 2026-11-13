use neo_ccs::Mat;
use neo_math::{Rq, D, ETA, F};
use neo_params::{nightstream_p64_k16 as profile, NeoParams};
use neo_reductions::PiCcsError;
use neo_reductions::{rot_rhos_to_mats, sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

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
fn test_profile_ring_expansion_factor() {
    let ring = RotRing::profile();

    let max_coeff = ring
        .alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap();
    let T_computed = (D as u128) * (max_coeff as u128);
    assert_eq!(
        T_computed,
        profile::T as u128,
        "the expansion factor is that of SuperNeo v1.2 Theorem 5 for X^D + 1"
    );

    let params = NeoParams::nightstream_p64_k16();
    assert_eq!(
        params.T(),
        profile::T,
        "the profile T is that of SuperNeo v1.2 Theorem 5 for X^D + 1"
    );
}

#[test]
fn test_sample_rot_rhos_succeeds_with_valid_params() {
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();
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
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();
    let state = Poseidon2Transcript::new_v1_2().state();
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state, 1);

    let error = sample_rot_rhos_n(&mut transcript, &params, &ring, 1)
        .expect_err("v1_2 sampling must reject a nonzero absorb cursor");
    assert!(error.to_string().contains("zero transcript absorb cursor"));
}

#[test]
fn test_rot_rhos_k1_matches_first_sampled_rho() {
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();
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
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();
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
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();

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
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();
    let mut tr = Poseidon2Transcript::new_v1_2();

    let result = sample_rot_rhos_n(&mut tr, &params, &ring, (params.k_rho() as usize) + 1);
    assert!(result.is_ok(), "the profile satisfies the ΠRLC bound");
}

#[test]
fn test_strong_sampling_set_check() {
    struct TestRing;
    impl TestRing {
        fn bad_alphabet() -> RotRing {
            const BAD_A: &[i8] = &[-127, 0, 127];

            RotRing {
                alphabet: BAD_A,
                binv_floor: Some(200),
            }
        }
    }

    let params = NeoParams::nightstream_p64_k16();
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
    let params = NeoParams::nightstream_p64_k16();
    let ring = RotRing::profile();

    let c_max = ring
        .alphabet
        .iter()
        .map(|&x| (x as i64).unsigned_abs())
        .max()
        .unwrap();
    let T_computed = (D as u64) * c_max;

    assert_eq!(
        params.T() as u64,
        T_computed,
        "NeoParams.T should match computed expansion factor"
    );
}

#[test]
fn sampled_rho_is_multiplication_in_the_negacyclic_ring() {
    let params = NeoParams::nightstream_p64_k16();
    assert_eq!((params.d() as usize, params.eta() as usize), (D, ETA));
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let rhos = sample_rot_rhos_n(&mut transcript, &params, &RotRing::profile(), 3).unwrap();
    for (index, rho) in rhos.iter().enumerate() {
        let challenge = Rq(core::array::from_fn(|row| rho[(row, 0)]));
        let operand = Rq(core::array::from_fn(|row| {
            F::from_u64((7 * index + 3 * row + 1) as u64)
        }));
        let product = challenge.mul(&operand);
        for row in 0..D {
            let by_matrix: F = (0..D)
                .map(|column| rho[(row, column)] * operand.0[column])
                .sum();
            assert_eq!(by_matrix, product.0[row], "rho {index}, row {row}");
        }
        // X^D = −1: the last column times X is minus the first column.
        assert_eq!(rho[(0, 1)], -rho[(D - 1, 0)]);
    }
}

#[test]
fn checked_rho_rejects_the_cyclic_rotation() {
    let params = NeoParams::nightstream_p64_k16();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    let rho = sample_rot_rhos_n(&mut transcript, &params, &RotRing::profile(), 1)
        .unwrap()
        .remove(0);
    let mut cyclic = rho.clone();
    for column in 1..D {
        for row in 0..D {
            cyclic[(row, column)] = cyclic[((row + D - 1) % D, column - 1)];
        }
    }
    assert_ne!(cyclic, rho, "the sampled challenge has a nonzero wrapped coefficient");
    assert!(neo_reductions::rot_rhos_from_mats(&params, &[rho], "rho").is_ok());
    assert!(neo_reductions::rot_rhos_from_mats(&params, &[cyclic], "rho").is_err());
}
