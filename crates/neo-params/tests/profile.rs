//! The profile constants against independent recomputation
//! (`decisions/field-2p64-59-ring-x64.md`).

use neo_params::{nightstream_p64_k16 as profile, NeoParams, ParamsError};
use num_bigint::BigUint;

fn euler_phi(mut n: u64) -> u64 {
    let mut phi = n;
    let mut p = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            while n.is_multiple_of(p) {
                n /= p;
            }
            phi -= phi / p;
        }
        p += 1;
    }
    if n > 1 {
        phi -= phi / n;
    }
    phi
}

#[test]
fn modulus_is_two_to_the_64_minus_59() {
    assert_eq!(profile::Q, 18_446_744_073_709_551_557);
    assert_eq!(u128::from(profile::Q), (1u128 << 64) - 59);
    assert_eq!(profile::NAME, "Nightstream P64, k_rho = 16");
}

#[test]
fn ring_is_the_128th_cyclotomic_ring() {
    assert_eq!(profile::ETA, 128);
    assert_eq!(profile::D as u64, euler_phi(profile::ETA as u64));
    assert_eq!(u64::from(profile::T), 2 * euler_phi(profile::ETA as u64));

    let mut phi = [0i32; profile::D];
    phi[0] = 1;
    assert_eq!(profile::PHI_COEFFS, phi, "X^64 + 1 below degree 64");
}

#[test]
fn x64_plus_1_has_two_factors_of_degree_32() {
    assert_eq!(profile::Q % 8, 5);
    let eta = profile::ETA as u128;
    let order = (1..=eta)
        .find(|&k| (0..k).fold(1u128, |power, _| power * u128::from(profile::Q) % eta) == 1)
        .unwrap();
    assert_eq!(order, 32);
    assert_eq!(profile::D as u128 / order, 2);
}

#[test]
fn invertibility_floor_is_the_floor_of_sqrt_q_over_sqrt_2() {
    let floor = u128::from(profile::B_INV_FLOOR);
    let q = u128::from(profile::Q);
    assert!(2 * floor * floor <= q);
    assert!(2 * (floor + 1) * (floor + 1) > q);
}

#[test]
fn challenge_set_is_a_strong_sampling_set_above_128_bits() {
    let alphabet = profile::CHALLENGE_ALPHABET;
    assert_eq!(alphabet, [-2, -1, 0, 1, 2]);
    let spread = (alphabet.iter().max().unwrap() - alphabet.iter().min().unwrap()) as u64;
    assert!(spread <= profile::B_INV_FLOOR);

    let size = BigUint::from(alphabet.len()).pow(profile::D as u32);
    assert!(size > BigUint::from(1u8) << 128);
    assert_eq!(size.bits() - 1, u64::from(profile::CHALLENGE_SET_LOG2_FLOOR));
}

#[test]
fn decomposition_is_binary_with_k_rho_16() {
    assert_eq!(profile::B_BASE, 2);
    assert_eq!(profile::K_RHO, 16);
    assert_eq!(profile::B, 1 << 16);
    assert_eq!(profile::KAPPA, 18);
    assert_eq!(profile::EXTENSION_DEGREE, 2);
}

#[test]
fn rlc_guard_holds_and_bounds_the_fresh_count() {
    let p = NeoParams::nightstream_p64_k16();
    let fresh = u64::from(p.max_fresh_count_from_rlc_guard().unwrap());
    let per_input = u64::from(profile::T) * u64::from(profile::B_BASE - 1);
    let k = u64::from(profile::K_RHO);

    assert_eq!(fresh, 495);
    assert!((fresh + k) * per_input < profile::B);
    assert!((fresh + 1 + k) * per_input >= profile::B);
    assert!((k + 1) * per_input < profile::B);
}

#[test]
fn constructor_returns_the_profile_constants() {
    let p = NeoParams::nightstream_p64_k16();
    assert_eq!(
        (p.q(), p.eta(), p.d(), p.kappa(), p.m()),
        (
            profile::Q,
            profile::ETA as u32,
            profile::D as u32,
            profile::KAPPA,
            profile::M
        )
    );
    assert_eq!(
        (p.b(), p.k_rho(), p.B(), p.T(), p.s(), p.lambda()),
        (
            profile::B_BASE,
            profile::K_RHO,
            profile::B,
            profile::T,
            profile::EXTENSION_DEGREE,
            profile::TARGET_LAMBDA
        )
    );
}

#[test]
fn maximum_geometry_padded_row_census_matches_formula() {
    let alphabet = profile::CHALLENGE_ALPHABET.len() as u32;
    let columns = (profile::M as usize / profile::D) * profile::D;
    let p = NeoParams::nightstream_p64_k16().with_lambda(115).unwrap();
    let summary = p
        .padded_row_security_check_for_shape(1 << 26, columns, 3, 2, alphabet)
        .expect("115-bit padded-row census");

    let (fresh, running, d, matrices) = (495u128, 16u128, 64u128, 3u128);
    let mixing = fresh + (fresh + running) + running * (matrices + 1) * d - 1 + 30;
    assert_eq!(summary.cube_variables, 30);
    assert_eq!(summary.verifier_degree, 4);
    assert_eq!(summary.sumcheck_factor, 30 * 4);
    assert_eq!(summary.mixing_factor, mixing);
    assert_eq!(summary.field_factor, 120 + mixing);
    assert_eq!(summary.fork_factor, fresh + running + 1);
    assert_eq!(summary.challenge_set_cardinality, BigUint::from(5u8).pow(64));

    // 2^bits · (field/q^2 + fork/5^64) <= 1 < 2^(bits + 1) · (the same sum).
    let field = BigUint::from(profile::Q).pow(2);
    let numerator = BigUint::from(summary.field_factor) * &summary.challenge_set_cardinality
        + BigUint::from(summary.fork_factor) * &field;
    let denominator = field * &summary.challenge_set_cardinality;
    assert_eq!(summary.security_bits, 115);
    assert!(&numerator << 115 <= denominator);
    assert!(numerator << 116 > denominator);
    assert_eq!(summary.slack_bits, 0);

    let err = p
        .with_lambda(116)
        .unwrap()
        .padded_row_security_check_for_shape(1 << 26, columns, 3, 2, alphabet)
        .unwrap_err();
    assert_eq!(
        err,
        ParamsError::InsufficientStatisticalSecurity {
            required: 116,
            available: 115
        }
    );
}

#[test]
fn census_charges_matrix_count_and_polynomial_degree() {
    let alphabet = profile::CHALLENGE_ALPHABET.len() as u32;
    let p = NeoParams::nightstream_p64_k16();
    let census = |matrices, degree| {
        p.padded_row_security_summary_for_shape(60, 60, matrices, degree, alphabet)
            .unwrap()
    };

    assert!(census(8, 2).mixing_factor > census(3, 2).mixing_factor);
    assert_eq!(census(3, 2).verifier_degree, 4);
    assert_eq!(census(3, 7).verifier_degree, 8);
    assert!(census(3, 7).sumcheck_factor > census(3, 2).sumcheck_factor);
}
