#![allow(clippy::uninlined_format_args)]
use neo_params::{nightstream_p64_k14 as profile, NeoParams, ParamsError};
use num_bigint::BigUint;

#[test]
fn params_reject_lambda_above_strong_set_entropy() {
    let challenge_set_size = BigUint::from(5u8).pow(profile::D as u32);
    let max_whole_bits = challenge_set_size.bits() as u32 - 1;
    let claimed_lambda = max_whole_bits + 1;

    let params = NeoParams::nightstream_p64_k14()
        .with_certified_lambda(claimed_lambda)
        .expect("positive bits");
    let statistical_policy =
        params.padded_row_security_check_for_shape(1, 1, 1, 0, profile::CHALLENGE_ALPHABET.len() as u32);

    assert!(
        matches!(
            statistical_policy,
            Err(ParamsError::InsufficientStatisticalSecurity { .. })
        ),
        "soundness-policy failure: accepted lambda={claimed_lambda} above floor(log2(5^64))={max_whole_bits}: {statistical_policy:?}"
    );
}

#[test]
fn guard_rejects_tight_or_overflowing_profiles() {
    let (b, k, d, eta, kappa, m, s) = (
        profile::B_BASE,
        profile::K_RHO,
        profile::D as u32,
        profile::ETA as u32,
        profile::KAPPA,
        profile::M,
        profile::EXTENSION_DEGREE,
    );
    let q = profile::Q;
    let t = (profile::B / (profile::K_RHO as u64 + 1) + 1) as u32;
    let err = NeoParams::new(q, eta, d, kappa, m, b, k, t, s).unwrap_err();
    assert!(matches!(err, ParamsError::GuardInequality));
    println!("✅ RED TEAM: Guard correctly rejects tight inequality");

    let large_b = 256u32;
    let large_k = 9u32;
    let err2 = NeoParams::new(q, eta, d, kappa, m, large_b, large_k, 10, 2).unwrap_err();
    assert!(matches!(err2, ParamsError::Invalid(_)));
    println!("✅ RED TEAM: B overflow correctly rejected");
}

#[test]
fn params_reject_when_combined_bound_reaches_half_the_field() {
    let result = NeoParams::new(16, 3, 2, 1, 1, 2, 3, 1, 2);
    assert!(matches!(
        result,
        Err(ParamsError::Invalid("2*B must be strictly smaller than q"))
    ));
}

#[test]
fn parameter_boundary_conditions() {
    let base_params = (
        profile::Q,
        profile::ETA as u32,
        profile::D as u32,
        profile::KAPPA,
        profile::M,
        profile::B_BASE,
        profile::K_RHO,
        profile::T,
        profile::EXTENSION_DEGREE,
    );
    let (q, eta, d, kappa, m, b, k, t, s) = base_params;

    assert!(matches!(
        NeoParams::new(0, eta, d, kappa, m, b, k, t, s).unwrap_err(),
        ParamsError::Invalid("q must be nonzero")
    ));
    assert!(matches!(
        NeoParams::new(q, 0, d, kappa, m, b, k, t, s).unwrap_err(),
        ParamsError::Invalid("eta must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, 0, kappa, m, b, k, t, s).unwrap_err(),
        ParamsError::Invalid("d must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, 0, m, b, k, t, s).unwrap_err(),
        ParamsError::Invalid("kappa must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, 0, b, k, t, s).unwrap_err(),
        ParamsError::Invalid("m must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, 1, k, t, s).unwrap_err(),
        ParamsError::Invalid("Nightstream fixes the decomposition base b = 2")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, 0, t, s).unwrap_err(),
        ParamsError::Invalid("k_rho must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, k, 0, s).unwrap_err(),
        ParamsError::Invalid("T must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, k, t, 3).unwrap_err(),
        ParamsError::UnsupportedExtension { required: 3 }
    ));
    assert!(matches!(
        NeoParams::nightstream_p64_k14()
            .with_certified_lambda(0)
            .unwrap_err(),
        ParamsError::Invalid("lambda must be > 0")
    ));

    println!("✅ RED TEAM: All parameter boundary conditions correctly enforced");
}

#[test]
fn profile_security_invariants() {
    let p = NeoParams::nightstream_p64_k14();

    let lhs = (p.k_rho() as u128 + 1) * (p.T() as u128) * ((p.b() as u128) - 1);
    let rhs = p.B() as u128;
    assert!(lhs < rhs, "Guard inequality must hold: {lhs} < {rhs}");

    let margin = rhs - lhs;
    let margin_ratio = (margin as f64) / (rhs as f64);

    assert_eq!(p.q(), profile::Q);
    assert_eq!(p.s(), profile::EXTENSION_DEGREE);
    assert_eq!(p.certified_lambda(), None);

    println!("✅ RED TEAM: The profile satisfies all security invariants");
    println!(
        "   Guard margin: {:.1}% ({} out of {})",
        margin_ratio * 100.0,
        margin,
        rhs
    );
}
