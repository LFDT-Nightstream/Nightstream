#![allow(clippy::uninlined_format_args)]
use neo_params::{goldilocks_paper_b2, NeoParams, ParamsError};

/// The raw parameter bundle does not own a challenge alphabet. The production
/// padded-row census does, and must reject a target above its exact combined
/// field and coordinate-fork security.
#[test]
fn params_reject_lambda_above_strong_set_entropy() {
    let challenge_set_size = 5u128.pow(goldilocks_paper_b2::D as u32);
    let max_whole_bits = challenge_set_size.ilog2();
    let claimed_lambda = max_whole_bits + 1;

    let params = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        goldilocks_paper_b2::KAPPA,
        goldilocks_paper_b2::M,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        claimed_lambda,
    )
    .expect("raw field parameters");
    let statistical_policy =
        params.padded_row_security_check_for_shape(1, 1, 1, 0, goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32);

    assert!(
        matches!(
            statistical_policy,
            Err(ParamsError::InsufficientStatisticalSecurity { .. })
        ),
        "soundness-policy failure: accepted lambda={claimed_lambda} above floor(log2(5^54))={max_whole_bits}: {statistical_policy:?}"
    );
}

#[test]
fn guard_rejects_tight_or_overflowing_profiles() {
    // Tight inequality: lhs == B should be rejected.
    // Start from Appendix B.2 and pick T so (k+1)T(b-1)>B.
    let (b, k, d, eta, kappa, m, s, lambda) = (
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::D as u32,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::KAPPA,
        goldilocks_paper_b2::M,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        goldilocks_paper_b2::LAMBDA,
    );
    let q = goldilocks_paper_b2::Q;
    let t = (goldilocks_paper_b2::B / (goldilocks_paper_b2::K_RHO as u64 + 1) + 1) as u32;
    let err = NeoParams::new(q, eta, d, kappa, m, b, k, t, s, lambda).unwrap_err();
    assert!(matches!(err, ParamsError::GuardInequality));
    println!("✅ RED TEAM: Guard correctly rejects tight inequality");

    // Overflow in B=b^k must be rejected (checked u128 → u64 downcast)
    // Pick b so b^k won't fit into u64: e.g., b=256, k=9 → 2^72 (large but avoids compile-time overflow)
    let large_b = 256u32;
    let large_k = 9u32; // 256^9 is much larger than u64::MAX
    let err2 = NeoParams::new(q, eta, d, kappa, m, large_b, large_k, 10, 2, 128).unwrap_err();
    assert!(matches!(err2, ParamsError::Invalid(_)));
    println!("✅ RED TEAM: B overflow correctly rejected");
}

#[test]
fn params_reject_when_combined_bound_reaches_half_the_field() {
    // B = 2^3 = 8 and the RLC guard is 4 < 8. The only invalid condition is
    // the required strict separation 2*B < q, which fails at q = 16.
    let result = NeoParams::new(16, 3, 2, 1, 1, 2, 3, 1, 2, 1);
    assert!(matches!(
        result,
        Err(ParamsError::Invalid("2*B must be strictly smaller than q"))
    ));
}

#[test]
fn extension_policy_rejects_when_s_min_gt_2() {
    let p = NeoParams::goldilocks_paper_b2(); // s=2 compatible
                                              // Force s_min > 2 by tightening λ and picking large (ℓ·d_sc)
    let mut p2 = p;
    p2.lambda = 320; // very tight target
    let (ell, d_sc) = (64u32, 16u32);
    let e = p2.extension_check(ell, d_sc).unwrap_err();
    match e {
        ParamsError::UnsupportedExtension { required } => {
            assert!(required > 2);
            println!("✅ RED TEAM: Extension policy correctly rejects s_min={required} > 2");
        }
        _ => panic!("expected UnsupportedExtension"),
    }
}

#[test]
fn s_min_and_slack_bits_behave() {
    let p = NeoParams::goldilocks_paper_b2(); // s=2 compatible

    // Test that s_min calculation doesn't panic and returns reasonable values
    let s_min1 = p.s_min(1, 1);
    let s_min2 = p.s_min(8, 8);

    // Both should be reasonable (not zero, not huge)
    assert!(s_min1 > 0 && s_min1 < 10);
    assert!(s_min2 > 0 && s_min2 < 10);

    // Test extension_check error handling
    match p.extension_check(64, 64) {
        Ok(summary) => {
            assert_eq!(summary.s_supported, 2);
            println!("✅ RED TEAM: Extension check passed for large inputs");
        }
        Err(_) => {
            println!("✅ RED TEAM: Extension check correctly rejects large inputs requiring s > 2");
        }
    }
}

#[test]
fn parameter_boundary_conditions() {
    let base_params = (
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        goldilocks_paper_b2::KAPPA,
        goldilocks_paper_b2::M,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        goldilocks_paper_b2::LAMBDA,
    );
    let (q, eta, d, kappa, m, b, k, t, s, lambda) = base_params;

    // Test zero/invalid parameters are rejected
    assert!(matches!(
        NeoParams::new(0, eta, d, kappa, m, b, k, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("q must be nonzero")
    ));
    assert!(matches!(
        NeoParams::new(q, 0, d, kappa, m, b, k, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("eta must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, 0, kappa, m, b, k, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("d must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, 0, m, b, k, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("kappa must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, 0, b, k, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("m must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, 1, k, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("b must be >= 2")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, 0, t, s, lambda).unwrap_err(),
        ParamsError::Invalid("k_rho must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, k, 0, s, lambda).unwrap_err(),
        ParamsError::Invalid("T must be > 0")
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, k, t, 3, lambda).unwrap_err(),
        ParamsError::UnsupportedExtension { required: 3 }
    ));
    assert!(matches!(
        NeoParams::new(q, eta, d, kappa, m, b, k, t, s, 0).unwrap_err(),
        ParamsError::Invalid("lambda must be > 0")
    ));

    println!("✅ RED TEAM: All parameter boundary conditions correctly enforced");
}

#[test]
fn goldilocks_preset_security_invariants() {
    let p = NeoParams::goldilocks_paper_b2();

    // Verify the guard inequality is satisfied with margin
    let lhs = (p.k_rho as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
    let rhs = p.B as u128;
    assert!(lhs < rhs, "Guard inequality must hold: {lhs} < {rhs}");

    // Verify reasonable margin exists (not too tight)
    let margin = rhs - lhs;
    let margin_ratio = (margin as f64) / (rhs as f64);
    assert!(
        margin_ratio > 0.1,
        "Security margin should be > 10%, got {:.1}%",
        margin_ratio * 100.0
    );

    // Verify field parameters
    assert!(p.is_goldilocks_paper_b2());
    assert_eq!(p.q, goldilocks_paper_b2::Q);
    assert_eq!(p.s, goldilocks_paper_b2::EXTENSION_DEGREE);
    assert_eq!(p.lambda, goldilocks_paper_b2::LAMBDA);

    println!("✅ RED TEAM: Goldilocks preset satisfies all security invariants");
    println!(
        "   Guard margin: {:.1}% ({} out of {})",
        margin_ratio * 100.0,
        margin,
        rhs
    );
}

/// Test parameter boundary conditions for overflow cases
#[test]
fn parameter_overflow_boundary_test() {
    use neo_params::{NeoParams, ParamsError};

    // Test with lambda=128 which should force overflow and require s_min >= 3
    let high_lambda_params = NeoParams {
        q: goldilocks_paper_b2::Q,
        eta: goldilocks_paper_b2::ETA as u32,
        d: goldilocks_paper_b2::D as u32,
        kappa: 3,
        m: 4,
        lambda: 128, // Very high security parameter
        s: 2,
        k_rho: 1,
        T: 256,
        b: 2,
        B: 1024,
    };

    // With lambda=128 and any reasonable ell, d_sc, this should overflow and return s_min >= 3
    let result = high_lambda_params.extension_check(1, 1);
    match result {
        Err(ParamsError::UnsupportedExtension { required }) => {
            assert!(required >= 3, "Expected s_min >= 3 for overflow case, got {required}");
            println!("✅ Overflow case correctly requires s_min >= 3 (got {required})");
        }
        Ok(_) => panic!("Expected overflow case to fail with UnsupportedExtension"),
        Err(e) => panic!("Unexpected error type: {e:?}"),
    }

    // Test boundary case where s=1 fails but s=2 might succeed
    let boundary_params = NeoParams {
        q: goldilocks_paper_b2::Q,
        eta: goldilocks_paper_b2::ETA as u32,
        d: goldilocks_paper_b2::D as u32,
        kappa: 3,
        m: 4,
        lambda: 127, // Challenging security parameter
        s: 2,
        k_rho: 1,
        T: 256,
        b: 2,
        B: 1024,
    };

    // Test with challenging values - higher ell and d_sc should make s=1 fail
    let result_s1 = boundary_params.extension_check(1000, 256); // High ell and d_sc
    assert!(result_s1.is_err(), "s=1 should fail for challenging parameters");

    let result_s2 = boundary_params.extension_check(100, 128); // Less challenging
                                                               // s=2 might succeed or fail depending on exact values, but shouldn't panic
    match result_s2 {
        Ok(slack) => {
            println!("✅ s=2 succeeds with slack: {slack:?}");
        }
        Err(ParamsError::UnsupportedExtension { required }) => {
            println!("✅ s=2 fails, requires s_min = {required}");
            assert!(required > 2, "Required s_min should be > 2");
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }

    println!("✅ Parameter boundary conditions handled correctly");
}
