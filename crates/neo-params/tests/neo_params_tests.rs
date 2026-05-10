use neo_params::{goldilocks_paper_b2, NeoParams, ParamsError};

#[test]
fn goldilocks_paper_b2_matches_guard_and_b() {
    let p = NeoParams::goldilocks_paper_b2();
    assert!(p.is_goldilocks_paper_b2());
    assert_eq!(p.B, goldilocks_paper_b2::B);
    let lhs = (p.k_rho as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
    assert!(lhs < p.B as u128, "guard must hold");
}

#[test]
fn s_min_monotone_in_lambda() {
    let p = NeoParams::goldilocks_paper_b2();
    // Pick a modest (ℓ, d_sc) representative for small CCS polynomials
    let (ell, d_sc) = (32u32, 8u32);
    // With λ=128 in this synthetic setting, s_min may be ≥2; check monotonicity only.
    let s1 = p.s_min(ell, d_sc);
    let mut tighter = p;
    tighter.lambda = 192;
    let s2 = tighter.s_min(ell, d_sc);
    assert!(s2 >= s1);
}

#[test]
fn extension_policy_enforces_s_eq_2() {
    let mut p = NeoParams::goldilocks_paper_b2();
    // s!=2 not supported
    p.s = 3;
    assert_eq!(
        Err(ParamsError::UnsupportedExtension { required: 3 }),
        NeoParams::new(p.q, p.eta, p.d, p.kappa, p.m, p.b, p.k_rho, p.T, 3, p.lambda)
    );
}

#[test]
fn r1cs_auto_params_charge_full_superneo_d4_budget() {
    let p = NeoParams::goldilocks_auto_r1cs_ccs(60).expect("R1CS params");

    assert!(p.has_goldilocks_paper_b2_core());
    // With Appendix B.2's s=2, the D.4 Schwartz-Zippel term dominates. This
    // is intentionally lower than the old sumcheck-only helper result, but
    // above the default 100-bit floor.
    assert_eq!(p.lambda, 107);
}

#[test]
fn r1cs_auto_params_reject_120_bit_full_d4_budget_under_s2() {
    let err = NeoParams::goldilocks_auto_r1cs_ccs_with(60, 120, 2)
        .expect_err("s=2 cannot satisfy a 120-bit full-D4 floor for this profile");
    assert_eq!(err, ParamsError::UnsupportedExtension { required: 3 });
}

#[test]
fn serde_roundtrip() {
    let p = NeoParams::goldilocks_paper_b2();
    let s = serde_json::to_string(&p).unwrap();
    let back: NeoParams = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}
