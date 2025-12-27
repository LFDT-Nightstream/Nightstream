use neo_params::{NeoParams, ParamsError};

#[test]
fn goldilocks_128_matches_guard_and_b() {
    let p = NeoParams::goldilocks_127();
    assert_eq!(p.B, 4096);
    let lhs = (p.k_rho as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
    assert!(lhs < p.B as u128, "guard must hold");
}

#[test]
fn s_min_monotone_in_lambda() {
    let p = NeoParams::goldilocks_127();
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
    let mut p = NeoParams::goldilocks_127();
    // s!=2 not supported
    p.s = 3;
    assert_eq!(
        Err(ParamsError::UnsupportedExtension { required: 3 }),
        NeoParams::new(p.q, p.eta, p.d, p.kappa, p.m, p.b, p.k_rho, p.T, 3, p.lambda)
    );
}

#[test]
fn serde_roundtrip() {
    let p = NeoParams::goldilocks_127();
    let s = serde_json::to_string(&p).unwrap();
    let back: NeoParams = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}
