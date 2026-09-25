use neo_params::{
    goldilocks_paper_b2, nightstream_goldilocks_k16, pi_rlc_sampler_completeness_summary, NeoParams, ParamsError,
};

#[test]
fn goldilocks_paper_b2_matches_guard_and_b() {
    let p = NeoParams::goldilocks_paper_b2();
    assert!(p.is_goldilocks_paper_b2());
    assert_eq!(p.B, goldilocks_paper_b2::B);
    let lhs = (p.k_rho as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
    assert!(lhs < p.B as u128, "guard must hold");
}

#[test]
fn nightstream_goldilocks_k16_matches_the_frozen_binary_profile() {
    let p = NeoParams::nightstream_goldilocks_k16();

    assert!(p.is_nightstream_goldilocks_k16());
    assert!(!p.is_goldilocks_paper_b2());
    assert_eq!(p.b, 2);
    assert_eq!(p.k_rho, 16);
    assert_eq!(p.B, nightstream_goldilocks_k16::B);
    assert_eq!(p.B, 1 << 16);
    assert_eq!(p.max_fresh_count_from_rlc_guard().unwrap(), 287);
}

#[test]
fn pi_rlc_sampler_schedule_meets_appendix_b2_completeness() {
    let summary = pi_rlc_sampler_completeness_summary();

    assert_eq!(summary.digest_rounds, 8);
    assert_eq!(summary.field_lanes, 32);
    assert_eq!(summary.candidates, 64);
    assert_eq!(summary.required, goldilocks_paper_b2::D);
    assert_eq!(summary.completeness_bits, 136);
    assert_eq!(summary.slack_bits, 11);
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
fn r1cs_auto_params_charge_padded_row_field_and_fork_budget() {
    let p = NeoParams::goldilocks_auto_r1cs_ccs(60).expect("R1CS params");

    assert!(p.has_goldilocks_paper_b2_core());
    assert_eq!(p.lambda, 116);
}

#[test]
fn ccs_auto_params_charge_actual_matrix_count() {
    let r1cs = NeoParams::goldilocks_auto_r1cs_ccs_with(60, 100, 2).expect("R1CS params");
    let t8 = NeoParams::goldilocks_auto_ccs_with(60, 8, 2, 100, 2).expect("t=8 CCS params");

    assert_eq!(r1cs.lambda, 114);
    assert_eq!(
        t8.lambda, 113,
        "the larger carried-coordinate block needs one more bit of room"
    );
}

#[test]
fn ccs_auto_params_charge_actual_polynomial_degree() {
    let degree2 = NeoParams::goldilocks_auto_ccs_with(60, 3, 2, 100, 2).expect("quadratic CCS params");
    let degree7 = NeoParams::goldilocks_auto_ccs_with(60, 3, 7, 100, 2).expect("degree-7 CCS params");

    assert_eq!(degree2.lambda, 114);
    assert_eq!(
        degree7.lambda, 114,
        "mixing still dominates, but the joint SumCheck must charge the accepted degree"
    );
}

#[test]
fn r1cs_auto_params_reject_120_bit_combined_budget_under_s2() {
    let err = NeoParams::goldilocks_auto_r1cs_ccs_with(60, 120, 2)
        .expect_err("s=2 and the production challenge set cannot satisfy this floor");
    assert!(matches!(
        err,
        ParamsError::InsufficientStatisticalSecurity {
            required: 122,
            available: 116
        }
    ));
}

#[test]
fn maximum_geometry_padded_row_census_matches_formula() {
    let mut p = NeoParams::goldilocks_paper_b2();
    p.lambda = 116;
    let summary = p
        .padded_row_security_check_for_shape(
            1 << 26,
            (goldilocks_paper_b2::M as usize / goldilocks_paper_b2::D) * goldilocks_paper_b2::D,
            3,
            2,
            goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .expect("116-bit padded-row census");

    assert_eq!(summary.cube_variables, 30);
    assert_eq!(summary.verifier_degree, 4);
    assert_eq!(summary.sumcheck_factor, 120);
    assert_eq!(summary.mixing_factor, 3159);
    assert_eq!(summary.field_factor, 3279);
    assert_eq!(summary.fork_factor, 76);
    assert_eq!(
        summary.challenge_set_cardinality,
        goldilocks_paper_b2::CHALLENGE_SET_CARDINALITY
    );
    assert_eq!(summary.security_bits, 116);
    assert_eq!(summary.slack_bits, 0);
}

#[test]
fn serde_roundtrip() {
    let p = NeoParams::goldilocks_paper_b2();
    let s = serde_json::to_string(&p).unwrap();
    let back: NeoParams = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}
