//! Audit checks for the single production parameter profile.

use neo_fold_clean::{config, Params};

#[test]
fn production_params_match_superneo_goldilocks_b2() {
    let pp = config::production_params();

    assert_eq!(config::PRODUCTION_PROFILE, "superneo-appendix-b2-goldilocks-b2");
    assert!(pp.is_production());
    assert_eq!(pp.q(), config::Q);
    assert_eq!(pp.eta(), config::ETA as u32);
    assert_eq!(pp.d(), config::D as u32);
    assert_eq!(pp.kappa(), config::KAPPA);
    assert_eq!(pp.m(), config::M);
    assert_eq!(pp.b(), config::B_BASE);
    assert_eq!(pp.k_rho(), config::K_RHO);
    assert_eq!(pp.big_b(), config::BIG_B);
    assert_eq!(pp.T(), config::T);
    assert_eq!(pp.extension_degree(), config::EXTENSION_DEGREE);
    assert_eq!(pp.lambda(), config::LAMBDA);
}

#[test]
fn production_params_use_k_rho_for_b_power() {
    let pp = Params::production();
    assert_eq!(pp.k_rho(), 14);
    assert_eq!(pp.big_b(), (pp.b() as u64).pow(pp.k_rho()));
}

#[test]
fn production_params_satisfy_superneo_rlc_guard() {
    let pp = Params::production();
    let lhs = (pp.k_rho() as u128 + 1) * (pp.T() as u128) * ((pp.b() - 1) as u128);
    assert!(lhs < pp.big_b() as u128, "(k_rho + 1) * T * (b - 1) must be < B");
}

#[test]
fn r1cs_params_keep_production_core_and_make_effective_lambda_explicit() {
    let pp = config::r1cs_params(60, 54).expect("R1CS params");

    assert_eq!(
        config::R1CS_PROFILE,
        "superneo-appendix-b2-goldilocks-b2-r1cs-effective-lambda"
    );
    assert!(pp.has_production_core());
    assert_eq!(pp.k_rho(), config::K_RHO);
    assert_eq!(pp.big_b(), config::BIG_B);
    assert_eq!(pp.extension_degree(), config::EXTENSION_DEGREE);
    assert!((config::MIN_EFFECTIVE_LAMBDA..=config::LAMBDA).contains(&pp.lambda()));
    assert_eq!(
        pp.lambda(),
        107,
        "current Fibonacci-sized R1CS shape should choose the strongest s=2 lambda above the 100-bit floor"
    );
}

#[test]
fn r1cs_params_are_sized_by_rows_or_variables_whichever_is_larger() {
    let row_heavy = config::r1cs_params(60, 54).expect("row-heavy params");
    let same_shape = config::r1cs_params(60, 12).expect("same row-heavy params");
    let var_heavy = config::r1cs_params(12, 60).expect("var-heavy params");

    assert_eq!(row_heavy.lambda(), same_shape.lambda());
    assert_eq!(row_heavy.lambda(), var_heavy.lambda());
}

#[test]
fn ccs_params_charge_matrix_count_and_degree() {
    let r1cs = config::r1cs_params(60, 54).expect("R1CS params");
    let t8 = config::ccs_params(60, 54, 8, 2).expect("t=8 CCS params");
    let degree7 = config::ccs_params(60, 54, 3, 7).expect("degree-7 CCS params");

    assert_eq!(r1cs.lambda(), 107);
    assert_eq!(t8.lambda(), 106);
    assert_eq!(degree7.lambda(), 107);
}

#[test]
fn r1cs_params_reject_when_full_d4_floor_is_too_high_for_s2() {
    let err = Params::for_r1cs_shape_with(60, 108, config::EXTENSION_SAFETY_MARGIN_BITS)
        .expect_err("s=2 cannot satisfy a higher full-D4 floor for this profile");
    assert!(matches!(
        err,
        neo_params::ParamsError::UnsupportedExtension { required: 3 }
    ));
}
