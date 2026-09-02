//! Audit checks for the single production parameter profile.

use neo_fold_clean::{config, Params};
#[test]
fn production_params_match_nightstream_goldilocks_k16() {
    let pp = config::production_params();

    assert_eq!(config::PRODUCTION_PROFILE, "nightstream-goldilocks-b2-k16");
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
    assert_eq!(pp.k_rho(), 16);
    assert_eq!(pp.big_b(), (pp.b() as u64).pow(pp.k_rho()));
}

#[test]
fn production_params_satisfy_superneo_rlc_guard() {
    let pp = Params::production();
    let lhs = (pp.k_rho() as u128 + 1) * (pp.T() as u128) * ((pp.b() - 1) as u128);
    assert!(lhs < pp.big_b() as u128, "(k_rho + 1) * T * (b - 1) must be < B");
}

/// Golden vector shared with
/// `Nightstream.SuperNeo.Concrete.production_parameter_values`.
#[test]
fn production_params_match_lean_m1_profile() {
    let pp = Params::production();
    assert_eq!(pp.q(), 18_446_744_069_414_584_321);
    assert_eq!(pp.b(), 2);
    assert_eq!(pp.k_rho(), 16);
    assert_eq!(pp.big_b(), 65_536);
    assert_eq!(pp.T(), 216);
    assert_eq!(pp.max_fresh_count(), 287);
    assert_eq!(pp.eta(), 81);
    assert_eq!(pp.d(), 54);
    assert_eq!(pp.kappa(), 22);
    assert_eq!(pp.extension_degree(), 2);
    assert_eq!(pp.lambda(), 125);
}

#[test]
fn r1cs_params_keep_production_core_and_make_effective_lambda_explicit() {
    let pp = config::r1cs_params(60, 54).expect("R1CS params");

    assert_eq!(
        config::R1CS_PROFILE,
        "nightstream-goldilocks-b2-k16-shape-derived-lambda"
    );
    assert!(pp.has_production_core());
    assert_eq!(pp.k_rho(), config::K_RHO);
    assert_eq!(pp.big_b(), config::BIG_B);
    assert_eq!(pp.extension_degree(), config::EXTENSION_DEGREE);
    assert_eq!(
        pp.lambda(),
        115,
        "the header must bind the exact strongest lambda supported by this shape"
    );
}

#[test]
fn r1cs_params_charge_the_joint_cube_for_rows_and_the_padded_carrier() {
    let row_heavy = config::r1cs_params(60, 54).expect("row-heavy params");
    let narrow_columns = config::r1cs_params(60, 12).expect("narrow-column params");
    let var_heavy = config::r1cs_params(12, 60).expect("var-heavy params");

    let row_heavy_summary = row_heavy
        .validate_ccs_shape(60, 54, 3, 2)
        .expect("row-heavy census");
    let narrow_column_summary = narrow_columns
        .validate_ccs_shape(60, 12, 3, 2)
        .expect("narrow-column census");
    let var_heavy_summary = var_heavy
        .validate_ccs_shape(12, 60, 3, 2)
        .expect("var-heavy census");

    assert_eq!(row_heavy_summary.cube_variables, 6);
    assert_eq!(narrow_column_summary.cube_variables, 6);
    assert_eq!(var_heavy_summary.cube_variables, 7);
    assert_eq!(row_heavy_summary.field_factor, narrow_column_summary.field_factor);
    assert!(var_heavy_summary.field_factor > row_heavy_summary.field_factor);
}

#[test]
fn ccs_params_charge_matrix_count_and_degree() {
    let r1cs = config::r1cs_params(60, 54).expect("R1CS params");
    let t8 = config::ccs_params(60, 54, 8, 2).expect("t=8 CCS params");
    let degree7 = config::ccs_params(60, 54, 3, 7).expect("degree-7 CCS params");

    assert_eq!(r1cs.lambda(), 115);
    assert_eq!(t8.lambda(), 114);
    assert_eq!(degree7.lambda(), 115);
}

#[test]
fn actual_ccs_shape_validation_rejects_an_undercharged_matrix_count() {
    let params = Params::for_ccs_shape(1 << 24, 1, 1, 8).expect("small-t profile");

    params
        .validate_ccs_shape(1 << 24, 1, 1, 8)
        .expect("the selected shape must validate itself");
    assert!(
        params.validate_ccs_shape(1 << 24, 1, 1_000, 8).is_err(),
        "preprocessing must not reuse parameters selected for a much smaller t"
    );
}

#[test]
fn explicit_r1cs_minimum_rejects_an_unsupported_target() {
    let err = Params::for_r1cs_shape_with(60, 54, 117, 0)
        .expect_err("the combined padded-row census cannot provide 117 bits");
    assert!(matches!(
        err,
        neo_params::ParamsError::InsufficientStatisticalSecurity {
            required: 117,
            available: 115
        }
    ));
}
