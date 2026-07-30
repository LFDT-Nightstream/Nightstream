//! Audit checks for the single production parameter profile.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly};
use neo_fold_clean::{config, Params};
use neo_math::{D, F};

fn empty_structure(rows: usize, columns: usize, matrices: usize) -> CcsStructure<F> {
    let matrix = CcsMatrix::Csc(CscMat::from_triplets(Vec::new(), rows, columns));
    CcsStructure {
        matrices: vec![matrix; matrices],
        f: SparsePoly::new(matrices, Vec::new()),
        n: rows,
        m: columns,
    }
}

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

/// Golden vector shared with
/// `Nightstream.SuperNeo.Concrete.production_parameter_values`.
#[test]
fn production_params_match_lean_m1_profile() {
    let pp = Params::production();
    assert_eq!(pp.q(), 18_446_744_069_414_584_321);
    assert_eq!(pp.b(), 2);
    assert_eq!(pp.k_rho(), 14);
    assert_eq!(pp.big_b(), 16_384);
    assert_eq!(pp.T(), 216);
    assert_eq!(pp.max_fresh_count(), 61);
    assert_eq!(pp.eta(), 81);
    assert_eq!(pp.d(), 54);
    assert_eq!(pp.kappa(), 18);
    assert_eq!(pp.extension_degree(), 2);
    assert_eq!(pp.lambda(), 125);
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
fn actual_ccs_shape_validation_rejects_an_undercharged_matrix_count() {
    let params = Params::for_ccs_shape_with(
        1 << 24,
        1,
        8,
        config::MIN_EFFECTIVE_LAMBDA,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("small-t profile");

    params
        .validate_ccs_shape(1 << 24, 1, 8)
        .expect("the selected shape must validate itself");
    assert!(
        params.validate_ccs_shape(1 << 24, 1_000, 8).is_err(),
        "preprocessing must not reuse parameters selected for a much smaller t"
    );
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

#[test]
fn pending_family_profile_is_domain_derived_not_an_exact_width_sentinel() {
    use neo_fold_clean::paper::construction2::running::uses_pending_accumulator_family;

    for columns in [11_437_038, 11_636_028] {
        assert!(uses_pending_accumulator_family(&empty_structure(
            6_956_163, columns, 13
        )));
    }

    assert!(!uses_pending_accumulator_family(&empty_structure(
        6_956_163,
        D * (1 << 18) + 1,
        13,
    )));
    assert!(!uses_pending_accumulator_family(&empty_structure(
        6_956_163, 11_636_028, 12,
    )));
    assert!(!uses_pending_accumulator_family(&empty_structure(
        (1 << 23) + 1,
        11_636_028,
        13,
    )));
}
