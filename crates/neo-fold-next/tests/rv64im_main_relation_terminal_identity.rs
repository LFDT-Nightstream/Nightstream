use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, enforce_k_eq, KNum};
use neo_fold_next::rv64im::main_relation_circuit::terminal_identity::{
    dummy_claim, rhs_terminal_identity_fe, rhs_terminal_identity_nc,
};
use neo_math::F as GoldilocksF;
use neo_math::K as NeoK;
use neo_params::NeoParams;
use neo_reductions::optimized_engine::{
    rhs_terminal_identity_fe_with_k_mcs, rhs_terminal_identity_nc as native_rhs_terminal_identity_nc, Challenges,
};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn k(re: i64, im: i64) -> NeoK {
    neo_math::from_complex(GoldilocksF::from_i64(re), GoldilocksF::from_i64(im))
}

fn dense_matrix(rows: usize, cols: usize, entries: &[(usize, usize, u64)]) -> Mat<GoldilocksF> {
    let mut out = Mat::zero(rows, cols, GoldilocksF::ZERO);
    for (row, col, value) in entries {
        out[(*row, *col)] = GoldilocksF::from_u64(*value);
    }
    out
}

fn synthetic_structure() -> CcsStructure<GoldilocksF> {
    CcsStructure::new(
        vec![
            dense_matrix(2, 2, &[(0, 0, 1), (1, 1, 1)]),
            dense_matrix(2, 2, &[(0, 1, 2), (1, 0, 3)]),
        ],
        SparsePoly::new(
            2,
            vec![
                Term {
                    coeff: GoldilocksF::from_u64(3),
                    exps: vec![1, 0],
                },
                Term {
                    coeff: GoldilocksF::from_u64(5),
                    exps: vec![0, 1],
                },
                Term {
                    coeff: GoldilocksF::from_u64(7),
                    exps: vec![0, 0],
                },
            ],
        ),
    )
    .expect("synthetic structure")
}

fn synthetic_claims() -> Vec<neo_ccs::CeClaim<neo_ajtai::Commitment, GoldilocksF, NeoK>> {
    vec![
        dummy_claim(
            vec![vec![k(2, 0), k(1, 1)], vec![k(3, 0), k(0, 1)]],
            vec![k(2, 0), k(5, 0)],
            vec![k(1, 0), k(2, 0)],
            vec![k(3, 0)],
            vec![k(2, 0)],
        ),
        dummy_claim(
            vec![vec![k(1, 0), k(4, 0)], vec![k(0, 1), k(2, 0)]],
            vec![k(1, 0), k(4, 0)],
            vec![k(3, 0), k(1, 1)],
            vec![k(3, 0)],
            vec![k(2, 0)],
        ),
        dummy_claim(
            vec![vec![k(2, 1), k(1, 0)], vec![k(1, 0), k(3, 1)]],
            vec![k(2, 1), k(1, 0)],
            vec![k(0, 1), k(2, 0)],
            vec![k(3, 0)],
            vec![k(2, 0)],
        ),
    ]
}

fn synthetic_public_challenges() -> Challenges {
    Challenges {
        alpha: vec![k(2, 0)],
        beta_a: vec![k(1, 1)],
        beta_r: vec![k(3, 0)],
        beta_m: vec![k(2, 1)],
        gamma: k(4, 0),
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_terminal_identity_fe_matches_native_formula() {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let delta = SpartanF::from_canonical_u64(7);
    let structure = synthetic_structure();
    let claims = synthetic_claims();
    let public_challenges = synthetic_public_challenges();
    let r_prime = vec![k(5, 0)];
    let alpha_prime = vec![k(1, 0)];
    let me_inputs_r = vec![k(6, 0)];

    let claim_vars = claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| alloc_ce_claim(&mut cs, claim, &format!("claim_{idx}")).expect("claim"))
        .collect::<Vec<_>>();
    let alpha_vars = public_challenges
        .alpha
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("alpha_{idx}")).expect("alpha")
        })
        .collect::<Vec<_>>();
    let beta_a_vars = public_challenges
        .beta_a
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("beta_a_{idx}")).expect("beta_a")
        })
        .collect::<Vec<_>>();
    let beta_r_vars = public_challenges
        .beta_r
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("beta_r_{idx}")).expect("beta_r")
        })
        .collect::<Vec<_>>();
    let gamma_var = alloc_constant_k(&mut cs, KNum::from_neo_k(public_challenges.gamma), "gamma").expect("gamma");
    let r_prime_vars = r_prime
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("r_prime_{idx}")).expect("r"))
        .collect::<Vec<_>>();
    let alpha_prime_vars = alpha_prime
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("alpha_prime_{idx}")).expect("ap")
        })
        .collect::<Vec<_>>();
    let me_inputs_r_vars = me_inputs_r
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("me_inputs_r_{idx}")).expect("in_r")
        })
        .collect::<Vec<_>>();

    let (rhs_var, rhs_value) = rhs_terminal_identity_fe(
        &mut cs,
        &structure,
        &public_challenges,
        &alpha_vars,
        &beta_a_vars,
        &beta_r_vars,
        &gamma_var,
        &r_prime_vars,
        &r_prime,
        &alpha_prime_vars,
        &alpha_prime,
        &claim_vars,
        1,
        Some(&me_inputs_r_vars),
        Some(&me_inputs_r),
        delta,
        "fe",
    )
    .expect("circuit fe rhs");
    let expected = rhs_terminal_identity_fe_with_k_mcs(
        &structure,
        &NeoParams::goldilocks_127(),
        &public_challenges,
        &r_prime,
        &alpha_prime,
        &claims,
        1,
        Some(&me_inputs_r),
    );
    assert_eq!(rhs_value, expected);
    let expected_var = alloc_constant_k(&mut cs, KNum::from_neo_k(expected), "expected_fe").expect("expected");
    enforce_k_eq(&mut cs, &rhs_var, &expected_var, "fe_matches_native");
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_terminal_identity_nc_matches_native_formula() {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let delta = SpartanF::from_canonical_u64(7);
    let claims = synthetic_claims();
    let public_challenges = synthetic_public_challenges();
    let s_col_prime = vec![k(2, 0)];
    let alpha_prime = vec![k(1, 0)];
    let params = NeoParams::goldilocks_127();

    let claim_vars = claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| alloc_ce_claim(&mut cs, claim, &format!("claim_{idx}")).expect("claim"))
        .collect::<Vec<_>>();
    let beta_a_vars = public_challenges
        .beta_a
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("beta_a_{idx}")).expect("beta_a")
        })
        .collect::<Vec<_>>();
    let beta_m_vars = public_challenges
        .beta_m
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("beta_m_{idx}")).expect("beta_m")
        })
        .collect::<Vec<_>>();
    let gamma_var = alloc_constant_k(&mut cs, KNum::from_neo_k(public_challenges.gamma), "gamma").expect("gamma");
    let s_col_prime_vars = s_col_prime
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("s_col_prime_{idx}")).expect("s")
        })
        .collect::<Vec<_>>();
    let alpha_prime_vars = alpha_prime
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("alpha_prime_{idx}")).expect("ap")
        })
        .collect::<Vec<_>>();

    let (rhs_var, rhs_value) = rhs_terminal_identity_nc(
        &mut cs,
        &params,
        &public_challenges,
        &beta_a_vars,
        &beta_m_vars,
        &gamma_var,
        &s_col_prime_vars,
        &s_col_prime,
        &alpha_prime_vars,
        &alpha_prime,
        &claim_vars,
        delta,
        "nc",
    )
    .expect("circuit nc rhs");
    let expected = native_rhs_terminal_identity_nc(&params, &public_challenges, &s_col_prime, &alpha_prime, &claims);
    assert_eq!(rhs_value, expected);
    let expected_var = alloc_constant_k(&mut cs, KNum::from_neo_k(expected), "expected_nc").expect("expected");
    enforce_k_eq(&mut cs, &rhs_var, &expected_var, "nc_matches_native");
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
