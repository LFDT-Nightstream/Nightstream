use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::initial_sum::claimed_initial_sum_from_me_inputs;
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, enforce_k_eq, KNum};
use neo_math::F as GoldilocksF;
use neo_math::K as NeoK;
use neo_reductions::optimized_engine::{claimed_initial_sum_from_inputs_with_k_mcs, Challenges};
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
        neo_fold_next::rv64im::main_relation_circuit::terminal_identity::dummy_claim(
            vec![vec![k(2, 0), k(1, 1)], vec![k(3, 0), k(0, 1)]],
            vec![k(2, 0), k(5, 0)],
            vec![k(1, 0), k(2, 0)],
            vec![k(3, 0)],
            vec![k(2, 0)],
        ),
        neo_fold_next::rv64im::main_relation_circuit::terminal_identity::dummy_claim(
            vec![vec![k(1, 0), k(4, 0)], vec![k(0, 1), k(2, 0)]],
            vec![k(1, 0), k(4, 0)],
            vec![k(3, 0), k(1, 1)],
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
fn rv64im_main_relation_initial_sum_matches_native_formula() {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let delta = SpartanF::from_canonical_u64(7);
    let structure = synthetic_structure();
    let claims = synthetic_claims();
    let public_challenges = synthetic_public_challenges();

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
    let gamma_var = alloc_constant_k(&mut cs, KNum::from_neo_k(public_challenges.gamma), "gamma").expect("gamma");

    let (claimed_sum, claimed_sum_value) = claimed_initial_sum_from_me_inputs(
        &mut cs,
        &structure,
        &alpha_vars,
        &public_challenges.alpha,
        &gamma_var,
        public_challenges.gamma,
        1,
        &claim_vars,
        delta,
        "initial_sum",
    )
    .expect("initial sum gadget");
    let expected = claimed_initial_sum_from_inputs_with_k_mcs(&structure, &public_challenges, 1, &claims);
    assert_eq!(claimed_sum_value, expected);
    let expected_var = alloc_constant_k(&mut cs, KNum::from_neo_k(expected), "expected_initial_sum").expect("expected");
    enforce_k_eq(&mut cs, &claimed_sum, &expected_var, "initial_sum_matches_native");
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
