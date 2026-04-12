use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ccs::{SparsePoly, Term};
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, enforce_k_eq, KNum};
use neo_fold_next::rv64im::main_relation_circuit::terminal_common::{eq_points, eval_sparse_poly_in_k, range_product};
use neo_math::F as GoldilocksF;
use neo_math::K as NeoK;
use neo_reductions::engines::optimized_engine::eq_points as native_eq_points;
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn k(re: u64, im: u64) -> NeoK {
    neo_math::from_complex(GoldilocksF::from_u64(re), GoldilocksF::from_u64(im))
}

fn native_range_product(value: NeoK, b: u32) -> NeoK {
    let mut acc = NeoK::ONE;
    for t in -((b as i64) - 1)..=((b as i64) - 1) {
        acc *= value - NeoK::from(GoldilocksF::from_i64(t));
    }
    acc
}

#[test]
fn rv64im_main_relation_terminal_common_matches_native_algebra() {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let delta = SpartanF::from_canonical_u64(7);

    let p_values = vec![k(2, 1), k(5, 0)];
    let q_values = vec![k(3, 0), k(1, 2)];
    let p_vars = p_values
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("p_{idx}")).expect("p"))
        .collect::<Vec<_>>();
    let q_vars = q_values
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("q_{idx}")).expect("q"))
        .collect::<Vec<_>>();
    let (eq_var, eq_value) =
        eq_points(&mut cs, &p_vars, &q_vars, &p_values, &q_values, delta, "eq").expect("eq_points");
    let expected_eq = native_eq_points(&p_values, &q_values);
    assert_eq!(eq_value, expected_eq);
    let expected_eq_var = alloc_constant_k(&mut cs, KNum::from_neo_k(expected_eq), "expected_eq").expect("expected eq");
    enforce_k_eq(&mut cs, &eq_var, &expected_eq_var, "eq_match");

    let value = k(4, 1);
    let value_var = alloc_constant_k(&mut cs, KNum::from_neo_k(value), "range_value").expect("range value");
    let (range_var, range_value) = range_product(&mut cs, &value_var, value, 3, delta, "range").expect("range");
    let expected_range = native_range_product(value, 3);
    assert_eq!(range_value, expected_range);
    let expected_range_var =
        alloc_constant_k(&mut cs, KNum::from_neo_k(expected_range), "expected_range").expect("expected range");
    enforce_k_eq(&mut cs, &range_var, &expected_range_var, "range_match");

    let poly = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: GoldilocksF::from_u64(3),
                exps: vec![1, 0],
            },
            Term {
                coeff: GoldilocksF::from_u64(5),
                exps: vec![0, 2],
            },
            Term {
                coeff: GoldilocksF::from_u64(7),
                exps: vec![0, 0],
            },
        ],
    );
    let inputs = vec![k(2, 0), k(1, 1)];
    let input_vars = inputs
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("input_{idx}")).expect("in"))
        .collect::<Vec<_>>();
    let (poly_var, poly_value) =
        eval_sparse_poly_in_k(&mut cs, &poly, &input_vars, &inputs, delta, "poly").expect("poly eval");
    let expected_poly = NeoK::from(GoldilocksF::from_u64(3)) * inputs[0]
        + NeoK::from(GoldilocksF::from_u64(5)) * inputs[1] * inputs[1]
        + NeoK::from(GoldilocksF::from_u64(7));
    assert_eq!(poly_value, expected_poly);
    let expected_poly_var =
        alloc_constant_k(&mut cs, KNum::from_neo_k(expected_poly), "expected_poly").expect("expected poly");
    enforce_k_eq(&mut cs, &poly_var, &expected_poly_var, "poly_match");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
