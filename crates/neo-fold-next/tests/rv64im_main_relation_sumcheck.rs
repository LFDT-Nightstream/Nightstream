use bellpepper_core::test_cs::TestConstraintSystem;
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, alloc_k, KNum};
use neo_fold_next::rv64im::main_relation_circuit::sumcheck::{sumcheck_eval_gadget, sumcheck_round_gadget};
use neo_math::F as GoldilocksF;
use neo_math::K as NeoK;
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn k(re: u64, im: u64) -> NeoK {
    neo_math::from_complex(GoldilocksF::from_u64(re), GoldilocksF::from_u64(im))
}

#[test]
fn rv64im_main_relation_sumcheck_gadgets_match_native_horner_logic() {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let delta = SpartanF::from_canonical_u64(7);
    let coeff_values = vec![k(3, 0), k(5, 0), k(7, 0)];
    let coeffs = coeff_values
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_k(&mut cs, Some(KNum::from_neo_k(*value)), &format!("coeff_{idx}")).expect("coeff"))
        .collect::<Vec<_>>();
    let claimed_sum = alloc_constant_k(&mut cs, KNum::from_neo_k(k(18, 0)), "claimed_sum").expect("claimed sum");
    sumcheck_round_gadget(&mut cs, &coeffs, &coeff_values, &claimed_sum, "round").expect("round gadget");

    let challenge = alloc_constant_k(&mut cs, KNum::from_neo_k(k(2, 0)), "challenge").expect("challenge");
    let eval =
        sumcheck_eval_gadget(&mut cs, &coeffs, &coeff_values, &challenge, k(2, 0), delta, "eval").expect("eval gadget");
    let expected_eval = alloc_constant_k(&mut cs, KNum::from_neo_k(k(41, 0)), "expected_eval").expect("expected eval");
    neo_fold_next::rv64im::main_relation_circuit::k_field::enforce_k_eq(&mut cs, &eval, &expected_eval, "eval_eq");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
