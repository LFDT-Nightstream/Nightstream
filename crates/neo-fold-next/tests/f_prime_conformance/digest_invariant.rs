use neo_fold_next::rv32im::audit::{
    evaluate_rv32im_main_recursion_f_prime_advice,
    rv32im_main_recursion_advice_tamper_folded_accumulator_input_digest_first_byte,
};

use super::support::single_step_advices;

#[test]
fn f_prime_recomputes_folded_accumulator_input_digest_from_state_in() {
    let advice = &single_step_advices()[0];
    evaluate_rv32im_main_recursion_f_prime_advice(advice).expect("baseline F' advice must evaluate");

    let mut tampered = advice.clone();
    rv32im_main_recursion_advice_tamper_folded_accumulator_input_digest_first_byte(&mut tampered);
    let err = evaluate_rv32im_main_recursion_f_prime_advice(&tampered)
        .expect_err("F' must reject a folded-accumulator digest that is not recomputed from state_in");

    assert!(
        err.to_string()
            .contains("folded accumulator input digest does not match state_in"),
        "unexpected error for folded-accumulator digest tamper: {err}"
    );
}
