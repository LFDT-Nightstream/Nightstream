//! HyperNova Construction 2 §6.3 base case: at step i = 0 the verifier
//! unconditionally accepts, but ONLY when `z_0 == z_i` AND `U_i == U_⊥`.
//! Any deviation in z or U at the base case must cause native F' to reject.
//!
//! This test exercises both limbs of that gate:
//!   * tampering z_i on the base-case advice must cause F' to reject;
//!   * tampering the running accumulator (by flipping the carried terminal
//!     handle, which directly feeds U_i) must cause F' to reject.

use neo_fold_next::rv64im::audit::{
    evaluate_rv64im_main_recursion_f_prime_advice, rv64im_main_recursion_advice_retarget_x_hash_to_current_accumulator,
    rv64im_main_recursion_advice_tamper_construction2_input_fresh_instance_x_first_byte,
    rv64im_main_recursion_advice_tamper_running_state_first_claim_commitment_first_word,
    rv64im_main_recursion_advice_tamper_z_i_first_byte,
};
use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_f_prime_public_output, verify_rv64im_main_recursion_f_prime_public_output,
};

use super::support::single_step_advices;

#[test]
fn f_prime_base_case_rejects_tampered_z_i() {
    let advices = single_step_advices();
    let base_case = &advices[0];
    assert_eq!(
        base_case.chunk_count_in(),
        0,
        "expected advice[0] to be the base case (chunk_count_in == 0)"
    );
    evaluate_rv64im_main_recursion_f_prime_advice(base_case).expect("baseline base case must evaluate");

    let mut tampered = base_case.clone();
    rv64im_main_recursion_advice_tamper_z_i_first_byte(&mut tampered);
    assert!(
        evaluate_rv64im_main_recursion_f_prime_advice(&tampered).is_err(),
        "HN Construction-2 §6.3 base case must reject when z_0 != z_i"
    );
}

#[test]
fn f_prime_base_case_rejects_tampered_construction2_u_perp() {
    let advices = single_step_advices();
    let base_case = &advices[0];
    assert_eq!(
        base_case.chunk_count_in(),
        0,
        "expected advice[0] to be the base case (chunk_count_in == 0)"
    );
    evaluate_rv64im_main_recursion_f_prime_advice(base_case).expect("baseline base case must evaluate");

    let mut tampered = base_case.clone();
    rv64im_main_recursion_advice_tamper_construction2_input_fresh_instance_x_first_byte(&mut tampered);
    assert!(
        evaluate_rv64im_main_recursion_f_prime_advice(&tampered).is_err(),
        "HN Construction-2 §6.3 base case must reject when the threaded canonical u_perp drifts from the explicit default pair"
    );
}

#[test]
fn f_prime_base_case_rejects_tampered_running_u_perp_tuple_even_if_x_i_is_retargeted() {
    let advices = single_step_advices();
    let base_case = &advices[0];
    let public_output =
        build_rv64im_main_recursion_f_prime_public_output(base_case).expect("build baseline base-case public output");

    let mut tampered = base_case.clone();
    rv64im_main_recursion_advice_tamper_running_state_first_claim_commitment_first_word(&mut tampered);
    rv64im_main_recursion_advice_retarget_x_hash_to_current_accumulator(&mut tampered);

    let err = verify_rv64im_main_recursion_f_prime_public_output(&public_output, &tampered)
        .expect_err("HN Construction-2 §6.3 base case must reject when the running U_i tuple drifts from U_perp");
    assert!(
        err.to_string().contains("U_i = (u_perp) carry"),
        "expected paper-facing U_perp carry failure, got: {err}"
    );
}
