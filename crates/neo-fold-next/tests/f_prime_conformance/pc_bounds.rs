//! HyperNova Construction 2 §6.3 pc discipline: the control-lane pointer
//! `pc_i` must stay within the declared bounds of the IVC machine — in this
//! specialization, the RV64IM trivial control lane fixes `pc_i = 1`, so any
//! pc that drifts from the canonical trivial pc must be rejected by native
//! F'. A prover that could forge pc breaks partial-function addressing.

use neo_fold_next::rv64im::audit::{
    evaluate_rv64im_main_recursion_f_prime_advice, rv64im_main_recursion_advice_tamper_pc_i,
};

use super::support::single_step_advices;

#[test]
fn f_prime_rejects_out_of_bounds_pc_i() {
    let advices = single_step_advices();
    let advice = &advices[0];
    evaluate_rv64im_main_recursion_f_prime_advice(advice).expect("baseline advice must evaluate");

    let mut tampered = advice.clone();
    rv64im_main_recursion_advice_tamper_pc_i(&mut tampered);
    assert!(
        evaluate_rv64im_main_recursion_f_prime_advice(&tampered).is_err(),
        "HN Construction-2 §6.3: native F' must reject when pc_i is outside the canonical trivial \
         control lane (tamper helper sets pc_i = 0)"
    );
}
