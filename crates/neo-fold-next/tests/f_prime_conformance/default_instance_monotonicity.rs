//! HyperNova Construction 2 in the current RV64IM specialization runs in the
//! single-program-counter regime `ℓ = 1`. The authoritative native Goal-1
//! surface therefore has one paper-facing running slot: the threaded
//! Construction-2 fresh input `u_i`.
//!
//! This test locks the monotonicity rule on that owned slot:
//! - step 0 must start from the canonical witness-backed `u_perp`,
//! - step n+1 must carry exactly the `u_{i+1}` produced by step n.

use neo_fold_next::rv64im::{
    build_rv64im_main_recursion_construction2_default_fresh_instance, evaluate_rv64im_main_recursion_f_prime_advice,
};

use super::support::{default_full_width_from_advice, single_step_advices};

#[test]
fn f_prime_single_slot_accumulator_threads_monotonically() {
    let advices = single_step_advices();
    let mut expected_u_i = build_rv64im_main_recursion_construction2_default_fresh_instance(
        advices[0].verifier_key_fs(),
        default_full_width_from_advice(&advices[0]),
    )
    .expect("build canonical u_perp");

    for (step, advice) in advices.iter().enumerate() {
        assert_eq!(
            advice.construction2_input_fresh_instance(),
            Some(&expected_u_i),
            "step {step}: the authoritative native Goal-1 surface must thread the single Construction-2 slot monotonically"
        );

        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)
            .unwrap_or_else(|err| panic!("step {step}: evaluate advice: {err}"));
        expected_u_i = step_image.construction2_u_next().clone();
    }
}
