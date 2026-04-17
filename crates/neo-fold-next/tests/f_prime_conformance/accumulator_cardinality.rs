//! HyperNova Construction 2 carries a fixed-width running accumulator `U_i`.
//! In the current RV64IM specialization the control function is pinned to the
//! single trivial lane (`ℓ = 1`), so the authoritative native Goal-1 surface
//! owns exactly one paper-facing running slot per step: the threaded
//! Construction-2 fresh input `u_i`.
//!
//! This test locks that width contract directly on the public native advice
//! surface instead of on legacy internal cargo.

use neo_fold_next::rv64im::evaluate_rv64im_main_recursion_f_prime_advice;

use super::support::single_step_advices;

#[test]
fn f_prime_authoritative_accumulator_surface_has_fixed_single_slot_shape() {
    let advices = single_step_advices();

    for (step, advice) in advices.iter().enumerate() {
        assert!(
            advice.construction2_input_fresh_instance().is_some(),
            "step {step}: the authoritative native Goal-1 surface must expose exactly one Construction-2 running slot"
        );
        assert_eq!(
            advice.pc_i(),
            1,
            "step {step}: the current RV64IM Construction-2 specialization is single-slot and must keep pc_i on lane 1"
        );

        let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)
            .unwrap_or_else(|err| panic!("step {step}: evaluate advice: {err}"));
        assert_eq!(
            step_image.pc_next(),
            1,
            "step {step}: the authoritative native Goal-1 output must stay on the single Construction-2 slot"
        );
    }
}
