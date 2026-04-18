//! SuperNeo arch §15 (replayability): native F' and circuit F' must produce
//! bit-exact outputs on the same inputs. This is the invariant that lets the
//! IVC verifier rebuild the recursive state independently and compare.
//!
//! Testing native-vs-circuit parity requires synthesizing the circuit which
//! is expensive. As a minimum lock-in, this test asserts the weaker but still
//! load-bearing determinism property: evaluating the native F' twice on the
//! same advice must produce bit-exact step images across every public field.

use neo_fold_next::rv64im::audit::evaluate_rv64im_main_recursion_f_prime_advice;

use super::support::single_step_advices;

#[test]
fn f_prime_native_is_deterministic_across_evaluations() {
    let advices = single_step_advices();
    for (step, advice) in advices.iter().enumerate() {
        let first = evaluate_rv64im_main_recursion_f_prime_advice(advice)
            .unwrap_or_else(|err| panic!("step {step}: first evaluation failed: {err}"));
        let second = evaluate_rv64im_main_recursion_f_prime_advice(advice)
            .unwrap_or_else(|err| panic!("step {step}: second evaluation failed: {err}"));

        assert_eq!(
            first.chunk_count(),
            second.chunk_count(),
            "step {step}: chunk_count drift"
        );
        assert_eq!(first.z_next(), second.z_next(), "step {step}: z_next drift");
        assert_eq!(first.pc_next(), second.pc_next(), "step {step}: pc_next drift");
        assert_eq!(
            first.folded_accumulator_digest(),
            second.folded_accumulator_digest(),
            "step {step}: folded_accumulator_digest drift"
        );
        assert_eq!(first.x_out(), second.x_out(), "step {step}: x_out drift");
        assert_eq!(
            first.construction2_u_next(),
            second.construction2_u_next(),
            "step {step}: construction2_u_next drift"
        );
    }
}
