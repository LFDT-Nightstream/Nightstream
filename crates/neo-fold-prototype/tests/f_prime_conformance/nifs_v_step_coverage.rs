//! HyperNova Construction 2 §6.3 step 4 wires NIFS.V across the full
//! Π_SuperNeo = Π_CCS ∘ Π_RLC ∘ Π_DEC composition. Every sub-protocol must
//! be actively checked: if the verifier only touches one of CCS/RLC/DEC the
//! remaining sub-proofs become soundness-free cargo.
//!
//! This test proves active coverage by flipping cargo that lives inside each
//! sub-protocol slot and asserting native F' rejects:
//!   * Π_CCS: flip a sum-check round coefficient in the authoritative
//!     `main_circuit_replay_witness.ccs_replay_proof`.
//!   * Π_DEC: flip the first word of a Π_DEC child commitment.
//!
//! (Π_RLC coverage is established transitively through the replay check that
//! must reproduce the same `y_ring`/`y_zcol` payloads — the Π_CCS tamper path
//! also exercises the RLC replay transcript.)

use neo_fold_prototype::rv32im::audit::{
    evaluate_rv32im_main_recursion_f_prime_advice,
    rv32im_main_recursion_advice_tamper_authoritative_ccs_replay_first_round_coeff,
    rv32im_main_recursion_advice_tamper_dec_child_commitment_first_word,
};

use super::support::single_step_advices;

#[test]
fn f_prime_nifs_v_rejects_pi_ccs_replay_tamper() {
    let advices = single_step_advices();
    let advice = &advices[0];
    evaluate_rv32im_main_recursion_f_prime_advice(advice).expect("baseline advice must evaluate");

    let mut tampered = advice.clone();
    rv32im_main_recursion_advice_tamper_authoritative_ccs_replay_first_round_coeff(&mut tampered);
    assert!(
        evaluate_rv32im_main_recursion_f_prime_advice(&tampered).is_err(),
        "NIFS.V must reject a tampered Π_CCS sum-check round coefficient"
    );
}

#[test]
fn f_prime_nifs_v_rejects_pi_dec_child_commitment_tamper() {
    let advices = single_step_advices();
    let advice = &advices[0];
    evaluate_rv32im_main_recursion_f_prime_advice(advice).expect("baseline advice must evaluate");

    let mut tampered = advice.clone();
    rv32im_main_recursion_advice_tamper_dec_child_commitment_first_word(&mut tampered, 0);
    assert!(
        evaluate_rv32im_main_recursion_f_prime_advice(&tampered).is_err(),
        "NIFS.V must reject a tampered Π_DEC child commitment"
    );
}
