//! SuperNeo §7.4 Π_RLC step 1: the verifier samples `K + k` transcript-bound
//! `ρ_i` challenges, one per CE claim entering the RLC stage.
//!
//! This test locks the owned native boundary we can actually observe:
//! - the sampled vector length must equal `K + k` for the carried step, and
//! - the sampled vector must be deterministic for a fixed transcript but change
//!   when the verifier transcript state changes.

use neo_fold_next::rv64im::audit::{
    audit_rv64im_main_recursion_construction2_pi_rlc_rho_digests,
    rv64im_main_recursion_advice_tamper_running_state_transcript_state_first_field,
};

use super::support::single_step_advices;

#[test]
fn f_prime_pi_rlc_samples_k_plus_k_transcript_bound_rho() {
    let advices = single_step_advices();
    let advice = &advices[0];

    let baseline = audit_rv64im_main_recursion_construction2_pi_rlc_rho_digests(advice)
        .expect("sample baseline Pi_RLC rho digests");
    let replay = audit_rv64im_main_recursion_construction2_pi_rlc_rho_digests(advice)
        .expect("resample baseline Pi_RLC rho digests");

    assert_eq!(
        baseline, replay,
        "Pi_RLC rho sampling must be deterministic for a fixed verifier transcript"
    );
    assert_eq!(
        baseline.len(),
        1 + advice.running_state().carry.main.claims.len(),
        "Pi_RLC must sample one rho per CE claim entering the RLC stage (K + k in SuperNeo §7.4)"
    );

    let mut tampered = advice.clone();
    rv64im_main_recursion_advice_tamper_running_state_transcript_state_first_field(&mut tampered);
    let tampered_rhos = audit_rv64im_main_recursion_construction2_pi_rlc_rho_digests(&tampered)
        .expect("sample tampered Pi_RLC rho digests");

    assert_ne!(
        baseline, tampered_rhos,
        "Pi_RLC rho sampling must be driven by the verifier transcript, not a fixed or ignored placeholder path"
    );
}
