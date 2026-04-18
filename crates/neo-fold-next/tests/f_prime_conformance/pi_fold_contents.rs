//! SuperNeo §7.4: the folding-proof `π_fold` carries only the sum-check round
//! polynomials, the reduced evaluations `y'_{i,j}`, and the Π_DEC child
//! commitments `(c_i, {y_{i,j}})`. Fiat–Shamir challenges (α, γ, sum-check
//! round r_t, RLC ρ_i) are NEVER serialized into `π_fold` — they are squeezed
//! from the transcript at verify time. Putting a challenge into the proof
//! bytes means the prover can forge it.
//!
//! This test formats the F' advice (which recursively includes `π_fold` via
//! its `Debug` impl) and asserts that it does not name any Fiat–Shamir
//! challenge field. It is a pragmatic name-level firewall: any future change
//! that sneaks a challenge field into the proof struct under a recognizable
//! name (`alpha`, `gamma`, `rho_`, `challenge_`) will trip this test.

use neo_fold_next::rv64im::audit::audit_rv64im_main_recursion_construction2_pi_fold_debug_dump;

use super::support::single_step_advices;

#[test]
fn f_prime_pi_fold_does_not_carry_fiat_shamir_challenges() {
    let advices = single_step_advices();
    let dump = audit_rv64im_main_recursion_construction2_pi_fold_debug_dump(&advices[0]).to_ascii_lowercase();

    for forbidden in [
        "alpha:",
        "gamma:",
        "rho_i:",
        "rho_vec",
        "challenge_",
        "fs_alpha",
        "fs_gamma",
    ] {
        assert!(
            !dump.contains(forbidden),
            "SuperNeo §7.4: π_fold must not carry Fiat–Shamir challenges. Found field mention `{forbidden}` \
             in F' advice debug dump — α/γ/ρ must be squeezed from the transcript at verify time."
        );
    }
}
