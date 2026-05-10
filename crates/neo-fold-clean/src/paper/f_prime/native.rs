//! F' — the augmented function from Hypernova §6.3 Construction 2.
//!
//! ```text
//! F'_j(vk_fs, U_i, u_i, pc_i, (i, z_0, z_i), ω_i, π) → x:
//!   1. pc_{i+1} = φ(z_i, ω_i)
//!   2. z_{i+1}  = F_j(z_i, ω_i)
//!   3. base case (i = 0):  no NIFS.P; running = canonical default
//!   4. recursive case:     NIFS.V(vk_fs[pc_i], U_i, u_i, π) → U_{i+1}
//!   5. x = hash(vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1})
//! ```
//!
//! For ccs-direct (ℓ = 1):
//! - `pc` is constant `TRIVIAL_PC` (φ trivially returns 1).
//! - `z_{i+1}` is the chain of the chunk's public-instance digest.
//! - The recursive `NIFS.V` call lives in [`crate::paper::nifs::verify`].
//!
//! ## Control flow at extend(state, next_latest)
//!
//! ```text
//! match state.proof:
//!   Initial                               -> no fold; running = empty
//!   Active { running, latest }            -> NIFS.P(running, latest) -> next_running
//! advance_state -> compute_x_out
//! state.proof = Active { running: next_running, latest: next_latest }
//! ```
//!
//! The ENCODING step ("encode this F' as next_latest") is the frontend's
//! job in PR7. Here today: the caller passes `next_latest` directly.
//!
//! This file owns the *native* F' on both sides (prover and verifier). The
//! R1CS form lives in `engine::decider` (PR5).

use neo_ajtai::AjtaiSModule;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::{
    self, FoldProof, LatestInstance, ProofState, RunningInstance, State, StepProof, VerifierKey,
};
use crate::paper::nifs;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, DecMixer, RlcMixer, Structure};

pub use construction2::Error;

// ──────────────────────────────────────────────────────────────────────────
// F' prove (native)
// ──────────────────────────────────────────────────────────────────────────

/// One full F' invocation on the prover side.
///
/// Reads `state.proof` to decide base-vs-recursive case:
/// - **Initial** (i = 0): no NIFS.P runs. `next_running = empty`.
///   `FoldProof::NoFold`.
/// - **Active** (i ≥ 1): NIFS.P folds `state.proof.latest` into
///   `state.proof.running`. `FoldProof::Recursive(π)`.
///
/// The new `next_latest` (the K instances the *next* step will fold) comes
/// from the caller. In strict Construction 2 (PR5+) it would be synthesized
/// internally from the F' encoder; in the direct-CCS interim it's the
/// caller's batch of CcsInstances.
pub fn prove(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
) -> Result<(State, StepProof), Error> {
    construction2::enforce_pc_in_range(&state)?;
    construction2::state_base_case_check(&state)?;

    let fresh_count = next_latest.len() as u64;
    let chunk_digest = construction2::chunk_public_digest_for_step(state.step_count, &next_latest);

    // Destructure proof out of state up front so the rest can move the
    // remaining fields freely.
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest,
        public_trace,
        proof: prev_proof,
    } = state;

    // F' fold step — branch on the tagged ProofState.
    let (next_running, fold) = match prev_proof {
        ProofState::Initial => {
            // i = 0: no NIFS.P; the running accumulator stays empty.
            (RunningInstance::default(), FoldProof::NoFold)
        }
        ProofState::Active { running, latest } => {
            let (next_running, nifs_proof) = nifs::prove(
                tr,
                pp,
                s,
                log,
                mix_rhos_commits,
                combine_b_pows,
                latest.instances,
                &running,
            )?;
            (next_running, FoldProof::Recursive(nifs_proof))
        }
    };

    // Build next ProofState: running advances, latest is what the caller just supplied.
    let new_proof = ProofState::Active {
        running: next_running,
        latest: LatestInstance::from_instances(next_latest),
    };

    // F' steps 1, 2, 5 — advance state and compute x_out.
    let prev_state_for_advance = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest,
        public_trace,
        proof: ProofState::Initial, // placeholder; advance_state reads new_proof for the new state
    };
    let next_state = construction2::advance_state(pp, prev_state_for_advance, new_proof, fresh_count, chunk_digest);
    let x_out = construction2::compute_x_out(vk, pp, s, &next_state);

    Ok((next_state, StepProof { fold, x_out }))
}

// ──────────────────────────────────────────────────────────────────────────
// F' verify (native)
// ──────────────────────────────────────────────────────────────────────────

/// One full F' invocation on the verifier side.
///
/// Mirrors `prove`:
/// - Reads `state.proof` to decide base-vs-recursive.
/// - For recursive: replays NIFS.V over `proof.fold` against
///   `proof.folded_claims` and the running claims in `state.proof`.
/// - Advances state, recomputes x_out, asserts it matches `proof.x_out`.
pub fn verify(
    tr: &mut Transcript,
    pp: &Params,
    s: &Structure,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest_claims: &[CcsClaim],
    proof: &StepProof,
) -> Result<State, Error> {
    construction2::enforce_pc_in_range(&state)?;
    construction2::state_base_case_check(&state)?;

    let fresh_count = next_latest_claims.len() as u64;
    let chunk_digest = construction2::chunk_public_digest_from_claims(state.step_count, next_latest_claims);

    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest,
        public_trace,
        proof: prev_proof,
    } = state;

    // F' fold-step verifier — branch on (prev_proof, proof.fold).
    let next_running_claims = match (prev_proof, &proof.fold) {
        (ProofState::Initial, FoldProof::NoFold) => {
            // i = 0: no NIFS.V; running stays empty.
            Vec::new()
        }
        (ProofState::Active { running, latest }, FoldProof::Recursive(nifs_proof)) => nifs::verify(
            tr,
            pp,
            s,
            mix_rhos_commits,
            combine_b_pows,
            &latest.claims(),
            &running.claims,
            nifs_proof,
        )?,
        _ => return Err(Error::FoldProofVariantMismatch),
    };

    // Build next ProofState (verifier-side: witnesses empty).
    let next_running = RunningInstance {
        claims: next_running_claims,
        witnesses: Vec::new(),
    };
    let new_proof = ProofState::Active {
        running: next_running,
        latest: latest_from_claims_for_verifier(next_latest_claims),
    };

    // F' steps 1, 2, 5 — advance and compare.
    let prev_state_for_advance = State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        acc_digest,
        public_trace,
        proof: ProofState::Initial, // placeholder; advance reads new_proof
    };
    let next_state = construction2::advance_state(pp, prev_state_for_advance, new_proof, fresh_count, chunk_digest);
    let x_out = construction2::compute_x_out(vk, pp, s, &next_state);
    if x_out != proof.x_out {
        return Err(Error::XOutMismatch);
    }
    Ok(next_state)
}

/// Verifier-side reconstruction of `LatestInstance` — claims only, with
/// shape-only witness placeholders. Verifier-side state never reads the
/// witnesses; they're carried for type uniformity with the prover side.
fn latest_from_claims_for_verifier(claims: &[CcsClaim]) -> LatestInstance {
    LatestInstance::from_instances(
        claims
            .iter()
            .map(|c| CcsInstance {
                claim: c.clone(),
                witness: crate::paper::relations::CcsWitness {
                    w: Vec::new(),
                    Z: neo_ccs::matrix::Mat::zero(0, 0, neo_math::F::default()),
                },
            })
            .collect(),
    )
}

// PR5 will add the in-circuit gadget mirror of `prove` / `verify`. The
// gadget reads Soundness Invariant I-5 — the absorb sequence in
// `paper::digest::state_x_out_digest` — and the `nifs::verify`
// composition above; both must move in lockstep with their R1CS
// counterparts.
