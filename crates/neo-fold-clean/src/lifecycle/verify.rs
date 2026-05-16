//! Verifier-side lifecycle: `verify_uncompressed` + `verify_uncompressed_audit`.
//!
//! `verify` (the compressed variant) lives in `compress.rs` next to
//! `compress` because they share the decider statement-builder helpers.
//!
//! ## Two verifiers, two input types
//!
//! - [`verify_uncompressed`] **(non-replay IVC verifier)** consumes the
//!   terminal-only [`Uncompressed`]. Constant verifier work in chain
//!   length. Authenticates the chain through a re-run of the terminal
//!   NIFS fold, never iterating per-step proofs.
//! - [`verify_uncompressed_audit`] **(chain-replay / audit verifier)**
//!   consumes the audit-bearing [`crate::lifecycle::UncompressedAudit`].
//!   Linear in chain length. Replays every `extend`'s NIFS.V to catch
//!   audit-trail tampers (`steps`, `public_batches`) the IVC verifier
//!   intentionally ignores.
//!
//! The Spartan decider statement is built from
//! [`crate::lifecycle::UncompressedAudit`] for the same reason: the
//! audit trail binds the public image to a verifiable history. See
//! [`crate::lifecycle::build_decider_statement`].
//!
//! ## Contract — non-replay IVC verifier (Phase 1.7)
//!
//! [`verify_uncompressed`] is the proper IVC verifier: its work is constant
//! in chain length. It authenticates the IVC chain through the **terminal
//! NIFS fold** (HyperNova §6.3 Construction 2 + SuperNeo §7), without
//! ever iterating per-step `StepProof`s. The walk over per-step F' proofs
//! lives in [`verify_uncompressed_audit`] (and the decider's
//! `validate_witness`); audit-trail tampers are caught there, not here.
//!
//! Specifically, given a finalized proof with the terminal-fold inputs
//! the prover stored in `final_fold.terminal_inputs`, the verifier:
//!
//! 1. Reconstructs the pre-final-fold `State` from
//!    `proof.state`'s chain coordinates + `terminal_inputs.pre_final_running`
//!    + `terminal_inputs.latest`.
//! 2. Calls [`construction2::verify_final_fold`], which:
//!    - runs Π_CCS / Π_RLC / Π_DEC on `(pre_final_running, latest)` with a
//!      verifier-driven transcript → derives `post_running` whose
//!      sumcheck point `r` is verifier-bound (not prover-supplied);
//!    - asserts the resulting `state_after.x_out` equals
//!      `final_fold.x_out` (chain-coordinate binding).
//! 3. Binds the verifier-derived `post_running.claims` to the prover's
//!    recorded `proof.state.running.claims`.
//! 4. Re-derives `acc_digest` from `proof.state.running.claims` and
//!    asserts it matches `proof.state.acc_digest`.
//! 5. Cross-checks the prover's `proof.state.running.witnesses` open the
//!    verifier-derived claims and are low-norm. (CE `y_j` openings are
//!    already authenticated by step 3 — the verifier-derived claim
//!    carries the verifier-derived `r` and `y_j`.)
//!
//! The load-bearing soundness step is the Π_CCS sumcheck inside step 2:
//! at random row `α` it implies CCS satisfaction for the latest and
//! correct CE evaluation for `pre_final_running` at its (prover-supplied
//! but circuit-bound) `r` — and the latest's CCS being satisfied
//! transitively grounds the chain because the F' CCS structure encodes
//! `latest.x = hash(vk_fs, i, z_0, z_i, pre_final_running, pc)`.

use neo_ccs::traits::SModuleHomomorphism;
use neo_math::balanced::within_nc_bound;
use neo_reductions::common::project_x_from_witness_mat;

use crate::lifecycle::{Error, Preprocessing, Uncompressed, UncompressedAudit};
use crate::paper::construction2::{
    self, FinalFoldProof, LatestInstance, ProofState, RunningInstance, State, TerminalFoldInputs,
};
use crate::paper::decider;
use crate::paper::digest::{accumulator_digest_from_claims, accumulator_digest_from_parent_claim};
use crate::paper::relations::{CeClaim, WitnessMat};

/// Verify an uncompressed proof in O(1) verifier work (constant in chain length).
///
/// Authority comes from re-running the **terminal NIFS fold** under a
/// verifier-driven transcript, then binding the derived post-fold state
/// to the recorded proof.state. See module docs for the full check list.
pub fn verify_uncompressed(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    let (recorded_running, recorded_latest) = require_active_state(&proof.state.proof)?;
    if !recorded_latest.instances.is_empty() {
        return Err(Error::NotFinalized);
    }
    check_running_shape(recorded_running)?;

    // (1)–(2) Reconstruct pre-fold state from terminal_inputs and run the
    // terminal fold verifier. Three sub-cases mirror `prove_final_fold`:
    match &proof.final_fold {
        None => verify_no_terminal_fold_case(prep, proof, recorded_running)?,
        Some(final_fold) => verify_terminal_fold_case(prep, proof, recorded_running, final_fold)?,
    }

    // (5) Witness-side authority: each prover-stored witness must open
    // the (now verifier-authenticated) claim, project to claim.X, and be
    // low-norm. y_j matching is already covered by step (3) above.
    check_running_witnesses_authority(prep, recorded_running)?;

    // (4) acc_digest is recomputed from the just-authenticated claims.
    check_recorded_acc_digest(prep, recorded_running, &proof.state.acc_digest)?;
    Ok(())
}

// ── State-shape gates ─────────────────────────────────────────────────────

fn require_active_state(state: &ProofState) -> Result<(&RunningInstance, &LatestInstance), Error> {
    match state {
        ProofState::Initial => Err(Error::NotFinalized),
        ProofState::Active { running, latest } => Ok((running, latest)),
    }
}

fn check_running_shape(running: &RunningInstance) -> Result<(), Error> {
    if !running.shape_ok() {
        return Err(Error::FinalAccumulatorWitnessShapeMismatch);
    }
    Ok(())
}

// ── Terminal-fold path: re-run NIFS.V and bind the result ─────────────────

fn verify_terminal_fold_case(
    prep: &Preprocessing,
    proof: &Uncompressed,
    recorded_running: &RunningInstance,
    final_fold: &FinalFoldProof,
) -> Result<(), Error> {
    let pre_state = build_pre_final_state(prep, &proof.state, &final_fold.terminal_inputs);
    let derived_state = construction2::verify_final_fold(
        &prep.params,
        &prep.structure,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        pre_state,
        Some(final_fold),
    )?;
    bind_derived_state_to_recorded(&derived_state, &proof.state)?;
    // Cross-check the derived running matches the recorded one. The
    // step above already binds chain coordinates + x_out + acc_digest;
    // this asserts the same on the claim-level data (commitments, X,
    // r, y_j).
    if derived_state
        .proof
        .running_claims_for_binding()
        .map_err(|_| Error::PostStateMismatch)?
        != recorded_running.claims
    {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

fn verify_no_terminal_fold_case(
    prep: &Preprocessing,
    proof: &Uncompressed,
    recorded_running: &RunningInstance,
) -> Result<(), Error> {
    // `prove_final_fold` returns `None` only when nothing needed flushing
    // (state was already `Initial` or `Active { running, latest: empty }`).
    // For the `Initial` half, `require_active_state` above already
    // rejected. So at this point either the prover ran zero extends + one
    // finalize (Initial → Initial, already rejected), or the input state
    // was an `Active { running, latest: empty }` produced by some
    // adversarial path — there is no terminal NIFS proof binding
    // `running` to anything verifiable. Reject.
    if !recorded_running.claims.is_empty() {
        return Err(Error::MissingTerminalFoldProof);
    }
    // Empty running with `final_fold = None` is the only case worth
    // accepting: chain produced no folds at all. acc_digest must be the
    // empty-claims digest, which the trailing acc_digest check enforces.
    let _ = (prep, proof);
    Ok(())
}

/// Construct the pre-final-fold `State` from chain coords (which are
/// unchanged by finalization) + the snapshotted pre-fold inputs.
fn build_pre_final_state(prep: &Preprocessing, post: &State, terminal: &TerminalFoldInputs) -> State {
    let pre_acc_digest = pre_fold_acc_digest(prep, &terminal.pre_final_running);
    State {
        chunk_count: post.chunk_count,
        step_count: post.step_count,
        z_0: post.z_0,
        z_i: post.z_i,
        pc: post.pc,
        acc_digest: pre_acc_digest,
        public_trace: post.public_trace,
        proof: ProofState::Active {
            running: terminal.pre_final_running.clone(),
            latest: terminal.latest.clone(),
        },
    }
}

/// `acc_digest` of a pre-finalization running. Mirrors the formula in
/// [`construction2::prove_final_fold`] / [`construction2::verify_final_fold`].
fn pre_fold_acc_digest(prep: &Preprocessing, pre_running: &RunningInstance) -> [u8; 32] {
    if pre_running.claims.is_empty() {
        accumulator_digest_from_claims(prep.params.b(), &[])
    } else if let Some(parent) = pre_running.parent_authority.as_ref() {
        accumulator_digest_from_parent_claim(pre_running.claims.len(), parent)
    } else {
        // Non-empty running without parent_authority can never have been
        // produced by an honest chain. The digest cannot be reconstructed
        // here, so we deliberately emit a value that will not match any
        // valid `proof.state.acc_digest` — the downstream binding check
        // surfaces the rejection.
        [0u8; 32]
    }
}

fn bind_derived_state_to_recorded(derived: &State, recorded: &State) -> Result<(), Error> {
    if derived.chunk_count != recorded.chunk_count
        || derived.step_count != recorded.step_count
        || derived.z_0 != recorded.z_0
        || derived.z_i != recorded.z_i
        || derived.pc != recorded.pc
        || derived.public_trace != recorded.public_trace
        || derived.acc_digest != recorded.acc_digest
    {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

// ── Witness-side authority ────────────────────────────────────────────────

fn check_running_witnesses_authority(prep: &Preprocessing, running: &RunningInstance) -> Result<(), Error> {
    let b = prep.params.b();
    for (index, (claim, witness)) in running.claims.iter().zip(&running.witnesses).enumerate() {
        if prep.log.commit(witness) != claim.c {
            return Err(Error::FinalAccumulatorWitnessCommitmentMismatch { index });
        }
        let projected = project_x_from_witness_mat(witness, prep.structure.m, claim.m_in)
            .map_err(|_| Error::FinalAccumulatorPublicInputMismatch { index })?;
        if projected != claim.X {
            return Err(Error::FinalAccumulatorPublicInputMismatch { index });
        }
        check_low_norm(index, witness, b)?;
    }
    Ok(())
}

fn check_low_norm(index: usize, witness: &WitnessMat, b: u32) -> Result<(), Error> {
    for (offset, entry) in witness.as_slice().iter().enumerate() {
        if !within_nc_bound(*entry, b) {
            let row = offset / witness.cols();
            let col = offset % witness.cols();
            return Err(Error::FinalAccumulatorLowNormViolation { index, row, col });
        }
    }
    Ok(())
}

// ── acc_digest consistency ────────────────────────────────────────────────

fn check_recorded_acc_digest(
    prep: &Preprocessing,
    running: &RunningInstance,
    recorded: &[u8; 32],
) -> Result<(), Error> {
    let recomputed = if running.claims.is_empty() {
        accumulator_digest_from_claims(prep.params.b(), &[])
    } else if let Some(parent) = running.parent_authority.as_ref() {
        accumulator_digest_from_parent_claim(running.claims.len(), parent)
    } else {
        return Err(Error::AccDigestMismatch);
    };
    if recomputed != *recorded {
        return Err(Error::AccDigestMismatch);
    }
    Ok(())
}

// ── Helper trait: avoids cloning the whole ProofState to read its claims ──

trait ProofStateBinding {
    fn running_claims_for_binding(&self) -> Result<Vec<CeClaim>, ()>;
}

impl ProofStateBinding for ProofState {
    fn running_claims_for_binding(&self) -> Result<Vec<CeClaim>, ()> {
        match self {
            ProofState::Initial => Err(()),
            ProofState::Active { running, latest } => {
                if !latest.instances.is_empty() {
                    return Err(());
                }
                Ok(running.claims.clone())
            }
        }
    }
}

// ── Chain-replay / audit verifier ─────────────────────────────────────────

/// Diagnostic / chain-replay verifier — replays every step's NIFS.V on
/// top of the terminal-fold check.
///
/// **Not the production IVC verifier.** It does the same work as
/// `paper::decider::validate_witness`, walking `audit.steps` and
/// `audit.public_batches` step by step, so its cost is **linear in
/// chain length**. Production callers want the constant-cost
/// [`verify_uncompressed`] instead.
///
/// Reach for `verify_uncompressed_audit` only when you need to detect
/// tampers on the per-step audit trail (`steps`, `public_batches`,
/// `final_fold.nifs`) that an attacker might attempt while leaving the
/// final running accumulator self-consistent. Concretely that means:
/// red-team tests for the audit trail, the Spartan compressed-decider
/// preflight, and chain-replay debugging.
pub fn verify_uncompressed_audit(prep: &Preprocessing, audit: &UncompressedAudit) -> Result<(), Error> {
    let statement = super::build_decider_statement(prep, audit);
    decider::validate_witness(
        &prep.params,
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        prep.public_input_len,
        &statement,
    )
    .map_err(Error::from)
}
