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
//! 5. Discharges every terminal witness-authority obligation against
//!    each `(claim, witness Z)` in `proof.state.running`:
//!    - `commit(Z) == claim.c` (Ajtai opening),
//!    - `project_x(Z) == claim.X` (public-input projection),
//!    - `||Z||_∞ < b` (low-norm),
//!    - `claim.y_ring[j] == multilinear_eval(M_j · Z, claim.r)` for
//!      every CCS matrix (CE-relation closure),
//!    - `claim.ct[j] == constant_term(claim.y_ring[j])` (the SuperNeo
//!      scalar view of the same `M_j · Z(r)`).
//!
//! ## What this is and isn't
//!
//! `verify_uncompressed` **executes the SuperNeo verifier equations
//! directly over the folded CCS/CE circuit relation.** Rust is the
//! executor; SuperNeo is the source of soundness. A consumer that
//! runs this function gets the SuperNeo verifier's terminal-fold
//! check + the full CE-relation closure against the opened witnesses.
//!
//! It is NOT the soundness contract for a consumer that verifies a
//! *compressed* artifact (the decider R1CS + a SNARK over it). For
//! that consumer, the CE-relation obligation has to live as
//! constraint rows in the decider R1CS — that's a separate parallel
//! obligation tracked by `paper::decider_ce_relation` (reference
//! gadget). The check in step 5 of THIS verifier does not substitute
//! for the in-circuit version.
//!
//! The load-bearing soundness step in §1–4 is the Π_CCS sumcheck inside
//! step 2: at random row `α` it implies CCS satisfaction for the latest
//! and correct CE evaluation for `pre_final_running` at its
//! (prover-supplied but circuit-bound) `r` — and the latest's CCS being
//! satisfied transitively grounds the chain because the F' CCS structure
//! encodes `latest.x = hash(vk_fs, i, z_0, z_i, pre_final_running, pc)`.
//! Step 5's CE-relation check is what binds those transcript-derived
//! `y_j` values back to the *opened* witness `Z`.

use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::utils::tensor_point;
use neo_math::balanced::within_nc_bound;
use neo_math::K;
use neo_reductions::common::project_x_from_witness_mat;
use neo_reductions::superneo_eval::{SuperneoEvalCache, SuperneoRingLinearForm, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

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

    // (0a) Initial-semantic-state anchor. The decider preflight catches
    // a tampered `statement.public.initial_semantic_state_digest` via the
    // anchor cross-check in `validate_witness`, but `verify_uncompressed`
    // takes a raw `Uncompressed` proof — `state.initial_semantic_state_digest`
    // is prover-supplied and never cross-checked elsewhere in this verifier
    // path. `vk_fs_digest` absorbs the verifier-owned anchor, so a chain
    // whose actual initial differs from `prep.initial_semantic_state_digest()`
    // also breaks `XOutMismatch`; this dedicated check exists so the prover
    // sees the precise invariant they violated.
    check_initial_semantic_anchor(prep, proof)?;

    // (0b) Stateless semantic invariant — checked next so a tampered
    // `semantic_state_digest` produces a precise
    // `StatelessSemanticInvariantViolated` rather than an opaque
    // `XOutMismatch` from the terminal-fold re-run. For stateless plans
    // the F' image's CCS structure has no Poseidon2 binding rows for
    // the `semantic_state_digest` lane, so a malicious prover could
    // otherwise self-consistently inject arbitrary bytes there.
    // Stateful plans skip this — terminal Π_CCS sumcheck authenticates
    // the field inductively via the binding rows.
    check_stateless_semantic_invariant(prep, proof)?;

    // (1)–(2) Reconstruct pre-fold state from terminal_inputs and run the
    // terminal fold verifier. Three sub-cases mirror `prove_final_fold`:
    match &proof.final_fold {
        None => verify_no_terminal_fold_case(prep, proof, recorded_running)?,
        Some(final_fold) => verify_terminal_fold_case(prep, proof, recorded_running, final_fold)?,
    }

    // (5) Witness-side authority: each prover-stored witness must
    // satisfy ALL five terminal CE obligations against its claim —
    // commit / X / low-norm / `y_ring == M_j · Z(r)` / `ct ==
    // constant-term(y_ring)`. These are the SuperNeo verifier
    // equations on the folded CE relation; this Rust function
    // executes them directly. See module docs for the layering
    // boundary with the decider R1CS path.
    check_running_witnesses_authority(prep, recorded_running)?;

    // (4) acc_digest is recomputed from the just-authenticated claims.
    check_recorded_acc_digest(prep, recorded_running, &proof.state.acc_digest)?;
    Ok(())
}

fn check_initial_semantic_anchor(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    if proof.state.initial_semantic_state_digest != prep.initial_semantic_state_digest() {
        return Err(Error::InitialSemanticStateAnchorMismatch);
    }
    Ok(())
}

fn check_stateless_semantic_invariant(prep: &Preprocessing, proof: &Uncompressed) -> Result<(), Error> {
    if !matches!(
        prep.semantic_state_mode,
        crate::paper::construction2::SemanticStateMode::Stateless
    ) {
        return Ok(());
    }
    let expected = match &proof.final_fold {
        None => {
            // No terminal fold ran (Initial or empty-latest path). The
            // current acc_digest IS the pre-terminal acc_digest.
            proof.state.acc_digest
        }
        Some(final_fold) => pre_fold_acc_digest(prep, &final_fold.terminal_inputs.pre_final_running),
    };
    if proof.state.semantic_state_digest != expected {
        return Err(Error::StatelessSemanticInvariantViolated);
    }
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
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
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
        initial_semantic_state_digest: post.initial_semantic_state_digest,
        semantic_state_digest: post.semantic_state_digest,
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
        || derived.initial_semantic_state_digest != recorded.initial_semantic_state_digest
        || derived.semantic_state_digest != recorded.semantic_state_digest
        || derived.public_trace != recorded.public_trace
        || derived.acc_digest != recorded.acc_digest
    {
        return Err(Error::PostStateMismatch);
    }
    Ok(())
}

// ── Witness-side authority ────────────────────────────────────────────────

/// Step (5) of [`verify_uncompressed`]: every running witness must
/// satisfy the **SuperNeo terminal CE relation** against its claim.
///
/// Paper-level CE relation (SuperNeo Theorem 5, §5):
///
/// 1. `commit_Ajtai(Z) == claim.c`
/// 2. `project_x(Z) == claim.X`
/// 3. every entry of `Z` is low-norm: `|z| < b`
/// 4. `claim.y_ring[j] == multilinear_eval(M_j · Z, claim.r)` for every
///    CCS matrix `M_j`
///
/// Implementation-consistency obligation (the SuperNeo paper's
/// `ct(y_j) = M̄_j z(r)` identity, made checkable from cached state):
///
/// 5. `claim.ct[j] == constant_term(claim.y_ring[j])` — the lane-0
///    K-element of `y_ring[j]`. `ct` is the scalar/constant-term view
///    of `y_ring`; if `y_ring` matches `M_j · Z(r)` and `ct` is the
///    constant term of `y_ring`, then `ct == M_j z(r)` transitively.
///
/// (4) and (5) close the CE relation against the opened witness.
/// Without them, the F'-chain `acc_digest` (commitment-only) would
/// let a malformed `y_ring` or `ct` slip through the binding
/// pipeline. The Rust code below faithfully executes the SuperNeo
/// verifier equations; it does not invent a new check.
///
/// **Layering note.** This makes `verify_uncompressed` sound for any
/// consumer that runs it. It does NOT substitute for the parallel
/// obligation on a future decider R1CS / SNARK consumer; that lives
/// in `paper::decider_ce_relation` (reference gadget for the
/// in-circuit version).
///
/// Exposed `pub` via [`validate_final_witness_authority`] so isolation
/// tests can exercise the five obligations against a hand-crafted
/// `(claim, witness)` pair without driving the full
/// `verify_uncompressed` binding pipeline.
fn check_running_witnesses_authority(prep: &Preprocessing, running: &RunningInstance) -> Result<(), Error> {
    check_running_shape(running)?;

    let b = prep.params.b();
    let ell_d = ell_d_for_ce_check();
    let superneo_cache = prep.optimized_cache().superneo();
    let mut cached_forms: Option<(Vec<K>, Vec<SuperneoRingLinearForm>)> = None;
    for (index, (claim, witness)) in running.claims.iter().zip(&running.witnesses).enumerate() {
        if prep.log.commit(witness) != claim.c {
            return Err(Error::FinalAccumulatorWitnessCommitmentMismatch { index });
        }
        let projected = project_x_from_witness_mat(witness, prep.structure().m, claim.m_in)
            .map_err(|_| Error::FinalAccumulatorPublicInputMismatch { index })?;
        if projected != claim.X {
            return Err(Error::FinalAccumulatorPublicInputMismatch { index });
        }
        check_low_norm(index, witness, b)?;
        let expected_r_len = expected_row_point_len(prep);
        if claim.r.len() != expected_r_len {
            return Err(Error::FinalAccumulatorEvaluationPointShapeMismatch {
                index,
                expected: expected_r_len,
                got: claim.r.len(),
            });
        }
        let forms = ring_linear_forms_for_claim_r(prep, superneo_cache, &mut cached_forms, &claim.r);
        check_ce_relation(prep, index, claim, witness, ell_d, forms)?;
    }
    Ok(())
}

/// Public entry that runs the five-obligation witness-authority block
/// from [`check_running_witnesses_authority`] against a caller-provided
/// `RunningInstance`. Used by tests that want to isolate the CE-relation
/// obligation without first passing the chain-replay + binding steps
/// `verify_uncompressed` does up-front.
pub fn validate_final_witness_authority(prep: &Preprocessing, running: &RunningInstance) -> Result<(), Error> {
    check_running_witnesses_authority(prep, running)
}

/// `ell_d = log2(next_power_of_two(D))`, matching the prover's
/// `compute_y_from_Z_and_r` padding so the verifier's expected
/// `y_ring` lengths align with the proof's.
#[inline]
fn ell_d_for_ce_check() -> usize {
    neo_math::D.next_power_of_two().trailing_zeros() as usize
}

#[inline]
fn expected_row_point_len(prep: &Preprocessing) -> usize {
    prep.structure()
        .n
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize
}

fn ring_linear_forms_for_claim_r<'a>(
    prep: &Preprocessing,
    superneo_cache: &SuperneoEvalCache,
    cached_forms: &'a mut Option<(Vec<K>, Vec<SuperneoRingLinearForm>)>,
    r: &[K],
) -> &'a [SuperneoRingLinearForm] {
    let needs_rebuild = cached_forms
        .as_ref()
        .is_none_or(|(cached_r, _)| cached_r.as_slice() != r);
    if needs_rebuild {
        let rb = tensor_point::<K>(r);
        let n_eff = core::cmp::min(prep.structure().n, rb.len());
        *cached_forms = Some((r.to_vec(), superneo_cache.build_ring_linear_forms(&rb, n_eff)));
    }
    cached_forms
        .as_ref()
        .expect("ring-linear forms must be cached")
        .1
        .as_slice()
}

/// Verify the two CE-relation obligations (4th and 5th) against the
/// opened witness: `claim.y_ring[j] == multilinear_eval(M_j · Z,
/// claim.r)` for every CCS matrix `M_j`, then `claim.ct[j] ==
/// constant_term(claim.y_ring[j])` (recomputed as the lane-0 view of
/// the same `M_j · Z(r)`). Both make the standalone Rust verifier
/// sound. See [`check_running_witnesses_authority`] for the full step
/// (5) checklist.
fn check_ce_relation(
    prep: &Preprocessing,
    index: usize,
    claim: &CeClaim,
    witness: &WitnessMat,
    ell_d: usize,
    ring_linear_forms: &[SuperneoRingLinearForm],
) -> Result<(), Error> {
    // ── y_ring closure ────────────────────────────────────────────────
    if ring_linear_forms.len() != claim.y_ring.len() {
        return Err(Error::FinalAccumulatorCeRelationViolation {
            index,
            matrix_index: ring_linear_forms.len().min(claim.y_ring.len()),
        });
    }
    let d_pad = 1usize << ell_d;
    let z_blocks = SuperneoZBlocks::from_witness_mat(witness, prep.structure().m)
        .expect("check_ce_relation: witness shape was validated before CE closure");
    let mut expected_ct = Vec::with_capacity(ring_linear_forms.len());
    for (matrix_index, (form, recorded)) in ring_linear_forms
        .iter()
        .zip(claim.y_ring.iter())
        .enumerate()
    {
        let coeffs = form.eval_real_z_blocks(&z_blocks);
        let mut expected = coeffs.to_vec();
        if expected.len() < d_pad {
            expected.resize(d_pad, K::ZERO);
        }
        if expected.as_slice() != recorded.as_slice() {
            return Err(Error::FinalAccumulatorCeRelationViolation { index, matrix_index });
        }
        expected_ct.push(expected[0]);
    }

    // ── ct closure ────────────────────────────────────────────────────
    // `ct` is the SuperNeo scalar view of `y_ring` (the constant term
    // of each `y_ring[j]`). It enters downstream consistency checks
    // (`Σ c_S · Π ct[j]`), so leaving it unauthenticated would let a
    // prover diverge `ct` from `y_ring` without this verifier noticing.
    if expected_ct.len() != claim.ct.len() {
        return Err(Error::FinalAccumulatorCtMismatch {
            index,
            matrix_index: expected_ct.len().min(claim.ct.len()),
        });
    }
    for (matrix_index, (expected, recorded)) in expected_ct.iter().zip(claim.ct.iter()).enumerate() {
        if expected != recorded {
            return Err(Error::FinalAccumulatorCtMismatch { index, matrix_index });
        }
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
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &prep.vk,
        prep.public_input_len,
        prep.semantic_state_mode,
        prep.initial_semantic_state_digest(),
        &statement,
    )
    .map_err(Error::from)
}
