//! F' — the augmented function from Hypernova §6.3 Construction 2.
//!
//! ```text
//! F'_j(vk_fs, U_i, u_i, pc_i, (i, z_0, z_i), ω_i, π) → x:
//!   1. pc_{i+1} = φ(z_i, ω_i)
//!   2. z_{i+1}  = F_j(z_i, ω_i)
//!   3. base case (i = 0):  no NIFS.P; running = canonical default
//!   4. recursive case:     NIFS.V(vk_fs[pc_i], U_i, u_i, π) → U_{i+1}
//!   5. x = state_x_out_digest(vk_fs, i+1, z_{i+1},
//!      semantic_state_digest_{i+1}, U_{i+1})
//! ```
//!
//! For ccs-direct (ℓ = 1):
//! - `pc` is constant `TRIVIAL_PC` (φ trivially returns 1).
//!   It is checked as state and absorbed into `state_x_out`, matching
//!   HyperNova's recursive-link preimage even though the selector is
//!   currently constant.
//! - `z_0` is verifier-derived, pinned at the base, and linked as state.
//!   It is not absorbed directly by `state_x_out`: it is
//!   `initial_boundary_digest(structure_digest, public_input_len)`, and both
//!   inputs are already absorbed into `vk_fs_digest`.
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
use neo_math::F;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::{
    self, FoldProof, LatestInstance, ProofState, RunningInstance, SemanticStateAdvance, SemanticStateMode, State,
    StepProof, VerifierKey,
};
use crate::paper::digest::digest32_as_fields;
use crate::paper::nifs;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, DecMixer, RlcMixer, Structure};

pub use construction2::Error;

/// Canonical transcript label for one F' step.
///
/// Used by `paper::f_prime::{prove, verify}` (native) and must match the
/// `cfg.transcript_label` of [`crate::paper::f_prime::r1cs::FPrimeStepConfig`]
/// when the in-circuit F' R1CS verifies the same step. Both sides initialize
/// their transcript with this label and absorb the state-bound F'-step
/// context below before NIFS.V; if either diverges, Fiat–Shamir challenges
/// disagree at the first absorb and the F' R1CS rejects.
pub const F_PRIME_STEP_TRANSCRIPT_LABEL: &[u8] = b"neo.fold.clean/f_prime/step/v1";

/// Absorb the F'-step context into a transcript.
///
/// Order is fixed and matches `enforce_f_prime_recursive_step_circuit` in
/// `paper::f_prime::r1cs`; do not reorder without updating the in-circuit
/// transcript prefix as well. `structure_digest` is the caller's cached
/// `paper::digest::structure_digest(&prep.structure)` value.
fn absorb_f_prime_step_context(
    tr: &mut Transcript,
    vk: &VerifierKey,
    structure_digest: &[F; 4],
    state: &State,
    chunk_digest: [F; 4],
) {
    tr.append_fields(b"f_prime/vk_fs", &digest32_as_fields(vk.digest()));
    tr.append_fields(b"f_prime/structure", structure_digest);
    tr.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count)]);
    tr.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count)]);
    tr.append_fields(b"f_prime/z_0", &digest32_as_fields(state.z_0));
    tr.append_fields(b"f_prime/z_i_in", &digest32_as_fields(state.z_i));
    tr.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    tr.append_fields(
        b"f_prime/semantic_state_in",
        &digest32_as_fields(state.semantic_state_digest),
    );
    tr.append_fields(b"f_prime/acc_digest_in", &digest32_as_fields(state.acc_digest));
    tr.append_fields(b"f_prime/public_trace_in", &digest32_as_fields(state.public_trace));
    tr.append_fields(b"f_prime/chunk_digest", &chunk_digest);
}

/// Build a fresh per-step F' transcript, initialized with
/// [`F_PRIME_STEP_TRANSCRIPT_LABEL`] and the F'-step context absorbs.
///
/// `state` is the state **input to this step** (i.e. before `advance_state`
/// runs), so its `z_i`, `public_trace`, etc. match the F' R1CS's `state-in`
/// fields. `chunk_digest` is computed from `next_latest`, the new batch
/// being deposited as `latest` (not the `latest` currently being folded).
/// `structure_digest` is the caller's cached
/// `paper::digest::structure_digest(&prep.structure)`.
pub fn f_prime_step_transcript(
    vk: &VerifierKey,
    structure_digest: &[F; 4],
    state: &State,
    chunk_digest: [F; 4],
) -> Transcript {
    let mut tr = Transcript::with_label(F_PRIME_STEP_TRANSCRIPT_LABEL);
    absorb_f_prime_step_context(&mut tr, vk, structure_digest, state, chunk_digest);
    tr
}

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
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
) -> Result<(State, StepProof), Error> {
    prove_with_semantic_state(
        pp,
        s,
        cache,
        structure_digest,
        log,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        next_latest,
        SemanticStateAdvance::Stateless,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prove_with_semantic_state(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest: Vec<CcsInstance>,
    semantic_advance: SemanticStateAdvance,
) -> Result<(State, StepProof), Error> {
    construction2::enforce_pc_in_range(&state)?;
    construction2::state_base_case_check(&state)?;
    if next_latest.is_empty() {
        return Err(Error::EmptyStep);
    }

    let fresh_count = next_latest.len() as u64;
    let chunk_digest = construction2::f_prime_chunk_public_digest_for_step(state.step_count, &next_latest);

    // Destructure proof out of state up front so the rest can move the
    // remaining fields freely.
    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
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
            // Fresh per-step F' transcript: init label + state-in context
            // absorbs that the in-circuit F' R1CS replays bit-for-bit.
            let state_in = State {
                chunk_count,
                step_count,
                z_0,
                z_i,
                pc,
                initial_semantic_state_digest,
                semantic_state_digest,
                acc_digest,
                public_trace,
                proof: ProofState::Initial,
            };
            let mut tr = f_prime_step_transcript(vk, structure_digest, &state_in, chunk_digest);
            let (next_running, nifs_proof) = nifs::prove(
                &mut tr,
                pp,
                s,
                cache,
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
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: ProofState::Initial, // placeholder; advance_state reads new_proof for the new state
    };
    let next_state = construction2::advance_state(
        pp,
        prev_state_for_advance,
        new_proof,
        fresh_count,
        chunk_digest,
        semantic_advance,
    );
    let semantic_mode = match semantic_advance {
        SemanticStateAdvance::Stateless => SemanticStateMode::Stateless,
        SemanticStateAdvance::Stateful(_) => SemanticStateMode::Stateful,
    };
    let x_out = construction2::compute_x_out(vk, pp, structure_digest, &next_state, semantic_mode);

    Ok((
        next_state.clone(),
        StepProof {
            fold,
            semantic_state_digest: next_state.semantic_state_digest,
            x_out,
        },
    ))
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
///
/// `semantic_mode` is verifier-owned (set on `Preprocessing` at
/// preprocess time, derived from the frontend plan). For
/// [`SemanticStateMode::Stateless`] chains the F' image's CCS
/// structure has no Poseidon2 binding rows for the semantic lane, so
/// `proof.semantic_state_digest` is **not** authenticated by the
/// in-circuit verifier; this function instead enforces the protocol
/// invariant `proof.semantic_state_digest == new_acc_digest` directly
/// and returns [`Error::StatelessSemanticInvariantViolated`] on
/// mismatch. For [`SemanticStateMode::Stateful`] chains the binding
/// rows are part of the structure, so the digest is authenticated
/// inductively by terminal Π_CCS sumcheck and this function trusts
/// `proof.semantic_state_digest` as the new chain coordinate.
pub fn verify(
    pp: &Params,
    s: &Structure,
    cache: &OptimizedStructureCache,
    structure_digest: &[F; 4],
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    state: State,
    next_latest_claims: &[CcsClaim],
    proof: &StepProof,
    semantic_mode: SemanticStateMode,
) -> Result<State, Error> {
    construction2::enforce_pc_in_range(&state)?;
    construction2::state_base_case_check(&state)?;
    if next_latest_claims.is_empty() {
        return Err(Error::EmptyStep);
    }

    let fresh_count = next_latest_claims.len() as u64;
    let chunk_digest = construction2::f_prime_chunk_public_digest_from_claims(state.step_count, next_latest_claims);

    let State {
        chunk_count,
        step_count,
        z_0,
        z_i,
        pc,
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: prev_proof,
    } = state;

    // F' fold-step verifier — branch on (prev_proof, proof.fold).
    let next_running = match (prev_proof, &proof.fold) {
        (ProofState::Initial, FoldProof::NoFold) => {
            // i = 0: no NIFS.V; running stays empty.
            RunningInstance::default()
        }
        (ProofState::Active { running, latest }, FoldProof::Recursive(nifs_proof)) => {
            // Same fresh per-step F' transcript the prover used.
            let state_in = State {
                chunk_count,
                step_count,
                z_0,
                z_i,
                pc,
                initial_semantic_state_digest,
                semantic_state_digest,
                acc_digest,
                public_trace,
                proof: ProofState::Initial,
            };
            let mut tr = f_prime_step_transcript(vk, structure_digest, &state_in, chunk_digest);
            nifs::verify(
                &mut tr,
                pp,
                s,
                cache,
                mix_rhos_commits,
                combine_b_pows,
                &latest.claims(),
                &running,
                nifs_proof,
            )?
        }
        _ => return Err(Error::FoldProofVariantMismatch),
    };

    // Build next ProofState (verifier-side: witnesses empty).
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
        initial_semantic_state_digest,
        semantic_state_digest,
        acc_digest,
        public_trace,
        proof: ProofState::Initial, // placeholder; advance reads new_proof
    };
    let semantic_advance = match semantic_mode {
        // Stateless plans have no F' image binding rows for the semantic
        // lane. The verifier therefore drives `advance_state` with
        // `Stateless` (which sets `semantic_state_digest = new_acc_digest`)
        // and then explicitly cross-checks the prover's claim against the
        // resulting deterministic value. A mismatch is surfaced with a
        // dedicated error instead of being implicitly caught by the x_out
        // chain check below — the prover should see exactly which
        // invariant they violated.
        SemanticStateMode::Stateless => SemanticStateAdvance::Stateless,
        SemanticStateMode::Stateful => SemanticStateAdvance::Stateful(proof.semantic_state_digest),
    };
    let next_state = construction2::advance_state(
        pp,
        prev_state_for_advance,
        new_proof,
        fresh_count,
        chunk_digest,
        semantic_advance,
    );
    if matches!(semantic_mode, SemanticStateMode::Stateless)
        && next_state.semantic_state_digest != proof.semantic_state_digest
    {
        return Err(Error::StatelessSemanticInvariantViolated);
    }
    let x_out = construction2::compute_x_out(vk, pp, structure_digest, &next_state, semantic_mode);
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
