//! Construction 2 state transition primitives.
//!
//! These three helpers together advance one IVC step's state and compute
//! its hash-chain output. Used by `paper::f_prime::{prove, verify}`.

use neo_math::F;

use crate::paper::construction2::proof_state::ProofState;
use crate::paper::construction2::state::State;
use crate::paper::construction2::verifier_key::VerifierKey;
use crate::paper::construction2::{enc_inst::EncInst, Error, TRIVIAL_PC};
use crate::paper::digest;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance};

/// How `advance_state` should derive the next `semantic_state_digest`.
///
/// Modelled as an explicit enum rather than `Option<[u8; 32]>` so a
/// caller must consciously declare the chain's mode — no silent fallback
/// hides a forgotten `Some(..)` argument.
///
/// - **Stateless**: the chain carries no application state. The advanced
///   `semantic_state_digest` is set to the new accumulator digest, which
///   keeps the `semantic_acc == construction2_acc` x_out layout the
///   pre-stateful build relied on. Verifier-side, `f_prime::verify`
///   enforces `proof.semantic_state_digest == new_acc_digest` so a
///   malicious prover cannot inject arbitrary self-consistent bytes
///   into the `PublicImage.semantic_state_digest` field.
/// - **Stateful**: the chain carries app state. The advanced
///   `semantic_state_digest` is the caller-supplied digest. The F'
///   image's CCS structure must authenticate that digest. There are two
///   supported shapes today:
///   explicit transition state, where the structure binds
///   `H(state_in_vars)` / `H(state_out_vars)` to the semantic lanes, and
///   output/public-only state, where the structure binds the app-public
///   output bits into the semantic digest. In both cases terminal Π_CCS
///   sumcheck authenticates the digest against F' wires; `Stateful` does
///   not by itself imply a linked `state_in -> state_out` transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticStateAdvance {
    Stateless,
    Stateful([u8; 32]),
}

/// Verifier-owned, structure-derived discriminator for whether a chain
/// carries app state.
///
/// Lives on [`crate::lifecycle::Preprocessing::semantic_state_mode`] —
/// a `pub(crate)` field with no public setter. In-crate stateful
/// frontends (today: R1CS-F') flip this to `Stateful` at preprocess
/// time when their plan declares either explicit semantic-state indices
/// or app-public-output binding. In both cases the resulting F' image's
/// CCS structure must carry Poseidon2 binding rows over the wires that
/// define the semantic digest. External callers cannot construct a
/// `Stateful` preprocessing: the only public path
/// (`lifecycle::preprocess` + `lifecycle::preprocess_with_test_log`)
/// always returns `Stateless`. This ownership boundary is what makes
/// `Stateful` a real verifier contract — without it, a caller could
/// flip the bit, `verify_uncompressed` would skip the stateless
/// invariant, and the prover would be free to inject arbitrary
/// `semantic_state_digest` bytes that no constraint authenticates.
///
/// Verifier consequences:
/// - `Stateless`: `verify_uncompressed` / `verify_step` enforce
///   `proof.semantic_state_digest == accumulator digest carried
///   through finalization`. A mismatch is surfaced as
///   [`Error::StatelessSemanticInvariantViolated`].
/// - `Stateful`: the verifier treats `proof.semantic_state_digest` as
///   the new chain coordinate; authenticity is established by the
///   terminal NIFS.V re-run inside `verify_uncompressed`, which runs
///   Π_CCS sumcheck on the last F' image. That image's CCS structure
///   carries the binding rows (by virtue of how the frontend builds
///   it), and the IVC inductive argument propagates the binding to
///   prior steps through each step's in-circuit NIFS.V verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SemanticStateMode {
    #[default]
    Stateless,
    Stateful,
}

/// 1 ≤ pc_i ≤ ℓ. ℓ=1 in this build.
pub(crate) fn enforce_pc_in_range(state: &State) -> Result<(), Error> {
    if state.pc == TRIVIAL_PC {
        Ok(())
    } else {
        Err(Error::PcOutOfRange)
    }
}

/// Branch coherence for HyperNova Construction 2.
///
/// The `NoFold`/Initial branch is valid only for the true initial state.
/// Once either counter has advanced, the verifier must take the Active
/// recursive branch so NIFS.V replays the running/latest fold.
pub(crate) fn state_base_case_check(state: &State) -> Result<(), Error> {
    match &state.proof {
        ProofState::Initial => {
            if state.chunk_count != 0 || state.step_count != 0 || state.z_0 != state.z_i {
                return Err(Error::BaseCaseMismatch);
            }
        }
        ProofState::Active { .. } => {
            if state.chunk_count == 0 || state.step_count == 0 {
                return Err(Error::BaseCaseMismatch);
            }
        }
    }
    Ok(())
}

/// Advance the IVC carrier by one step: bump counters, carry the
/// chunk's public-instance digest as `z_i`, mirror `public_trace` to
/// `z_i`, and re-derive `acc_digest` from the new running accumulator.
pub(crate) fn advance_state(
    _pp: &Params,
    prev: State,
    new_proof: ProofState,
    fresh_count: u64,
    chunk_digest: [F; 4],
    semantic_advance: SemanticStateAdvance,
    nebula_next: Option<crate::paper::construction2::NebulaLane>,
) -> State {
    let new_z_i = digest::digest_fields_as_digest32(chunk_digest);
    // `public_trace` has the same domain-separation role as `z_i` in
    // this direct-CCS build. Keep the state field for existing public
    // image shape, but do not spend a second Poseidon2 chain on it.
    let new_public_trace = new_z_i;
    let new_acc_digest = match &new_proof {
        ProofState::Initial => digest::AccumulatorHandle::empty().digest(),
        ProofState::Active { running, .. } if running.claims.is_empty() => digest::AccumulatorHandle::empty().digest(),
        ProofState::Active { running, .. } => {
            let parent = running
                .parent_authority
                .as_ref()
                .expect("non-empty running accumulator must carry its Pi_RLC parent authority");
            digest::AccumulatorHandle::from_running_parts(&running.claims, Some(parent)).digest()
        }
    };
    let new_semantic_state_digest = match semantic_advance {
        SemanticStateAdvance::Stateless => new_acc_digest,
        SemanticStateAdvance::Stateful(digest) => digest,
    };
    State {
        chunk_count: prev.chunk_count + 1,
        step_count: prev.step_count + fresh_count,
        z_0: prev.z_0,
        z_i: new_z_i,
        pc: prev.pc,
        initial_semantic_state_digest: prev.initial_semantic_state_digest,
        semantic_state_digest: new_semantic_state_digest,
        acc_digest: new_acc_digest,
        public_trace: new_public_trace,
        proof: new_proof,
        // A Nebula step installs the advanced lane (spec §6.3); plain
        // steps carry the previous lane coordinate unchanged.
        nebula: nebula_next.or(prev.nebula),
    }
}

/// F'-step chunk digest from `&[CcsInstance]`. Uses
/// [`digest::f_prime_chunk_public_digest`] (which excludes `claim.x`)
/// so the chain advance is independent of the fresh public input — see
/// that function's doc for the recursive-link fixed-point rationale.
pub(crate) fn f_prime_chunk_public_digest_for_step(start_index: u64, fresh: &[CcsInstance]) -> [F; 4] {
    let claims: Vec<_> = fresh.iter().map(|i| i.claim.clone()).collect();
    digest::f_prime_chunk_public_digest(start_index, &claims)
}

/// Verifier-side variant of [`f_prime_chunk_public_digest_for_step`] that
/// consumes claims directly (witnesses are prover-only).
pub(crate) fn f_prime_chunk_public_digest_from_claims(start_index: u64, fresh_claims: &[CcsClaim]) -> [F; 4] {
    digest::f_prime_chunk_public_digest(start_index, fresh_claims)
}

/// `x_{i+1}` — Construction-2 hash-chain output (Soundness Invariant I-5).
///
/// `structure_digest` is the caller's cached
/// `paper::digest::structure_digest(&prep.structure)`; passing it in
/// avoids walking the structure on every step.
pub(crate) fn compute_x_out(
    vk: &VerifierKey,
    _pp: &Params,
    structure_digest: &[F; 4],
    state: &State,
    semantic_mode: SemanticStateMode,
) -> EncInst {
    let mode = match semantic_mode {
        SemanticStateMode::Stateless => digest::StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => digest::StateXOutDigestMode::Stateful,
    };
    let bytes = digest::state_x_out_digest_with_mode(
        mode,
        vk.digest(),
        vk.pi_ccs_header_bundle(),
        structure_digest,
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        state.nebula.as_ref().map(|lane| lane.digest()),
    );
    EncInst::from_digest(bytes)
}
