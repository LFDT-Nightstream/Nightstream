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
///   image's CCS structure carries Poseidon2 binding rows
///   (`H(state_in_vars) == semantic_state_digest_in_lane` and the
///   state_out counterpart) so terminal Π_CCS sumcheck authenticates the
///   digest against the actual app-state wires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticStateAdvance {
    Stateless,
    Stateful([u8; 32]),
}

/// Verifier-owned discriminator for whether a chain carries app state.
///
/// Lives on [`crate::lifecycle::Preprocessing`]; the frontend sets it
/// once at preprocess time based on whether its plan declares
/// `semantic_state_in/out_var_indices`. The verifier consults this bit
/// to decide whether a prover-supplied `StepProof.semantic_state_digest`
/// must equal the accumulator digest (stateless invariant — no F' image
/// binding rows would otherwise authenticate the field) or whether the
/// F' image's Poseidon2 binding rows are responsible for it (stateful).
///
/// Without this bit, a malicious prover on a stateless plan could
/// inject arbitrary self-consistent bytes into
/// `PublicImage.semantic_state_digest` because the F' image's CCS
/// structure has no binding constraint for that lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SemanticStateMode {
    /// Chain has no application state. `proof.semantic_state_digest`
    /// must equal the accumulator digest.
    #[default]
    Stateless,
    /// Chain carries application state. The F' image's CCS structure
    /// has Poseidon2 binding rows over the app-state wires; terminal
    /// Π_CCS sumcheck authenticates them.
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

/// Base case: at i = 0, z_0 must equal z_i.
pub(crate) fn state_base_case_check(state: &State) -> Result<(), Error> {
    if state.chunk_count == 0 && state.z_0 != state.z_i {
        return Err(Error::BaseCaseMismatch);
    }
    Ok(())
}

/// Advance the IVC carrier by one step: bump counters, chain `z_i` and
/// `public_trace` from the chunk's public-instance digest, and re-derive
/// `acc_digest` from the new running accumulator.
pub(crate) fn advance_state(
    pp: &Params,
    prev: State,
    new_proof: ProofState,
    fresh_count: u64,
    chunk_digest: [F; 4],
    semantic_advance: SemanticStateAdvance,
) -> State {
    let new_z_i = digest::boundary_update_digest(prev.z_i, chunk_digest);
    let new_public_trace = digest::public_trace_update_digest(prev.public_trace, chunk_digest);
    let new_acc_digest = match &new_proof {
        ProofState::Initial => digest::accumulator_digest_from_claims(pp.b(), &[]),
        ProofState::Active { running, .. } if running.claims.is_empty() => {
            digest::accumulator_digest_from_claims(pp.b(), &[])
        }
        ProofState::Active { running, .. } => {
            let parent = running
                .parent_authority
                .as_ref()
                .expect("non-empty running accumulator must carry its Pi_RLC parent authority");
            digest::accumulator_digest_from_parent_claim(running.claims.len(), parent)
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
pub(crate) fn compute_x_out(vk: &VerifierKey, _pp: &Params, structure_digest: &[F; 4], state: &State) -> EncInst {
    let bytes = digest::state_x_out_digest(
        vk.digest(),
        structure_digest,
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
    );
    EncInst::from_digest(bytes)
}
