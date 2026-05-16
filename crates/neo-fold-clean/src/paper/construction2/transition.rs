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
    State {
        chunk_count: prev.chunk_count + 1,
        step_count: prev.step_count + fresh_count,
        z_0: prev.z_0,
        z_i: new_z_i,
        pc: prev.pc,
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
        state.acc_digest,
        state.acc_digest, // single-step build: construction2_acc == semantic_acc
        state.public_trace,
    );
    EncInst::from_digest(bytes)
}
