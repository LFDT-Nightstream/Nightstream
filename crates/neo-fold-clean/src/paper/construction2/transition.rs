//! Construction 2 state transition primitives.
//!
//! These three helpers together advance one IVC step's state and compute
//! its hash-chain output. Used by `paper::f_prime::{prove, verify}`.

use neo_math::F;

use crate::paper::construction2::nebula_lane::NebulaLane;
use crate::paper::construction2::proof_state::ProofState;
use crate::paper::construction2::running::RunningInstance;
use crate::paper::construction2::state::State;
use crate::paper::construction2::verifier_key::VerifierKey;
use crate::paper::construction2::{enc_inst::EncInst, Error, TRIVIAL_PC};
use crate::paper::digest;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, CcsInstance, Structure};

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

/// Crate-private observation seam for the operations performed while the
/// native verifier advances and hashes one state.
///
/// The ordinary path uses the zero-sized no-op implementation. The audited
/// verifier implements this trait with an owned public-data receipt.
pub(crate) trait VerifyTransitionRecorder {
    fn running_digest(&mut self, _running: &RunningInstance, _relation_columns: usize, _output: [u8; 32]) {}

    fn state_advanced(&mut self, _output: &State) {}

    fn verifier_digest_read(&mut self, _output: [u8; 32]) {}

    fn pi_ccs_header_read(&mut self, _output: [F; 4]) {}

    fn nebula_digest(&mut self, _lane: &NebulaLane, _output: [F; 4]) {}

    fn state_x_out_hash(&mut self, _preimage: &[F], _output_digest: [u8; 32], _output: &EncInst) {}
}

pub(crate) struct NoopVerifyTransitionRecorder;

impl VerifyTransitionRecorder for NoopVerifyTransitionRecorder {}

/// A canonical accumulator digest that cannot be constructed from a
/// caller-supplied byte string.
struct CanonicalAccumulatorDigest([u8; 32]);

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
pub(crate) fn advance_state_recorded<R: VerifyTransitionRecorder>(
    prev: State,
    new_proof: ProofState,
    structure: &Structure,
    fresh_count: u64,
    chunk_digest: [F; 4],
    semantic_advance: SemanticStateAdvance,
    nebula_next: Option<crate::paper::construction2::NebulaLane>,
    recorder: &mut R,
) -> Result<State, Error> {
    let (chunk_count, step_count) = checked_advanced_counts(&prev, fresh_count)?;
    let canonical = canonical_accumulator_digest(&new_proof, structure)?;
    if let ProofState::Active { running, .. } = &new_proof {
        let running = running
            .as_materialized()
            .ok_or(Error::AccumulatorDigestOverrideMismatch)?;
        recorder.running_digest(running, structure.m, canonical.0);
    }
    let state = finish_state_advance(
        prev,
        new_proof,
        chunk_count,
        step_count,
        chunk_digest,
        semantic_advance,
        canonical.0,
        nebula_next,
    );
    recorder.state_advanced(&state);
    Ok(state)
}

pub(crate) fn advance_state_with_acc_digest(
    _pp: &Params,
    structure: &Structure,
    prev: State,
    new_proof: ProofState,
    fresh_count: u64,
    chunk_digest: [F; 4],
    semantic_advance: SemanticStateAdvance,
    acc_digest_override: Option<[u8; 32]>,
    nebula_next: Option<crate::paper::construction2::NebulaLane>,
) -> Result<State, Error> {
    let (chunk_count, step_count) = checked_advanced_counts(&prev, fresh_count)?;
    let canonical = canonical_accumulator_digest_optional(&new_proof, structure)?;
    let new_acc_digest = match (canonical.map(|digest| digest.0), acc_digest_override) {
        (Some(canonical), Some(supplied)) if canonical != supplied => {
            return Err(Error::AccumulatorDigestOverrideMismatch);
        }
        (Some(canonical), _) => canonical,
        (None, Some(supplied)) => supplied,
        (None, None) => {
            return Err(Error::AccumulatorDigestOverrideMismatch);
        }
    };
    Ok(finish_state_advance(
        prev,
        new_proof,
        chunk_count,
        step_count,
        chunk_digest,
        semantic_advance,
        new_acc_digest,
        nebula_next,
    ))
}

fn checked_advanced_counts(prev: &State, fresh_count: u64) -> Result<(u64, u64), Error> {
    let chunk_count = prev
        .chunk_count
        .checked_add(1)
        .ok_or(Error::CounterOverflow { counter: "chunk_count" })?;
    let step_count = prev
        .step_count
        .checked_add(fresh_count)
        .ok_or(Error::CounterOverflow { counter: "step_count" })?;
    Ok((chunk_count, step_count))
}

fn canonical_accumulator_digest(
    new_proof: &ProofState,
    structure: &Structure,
) -> Result<CanonicalAccumulatorDigest, Error> {
    canonical_accumulator_digest_optional(new_proof, structure)?.ok_or(Error::AccumulatorDigestOverrideMismatch)
}

fn canonical_accumulator_digest_optional(
    new_proof: &ProofState,
    structure: &Structure,
) -> Result<Option<CanonicalAccumulatorDigest>, Error> {
    match new_proof {
        ProofState::Initial => Ok(Some(CanonicalAccumulatorDigest(
            digest::AccumulatorHandle::empty().digest(),
        ))),
        ProofState::Active { running, .. } => match running.as_materialized() {
            Some(running) => Ok(Some(CanonicalAccumulatorDigest(running.accumulator_digest(structure)?))),
            None if crate::paper::construction2::running::uses_pending_accumulator_family(structure) => {
                Err(Error::DeferredPendingAccumulatorUnsupported)
            }
            None => Ok(None),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_state_advance(
    prev: State,
    new_proof: ProofState,
    chunk_count: u64,
    step_count: u64,
    chunk_digest: [F; 4],
    semantic_advance: SemanticStateAdvance,
    new_acc_digest: [u8; 32],
    nebula_next: Option<crate::paper::construction2::NebulaLane>,
) -> State {
    let new_z_i = digest::digest_fields_as_digest32(chunk_digest);
    // `public_trace` has the same domain-separation role as `z_i` in
    // this direct-CCS build. Keep the state field for existing public
    // image shape, but do not spend a second Poseidon2 chain on it.
    let new_public_trace = new_z_i;
    let new_semantic_state_digest = match semantic_advance {
        SemanticStateAdvance::Stateless => new_acc_digest,
        SemanticStateAdvance::Stateful(digest) => digest,
    };
    State {
        chunk_count,
        step_count,
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
    let mut recorder = NoopVerifyTransitionRecorder;
    compute_x_out_recorded(vk, structure_digest, state, semantic_mode, &mut recorder)
}

pub(crate) fn compute_x_out_recorded<R: VerifyTransitionRecorder>(
    vk: &VerifierKey,
    _structure_digest: &[F; 4],
    state: &State,
    semantic_mode: SemanticStateMode,
    recorder: &mut R,
) -> EncInst {
    let mode = match semantic_mode {
        SemanticStateMode::Stateless => digest::StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => digest::StateXOutDigestMode::Stateful,
    };
    let verifier_digest = vk.digest();
    recorder.verifier_digest_read(verifier_digest);
    let pi_ccs_header = vk.pi_ccs_header_bundle();
    recorder.pi_ccs_header_read(pi_ccs_header);
    let nebula_digest = state.nebula.as_ref().map(|lane| {
        let output = lane.digest();
        recorder.nebula_digest(lane, output);
        output
    });
    let preimage = digest::state_x_out_preimage_with_mode(
        mode,
        verifier_digest,
        pi_ccs_header,
        state.chunk_count,
        state.step_count,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        nebula_digest,
    );
    let output_digest = digest::state_x_out_digest_from_preimage(&preimage);
    let output = EncInst::from_digest(output_digest);
    recorder.state_x_out_hash(&preimage, output_digest, &output);
    output
}
