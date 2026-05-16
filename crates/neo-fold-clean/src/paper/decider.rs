//! Spartan terminal-compression contract.
//!
//! Owns the *statement* the SNARK proves and a non-SNARK
//! `validate_witness` that runs the **chain-replay** authority path —
//! walking every per-step NIFS.V plus the terminal fold. This is a
//! superset of `lifecycle::verify_uncompressed` (non-replay IVC, which
//! authenticates only the terminal fold) and is the gatekeeper the
//! Spartan SNARK must reproduce in zero knowledge. Spartan itself is
//! wired in a later PR (`prove` / `verify` are `Unsupported`
//! placeholders until then).
//!
//! ## Authority boundary
//!
//! - **Public**: `PublicImage` — the chain-binding coordinates the
//!   verifier recomputes from preprocessing (vk_fs_digest, counters, z_0,
//!   z_i, pc, acc_digest, public_trace, x_out).
//! - **Witness**: the prover-side material the verifier walks to derive
//!   the same public image — step proofs, public batches, the terminal
//!   fold proof, and the post-finalization state (with its final running
//!   accumulator and witness matrices).
//!
//! `validate_witness` ties the two together: replay every step + the
//! final fold, recompute the public image from the resulting verifier
//! state, and assert it matches `statement.public`. It also checks that
//! the final running's witness matrices commit to the claims' commitments,
//! so a prover that supplies a public image disconnected from any witness
//! is rejected before Spartan is even invoked.

use neo_ajtai::AjtaiSModule;
use neo_ccs::traits::SModuleHomomorphism;
use thiserror::Error;

use crate::paper::construction2::{
    self, EncInst, FinalFoldProof, ProofState, RunningInstance, State, StepProof, VerifierKey,
};
use crate::paper::digest::{
    accumulator_digest_from_claims, initial_boundary_digest, public_trace_seed_digest, structure_digest,
};
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, DecMixer, RlcMixer, Structure};

#[derive(Debug, Error)]
pub enum Error {
    #[error("decider: Spartan terminal compression is not implemented yet")]
    Unsupported,
    #[error("decider: validation walk failed: {0}")]
    WalkFailed(String),
    #[error("decider: public image derived from witness ≠ statement.public")]
    PublicImageMismatch,
    #[error("decider: witness shape mismatch (proof.state must be Active with empty latest after finalization)")]
    WitnessShape,
    #[error("decider: final running claims/witnesses length mismatch")]
    WitnessLengthMismatch,
    #[error("decider: final running claim {index} witness commitment ≠ claim.c")]
    WitnessCommitmentMismatch { index: usize },
    #[error("decider: public_batches / steps length mismatch (got {batches} batches, {steps} steps)")]
    StepsBatchesLengthMismatch { batches: usize, steps: usize },
}

/// Public coordinates the Spartan SNARK binds — same fields the verifier
/// recomputes from preprocessing. Paper-named; matches the absorb order
/// of [`paper::construction2::compute_x_out`].
///
/// **Content authority**: under direct-CCS with
/// `f_prime_chunk_public_digest` dropping `claim.x` and `claim.c.data`,
/// `z_i` and `public_trace` are step/shape digests only — they bind step
/// indices and protocol shape, not chunk contents. `acc_digest` is the
/// content-binding public coordinate, independently recomputed by
/// [`validate_witness`] from the NIFS.V chain over `Witness.public_batches`
/// + the final fold.
///
/// [`paper::construction2::compute_x_out`]: crate::paper::construction2
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicImage {
    pub vk_fs_digest: [u8; 32],
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub acc_digest: [u8; 32],
    pub public_trace: [u8; 32],
    pub x_out: EncInst,
}

/// Prover-side witness for one compressed proof. Drives
/// [`validate_witness`]'s authority walk; the resulting verifier state's
/// public image must equal `Statement.public`.
#[derive(Clone, Debug)]
pub struct Witness {
    /// Per-step F' proofs, one per `extend` call.
    pub steps: Vec<StepProof>,
    /// Public claims of each step's deposited batch. Length must match `steps`.
    pub public_batches: Vec<Vec<CcsClaim>>,
    /// Final NIFS proof that flushes the trailing latest. `None` only if
    /// finalization had nothing to flush (Initial / empty latest).
    pub final_fold: Option<FinalFoldProof>,
    /// Post-finalization state. Carries the final running accumulator's
    /// claims and witness matrices; `validate_witness` requires
    /// `proof = Active { running, latest: empty }`.
    pub final_state: State,
}

/// What the Spartan SNARK proves. Bundles the public coordinates and the
/// prover-side witness. The verifier-side `compress::verify` consumes only
/// `public` (the witness stays prover-private).
#[derive(Clone, Debug)]
pub struct Statement {
    pub public: PublicImage,
    pub witness: Witness,
}

/// Non-SNARK preflight for one compressed-proof statement.
///
/// Replays every step (`construction2::verify_step`) and the terminal
/// fold (`construction2::verify_final_fold`) starting from
/// preprocessing-derived base state, derives the resulting
/// [`PublicImage`], and asserts it equals `statement.public`. Also
/// verifies the final running's witness matrices commit to the claims'
/// commitments under `log`.
///
/// This is the contract the Spartan SNARK must reproduce in zero
/// knowledge. Building Spartan around an underspecified statement risks
/// soundness gaps that `validate_witness` cannot catch later, so this
/// non-SNARK check is the gatekeeper.
pub fn validate_witness(
    params: &Params,
    structure: &Structure,
    log: &AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    vk: &VerifierKey,
    public_input_len: Option<usize>,
    statement: &Statement,
) -> Result<(), Error> {
    let Witness {
        steps,
        public_batches,
        final_fold,
        final_state,
    } = &statement.witness;

    if steps.len() != public_batches.len() {
        return Err(Error::StepsBatchesLengthMismatch {
            batches: public_batches.len(),
            steps: steps.len(),
        });
    }

    // Rebuild verifier state from preprocessing.
    let structure_digest_v = structure_digest(structure);
    let z_0 = initial_boundary_digest(&structure_digest_v, public_input_len);
    let public_trace = public_trace_seed_digest(&structure_digest_v);
    let acc_digest = accumulator_digest_from_claims(params.b(), &[]);
    let mut state = State::base(z_0, public_trace, acc_digest);

    // Walk each step through F'.verify.
    for (public_batch, step_proof) in public_batches.iter().zip(steps) {
        state = construction2::verify_step(
            params,
            structure,
            mix_rhos_commits,
            combine_b_pows,
            vk,
            state,
            public_batch,
            step_proof,
        )
        .map_err(|e| Error::WalkFailed(format!("step: {e}")))?;
    }

    // Flush trailing latest through the terminal fold.
    state = construction2::verify_final_fold(
        params,
        structure,
        mix_rhos_commits,
        combine_b_pows,
        vk,
        state,
        final_fold.as_ref(),
    )
    .map_err(|e| Error::WalkFailed(format!("final_fold: {e}")))?;

    // Derive the public image from the walked state and compare to the
    // statement's declared public.
    let x_out = construction2::compute_x_out(vk, params, structure, &state);
    let derived = PublicImage {
        vk_fs_digest: vk.digest(),
        chunk_count: state.chunk_count,
        step_count: state.step_count,
        z_0: state.z_0,
        z_i: state.z_i,
        pc: state.pc,
        acc_digest: state.acc_digest,
        public_trace: state.public_trace,
        x_out,
    };
    if derived != statement.public {
        return Err(Error::PublicImageMismatch);
    }

    // Bind `statement.witness.final_state` to the walked state. Without
    // this, a prover could supply self-consistent witness matrices for an
    // unrelated running accumulator that happens to share the public image
    // — the witness openings would type-check but they would not open
    // the *walked* commitments. Three checks pin the binding:
    //
    //   (a) Canonical fields equal: chunk/step counters, z_0, z_i, pc,
    //       acc_digest, public_trace.
    //   (b) `final_state.proof` is Active with empty `latest` (same shape
    //       as `verify_final_fold`'s output).
    //   (c) `final_state.proof.running.claims == walked.proof.running.claims`
    //       — only then do the witness openings authenticate the walked
    //       accumulator.
    if final_state.chunk_count != state.chunk_count
        || final_state.step_count != state.step_count
        || final_state.z_0 != state.z_0
        || final_state.z_i != state.z_i
        || final_state.pc != state.pc
        || final_state.acc_digest != state.acc_digest
        || final_state.public_trace != state.public_trace
    {
        return Err(Error::WitnessShape);
    }
    let walked_running = final_running(&state)?;
    let prover_running = final_running(final_state)?;
    if prover_running.claims != walked_running.claims
        || prover_running.parent_authority != walked_running.parent_authority
    {
        return Err(Error::WitnessShape);
    }

    // Now check that the prover's witness matrices open the walked CE
    // claims' commitments under `log`.
    if prover_running.claims.len() != prover_running.witnesses.len() {
        return Err(Error::WitnessLengthMismatch);
    }
    for (index, (claim, witness)) in prover_running
        .claims
        .iter()
        .zip(&prover_running.witnesses)
        .enumerate()
    {
        if log.commit(witness) != claim.c {
            return Err(Error::WitnessCommitmentMismatch { index });
        }
    }
    Ok(())
}

/// Extract the running accumulator from a post-finalization state: must be
/// `Active { running, latest: empty }`, anything else is a witness-shape
/// error.
fn final_running(state: &State) -> Result<&RunningInstance, Error> {
    match &state.proof {
        ProofState::Active { running, latest } if latest.instances.is_empty() => Ok(running),
        _ => Err(Error::WitnessShape),
    }
}

/// The compressed proof handed to the verifier.
///
/// PR5 will populate this with the Spartan SNARK bytes (and any auxiliary
/// public-IO fields the decider's R1CS exposes). Today it is a placeholder
/// type so the lifecycle wiring compiles end-to-end.
#[derive(Clone, Debug, Default)]
pub struct Proof;

/// Verifier key digest (32 bytes). Compared by the caller against an expected
/// value, never trusted as authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKeyDigest(pub [u8; 32]);

/// Run the Spartan terminal compression on the IVC statement. Placeholder
/// until Spartan is wired; the contract `validate_witness` enforces is the
/// relation that the SNARK must reproduce in zero knowledge.
pub fn prove(_statement: &Statement) -> Result<(Proof, VerifierKeyDigest), Error> {
    Err(Error::Unsupported)
}

/// Verify a Spartan-compressed proof against the expected public image.
/// Placeholder until Spartan is wired.
pub fn verify(_public: &PublicImage, _vk_digest: &VerifierKeyDigest, _proof: &Proof) -> Result<(), Error> {
    Err(Error::Unsupported)
}
