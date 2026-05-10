//! Public lifecycle: `preprocess → prove → extend → compress → verify`.
//!
//! This is the *only* public surface a frontend or downstream consumer
//! should know about. Everything below is in `paper/` (auditable) or
//! `engine/` (implementation).
//!
//! See the crate-level docs in `lib.rs` for the canonical example.
//!
//! ## Pipeline
//!
//! Prover and verifier execute the same numbered sequence (Jolt-style index;
//! every line maps to one paper section so an auditor can follow along).
//!
//! ```text
//! 1. preprocess              (one-time)
//!    └─ derive vk_fs from (params, structure)            [Construction 2]
//!
//! 2. start                    (per session)
//!    └─ State::base(z_0): empty proof state, pc = TRIVIAL_PC
//!                                                        [Construction 2 initial case]
//!
//! 3. extend                   (per IVC step)
//!    ├─ if state.proof = Active: Π_CCS / Π_RLC / Π_DEC fold latest into running
//!    ├─ else (Initial):         no fold; running stays empty
//!    ├─ advance counters / digests
//!    ├─ x_out                  H(vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc)
//!    └─ store next batch as state.proof.latest for the *next* fold
//!
//! 4. finish_uncompressed      (optional before compression)
//!    └─ fold the trailing latest into running and retain terminal NIFS proof
//!
//! 5. compress                 (one-time; PR5)
//!    └─ Spartan terminal compression of the final F' step       [decider]
//!
//! 6. verify                   (one-time; PR5)
//!    └─ Spartan SNARK verify against PublicImage                [decider]
//! ```
//!
//! ## What this module owns
//!
//! - `mod.rs` (this file) — public types (Preprocessing, Uncompressed,
//!   Compressed, PublicImage), the `Error` enum, and `preprocess`.
//! - `prove.rs` — `prove`, `extend`, and the `start_proof` helper.
//! - `verify.rs` — `verify`, `verify_uncompressed`, and shape/state checks.
//! - `compress.rs` — `finish_uncompressed`, `compress`, public-image and
//!   decider-statement builders.
//! - `schedule.rs` — `FoldSchedule`, `partition<T>`, `ScheduleError`.

pub mod compress;
pub mod prove;
pub mod schedule;
pub mod verify;

use neo_ajtai::AjtaiSModule;
use neo_math::D;
use thiserror::Error;

use crate::engine::transcript::Transcript;
use crate::paper::construction2::{EncInst, FinalFoldProof, State, StepProof, VerifierKey};
use crate::paper::decider;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, DecMixer, RlcMixer, Structure};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Construction2(#[from] crate::paper::construction2::Error),
    #[error(transparent)]
    Decider(#[from] decider::Error),
    #[error("verify_uncompressed: |steps| ({steps}) \u{2260} |batches| ({batches})")]
    UncompressedShapeMismatch { steps: usize, batches: usize },
    #[error("verify_uncompressed: replayed final state did not match the prover's recorded state")]
    UncompressedStateMismatch,
    #[error("verify_uncompressed: recorded final accumulator witness shape is inconsistent")]
    FinalAccumulatorWitnessShapeMismatch,
    #[error("verify_uncompressed: recorded final accumulator witness commitment mismatch at index {index}")]
    FinalAccumulatorWitnessCommitmentMismatch { index: usize },
    #[error("extend: cannot extend an already-finalized uncompressed proof")]
    AlreadyFinalized,
    #[error("finish_uncompressed: already-finalized proof is internally inconsistent")]
    FinalizedProofInconsistent,
    #[error("lifecycle: public input length mismatch (expected {expected}, got {got})")]
    PublicInputLenMismatch { expected: usize, got: usize },
    #[error("preprocess: Ajtai setup dimension mismatch (expected d={expected_d}, cols={expected_cols}; got d={got_d}, cols={got_cols})")]
    AjtaiDimensionMismatch {
        expected_d: usize,
        expected_cols: usize,
        got_d: usize,
        got_cols: usize,
    },
    #[error("preprocess: Ajtai setup κ mismatch (expected {expected}, got {got})")]
    AjtaiKappaMismatch { expected: usize, got: usize },
    #[error("preprocess: canonical Ajtai setup unavailable ({0})")]
    AjtaiSetup(#[from] neo_ajtai::AjtaiError),
}

/// Verifier-owned protocol context. Built once per program and reused
/// across many proofs.
///
/// The verifier does not know which Ajtai setup the prover used internally.
/// It fixes this context locally and accepts only proofs that verify under
/// these params/setup. Proofs must never carry or choose params/setup.
pub struct Preprocessing {
    pub params: Params,
    pub structure: Structure,
    pub log: AjtaiSModule,
    pub vk: VerifierKey,
    pub mix_rhos_commits: RlcMixer,
    pub combine_b_pows: DecMixer,
    /// Program-fixed public-input length; absorbed into `vk_fs_digest` so
    /// the chain binds to a specific m_in. `None` means "unfixed at the
    /// program level" — encoded as `u64::MAX` in the absorb.
    pub public_input_len: Option<usize>,
}

/// Uncompressed proof state. Before `finish_uncompressed`, the final batch is
/// still held in `state.proof.latest`; after finishing, `final_fold` verifies
/// that trailing batch and `state` is the post-finalization state.
pub struct Uncompressed {
    pub state: State,
    pub steps: Vec<StepProof>,
    /// The K instances each `extend` stored as the next-step's latest,
    /// claims-only (witnesses are prover-private). Length matches `steps`.
    pub public_batches: Vec<Vec<CcsClaim>>,
    /// Final NIFS proof that flushes the trailing latest into the running
    /// accumulator without advancing chunk counters.
    pub final_fold: Option<FinalFoldProof>,
    pub transcript: Transcript,
}

/// The final proof bundle.
pub struct Compressed {
    pub proof: decider::Proof,
    pub vk: decider::VerifierKeyDigest,
    pub public_image: PublicImage,
}

/// What the verifier sees. Paper-named; matches the absorb order of
/// [`crate::paper::construction2::compute_x_out`].
#[derive(Clone, Debug)]
pub struct PublicImage {
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub acc_digest: [u8; 32],
    pub public_trace: [u8; 32],
    pub x_out: EncInst,
    pub vk_fs_digest: [u8; 32],
}

// ──────────────────────────────────────────────────────────────────────────
// Public entry-point re-exports + preprocess (the only one-line entry).
// ──────────────────────────────────────────────────────────────────────────

pub use compress::{compress, finish_uncompressed, verify};
pub use prove::{extend, prove};
pub use schedule::{FoldSchedule, ScheduleError};
pub use verify::verify_uncompressed;

/// Build the verifier-owned preprocessing once and reuse it.
///
/// This entry uses the process-global Ajtai setup registered for the
/// expected `(D, cols)` shape. The proof never chooses this setup.
pub fn preprocess(
    params: Params,
    structure: Structure,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    public_input_len: Option<usize>,
) -> Result<Preprocessing, Error> {
    let cols = structure.m.div_ceil(D);
    let log = AjtaiSModule::from_global_for_dims(D, cols)?;
    preprocess_with_test_log(
        params,
        structure,
        log,
        mix_rhos_commits,
        combine_b_pows,
        public_input_len,
    )
}

/// Build preprocessing with an explicitly supplied Ajtai module.
///
/// This is for tests and adversarial fixtures only. Production callers use
/// [`preprocess`], which obtains the verifier-owned canonical setup from the
/// global Ajtai registry.
#[doc(hidden)]
pub fn preprocess_with_test_log(
    params: Params,
    structure: Structure,
    log: AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    public_input_len: Option<usize>,
) -> Result<Preprocessing, Error> {
    validate_ajtai_context(&params, &structure, &log)?;
    let vk = VerifierKey::derive(&params, &structure, public_input_len);
    Ok(Preprocessing {
        params,
        structure,
        log,
        vk,
        mix_rhos_commits,
        combine_b_pows,
        public_input_len,
    })
}

fn validate_ajtai_context(params: &Params, structure: &Structure, log: &AjtaiSModule) -> Result<(), Error> {
    // This is a local consistency guard for the global, verifier-owned
    // Ajtai setup. It is deliberately not a proof-carried digest: the
    // verifier builds `Preprocessing` from its own canonical config and
    // rejects any local setup that cannot commit witnesses of this shape.
    let expected_d = D;
    let expected_cols = structure.m.div_ceil(D);
    let (got_d, got_cols) = log.dims();
    if got_d != expected_d || got_cols != expected_cols {
        return Err(Error::AjtaiDimensionMismatch {
            expected_d,
            expected_cols,
            got_d,
            got_cols,
        });
    }

    let expected = params.kappa() as usize;
    let got = log.kappa();
    if got != expected {
        return Err(Error::AjtaiKappaMismatch { expected, got });
    }

    Ok(())
}

pub(crate) fn validate_public_input_len(prep: &Preprocessing, claims: &[CcsClaim]) -> Result<(), Error> {
    let Some(expected) = prep.public_input_len else {
        return Ok(());
    };
    for claim in claims {
        if claim.m_in != expected {
            return Err(Error::PublicInputLenMismatch {
                expected,
                got: claim.m_in,
            });
        }
    }
    Ok(())
}
