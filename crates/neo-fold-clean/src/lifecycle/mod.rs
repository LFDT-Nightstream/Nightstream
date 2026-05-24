//! Public lifecycle.
//!
//! This is the *only* public surface a frontend or downstream consumer
//! should know about. Everything below is in `paper/` (auditable) or
//! `engine/` (implementation).
//!
//! See the crate-level docs in `lib.rs` for the canonical example.
//!
//! ## Two paths, two verifier types
//!
//! The Phase 1.7 type split makes the verifier-authority boundary
//! structural. Production code wires the non-replay path; the audit
//! path is for diagnostics, the Spartan decider statement, and red-team
//! tests that need to mutate the per-step audit trail.
//!
//! ```text
//! Production (non-replay IVC):
//!   preprocess              one-time
//!     └─ derive vk_fs from (params, structure)
//!   prove(prep, batches)  → UncompressedAudit                (per-session)
//!     └─ runs Π_CCS / Π_RLC / Π_DEC on each batch, accumulating audit
//!   extend(prep, audit, batch) → UncompressedAudit            (optional)
//!     └─ one more F' step
//!   finish_uncompressed(prep, audit) → Uncompressed
//!     └─ flush trailing latest, DROP audit trail
//!   verify_uncompressed(prep, &Uncompressed) → Result<()>
//!     └─ constant-time IVC verification via terminal-fold re-run
//!        (HyperNova §6.3 Construction 2 + SuperNeo §7)
//!
//! Audit / decider (chain replay, Spartan):
//!   ... prove + extend as above ...
//!   finish_uncompressed_with_audit(prep, audit) → UncompressedAudit
//!     └─ flush trailing latest, KEEP audit trail
//!   verify_uncompressed_audit(prep, &UncompressedAudit) → Result<()>
//!     └─ linear-time chain replay; catches audit-trail tampers
//!   build_decider_statement(prep, &UncompressedAudit) → decider::Statement
//!     └─ feeds the (PR5) Spartan compress / verify SNARK
//!   compress(prep, UncompressedAudit) → Compressed              (PR5)
//!   verify(prep, &Compressed) → Result<()>                      (PR5)
//! ```
//!
//! ## What this module owns
//!
//! - `mod.rs` (this file) — public types ([`Preprocessing`],
//!   [`Uncompressed`], [`UncompressedAudit`], [`Compressed`],
//!   [`PublicImage`]), the [`Error`] enum, and [`preprocess`].
//! - `prove.rs` — [`prove`], [`extend`], and the `start_proof` helper.
//! - `verify.rs` — [`verify`] (compressed), [`verify_uncompressed`]
//!   (non-replay IVC), [`verify_uncompressed_audit`] (chain replay).
//! - `compress.rs` — [`finish_uncompressed`] / [`finish_uncompressed_with_audit`],
//!   [`compress`], public-image / decider-statement builders.
//! - `schedule.rs` — [`FoldSchedule`], `partition<T>`, [`ScheduleError`].

pub mod compress;
pub mod prove;
pub mod schedule;
pub mod verify;

use neo_ajtai::AjtaiSModule;
use neo_math::{D, F};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use thiserror::Error;

use crate::paper::construction2::{FinalFoldProof, SemanticStateMode, State, StepProof, VerifierKey};
use crate::paper::decider;
use crate::paper::params::Params;
use crate::paper::relations::{CcsClaim, DecMixer, RlcMixer, Structure};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Construction2(#[from] crate::paper::construction2::Error),
    #[error(transparent)]
    Decider(#[from] decider::Error),
    #[error("verify_uncompressed: proof is not finalized (state is Initial, or trailing latest is non-empty)")]
    NotFinalized,
    #[error("verify_uncompressed: recorded final accumulator witness shape is inconsistent")]
    FinalAccumulatorWitnessShapeMismatch,
    #[error("verify_uncompressed: recorded final accumulator witness commitment mismatch at index {index}")]
    FinalAccumulatorWitnessCommitmentMismatch { index: usize },
    #[error("verify_uncompressed: recorded final accumulator claim {index} public-input X does not match the projection from witness Z")]
    FinalAccumulatorPublicInputMismatch { index: usize },
    #[error("verify_uncompressed: recorded final accumulator witness {index} has an entry outside the low-norm bound at row={row}, col={col}")]
    FinalAccumulatorLowNormViolation {
        index: usize,
        row: usize,
        col: usize,
    },
    #[error(
        "verify_uncompressed: state after re-running the terminal NIFS fold disagrees with the recorded proof.state"
    )]
    PostStateMismatch,
    #[error("verify_uncompressed: finalized proof has a non-empty final running accumulator but carries no terminal-fold proof")]
    MissingTerminalFoldProof,
    #[error("verify_uncompressed: recorded acc_digest does not match the digest of the recorded final running claims")]
    AccDigestMismatch,
    #[error(
        "verify_uncompressed: stateless chain's terminal semantic_state_digest does not equal \
         the pre-terminal accumulator digest (stateless plans require \
         `semantic_state_digest == acc_digest` to be carried unchanged through finalization)"
    )]
    StatelessSemanticInvariantViolated,
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
    #[error("preprocess: optimized engine cache build failed ({0})")]
    OptimizedCacheBuild(#[from] neo_reductions::error::PiCcsError),
    #[error(
        "preprocess: cached structure_digest / optimized_cache no longer matches `prep.structure`. \
         This is a developer footgun: internal code desynchronized preprocessing caches after construction. \
         Rebuild `Preprocessing` via `preprocess` instead of mutating fields."
    )]
    StructureCacheMismatch,
}

/// Verifier-owned protocol context. Built once per program and reused
/// across many proofs.
///
/// The verifier does not know which Ajtai setup the prover used internally.
/// It fixes this context locally and accepts only proofs that verify under
/// these params/setup. Proofs must never carry or choose params/setup.
pub struct Preprocessing {
    pub params: Params,
    structure: Structure,
    pub log: AjtaiSModule,
    pub vk: VerifierKey,
    pub mix_rhos_commits: RlcMixer,
    pub combine_b_pows: DecMixer,
    /// Program-fixed public-input length; absorbed into `vk_fs_digest` so
    /// the chain binds to a specific m_in. `None` means "unfixed at the
    /// program level" — encoded as `u64::MAX` in the absorb.
    pub public_input_len: Option<usize>,
    /// Verifier-owned semantic-state mode. Default `Stateless`; the
    /// R1CS-F' frontend (or any other stateful frontend) sets this to
    /// `Stateful` at its own preprocess time if its plan declares
    /// `semantic_state_in/out_var_indices`. The verifier consults this
    /// bit in `verify_uncompressed` / `verify_uncompressed_audit` so a
    /// malicious prover on a stateless plan cannot inject self-consistent
    /// arbitrary bytes into `PublicImage.semantic_state_digest`.
    pub semantic_state_mode: SemanticStateMode,
    /// Memoized 4-limb digest of the full CCS structure
    /// (`paper::digest::structure_digest(&structure)`). Verifier-owned,
    /// computed once at preprocess time; protocol code reads this field
    /// instead of recomputing the digest on every step.
    structure_digest: [F; 4],
    /// Memoized optimized-engine cache for this structure (sparse + SuperNeo
    /// eval tables + matrix digest). Verifier-derived; built once at
    /// preprocess time so `engine::optimized::{prove_pi_ccs, verify_pi_ccs}`
    /// don't rebuild it on every fold.
    optimized_cache: OptimizedStructureCache,
}

impl Preprocessing {
    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    /// Frontend-side hook to declare that the chain built on top of this
    /// preprocessing carries application state (i.e., the plan declares
    /// `semantic_state_in/out_var_indices`). Verifier checks consult the
    /// resulting mode to decide whether `proof.semantic_state_digest`
    /// must equal the accumulator digest (Stateless) or whether the F'
    /// image's Poseidon2 binding rows authenticate it (Stateful).
    ///
    /// Called once by the stateful frontend's `preprocess`; idempotent.
    pub fn with_semantic_state_mode(mut self, mode: SemanticStateMode) -> Self {
        self.semantic_state_mode = mode;
        self
    }

    pub fn structure_digest(&self) -> &[F; 4] {
        &self.structure_digest
    }

    pub fn optimized_cache(&self) -> &OptimizedStructureCache {
        &self.optimized_cache
    }

    /// Cheap integrity check that the memoized `structure_digest` and
    /// `optimized_cache` still describe the live `structure`. Compares
    /// the cache's shape fingerprint `(n, m, t)` and re-runs
    /// `structure_digest` from `structure`. The protocol-bound digest
    /// is the recomputed value; the stored field is only authority by
    /// preprocessing-time construction.
    ///
    /// Returns [`Error::StructureCacheMismatch`] if internal code somehow
    /// desynchronized `structure` and the derived caches after construction.
    /// Production paths don't call this on every step; it's a developer
    /// footgun gate.
    pub fn validate_cached_structure(&self) -> Result<(), Error> {
        let live_shape = (self.structure.n, self.structure.m, self.structure.t());
        if self.optimized_cache.shape() != live_shape {
            return Err(Error::StructureCacheMismatch);
        }
        let live_digest = crate::paper::digest::structure_digest(&self.structure);
        if live_digest != self.structure_digest {
            return Err(Error::StructureCacheMismatch);
        }
        Ok(())
    }
}

/// Terminal-only uncompressed proof — the **non-replay IVC verifier**'s
/// input.
///
/// Carries exactly the fields `verify_uncompressed` reads: the
/// post-finalization `State` (chain coordinates + final running
/// accumulator with witnesses) and the terminal `FinalFoldProof`
/// (whose `terminal_inputs` snapshot is what authentiticates the chain
/// through a verifier-driven NIFS.V re-run; see
/// [`verify::verify_uncompressed`]).
///
/// The per-step audit trail (`steps`, `public_batches`) is **not** part
/// of this type — it lives in [`UncompressedAudit`] and is consumed by
/// the chain-replay verifier ([`verify::verify_uncompressed_audit`])
/// and the Spartan decider.
///
/// There is no session-wide transcript on the proof. Each F' step owns
/// its own per-step transcript inside `paper::f_prime::{prove, verify}`,
/// and the terminal fold owns its own inside
/// `paper::construction2::{prove_final_fold, verify_final_fold}`.
#[derive(Clone, Debug)]
pub struct Uncompressed {
    pub state: State,
    /// Final NIFS proof that flushed the trailing latest into the running
    /// accumulator at finalization, plus the prover-snapshotted
    /// `terminal_inputs` (pre-fold running + latest) the verifier
    /// re-runs NIFS.V against. `None` only when the chain had nothing
    /// to flush at finalize.
    pub final_fold: Option<FinalFoldProof>,
}

/// Uncompressed proof **with audit trail** — the chain-replay verifier's
/// input and the Spartan decider's witness source.
///
/// Wraps the terminal-only [`Uncompressed`] with the per-step
/// `StepProof`s and public batches each `extend` produced. The wrapping
/// (rather than flat fields) makes the verifier-authority boundary
/// explicit at the type level: anything inside `proof` is the terminal
/// IVC verifier's authority surface; `steps` / `public_batches` are
/// audit-trail metadata.
///
/// Pre-finalize this type is `UncompressedAudit { proof: Uncompressed
/// { state, final_fold: None }, steps, public_batches }` — the terminal
/// fold hasn't run yet. After [`finish_uncompressed_with_audit`] the
/// inner `proof.final_fold` is `Some`.
#[derive(Clone, Debug)]
pub struct UncompressedAudit {
    pub proof: Uncompressed,
    pub steps: Vec<StepProof>,
    /// The K instances each `extend` stored as the next-step's latest,
    /// claims-only (witnesses are prover-private). Length matches `steps`.
    pub public_batches: Vec<Vec<CcsClaim>>,
}

/// The final proof bundle.
pub struct Compressed {
    pub proof: decider::Proof,
    pub vk: decider::VerifierKeyDigest,
    pub public_image: PublicImage,
}

// Public image lives with the decider contract; re-export so lifecycle
// callers can name it without reaching into `paper::decider`.
pub use crate::paper::decider::PublicImage;

// ──────────────────────────────────────────────────────────────────────────
// Public entry-point re-exports + preprocess (the only one-line entry).
// ──────────────────────────────────────────────────────────────────────────

// Production path — non-replay IVC.
pub use compress::finish_uncompressed;
pub use prove::{extend, prove};
pub use verify::verify_uncompressed;

// Audit / decider path — chain replay, Spartan, diagnostic tests.
pub use compress::{build_decider_statement, compress, finish_uncompressed_with_audit, verify};
pub use verify::verify_uncompressed_audit;

pub use schedule::{FoldSchedule, ScheduleError};

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
    // Verifier-derived caches: pure functions of `structure`, computed
    // once here so engine seams + protocol-binding paths don't recompute
    // them on every fold/step. The optimized cache carries the Π_CCS
    // `mat_digest`, which `structure_digest` also binds, so derive the
    // structure digest from that same matrix digest instead of walking the
    // matrices twice during preprocess.
    let optimized_cache = OptimizedStructureCache::build(&structure)?;
    let structure_digest =
        crate::paper::digest::structure_digest_from_mat_digest(&structure, optimized_cache.mat_digest());
    let vk = VerifierKey::derive_from_structure_digest(&params, &structure_digest, public_input_len);
    Ok(Preprocessing {
        params,
        structure,
        log,
        vk,
        mix_rhos_commits,
        combine_b_pows,
        public_input_len,
        // Default to Stateless. Stateful frontends call
        // [`Preprocessing::with_semantic_state_mode`] after preprocess to
        // upgrade the mode based on their plan.
        semantic_state_mode: SemanticStateMode::Stateless,
        structure_digest,
        optimized_cache,
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
