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
//! Terminal-only IVC:
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
//!        Accepted only when the terminal fold starts from an empty
//!        running accumulator (a single chunk). Multi-chunk histories
//!        need the audit/decider path below because the evidence needed
//!        to bind earlier chunks' counters and boundary coordinates lives
//!        in per-step rows that `Uncompressed` intentionally drops.
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
use crate::paper::relations::{ajtai_dec_mixer, ajtai_rlc_mixer, CcsClaim, DecMixer, RlcMixer, Structure};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Construction2(#[from] crate::paper::construction2::Error),
    #[error("nebula: segment-open payload supplied but this preprocessing carries no Nebula plan")]
    NebulaNotConfigured,
    #[error("nebula: preprocessing/plan and chain-state lane presence disagree (config without lane, or lane without config)")]
    NebulaLanePresenceMismatch,
    #[error("nebula: externally accepted proofs must end at a closed segment (§6.3 finalization rule: idx == 0, γ == ⊥, header chains)")]
    NebulaSegmentOpenAtTerminal,
    #[error("nebula: terminal claim's adv tuple failed the lane slice-opening (spec §5.2 R3)")]
    NebulaSliceOpeningFailed,
    #[error("nebula: terminal claim carries no adv tuple on a Nebula chain (or a tuple on a plain chain)")]
    NebulaAdvPresenceMismatch,
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
        "verify_uncompressed: recorded final accumulator claim {index} CE relation violated — \
         `y_ring[{matrix_index}]` does not equal multilinear_eval(M_{matrix_index} · Z, r). \
         The SuperNeo verifier equation on the folded CE relation requires y_ring closure \
         against the opened witness; the F'-chain `acc_digest` commits to the public CE claim, \
         but does not by itself prove that the opened witness Z satisfies that claim."
    )]
    FinalAccumulatorCeRelationViolation { index: usize, matrix_index: usize },
    #[error(
        "verify_uncompressed: recorded final accumulator claim {index} `ct[{matrix_index}]` \
         does not equal the SuperNeo scalar view of `multilinear_eval(M_{matrix_index} · Z, r)` \
         (the constant term of y_ring[{matrix_index}]). Bound here so the prover can't lie about \
         `ct` independently of `y_ring` — `ct` enters the protocol's consistency checks downstream."
    )]
    FinalAccumulatorCtMismatch { index: usize, matrix_index: usize },
    #[error(
        "verify_uncompressed: recorded final accumulator claim {index} optional NC channel \
         `y_zcol` does not equal the projection `Z · chi(s_col)` from the opened witness. \
         `y_zcol` is not recursive accumulator-handle authority, but when present in the \
         terminal claim it must be recomputed from terminal witness authority rather than trusted."
    )]
    FinalAccumulatorNcChannelMismatch { index: usize },
    #[error(
        "verify_uncompressed: recorded final accumulator claim {index} evaluation point `r` has the \
         wrong length (expected {expected} = log2(next_pow2(structure.n).max(2)), got {got}). A \
         truncated `r` would silently shrink the multilinear evaluation domain and a padded `r` \
         would over-extend it, so the CE-relation closure rejects an off-shape `r` before computing M·Z(r)."
    )]
    FinalAccumulatorEvaluationPointShapeMismatch {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "verify_uncompressed: recorded final accumulator claim {index} carries unsupported sidecar field `{field}` \
         with length/value {got}. This clean SuperNeo path does not implement that metadata, so terminal witness \
         authority must reject it rather than let accumulator-digested data remain unconstrained."
    )]
    FinalAccumulatorUnsupportedSidecar {
        index: usize,
        field: &'static str,
        got: usize,
    },
    #[error(
        "verify_uncompressed: state after re-running the terminal NIFS fold disagrees with the recorded proof.state"
    )]
    PostStateMismatch,
    #[error(
        "verify_uncompressed: terminal latest claim {index} public input does not encode the pre-final state x_out"
    )]
    TerminalLatestPublicInputMismatch { index: usize },
    #[error(
        "verify_uncompressed: finalized proofs must carry a terminal-fold proof; \
         `final_fold = None` has no verifier-driven NIFS proof binding the recorded state"
    )]
    MissingTerminalFoldProof,
    #[error(
        "verify_uncompressed: terminal fold inputs carry a non-empty pre-final running accumulator without \
         parent authority. That shape cannot be produced by an honest NIFS chain and must fail before any \
         pre-fold accumulator digest is reconstructed."
    )]
    PreFinalAccumulatorMissingParentAuthority,
    #[error("verify_uncompressed: recorded acc_digest does not match the digest of the recorded final running claims")]
    AccDigestMismatch,
    #[error(
        "verify_uncompressed: stateless chain's terminal semantic_state_digest does not equal \
         the pre-terminal accumulator digest (stateless plans require \
         `semantic_state_digest == acc_digest` to be carried unchanged through finalization)"
    )]
    StatelessSemanticInvariantViolated,
    #[error(
        "verify_uncompressed: proof.state.initial_semantic_state_digest \u{2260} \
         prep.initial_semantic_state_digest() (the verifier-owned preprocessing anchors the \
         chain's initial app-state seed; vk_fs_digest absorbs it, so a mismatched proof field \
         cannot be silently relabelled)"
    )]
    InitialSemanticStateAnchorMismatch,
    #[error(
        "lifecycle: noncanonical semantic-state digest byte limb in {owner} at lane {lane}; \
         semantic digests are interpreted as four Goldilocks lanes in F' and must use canonical lane bytes"
    )]
    SemanticStateDigestCanonicality { owner: &'static str, lane: usize },
    #[error(
        "verify_uncompressed: terminal-only verification is supported only for a single F' chunk \
         until the compressed decider proves the recursive F' / NIFS.V induction (got chunk_count={chunk_count}). \
         Use finish_uncompressed_with_audit + verify_uncompressed_audit / build_decider_statement for multi-chunk F' chains."
    )]
    FPrimeNonReplayUnsupported { chunk_count: u64 },
    #[error(
        "verify_uncompressed: terminal-only verification is supported only when the terminal fold starts \
         from an empty running accumulator (single chunk). This proof carries chunk_count={chunk_count}; \
         use finish_uncompressed_with_audit + verify_uncompressed_audit / build_decider_statement for multi-chunk chains."
    )]
    TerminalOnlyMultiChunkUnsupported { chunk_count: u64 },
    #[error("extend: cannot extend an already-finalized uncompressed proof")]
    AlreadyFinalized,
    #[error("extend: cannot fold an empty batch; every extend must contribute at least one CCS instance")]
    EmptyBatch,
    #[error("extend: batch has {got} fresh instances, but this SuperNeo profile supports at most {max}")]
    BatchTooLarge { got: usize, max: usize },
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
    pub(crate) mix_rhos_commits: RlcMixer,
    pub(crate) combine_b_pows: DecMixer,
    /// Nebula memory-checking plan context (spec §6): the lane-commitment
    /// scheme, segment length, plan digest, and the verifier's ROM handle
    /// `D_init`. `None` for plain chains. Set by
    /// [`Preprocessing::with_nebula`]; every extend on a Nebula
    /// preprocessing runs the §6.3 lane transition.
    pub(crate) nebula: Option<std::sync::Arc<crate::paper::construction2::NebulaConfig>>,
    /// Program-fixed public-input length; absorbed into `vk_fs_digest` so
    /// the chain binds to a specific m_in. `None` means "unfixed at the
    /// program level" — encoded as `u64::MAX` in the absorb.
    pub public_input_len: Option<usize>,
    /// Verifier-owned initial app/VM semantic-state digest.
    ///
    /// Absorbed into [`vk_fs_digest`] at preprocess time so every
    /// step's `state_x_out` transitively binds it. Default is the
    /// `empty_semantic_state_digest()` constant (stateless seed).
    /// Stateful frontends set the actual `H(initial_app_state)` via
    /// [`Preprocessing::with_initial_semantic_state_digest`], which
    /// **rebuilds `vk`** so the new value is bound from the very
    /// first step.
    ///
    /// `pub(crate)` so external callers can't quietly mutate the seed
    /// (which would silently rebuild `vk` underneath them). Read
    /// access goes through
    /// [`Preprocessing::initial_semantic_state_digest`].
    pub(crate) initial_semantic_state_digest: [u8; 32],
    /// Verifier-owned semantic-state mode — **structure-derived, not
    /// caller-settable**.
    ///
    /// Default `Stateless`; in-crate frontends (the R1CS-F' preprocess
    /// path) flip this to `Stateful` at their own preprocess time when
    /// their plan declares either explicit semantic-state indices or
    /// app-public-output binding. The resulting F' image's CCS
    /// structure must carry Poseidon2 binding rows over the wires that
    /// define the semantic digest. `Stateful` therefore means
    /// "independent semantic digest authenticated by F' constraints";
    /// it does not always mean "explicit transition state with both
    /// semantic input and semantic output variables."
    ///
    /// The field is `pub(crate)` rather than `pub` precisely because a
    /// public setter would break the ownership boundary: an external
    /// caller could mark a stateless `Preprocessing` `Stateful`,
    /// `verify_uncompressed` would skip the stateless invariant, and
    /// the resulting proof would carry a prover-chosen
    /// `semantic_state_digest` that no constraint authenticates. Read
    /// access goes through [`Preprocessing::semantic_state_mode`].
    pub(crate) semantic_state_mode: SemanticStateMode,
    /// Whether this verifier context owns an R1CS-F' recursive-link public
    /// input. This is not inferable from `public_input_len`: ordinary direct
    /// CCS programs may coincidentally have the same public width as F'.
    ///
    /// The field is verifier-owned and crate-private. R1CS-F' frontends set
    /// it during preprocess; generic CCS frontends leave it false.
    pub(crate) f_prime_recursive_link: bool,
    /// Memoized 4-limb digest of the full CCS structure
    /// (`paper::digest::structure_digest(&structure)`). Verifier-owned,
    /// computed once at preprocess time; protocol code reads this field
    /// instead of recomputing the digest on every step.
    structure_digest: [F; 4],
    /// SplitNc transcript header derived from `(params, structure, dims,
    /// matrix_digest)`. It is part of `vk_fs` and enters folded F' as
    /// witness data, never as a self-referential matrix constant.
    pi_ccs_header_bundle: [F; 4],
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

    /// Read-only view of the verifier-owned semantic-state mode. See
    /// the [`Preprocessing.semantic_state_mode`] field doc for the
    /// soundness argument.
    pub fn semantic_state_mode(&self) -> SemanticStateMode {
        self.semantic_state_mode
    }

    /// True when this preprocessing context must enforce HyperNova's F'
    /// recursive-link public input (`u_i.x == enc_inst(prior_x_out)`).
    pub fn enforces_f_prime_recursive_link(&self) -> bool {
        self.f_prime_recursive_link
    }

    /// Read-only view of the Nebula plan context; `None` for plain chains.
    pub fn nebula(&self) -> Option<&crate::paper::construction2::NebulaConfig> {
        self.nebula.as_deref()
    }

    /// Attach the Nebula memory-checking plan (spec §11 constants +
    /// `D_init`) to this preprocessing. Every subsequent chain started
    /// from it carries a `NebulaLane` from the base state, every extend
    /// runs the §6.3 transition over the deposited claims, and the
    /// verifiers enforce the finalization rule and the terminal
    /// slice-openings (spec §5.2 R3).
    pub fn with_nebula(mut self, cfg: crate::paper::construction2::NebulaConfig) -> Self {
        self.nebula = Some(std::sync::Arc::new(cfg));
        self
    }

    /// Read-only view of the verifier-owned initial app/VM
    /// semantic-state digest. Stateless preprocessings carry
    /// `empty_semantic_state_digest()`; stateful frontends carry
    /// `H(initial_app_state)`.
    pub fn initial_semantic_state_digest(&self) -> [u8; 32] {
        self.initial_semantic_state_digest
    }

    /// In-crate hook for stateful frontends to set the initial
    /// app-state seed. Rebuilds `vk` so the new value is absorbed
    /// into `vk_fs_digest` and transitively binds every step's
    /// `state_x_out`.
    ///
    /// **`pub(crate)` is load-bearing for soundness.** Stateful
    /// frontends must call this from preprocess time, with a value
    /// matching the structure-baked anchor (read from the same plan).
    /// Exposing it publicly would let a caller drift `vk_fs_digest`'s
    /// initial anchor away from the F' image's base-step constraint —
    /// the very gap the structural fix closes. The only legitimate
    /// caller is `frontends/r1cs_f_prime::preprocess`, which reads
    /// the anchor from `RecursiveStepImagePlan` and applies the same
    /// value here that the structure builder baked into the CCS.
    pub(crate) fn with_initial_semantic_state_digest(mut self, initial: [u8; 32]) -> Result<Self, Error> {
        validate_semantic_state_digest_canonical("initial_semantic_state_digest", initial)?;
        self.initial_semantic_state_digest = initial;
        self.vk = VerifierKey::derive_from_structure_digest(
            &self.params,
            &self.structure_digest,
            self.pi_ccs_header_bundle,
            self.public_input_len,
            initial,
        );
        Ok(self)
    }

    /// In-crate hook for stateful frontends to declare the chain's
    /// semantic mode at their own preprocess time. The frontend MUST
    /// derive `mode` from observable structure properties: either
    /// explicit semantic-state input/output variables or app-public
    /// output variables bound into the semantic digest. It MUST NOT
    /// take a caller-supplied value. The setter is `pub(crate)` so
    /// external code cannot lie about the mode to short-circuit
    /// `verify_uncompressed`'s stateless invariant.
    pub(crate) fn with_semantic_state_mode(mut self, mode: SemanticStateMode) -> Self {
        self.semantic_state_mode = mode;
        self
    }

    /// In-crate hook for R1CS-F' frontends to enable F'-specific recursive
    /// public-input checks. This must not be public: the mode is a verifier
    /// ownership boundary, not a prover/caller choice.
    pub(crate) fn with_f_prime_recursive_link(mut self) -> Self {
        self.f_prime_recursive_link = true;
        self
    }

    pub fn structure_digest(&self) -> &[F; 4] {
        &self.structure_digest
    }

    pub fn pi_ccs_header_bundle(&self) -> [F; 4] {
        self.pi_ccs_header_bundle
    }

    pub fn optimized_cache(&self) -> &OptimizedStructureCache {
        &self.optimized_cache
    }

    /// Verifier-circuit view of this preprocessing context. The dimensions
    /// and Split-NC header are derived from the same params, structure, and
    /// matrix cache used by native proving, so recursive frontends do not
    /// reconstruct protocol metadata through a parallel path.
    pub fn nifs_v_circuit_config(&self) -> Result<crate::paper::nifs::circuit::NifsVCircuitConfig<'_>, Error> {
        let dims = neo_reductions::engines::utils::build_dims_and_policy(self.params.inner(), &self.structure)?;
        Ok(crate::paper::nifs::circuit::NifsVCircuitConfig {
            pi_ccs: crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig {
                params: &self.params,
                structure: &self.structure,
                header_bundle: self.pi_ccs_header_bundle,
                ell_d: dims.ell_d,
                ell_n: dims.ell_n,
                ell_m: dims.ell_m,
                d_sc: dims.d_sc,
            },
        })
    }

    /// Low-level Π_RLC commitment action fixed by preprocessing.
    ///
    /// Exposed read-only for reduction tests and circuit builders that call
    /// NIFS directly. Public preprocessing always installs the canonical
    /// Ajtai action; callers cannot replace it after construction.
    pub fn mix_rhos_commits(&self) -> RlcMixer {
        self.mix_rhos_commits
    }

    /// Low-level Π_DEC commitment action fixed by preprocessing.
    ///
    /// Exposed read-only for reduction tests and circuit builders that call
    /// NIFS directly. Public preprocessing always installs the canonical
    /// Ajtai action; callers cannot replace it after construction.
    pub fn combine_b_pows(&self) -> DecMixer {
        self.combine_b_pows
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

pub(crate) fn validate_semantic_state_digest_canonical(owner: &'static str, digest: [u8; 32]) -> Result<(), Error> {
    if let Some(lane) = crate::paper::digest::noncanonical_digest32_lane(digest) {
        return Err(Error::SemanticStateDigestCanonicality { owner, lane });
    }
    Ok(())
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
    /// to flush inside the low-level finalization helper; public
    /// terminal/verifier surfaces reject `None` because it carries no
    /// verifier-driven terminal fold binding.
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

// Terminal-only lifecycle path.
pub use compress::finish_uncompressed;
pub use prove::{extend, extend_nebula_open, prove};
pub use verify::{validate_final_witness_authority, verify_uncompressed};

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
    public_input_len: Option<usize>,
) -> Result<Preprocessing, Error> {
    let cols = structure.m.div_ceil(D);
    let log = AjtaiSModule::from_global_for_dims(D, cols)?;
    preprocess_with_test_log(params, structure, log, public_input_len)
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
    public_input_len: Option<usize>,
) -> Result<Preprocessing, Error> {
    let optimized_cache = OptimizedStructureCache::build(&structure)?;
    preprocess_with_test_log_and_optimized_cache(
        params,
        structure,
        log,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        public_input_len,
        optimized_cache,
    )
}

/// Build preprocessing from a verifier-owned optimized cache.
///
/// This is intentionally crate-private. Callers must not be able to supply an
/// arbitrary `(structure, optimized_cache)` pair across a protocol boundary.
/// Frontends may use it only with cache material they just built from the same
/// verifier-derived structure artifact.
pub(crate) fn preprocess_with_test_log_and_optimized_cache(
    params: Params,
    structure: Structure,
    log: AjtaiSModule,
    mix_rhos_commits: RlcMixer,
    combine_b_pows: DecMixer,
    public_input_len: Option<usize>,
    optimized_cache: OptimizedStructureCache,
) -> Result<Preprocessing, Error> {
    validate_ajtai_context(&params, &structure, &log)?;
    let live_shape = (structure.n, structure.m, structure.t());
    if optimized_cache.shape() != live_shape {
        return Err(Error::StructureCacheMismatch);
    }
    // Verifier-derived cache: a pure function of `structure`, computed by the
    // frontend's prepared-structure constructor or by `preprocess_with_test_log`
    // above. The optimized cache carries the Π_CCS `mat_digest`, which
    // `structure_digest` also binds, so derive the structure digest from that
    // same matrix digest instead of walking the matrices twice here.
    let structure_digest =
        crate::paper::digest::structure_digest_from_mat_digest(&structure, optimized_cache.mat_digest());
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params.inner(), &structure)?;
    let pi_ccs_header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        params.inner(),
        &structure,
        dims,
        optimized_cache.mat_digest(),
    )?;
    // Default seed: `empty_semantic_state_digest()`. Stateful frontends
    // call [`Preprocessing::with_initial_semantic_state_digest`] after
    // preprocess to install their `H(initial_app_state)`; that setter
    // rebuilds `vk` so the new seed propagates through every step's
    // `state_x_out`.
    let initial_semantic_state_digest = crate::paper::digest::empty_semantic_state_digest();
    let vk = VerifierKey::derive_from_structure_digest(
        &params,
        &structure_digest,
        pi_ccs_header_bundle,
        public_input_len,
        initial_semantic_state_digest,
    );
    Ok(Preprocessing {
        params,
        structure,
        log,
        vk,
        mix_rhos_commits,
        combine_b_pows,
        public_input_len,
        initial_semantic_state_digest,
        // Default to Stateless. Stateful frontends call
        // [`Preprocessing::with_semantic_state_mode`] after preprocess to
        // upgrade the mode based on their plan.
        semantic_state_mode: SemanticStateMode::Stateless,
        f_prime_recursive_link: false,
        structure_digest,
        pi_ccs_header_bundle,
        optimized_cache,
        nebula: None,
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
