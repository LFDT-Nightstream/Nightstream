//! Construction 2 (Hypernova §6.3) over Neo's CCS folding scheme (SuperNeo §7).
//!
//! ## What this module owns
//!
//! - `state.rs` — `State`, the IVC carrier (counters, boundary digests, pc,
//!   and `proof: ProofState`).
//! - `proof_state.rs` — `enum ProofState { Initial, Active { running, latest } }`.
//!   Tagged so the initial case is structurally distinct from the active case.
//! - `running.rs` — `RunningInstance` (U_i, W_i).
//! - `latest.rs` — `LatestInstance` (u_i, w_i).
//! - `verifier_key.rs` — `VerifierKey` + vk_fs derivation.
//! - `enc_inst.rs` — `EncInst` (bit-decomposition of x_out).
//! - `step_proof.rs` — `StepProof` and `FoldProof::{NoFold, Recursive}`.
//! - `transition.rs` — `advance_state`, `compute_x_out`, pc/base-case checks.
//! - `finalization.rs` — flush trailing latest before Spartan compression.
//!
//! The IVC step entry points (`step` / `verify_step`) are re-exported from
//! `paper::f_prime` — that's where the math actually runs.
//!
//! ## ℓ = 1 specialization
//!
//! `pc` is constant-`TRIVIAL_PC` here. Per-opcode dispatch, where present in
//! a frontend, lives in the frontend, not in the IVC step.

pub mod enc_inst;
pub mod finalization;
pub mod latest;
pub mod nebula_lane;
pub mod proof_state;
pub mod running;
pub mod state;
pub mod step_proof;
pub mod transition;
pub mod verifier_key;

use thiserror::Error;

/// Construction 2 program counter — fixed to 1 in the ℓ=1 specialization.
pub const TRIVIAL_PC: u64 = 1;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Nifs(#[from] crate::paper::nifs::Error),
    #[error(transparent)]
    PiCcs(#[from] crate::paper::pi_ccs::Error),
    #[error(transparent)]
    PiRlc(#[from] crate::paper::pi_rlc::Error),
    #[error(transparent)]
    PiDec(#[from] crate::paper::pi_dec::Error),
    #[error(
        "Construction 2: base/recursive branch check failed \
         (Initial requires zero counters and z_0 = z_i; Active requires nonzero counters)"
    )]
    BaseCaseMismatch,
    #[error("Construction 2: F' step must carry at least one fresh/latest instance")]
    EmptyStep,
    #[error("Construction 2: {counter} counter overflow")]
    CounterOverflow { counter: &'static str },
    #[error(
        "Construction 2: pc out of range (expected 1 \u{2264} pc \u{2264} \u{2113}; this build hardcodes \u{2113}=1)"
    )]
    PcOutOfRange,
    #[error("Construction 2: x_out hash chain mismatch (proof's x_out != recomputed)")]
    XOutMismatch,
    #[error(
        "Construction 2: stateless chain claimed semantic_state_digest \u{2260} accumulator digest \
         (stateless plans have no F' image binding rows; the field must equal the new acc_digest)"
    )]
    StatelessSemanticInvariantViolated,
    #[error("Construction 2: FoldProof variant disagrees with ProofState (base/active mismatch)")]
    FoldProofVariantMismatch,
    #[error("Construction 2: step proof's nebula_open payload disagrees with the verifier's segment-open replay")]
    NebulaOpenMismatch,
    #[error(transparent)]
    Nebula(#[from] crate::paper::construction2::nebula_lane::NebulaError),
    #[error(transparent)]
    NebulaX(#[from] crate::paper::construction2::nebula_lane::NebulaXError),
    #[error("Construction 2: final fold proof missing while latest is non-empty")]
    MissingFinalFoldProof,
    #[error("Construction 2: final fold proof present but no trailing latest exists")]
    UnexpectedFinalFoldProof,
}

// Public re-exports: paper-named types live in submodules, but the auditor
// reaches them via `paper::construction2::Foo` (one path, no internal
// double-namespace).
pub use enc_inst::EncInst;
pub use finalization::FINAL_FOLD_TRANSCRIPT_LABEL;
pub use latest::LatestInstance;
pub use nebula_lane::{NebulaAdvance, NebulaConfig, NebulaError, NebulaLane, NebulaStepX, NebulaXError, StackShape};
pub use proof_state::ProofState;
pub use running::RunningInstance;
pub use state::State;
pub use step_proof::{FinalFoldProof, FoldProof, StepProof, TerminalFoldInputs};
pub use verifier_key::VerifierKey;

// Step entry points live in `paper::f_prime`; re-exported here under the
// canonical Construction-2 names so call sites use one path.
pub use crate::paper::f_prime::prove as step;
pub use crate::paper::f_prime::prove_with_semantic_state as step_with_semantic_state;
pub use crate::paper::f_prime::verify as verify_step;

// Transition + finalization helpers exposed `pub(crate)` for f_prime and lifecycle.
pub(crate) use finalization::{prove_final_fold, verify_final_fold};
pub(crate) use transition::{
    advance_state, compute_x_out, enforce_pc_in_range, f_prime_chunk_public_digest_for_step,
    f_prime_chunk_public_digest_from_claims, state_base_case_check,
};
pub use transition::{SemanticStateAdvance, SemanticStateMode};
