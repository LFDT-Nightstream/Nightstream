//! NIFS — the non-interactive folding scheme used by Construction 2.
//!
//! ## What `NIFS` means here
//!
//! In Construction 2 (Hypernova §6.3) the recursive verifier circuit `F'`
//! re-runs `NIFS.V` to fold the previous step's instance `u_i` into the
//! running accumulator `U_i`. Hypernova treats the underlying multi-folding
//! scheme as a black box; only its 4-tuple `(NIFS.G, NIFS.K, NIFS.P, NIFS.V)`
//! interface and the NIVC-compatibility properties (Definition 12) are
//! relevant to F'.
//!
//! ## SuperNeo's flavor differs from Hypernova's flavor
//!
//! Hypernova instantiates NIFS with [Construction 1 + Construction 3] from
//! the Hypernova paper: one round of sumcheck + RLC, producing **one**
//! linearized output instance. The strict NIVC-compatibility shape is
//! `(μ = 1, ν = 1)` — one carried, one fresh, one output.
//!
//! SuperNeo instantiates NIFS with the three-reduction chain
//!
//! ```text
//!     NIFS.V = Π_DEC.verify ∘ Π_RLC.verify ∘ Π_CCS.verify
//!     NIFS.P = Π_DEC.prove  ∘ Π_RLC.prove  ∘ Π_CCS.prove
//! ```
//!
//! producing **k** CE claims (the Π_DEC children). The extra `Π_DEC` step
//! is required for the lattice setting: Π_RLC produces a CE claim of norm
//! `B = b^k`, and Π_DEC splits it back to k claims of norm `b` so the
//! running accumulator stays low-norm. Hypernova's Pedersen commitments
//! don't need this and so don't have a Π_DEC step.
//!
//! ## Backend selection
//!
//! The caller selects one complete NIFS prover at this boundary:
//!
//! - [`PaperExactNifsProver`] uses PaperExact PiCCS, PiRLC, and PiDEC.
//! - [`OptimizedCpuNifsProver`] uses optimized PiCCS, PiRLC, and PiDEC.
//! - CUDA and Metal prover types implement the optimized NIFS adapter in
//!   their backend crates.
//! - [`CrosscheckNifsProver`] runs optimized CPU and PaperExact together.
//!
//! Reduction mode is not a second caller choice. An accelerator selection
//! therefore cannot be combined with PaperExact reductions.
//!
//! ## What this module owns
//!
//! - `proof.rs` — `NifsProof` wire-format type (the three sub-proofs F'
//!   will re-verify in-circuit).
//! - `prover.rs` — `prove(running, fresh) -> (next_running, NifsProof)`.
//!   The Π_CCS → Π_RLC → Π_DEC composition on the prover side.
//! - `verifier.rs` — `verify(running_claims, fresh_claims, &proof)
//!   -> next_running_claims`. Mirror of `prover.rs`.
//! - `work.rs` — witness-threading helpers used by the prover side.
//!
//! No IVC vocabulary here (no `z_i`, no `pc`, no `x_out`). Those belong to
//! `paper::f_prime` and `paper::construction2`.
//!
//! The math itself lives in `paper::reductions::{pi_ccs, pi_rlc, pi_dec}`.
//! This module selects one complete implementation and composes it.

mod backend;
pub mod circuit;
mod crosscheck;
mod fixed;
mod paper_exact;
mod proof;
mod prover;
mod verifier;
mod work;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    PiCcs(#[from] crate::paper::pi_ccs::Error),
    #[error(transparent)]
    PiRlc(#[from] crate::paper::pi_rlc::Error),
    #[error(transparent)]
    PiDec(#[from] crate::paper::pi_dec::Error),
    #[error(transparent)]
    Running(#[from] crate::paper::construction2::running::RunningInstanceError),
    #[error("fixed Construction-2 NIFS {what}: expected {expected}, got {got}")]
    FixedShape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("NIFS.P backend `{backend}` is not available in this build: {reason}")]
    BackendUnavailable {
        backend: &'static str,
        reason: &'static str,
    },
    #[error("NIFS.P backend `{backend}` failed during {phase}: {reason}")]
    BackendFailure {
        backend: &'static str,
        phase: &'static str,
        reason: String,
    },
    #[error("NIFS.P crosscheck executions differ at {boundary}")]
    CrosscheckMismatch { boundary: &'static str },
    #[error("NIFS.P crosscheck reference worker panicked")]
    CrosscheckWorkerPanic,
}

pub use backend::{
    AcceleratorCrosscheckNifsProver, CrosscheckNifsProver, NifsFreshInstancesRequest, NifsFreshSignedUnitAssignment,
    NifsFreshSignedUnitInstancesRequest, NifsProverAdapter, NifsProverRequest, OptimizedCpuNifsProver,
    OptimizedNifsProverAdapter, PaperExactNifsProver,
};
#[doc(hidden)]
pub use crosscheck::require_nifs_execution_match;
pub use fixed::{prove_fixed, verify_fixed, FixedNifsAccumulator};
pub use paper_exact::prove_paper_exact;
pub use proof::NifsProof;
#[doc(hidden)]
pub use prover::prove_with_joint_oracle_backend;
pub use prover::{prove, prove_with_adapter};
pub use verifier::{verify, verify_paper_exact};
