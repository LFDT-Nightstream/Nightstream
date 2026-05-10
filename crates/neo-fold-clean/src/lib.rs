//! `neo-fold-clean` — paper-faithful, audit-first SuperNeo IVC integrator.
//!
//! ## Public lifecycle
//!
//! ```ignore
//! use neo_fold_clean::{
//!     frontends::direct_ccs, prove, extend, finish_uncompressed,
//!     verify_uncompressed, CcsInstance, FoldSchedule,
//! };
//!
//! let prep = direct_ccs::preprocess_seeded(&r1cs, seed)?;
//!
//! // Build one CCS instance per row of user computation.
//! let rows: Vec<CcsInstance> = user_assignments.iter().map(|z| {
//!     direct_ccs::build_instance(&prep, &r1cs, z)
//! }).collect::<Result<Vec<_>, _>>()?;
//!
//! // Pick a batch size. Schedule.partition slices rows into fold steps.
//! //   FoldSchedule::RowsPerStep(1)  — one row per fold (default, lowest latency)
//! //   FoldSchedule::RowsPerStep(n)  — n rows per fold (amortise per-fold cost)
//! //   FoldSchedule::WholeRun        — all rows in one fold step
//! let steps = FoldSchedule::RowsPerStep(4).partition(rows)?;
//!
//! let mut proof = prove(&prep, steps)?;
//!
//! // More rows arrive later? Partition them and `extend`.
//! let extra = FoldSchedule::RowsPerStep(4).partition(more_rows)?;
//! for step in extra {
//!     proof = extend(&prep, proof, step)?;
//! }
//!
//! // Finish folds the trailing latest into the running accumulator, retaining
//! // the terminal NIFS proof. The IVC chain is verifiable before Spartan.
//! proof = finish_uncompressed(&prep, proof)?;
//! verify_uncompressed(&prep, &proof)?;
//!
//! // `compress` / compressed `verify` are reserved for the PR5 Spartan decider
//! // and currently return an explicit unsupported error.
//! ```
//!
//! ## Where do `(z, m_in)` come from?
//!
//! The caller. ccs-direct is a generic frontend; it does not run a VM and
//! does not know what computation you are proving.
//!
//! - **`z = [x, w]`** — your satisfying assignment, length `structure.m`.
//! - **`m_in`** — split point: `z[..m_in] = x` (public), `z[m_in..] = w` (private).
//!
//! ## Where to start reading
//!
//! Auditor: start with [`paper/mod.rs`](crate::paper) for the glossary,
//! then follow the public lifecycle into `paper/`.

pub mod config;
pub mod engine;
pub mod frontends;
pub mod lifecycle;
pub mod paper;

// ── Public lifecycle re-exports. Keep this surface small. ─────────────────
pub use lifecycle::{
    compress, extend, finish_uncompressed, preprocess, prove, verify, verify_uncompressed, Compressed, Error,
    Preprocessing, PublicImage, Uncompressed,
};

pub use lifecycle::{FoldSchedule, ScheduleError};
pub use paper::construction2::{
    FinalFoldProof, FoldProof, LatestInstance, ProofState, RunningInstance, State, StepProof, VerifierKey,
};
pub use paper::params::Params;
pub use paper::relations::{CcsInstance, CcsWitness, CeClaim, DecMixer, RlcMixer, Structure};
