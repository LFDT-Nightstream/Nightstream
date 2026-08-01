//! `neo-fold-clean` — paper-faithful, audit-first SuperNeo IVC integrator.
//!
//! ## Public lifecycle (terminal-only IVC path)
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
//! // `prove` returns an `UncompressedAudit` — the in-flight proof,
//! // with the per-step audit trail attached because `extend` may want
//! // to push more batches into it.
//! let mut audit = prove(&prep, steps)?;
//!
//! // More rows arrive later? Partition them and `extend`.
//! let extra = FoldSchedule::RowsPerStep(4).partition(more_rows)?;
//! for step in extra {
//!     audit = extend(&prep, audit, step)?;
//! }
//!
//! // Finalize and drop the per-step audit trail. Authoritative plain F'
//! // preserves HyperNova's running/latest pair; Nebula and legacy one-chunk
//! // relations flush latest through a terminal fold. Generic direct CCS has
//! // no terminal-induction capability, so this compact path verifies only a
//! // single chunk.
//! let proof = finish_uncompressed(&prep, audit)?;
//! verify_uncompressed(&prep, &proof)?;
//!
//! // A Lean-native CCS relation can instead use `finish_with_spartan` and
//! // `verify_spartan`. That path lowers only the terminal relation to R1CS
//! // and uses WHIR inside repeated Spartan proofs. The terminal statement is
//! // explicit and separate from the proof:
//! // let terminal = finish_uncompressed(&prep, audit)?;
//! // let (statement, proof) = finish_with_spartan(&prep, &manifest, terminal)?;
//! // verify_spartan(&prep, &manifest, &expected, &statement, &proof)?;
//! ```
//!
//! ## Audit / decider path (diagnostic + Spartan)
//!
//! Tests, the Spartan decider statement, and chain-replay debugging
//! need the per-step audit trail kept. Use the `_audit` variants:
//!
//! ```ignore
//! use neo_fold_clean::{finish_uncompressed_with_audit, verify_uncompressed_audit};
//!
//! let audit_finalized = finish_uncompressed_with_audit(&prep, audit)?;
//!
//! // Terminal-only check on the projected proof. Whether multi-chunk
//! // verification is allowed is a capability of the preprocessing relation,
//! // not a property callers can opt into.
//! verify_uncompressed(&prep, &audit_finalized.proof)?;
//!
//! // Linear-time chain replay — catches audit-trail tampers (steps,
//! // public_batches, final_fold.nifs) that the IVC verifier ignores
//! // by design.
//! verify_uncompressed_audit(&prep, &audit_finalized)?;
//! ```
//!
//! Generic callers that need multi-chunk verification keep the audit trail
//! (`finish_uncompressed_with_audit` + `verify_uncompressed_audit`). The
//! authoritative Nebula F' chain may drop it after finalization because its
//! preprocessing certifies the folded induction. Reach for audit variants for
//! diagnostics, the Spartan decider statement, or red-team tests that mutate
//! audit-trail fields.
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
mod heap;
pub mod lifecycle;
pub mod paper;

// ── Public lifecycle re-exports. Keep this surface small. ─────────────────

// Terminal-only lifecycle path.
pub use lifecycle::{
    extend, finish_uncompressed, preprocess, prove, verify_uncompressed, Compressed, Error, Preprocessing, PublicImage,
    Uncompressed,
};

// Audit / decider path — chain-replay verifier, Spartan statement, diagnostic
// tests. See the crate-level docs for when each is appropriate; reach for
// these only when the production names above don't fit the use case.
pub use lifecycle::{
    build_decider_statement, compress, finish_uncompressed_with_audit, verify, verify_uncompressed_audit,
    UncompressedAudit,
};

pub use frontends::r1cs_f_prime::{
    finish_with_spartan, verify_spartan, LeanNativeCcsError, LeanNativeCcsManifest, LeanNativeCcsPreprocessing,
    TerminalR1csError, TerminalSpartanProof, TerminalSpartanStatement,
};
pub use lifecycle::{FoldSchedule, ScheduleError};
pub use paper::construction2::{
    FinalFoldProof, FoldProof, LatestInstance, ProofState, RunningInstance, State, StepProof, VerifierKey,
};
pub use paper::params::Params;
pub use paper::relations::{CcsInstance, CcsWitness, CeClaim, DecMixer, RlcMixer, Structure};
