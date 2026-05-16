//! Lifecycle wrapper for encoded-F' chains.
//!
//! [`prove_encoded_steps`] is the production entry point that takes a
//! sequence of encoded F' steps and drives `lifecycle::prove` end-to-end,
//! one step per batch. Returns the pre-finalize
//! [`UncompressedAudit`] (per-step `StepProof`s + `public_batches`
//! still attached); callers chain
//! `lifecycle::finish_uncompressed` (terminal-only output) or
//! `lifecycle::finish_uncompressed_with_audit` (keeps audit trail) as
//! needed.

use crate::frontends::fibonacci_f_prime::encoder::EncodedFibonacciFPrimeStep;
use crate::frontends::fibonacci_f_prime::instance::build_instance;
use crate::frontends::fibonacci_f_prime::{Error, FibonacciFPrimePreprocessing};
use crate::lifecycle::UncompressedAudit;
use crate::paper::relations::CcsInstance;

/// Fold a sequence of encoded F' steps through `lifecycle::prove`,
/// one step per batch.
///
/// Each step is converted into a `CcsInstance` via [`build_instance`],
/// which enforces that every step's CCS structure matches the
/// preprocessing's structure (so the chain folds a homogeneous
/// relation).
pub fn prove_encoded_steps(
    prep: &FibonacciFPrimePreprocessing,
    steps: &[EncodedFibonacciFPrimeStep],
) -> Result<UncompressedAudit, Error> {
    let mut batches: Vec<Vec<CcsInstance>> = Vec::with_capacity(steps.len());
    for step in steps {
        batches.push(vec![build_instance(prep, step)?]);
    }
    Ok(crate::lifecycle::prove(&prep.prep, batches)?)
}
