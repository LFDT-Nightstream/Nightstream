//! Per-step `CcsInstance` construction for the encoded-F' frontend.
//!
//! Unlike [`neo_fold_clean::frontends::direct_ccs::instance::build_instance`]
//! (which builds an instance from a satisfying R1CS assignment), this
//! entry takes a fully-encoded `EncodedFPrimeStep` — the
//! encoder has already produced the strict low-norm witness — and
//! commits it under the matched preprocessing.

use super::{Error, FibonacciFPrimePreprocessing};
use neo_fold_clean::frontends::f_prime::encoder::EncodedFPrimeStep;
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::paper::relations::CcsInstance;
use std::sync::Arc;

/// Build one `CcsInstance` from an encoded F' step.
///
/// The step's CCS structure (`step.structure.ccs`) must match the
/// preprocessing's structure by `structure_digest` — every step in one
/// chain must share the same CCS shape. The witness is committed via
/// `prep.log` under the canonical public-input split
/// (`step.public_input_len()` is the `D`-aligned SuperNeo carrier containing
/// `[1 | boundary_bits | fixed-zero padding]`).
pub fn build_instance(prep: &FibonacciFPrimePreprocessing, step: &EncodedFPrimeStep) -> Result<CcsInstance, Error> {
    if !Arc::ptr_eq(&prep.structure, &step.structure) {
        let prep_digest = *prep.prep.structure_digest();
        let step_digest = structure_digest(&step.structure.ccs);
        if prep_digest != step_digest {
            return Err(Error::StructureMismatch {
                prep_digest,
                step_digest,
            });
        }
    }

    Ok(step.to_public_ccs_instance(&prep.prep.params, &prep.prep.log)?)
}
