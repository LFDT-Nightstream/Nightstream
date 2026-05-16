//! Per-step `CcsInstance` construction for the encoded-F' frontend.
//!
//! Unlike [`crate::frontends::direct_ccs::instance::build_instance`]
//! (which builds an instance from a satisfying R1CS assignment), this
//! entry takes a fully-encoded `EncodedFPrimeStep` — the
//! encoder has already produced the strict low-norm witness — and
//! commits it under the matched preprocessing.

use crate::frontends::f_prime_shell::encoder::EncodedFPrimeStep;
use crate::frontends::fibonacci_f_prime::{Error, FibonacciFPrimePreprocessing};
use crate::paper::digest::structure_digest;
use crate::paper::relations::CcsInstance;

/// Build one `CcsInstance` from an encoded F' step.
///
/// The step's CCS structure (`step.structure.ccs`) must match the
/// preprocessing's structure by `structure_digest` — every step in one
/// chain must share the same CCS shape. The witness is committed via
/// `prep.log` under the canonical public-input split
/// (`step.public_input_len()` = `1 + boundary_bits`).
pub fn build_instance(prep: &FibonacciFPrimePreprocessing, step: &EncodedFPrimeStep) -> Result<CcsInstance, Error> {
    let prep_digest = structure_digest(&prep.prep.structure);
    let step_digest = structure_digest(&step.structure.ccs);
    if prep_digest != step_digest {
        return Err(Error::StructureMismatch {
            prep_digest,
            step_digest,
        });
    }

    Ok(step.to_public_ccs_instance(&prep.prep.params, &prep.prep.log)?)
}
