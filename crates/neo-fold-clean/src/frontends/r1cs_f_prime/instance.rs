//! Per-step `CcsInstance` construction for the R1CS-encoded-F' frontend.

use crate::frontends::f_prime_shell::encoder::EncodedFPrimeStep;
use crate::frontends::r1cs_f_prime::{Error, R1csFPrimePreprocessing};
use crate::paper::digest::structure_digest;
use crate::paper::relations::CcsInstance;

/// Build one `CcsInstance` from an encoded R1CS-F' step.
///
/// The step's CCS structure (`step.structure.ccs`) must match the
/// preprocessing's structure by `structure_digest`. The witness is
/// committed via `prep.log` under the canonical public-input split
/// (`step.public_input_len()` = `1 + boundary_bits`).
pub fn build_instance(prep: &R1csFPrimePreprocessing, step: &EncodedFPrimeStep) -> Result<CcsInstance, Error> {
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
