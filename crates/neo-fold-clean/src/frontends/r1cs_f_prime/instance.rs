//! Per-step `CcsInstance` construction for the R1CS-encoded-F' frontend.

use std::sync::Arc;

use crate::frontends::f_prime::encoder::EncodedFPrimeStep;
use crate::frontends::r1cs_f_prime::{Error, R1csFPrimePreprocessing};
use crate::paper::digest::structure_digest;
use crate::paper::relations::CcsInstance;

/// Build one `CcsInstance` from an encoded R1CS-F' step.
///
/// The step's CCS structure (`step.structure.ccs`) must match the
/// preprocessing's structure by `structure_digest`. The witness is
/// committed via `prep.log` under the canonical public-input split
/// (`step.public_input_len()` = `1 + boundary_bits`).
///
/// Fast path: if `step.structure` is the same [`Arc`] as
/// `prep.structure` (the normal in-chain case), the digest check is
/// skipped — pointer equality already implies value equality and saves
/// the four-limb Poseidon recompute over the full sparse structure.
pub fn build_instance(prep: &R1csFPrimePreprocessing, step: &EncodedFPrimeStep) -> Result<CcsInstance, Error> {
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
