//! Per-step `CcsInstance` construction for the direct-CCS frontend.
//!
//! `build_instance(prep, r1cs, z)` validates that `z` satisfies the R1CS,
//! checks `‖z‖_∞ < b`, packs `z` into the Ajtai-shaped `Z` matrix, and
//! commits via `prep.log`. The R1CS satisfaction check happens *here*
//! (before commit) so witness bugs surface as
//! `FrontendError::Unsatisfied` instead of as cryptic Π_CCS sumcheck
//! failures later in the pipeline.

use neo_math::F;

use crate::frontends::direct_ccs::r1cs::R1cs;
use crate::frontends::direct_ccs::FrontendError;
use crate::lifecycle::Preprocessing;
use crate::paper::digest;
use crate::paper::relations::CcsInstance;

/// Build one `CcsInstance` from a satisfying R1CS assignment.
///
/// Order of validation:
/// 1. R1CS shape (`r1cs.validate_shape()`).
/// 2. Frontend/preprocessing consistency (`m_in` and R1CS-derived structure).
/// 3. Assignment length.
/// 4. R1CS satisfaction row-by-row.
/// 5. Norm bound (`‖z‖_∞ < pp.b()`, enforced by the paper-layer
///    `from_low_norm_assignment`).
///
/// On success returns a `CcsInstance` with a real Ajtai commitment in
/// `claim.c` and the packed witness in `witness.Z`, ready to fold.
pub fn build_instance(prep: &Preprocessing, r1cs: &R1cs, z: &[F]) -> Result<CcsInstance, FrontendError> {
    r1cs.validate_shape()?;
    ensure_preprocessing_matches_r1cs(prep, r1cs)?;
    r1cs.is_satisfied_by(z)?;
    let instance = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), z, r1cs.m_in)?;
    Ok(instance)
}

fn ensure_preprocessing_matches_r1cs(prep: &Preprocessing, r1cs: &R1cs) -> Result<(), FrontendError> {
    if prep.public_input_len != Some(r1cs.m_in) {
        return Err(FrontendError::PreprocessingPublicInputMismatch {
            r1cs_m_in: r1cs.m_in,
            prep_m_in: prep.public_input_len,
        });
    }

    // Prover/developer guardrail only: this digest is not verifier authority.
    // The verifier's real soundness check is Π_CCS over `prep.structure`.
    // Here we merely catch the local mistake "preprocess R1CS A, build an
    // instance with R1CS B" before committing a witness under the wrong
    // relation. We compare the full CCS structure digest ({M_j}, f), but do
    // not generalize this into a protocol proof boundary.
    let expected_structure = r1cs.to_structure();
    if digest::structure_digest(&expected_structure) != *prep.structure_digest() {
        return Err(FrontendError::PreprocessingStructureMismatch);
    }

    Ok(())
}
