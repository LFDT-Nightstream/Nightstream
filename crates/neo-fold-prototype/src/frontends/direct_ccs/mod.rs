//! Owns the generic non-VM direct CCS/R1CS IVC compression path.
//!
//! Start here for the direct frontend public API. Implementation ownership is
//! split below by adapter, native state, F', final-state proof components, and
//! recursive proof.

mod adapter;
mod f_prime;
mod public_image;
mod recursive;
mod snark;
mod state;
mod step;
mod terminal;
mod verify;

pub(crate) type DirectCcsProofState = recursive::DirectCcsRecursiveIvcState;

pub use adapter::{
    direct_ccs_program_from_sparse_r1cs, direct_ccs_program_from_sparse_r1cs_with_public_input_len,
    direct_sparse_r1cs_export_from_spartan_circuit, lower_sparse_r1cs_export_to_low_norm,
    lower_sparse_r1cs_export_to_low_norm_program_and_step, DirectLowNormLaneKind, DirectR1csLowNormLayout,
    DirectSparseR1csExport, DirectSparseR1csLowNormReport, DirectSparseR1csLowNormViolation,
};
pub use f_prime::{
    export_latest_direct_ccs_f_prime_verifier_body_r1cs, measure_latest_direct_ccs_f_prime_verifier_body,
    DirectCcsCompactFPrimeImage, DirectCcsFPrimeLowNormSourceImage, DirectCcsFPrimeLowNormSourceR1cs,
    DirectCcsFPrimeLowNormSourceR1csShape, DirectCcsFPrimeVerifierBodyNifsShape, DirectCcsFPrimeVerifierBodyShape,
    DirectCcsNativeFPrimeAdvice, DirectCcsNativeFPrimeStepImage,
};
pub use public_image::{DirectCcsIvcPublicImage, DirectCcsStatement, DIRECT_CCS_TRIVIAL_PC};
pub use recursive::{
    start_direct_ccs_proof_state, DirectCcsFPrimeLowNormSourceR1csSummary, DirectCcsFPrimeLowNormSourceSummary,
    DirectCcsFPrimeVerifierBodySummary, DirectCcsFPrimeVerifierNifsSummary, DirectCcsRecursiveFPrimeSummary,
    DirectCcsRecursiveIvcPublicImage, DirectCcsRecursiveIvcSnark, DirectCcsRecursiveIvcSnarkPerf,
    DirectCcsRecursiveIvcSnarkVerifierKey, DirectCcsRecursiveIvcState, DirectCcsRecursiveIvcSummary,
    DirectCcsRecursiveProofSummary, DirectCcsRecursiveSemanticSummary,
};
pub use snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
pub use state::{
    DirectCcsFPrimeChunkPerf, DirectCcsFPrimeCommittedPerf, DirectCcsFPrimeCommittedSourcePerf,
    DirectCcsFPrimeConstraintPerf, DirectCcsFPrimeFinalCePerf, DirectCcsFPrimeProofSizePerf, DirectCcsFPrimeR1csPerf,
    DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf, DirectCcsFPrimeSnarkProof, DirectCcsFPrimeTimingPerf,
    DirectCcsIvcState, DirectCcsLatestFPrimeSummary, DirectCcsProgram,
};
pub use step::{direct_ccs_step_from_low_norm_full_witness, DirectCcsStep};
pub use terminal::DirectCcsTerminalCommittedConstraintBreakdown;
pub use verify::verify_direct_ccs_statement;
