//! Owns the generic non-VM direct CCS/R1CS IVC compression path.
//!
//! Frontends provide same-shape CCS steps. This module threads them through
//! native SuperNeo and compresses the latest Construction-2 F' step with
//! Spartan, without depending on RV32IM machine semantics.

mod ce_bundle;
mod circuit_util;
mod construction2_fold;
mod f_prime;
mod f_prime_chain;
mod f_prime_r1cs;
mod f_prime_r1cs_poseidon;
mod f_prime_verifier_body;
mod final_ce;
mod ivc;
mod ivc_helpers;
mod public_image;
mod r1cs;
mod r1cs_export;
mod r1cs_low_norm;
mod recursive;
mod snark;
mod surface;
mod terminal_committed;
mod terminal_measure;
mod verify;
mod zero_carry;

pub use f_prime::{
    DirectCcsCompactFPrimeImage, DirectCcsFPrimeLowNormSourceImage, DirectCcsNativeFPrimeAdvice,
    DirectCcsNativeFPrimeStepImage,
};
pub use f_prime_r1cs::{DirectCcsFPrimeLowNormSourceR1cs, DirectCcsFPrimeLowNormSourceR1csShape};
pub use f_prime_verifier_body::{
    export_latest_direct_ccs_f_prime_verifier_body_r1cs, measure_latest_direct_ccs_f_prime_verifier_body,
    DirectCcsFPrimeVerifierBodyShape,
};
pub use ivc::{
    verify_direct_ccs_ivc_snark, DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkPerf, DirectCcsFPrimeSnarkProof,
    DirectCcsIvcState, DirectCcsLatestFPrimeSummary, DirectCcsProgram, DirectCcsStep,
};
pub use public_image::{DirectCcsIvcPublicImage, DirectCcsStatement, DIRECT_CCS_TRIVIAL_PC};
pub use r1cs::{
    direct_ccs_program_from_sparse_r1cs, direct_ccs_program_from_sparse_r1cs_with_public_input_len,
    direct_ccs_step_from_low_norm_full_witness,
};
pub use r1cs_export::{
    direct_sparse_r1cs_export_from_spartan_circuit, DirectSparseR1csExport, DirectSparseR1csLowNormReport,
    DirectSparseR1csLowNormViolation,
};
pub use r1cs_low_norm::{
    lower_sparse_r1cs_export_to_low_norm, lower_sparse_r1cs_export_to_low_norm_program_and_step, DirectLowNormLaneKind,
    DirectR1csLowNormLayout,
};
pub use recursive::{
    verify_direct_ccs_recursive_ivc_snark_public, DirectCcsRecursiveIvcPublicImage, DirectCcsRecursiveIvcSnark,
    DirectCcsRecursiveIvcSnarkPerf, DirectCcsRecursiveIvcSnarkVerifierKey, DirectCcsRecursiveIvcState,
    DirectCcsRecursiveIvcSummary,
};
pub use snark::{DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey};
pub use terminal_committed::DirectCcsTerminalCommittedConstraintBreakdown;
pub use verify::{verify_direct_ccs_ivc_snark_public, verify_direct_ccs_statement};
