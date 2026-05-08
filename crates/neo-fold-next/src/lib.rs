//! Owns the active Rust proving path for `neo-fold-next`.
//!
//! Ownership:
//! - `core`: shared SuperNeo, Construction-2, proof, opening, and run plumbing
//! - `circuit`: shared Bellpepper/SuperNeo gadgets
//! - `frontends::direct_ccs`: first-class direct CCS/R1CS frontend
//! - `frontends::rv32im`: first-class RV32IM frontend
//! - `public_proof`: published proof boundary and frontend adapters
//! - `decider`: Spartan/decider wrappers
//! - `vm`: static VM contracts

pub mod circuit;
pub mod core;
pub mod decider;
pub mod frontends;
pub mod public_proof;
pub mod vm;

pub(crate) use self::circuit::{superneo as superneo_circuit, superneo_nifs as superneo_nifs_circuit};
pub use self::core::{
    chunk_relation, construction2, finalize, ivc, opening, proof, prover, run, step_build, time_opening, verifier,
    witness_layout,
};
pub(crate) use self::core::{construction2_terminal, multilinear};
pub(crate) use self::decider::spartan_backend;
pub use self::frontends::{direct_ccs, rv32im};

pub use direct_ccs::{
    direct_ccs_program_from_sparse_r1cs, direct_ccs_program_from_sparse_r1cs_with_public_input_len,
    direct_ccs_step_from_low_norm_full_witness, direct_sparse_r1cs_export_from_spartan_circuit,
    export_latest_direct_ccs_f_prime_verifier_body_r1cs, lower_sparse_r1cs_export_to_low_norm,
    lower_sparse_r1cs_export_to_low_norm_program_and_step, measure_latest_direct_ccs_f_prime_verifier_body,
    verify_direct_ccs_ivc_snark, verify_direct_ccs_ivc_snark_public, verify_direct_ccs_recursive_ivc_snark_public,
    verify_direct_ccs_statement, DirectCcsCompactFPrimeImage, DirectCcsFPrimeLowNormSourceImage,
    DirectCcsFPrimeLowNormSourceR1cs, DirectCcsFPrimeLowNormSourceR1csShape, DirectCcsFPrimeSnarkError,
    DirectCcsFPrimeSnarkPerf, DirectCcsFPrimeSnarkProof, DirectCcsFPrimeVerifierBodyShape, DirectCcsIvcPublicImage,
    DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey, DirectCcsIvcState, DirectCcsLatestFPrimeSummary,
    DirectCcsNativeFPrimeAdvice, DirectCcsNativeFPrimeStepImage, DirectCcsProgram, DirectCcsRecursiveIvcPublicImage,
    DirectCcsRecursiveIvcSnark, DirectCcsRecursiveIvcSnarkPerf, DirectCcsRecursiveIvcSnarkVerifierKey,
    DirectCcsRecursiveIvcState, DirectCcsRecursiveIvcSummary, DirectCcsStatement, DirectCcsStep, DirectLowNormLaneKind,
    DirectR1csLowNormLayout, DirectSparseR1csExport, DirectSparseR1csLowNormReport, DirectSparseR1csLowNormViolation,
    DIRECT_CCS_TRIVIAL_PC,
};
