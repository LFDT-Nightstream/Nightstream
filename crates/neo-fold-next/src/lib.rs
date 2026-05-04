//! Owns the active Rust proving path for `neo-fold-next`.
//!
//! Ownership:
//! - `prover`, `verifier`: generic `Π_CCS -> Π_RLC -> Π_DEC`
//! - `construction2`: relation-neutral recursive public-image primitives
//! - `run`: session orchestration
//! - `ivc`: generic native SuperNeo IVC/NIFS accumulator carrier
//! - `proof`: generic session proof boundary
//! - `opening`: shared opening-claim and time-opening summary boundary
//! - `step_build`: frontend-produced step packaging and extension records
//! - `time_opening`, `finalize`: final opening and packaged-proof boundaries
//! - `witness_layout`: shared local packed witness layout helpers
//! - `vm`: static VM contracts
//! - `chip8`: current VM frontend and staged kernel

pub mod chip8;
pub mod chunk_relation;
pub mod construction2;
pub(crate) mod construction2_terminal;
pub mod decider;
pub mod direct_ccs;
pub mod finalize;
pub mod ivc;
pub mod nightstream;
pub mod opening;
pub mod proof;
pub mod prover;
pub mod run;
pub mod rv32im;
pub(crate) mod spartan_backend;
pub mod step_build;
pub(crate) mod superneo_circuit;
pub(crate) mod superneo_nifs_circuit;
pub mod time_opening;
pub mod verifier;
pub mod vm;
pub mod witness_layout;

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
