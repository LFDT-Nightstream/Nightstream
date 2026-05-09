//! Crate root for the active `neo-fold-next` proving path.
//!
//! The crate-level surface intentionally stays small: standard RV32IM proof
//! helpers and direct-CCS compression/verifier helpers live here, while
//! frontend-specific construction knobs stay under `rv32im` and `direct_ccs`.

pub mod circuit;
pub mod core;
pub mod decider;
pub mod frontends;
pub mod public_proof;
pub mod vm;

pub use self::frontends::{direct_ccs, rv32im};

/// Prove the standard RV32IM public proof from a prepared frontend input.
pub fn prove_rv32im(input: &rv32im::Rv32imProofInput) -> Result<rv32im::Rv32imProof, rv32im::SimpleKernelError> {
    rv32im::prove_rv32im_public_proof(input)
}

/// Verify an RV32IM proof against the input it claims to prove.
pub fn verify_rv32im(
    input: &rv32im::Rv32imProofInput,
    proof: &rv32im::Rv32imProof,
) -> Result<(), rv32im::SimpleKernelError> {
    rv32im::audit::audit_rv32im_public_proof_against_input(input, proof)
}

/// Compress a built terminal direct-CCS state into its verifier-facing SNARK.
pub fn prove_direct_ccs_terminal(
    state: &direct_ccs::DirectCcsIvcState,
) -> Result<
    (
        direct_ccs::DirectCcsIvcSnark,
        direct_ccs::DirectCcsIvcSnarkVerifierKey,
        direct_ccs::DirectCcsFPrimeSnarkPerf,
    ),
    direct_ccs::DirectCcsFPrimeSnarkError,
> {
    state.compress_snark()
}

/// Verify a terminal direct-CCS SNARK against its expected public image.
pub fn verify_direct_ccs_terminal(
    vk: &direct_ccs::DirectCcsIvcSnarkVerifierKey,
    expected_public_image: &direct_ccs::DirectCcsIvcPublicImage,
    snark: &direct_ccs::DirectCcsIvcSnark,
) -> Result<(), direct_ccs::DirectCcsFPrimeSnarkError> {
    snark.verify(vk, expected_public_image)
}

/// Compress a built recursive direct-CCS state, including carried F' authority.
pub fn prove_direct_ccs_recursive(
    state: &direct_ccs::DirectCcsRecursiveIvcState,
) -> Result<
    (
        direct_ccs::DirectCcsRecursiveIvcSnark,
        direct_ccs::DirectCcsRecursiveIvcSnarkVerifierKey,
        direct_ccs::DirectCcsRecursiveIvcSnarkPerf,
    ),
    direct_ccs::DirectCcsFPrimeSnarkError,
> {
    state.compress_recursive_snark()
}

/// Verify a recursive direct-CCS SNARK against its expected public image.
pub fn verify_direct_ccs_recursive(
    vk: &direct_ccs::DirectCcsRecursiveIvcSnarkVerifierKey,
    expected_public_image: &direct_ccs::DirectCcsRecursiveIvcPublicImage,
    snark: &direct_ccs::DirectCcsRecursiveIvcSnark,
) -> Result<(), direct_ccs::DirectCcsFPrimeSnarkError> {
    snark.verify(vk, expected_public_image)
}

pub(crate) use self::circuit::{superneo as superneo_circuit, superneo_nifs as superneo_nifs_circuit};
pub(crate) use self::core::multilinear;
pub(crate) use self::core::{
    chunk_folding, construction2, finalize, ivc, opening, proof, prover, session, step_build, verifier, witness_layout,
};
pub(crate) use self::decider::spartan_backend;
