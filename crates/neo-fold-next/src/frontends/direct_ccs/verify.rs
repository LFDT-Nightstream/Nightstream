//! Owns verifier-facing checks for direct CCS/R1CS terminal proofs.

use super::public_image::{DirectCcsIvcPublicImage, DirectCcsStatement};
use super::snark::DirectCcsIvcSnarkVerifierKey;
use super::state::{DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkProof};
use super::terminal::committed::verify_direct_ccs_terminal_committed_relation;

pub fn verify_direct_ccs_ivc_snark_public(
    vk: &DirectCcsIvcSnarkVerifierKey,
    expected_public_image: &DirectCcsIvcPublicImage,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    verify_direct_ccs_statement(vk, &expected_public_image.statement(), proof)
}

pub fn verify_direct_ccs_statement(
    vk: &DirectCcsIvcSnarkVerifierKey,
    statement: &DirectCcsStatement,
    proof: &DirectCcsFPrimeSnarkProof,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    statement
        .validate_final_construction2_public_boundary()
        .map_err(DirectCcsFPrimeSnarkError::Verify)?;
    if proof.construction2_u_i != statement.construction2_u_i {
        return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
    }
    verify_direct_ccs_terminal_committed_relation(
        vk.terminal_f_prime(),
        &statement.terminal_public_values(),
        &statement.construction2_u_i,
        &proof.terminal_f_prime_committed_step_proof,
    )
    .map_err(|err| DirectCcsFPrimeSnarkError::Verify(err.to_string()))
}
