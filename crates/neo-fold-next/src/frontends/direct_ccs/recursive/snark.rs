//! Owns recursive SNARK accessors and verifier-side boundary checks.

use super::*;

impl DirectCcsRecursiveIvcSnark {
    pub fn public_image(&self) -> &DirectCcsRecursiveIvcPublicImage {
        &self.public_image
    }

    pub fn terminal_snark(&self) -> &DirectCcsIvcSnark {
        &self.terminal
    }

    pub fn f_prime_final_claims(&self) -> &[CeClaim<Commitment, F, K>] {
        &self.f_prime_final_claims
    }

    pub fn f_prime_chain_snark(&self) -> Option<&DirectCcsIvcSnark> {
        self.f_prime_chain.as_ref()
    }

    pub fn verify(
        &self,
        vk: &DirectCcsRecursiveIvcSnarkVerifierKey,
        expected_public_image: &DirectCcsRecursiveIvcPublicImage,
    ) -> Result<(), DirectCcsFPrimeSnarkError> {
        expected_public_image.validate_recursive_boundary()?;
        if &self.public_image != expected_public_image {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        self.public_image.validate_recursive_boundary()?;
        if expected_public_image.f_prime_accumulator_base != vk.expected_f_prime_accumulator_base
            || expected_public_image.f_prime_final_ce_claims != vk.expected_f_prime_final_ce_claims
        {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        self.terminal
            .verify(&vk.terminal, &expected_public_image.terminal_public_image)?;
        match (
            expected_public_image.folded_f_prime_r2_steps,
            self.f_prime_chain.as_ref(),
            vk.f_prime_chain.as_ref(),
        ) {
            (0, None, None) => {
                if expected_public_image.proven_f_prime_accumulator_digest
                    != vk.expected_f_prime_default_accumulator_digest
                {
                    return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
                }
            }
            (0, _, _) => return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch),
            (folded_steps, Some(chain_snark), Some(chain_vk)) => {
                chain_snark.verify(chain_vk, chain_snark.public_image())?;
                let chain_image = chain_snark.public_image();
                if chain_image.accumulator_out_digest != expected_public_image.proven_f_prime_accumulator_digest
                    || chain_image.chunk_count_out != folded_steps
                    || chain_image.step_count_out != folded_steps
                {
                    return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
                }
            }
            _ => return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch),
        }
        match (
            expected_public_image.folded_f_prime_r2_steps,
            self.f_prime_final_claims.as_slice(),
            self.f_prime_final_ce_proof.as_ref(),
            vk.f_prime_final_ce.as_ref(),
        ) {
            (0, [], None, None) => Ok(()),
            (0, _, _, _) => Err(DirectCcsFPrimeSnarkError::PublicIoMismatch),
            (_, claims, Some(proof), Some(final_ce_vk)) => {
                if claims.len() as u64 != expected_public_image.f_prime_final_ce_claims {
                    return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
                }
                let digest = direct_accumulator_digest_from_claims_with_base(
                    expected_public_image.f_prime_accumulator_base,
                    claims,
                );
                if digest != expected_public_image.proven_f_prime_accumulator_digest {
                    return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
                }
                verify_direct_ce_bundle_relation(final_ce_vk, claims, proof)
            }
            _ => Err(DirectCcsFPrimeSnarkError::PublicIoMismatch),
        }
    }
}

pub fn verify_direct_ccs_recursive_ivc_snark_public(
    vk: &DirectCcsRecursiveIvcSnarkVerifierKey,
    expected_public_image: &DirectCcsRecursiveIvcPublicImage,
    snark: &DirectCcsRecursiveIvcSnark,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    snark.verify(vk, expected_public_image)
}
