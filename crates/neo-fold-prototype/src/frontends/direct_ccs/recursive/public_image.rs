//! Owns the recursive public-image boundary checks and digest.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCcsRecursiveIvcPublicImage {
    pub terminal_public_image: DirectCcsIvcPublicImage,
    pub proven_accumulator_digest: [u8; 32],
    pub proven_f_prime_accumulator_digest: [u8; 32],
    pub f_prime_accumulator_base: u32,
    pub proven_chunk_count: u64,
    pub proven_step_count: u64,
    pub folded_f_prime_r2_steps: u64,
    pub f_prime_final_ce_claims: u64,
}

impl DirectCcsRecursiveIvcPublicImage {
    pub fn from_terminal_and_f_prime_accumulator(
        terminal_public_image: DirectCcsIvcPublicImage,
        proven_f_prime_accumulator_digest: [u8; 32],
        f_prime_accumulator_base: u32,
        folded_f_prime_r2_steps: u64,
        f_prime_final_ce_claims: u64,
    ) -> Result<Self, DirectCcsFPrimeSnarkError> {
        terminal_public_image
            .validate_final_construction2_public_boundary()
            .map_err(DirectCcsFPrimeSnarkError::Verify)?;
        if f_prime_accumulator_base < 2 {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        let image = Self {
            proven_accumulator_digest: terminal_public_image.accumulator_out_digest,
            proven_f_prime_accumulator_digest,
            f_prime_accumulator_base,
            proven_chunk_count: terminal_public_image.chunk_count_out,
            proven_step_count: terminal_public_image.step_count_out,
            folded_f_prime_r2_steps,
            f_prime_final_ce_claims,
            terminal_public_image,
        };
        image.validate_recursive_boundary()?;
        Ok(image)
    }

    pub fn validate_recursive_boundary(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        self.terminal_public_image
            .validate_final_construction2_public_boundary()
            .map_err(DirectCcsFPrimeSnarkError::Verify)?;
        if self.proven_chunk_count != self.terminal_public_image.chunk_count_out {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.proven_step_count != self.terminal_public_image.step_count_out {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.proven_accumulator_digest != self.terminal_public_image.accumulator_out_digest {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.proven_f_prime_accumulator_digest != self.terminal_public_image.construction2_accumulator_digest {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.folded_f_prime_r2_steps.checked_add(1) != Some(self.proven_chunk_count) {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.f_prime_final_ce_claims == 0 {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        if self.f_prime_accumulator_base < 2 {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        Ok(())
    }

    pub fn expected_digest(&self) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        self.validate_recursive_boundary()?;
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/recursive_ivc_public_image");
        tr.append_message(b"neo.fold.next/direct_ccs/recursive_ivc_public_image/version", b"v1");
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_public_image/terminal",
            &self.terminal_public_image.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_public_image/accumulator",
            &self.proven_accumulator_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_public_image/f_prime_accumulator",
            &self.proven_f_prime_accumulator_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/recursive_ivc_public_image/counters",
            &[
                self.proven_chunk_count,
                self.proven_step_count,
                self.folded_f_prime_r2_steps,
                self.f_prime_final_ce_claims,
                self.f_prime_accumulator_base as u64,
            ],
        );
        Ok(tr.digest32())
    }
}
