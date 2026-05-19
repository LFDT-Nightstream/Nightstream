//! Owns recursive SNARK accessors and verifier-side boundary checks.

use super::*;

pub struct DirectCcsRecursiveIvcSnarkVerifierKey {
    pub(crate) terminal: DirectCcsIvcSnarkVerifierKey,
    pub(crate) f_prime_chain: Option<DirectCcsIvcSnarkVerifierKey>,
    pub(crate) f_prime_final_ce: Option<DirectCcsCeBundleVerifierKey>,
    pub(crate) expected_f_prime_default_accumulator_digest: [u8; 32],
    pub(crate) expected_f_prime_accumulator_base: u32,
    pub(crate) expected_f_prime_final_ce_claims: u64,
}

impl DirectCcsRecursiveIvcSnarkVerifierKey {
    pub fn expected_digest(&self) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        let terminal_digest = self.terminal.expected_digest()?;
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key");
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/version",
            b"v1",
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/terminal",
            &terminal_digest,
        );
        tr.append_message(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/default_f_prime_accumulator",
            &self.expected_f_prime_default_accumulator_digest,
        );
        match self.f_prime_chain.as_ref() {
            Some(vk) => {
                let f_prime_chain_digest = vk.expected_digest()?;
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_chain",
                    &[1],
                );
                tr.append_message(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/f_prime_chain",
                    &f_prime_chain_digest,
                );
            }
            None => {
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_chain",
                    &[0],
                );
            }
        }
        match self.f_prime_final_ce.as_ref() {
            Some(vk) => {
                let final_ce_digest = vk
                    .digest()
                    .map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?;
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_final_ce",
                    &[1],
                );
                tr.append_message(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/f_prime_final_ce",
                    &final_ce_digest,
                );
            }
            None => {
                tr.append_u64s(
                    b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/has_f_prime_final_ce",
                    &[0],
                );
            }
        }
        tr.append_u64s(
            b"neo.fold.next/direct_ccs/recursive_ivc_snark_verifier_key/f_prime_shape",
            &[
                self.expected_f_prime_accumulator_base as u64,
                self.expected_f_prime_final_ce_claims,
            ],
        );
        Ok(tr.digest32())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectCcsRecursiveIvcSnark {
    pub(crate) terminal: DirectCcsIvcSnark,
    pub(crate) f_prime_chain: Option<DirectCcsIvcSnark>,
    pub(crate) f_prime_final_claims: Vec<CeClaim<Commitment, F, K>>,
    pub(crate) f_prime_final_ce_proof: Option<DirectCcsCeBundleProof>,
    pub(crate) public_image: DirectCcsRecursiveIvcPublicImage,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DirectCcsRecursiveIvcSnarkPerf {
    pub terminal: DirectCcsFPrimeSnarkPerf,
    pub f_prime_chain: Option<DirectCcsFPrimeSnarkPerf>,
    pub f_prime_chain_setup_ms: f64,
    pub f_prime_chain_prove_ms: f64,
    pub f_prime_chain_verify_ms: f64,
    pub f_prime_chain_constraints: usize,
    pub f_prime_chain_proof_bytes: usize,
    pub f_prime_final_ce_setup_ms: f64,
    pub f_prime_final_ce_prove_ms: f64,
    pub f_prime_final_ce_verify_ms: f64,
    pub f_prime_final_ce_constraints: usize,
    pub f_prime_final_ce_digest_constraints: usize,
    pub f_prime_final_ce_digest_match_constraints: usize,
    pub f_prime_final_ce_relation_constraints: usize,
    pub f_prime_final_ce_public_inputs: usize,
    pub f_prime_final_ce_claims: usize,
    pub total_prove_ms: f64,
    pub total_verify_ms: f64,
    pub terminal_proof_bytes: usize,
    pub f_prime_final_ce_proof_bytes: usize,
    pub total_proof_bytes: usize,
}

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
