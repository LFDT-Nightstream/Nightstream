//! Owns the reusable direct-CCS terminal SNARK wrapper and verifier key.

use std::sync::Arc;

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use spartan2::traits::snark::DigestHelperTrait;

use super::public_image::{DirectCcsIvcPublicImage, DirectCcsStatement};
use super::state::{DirectCcsFPrimeSnarkError, DirectCcsFPrimeSnarkProof};
use super::verify::verify_direct_ccs_ivc_snark_public;
use crate::spartan_backend::NeoFoldDeciderVerifierKey;

pub struct DirectCcsIvcSnarkVerifierKey {
    pub(crate) terminal_f_prime: Arc<NeoFoldDeciderVerifierKey>,
}

impl DirectCcsIvcSnarkVerifierKey {
    pub(crate) fn from_terminal_f_prime(terminal_f_prime: Arc<NeoFoldDeciderVerifierKey>) -> Self {
        Self { terminal_f_prime }
    }

    pub(crate) fn terminal_f_prime(&self) -> &NeoFoldDeciderVerifierKey {
        &self.terminal_f_prime
    }

    pub fn expected_digest(&self) -> Result<[u8; 32], DirectCcsFPrimeSnarkError> {
        let terminal_f_prime_digest = self
            .terminal_f_prime
            .digest()
            .map_err(|err| DirectCcsFPrimeSnarkError::Encode(err.to_string()))?;
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/ivc_snark_verifier_key");
        tr.append_message(b"neo.fold.next/direct_ccs/ivc_snark_verifier_key/version", b"v1");
        tr.append_message(
            b"neo.fold.next/direct_ccs/ivc_snark_verifier_key/terminal_f_prime",
            &terminal_f_prime_digest,
        );
        Ok(tr.digest32())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectCcsIvcSnark {
    proof: DirectCcsFPrimeSnarkProof,
    public_image: DirectCcsIvcPublicImage,
}

impl DirectCcsIvcSnark {
    pub(crate) fn from_parts(proof: DirectCcsFPrimeSnarkProof, public_image: DirectCcsIvcPublicImage) -> Self {
        Self { proof, public_image }
    }

    pub fn proof(&self) -> &DirectCcsFPrimeSnarkProof {
        &self.proof
    }

    pub fn proof_mut(&mut self) -> &mut DirectCcsFPrimeSnarkProof {
        &mut self.proof
    }

    pub fn public_image(&self) -> &DirectCcsIvcPublicImage {
        &self.public_image
    }

    pub fn statement(&self) -> DirectCcsStatement {
        self.public_image.statement()
    }

    pub fn public_image_mut(&mut self) -> &mut DirectCcsIvcPublicImage {
        &mut self.public_image
    }

    pub fn verify(
        &self,
        vk: &DirectCcsIvcSnarkVerifierKey,
        expected_public_image: &DirectCcsIvcPublicImage,
    ) -> Result<(), DirectCcsFPrimeSnarkError> {
        if &self.public_image != expected_public_image {
            return Err(DirectCcsFPrimeSnarkError::PublicIoMismatch);
        }
        verify_direct_ccs_ivc_snark_public(vk, expected_public_image, &self.proof)
    }
}
