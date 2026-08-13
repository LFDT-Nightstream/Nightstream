//! Verifier-key digest types used by direct Spartan proofs.

use crate::{
  errors::SpartanError,
  traits::{Engine, Group, TranscriptReprTrait},
};

/// A Poseidon2 digest of a Spartan verifier key.
pub type SpartanDigest = [u8; 32];

/// Computes the digest bound into a Spartan proof transcript.
pub trait DigestHelperTrait<E: Engine> {
  /// Return the verifier-key digest.
  fn digest(&self) -> Result<SpartanDigest, SpartanError>;
}

impl<G: Group> TranscriptReprTrait<G> for SpartanDigest {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_vec()
  }
}
