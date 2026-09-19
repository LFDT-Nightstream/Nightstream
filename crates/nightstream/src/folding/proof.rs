//! The three native subproofs checked by the recursive verifier.

use super::{pi_ccs, pi_dec, pi_rlc};

#[derive(Clone, Debug, PartialEq)]
pub struct NifsProof {
    pub pi_ccs: pi_ccs::Proof,
    pub pi_rlc: pi_rlc::Proof,
    pub pi_dec: pi_dec::Proof,
}

#[cfg(test)]
#[path = "../../tests/lifecycle_native/proof_encoding.rs"]
mod encoding;
