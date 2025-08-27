//! # Neo Spartan Bridge: Last-Mile Compression Only
//!
//! This crate is the **only** place that sees FRI. Enforced invariants:
//! - **FRI confined to compression**: No FRI code in neo-fold or other crates
//! - **Real FRI only**: No simulated FRI - only the actual Spartan2+FRI implementation
//! - **One-way bridge**: Translates final ME(b,L) claims to Spartan2 linearized CCS

use neo_math::{F, ExtF};
use neo_ccs::MatrixEvaluation;

#[cfg(feature = "spartan2")]
mod spartan2_bridge;

/// Compress a final matrix evaluation claim to a succinct proof via Spartan2+FRI
#[cfg(feature = "spartan2")]
pub fn compress_to_spartan2(
    me_claim: &MatrixEvaluation,
    witness: &[F],
) -> Result<Vec<u8>, CompressionError> {
    spartan2_bridge::compress(me_claim, witness)
}

/// Verify a Spartan2 compressed proof
#[cfg(feature = "spartan2")]
pub fn verify_spartan2_proof(
    me_claim: &MatrixEvaluation,
    proof: &[u8],
) -> Result<bool, CompressionError> {
    spartan2_bridge::verify(me_claim, proof)
}

/// Placeholder when Spartan2 feature is disabled
#[cfg(not(feature = "spartan2"))]
pub fn compress_to_spartan2(
    _me_claim: &MatrixEvaluation,
    _witness: &[F],
) -> Result<Vec<u8>, CompressionError> {
    Err(CompressionError::FeatureDisabled("spartan2 feature not enabled".to_string()))
}

/// Error types for compression operations
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("Spartan2 compression failed: {0}")]
    Spartan2Failed(String),
    
    #[error("FRI proof generation failed: {0}")]
    FriFailed(String),
    
    #[error("Feature disabled: {0}")]
    FeatureDisabled(String),
}
