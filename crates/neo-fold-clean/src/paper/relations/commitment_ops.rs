//! Commitment-action closure types the caller hands to Π_RLC and Π_DEC.
//!
//! These are not paper Definitions — they're API contracts for "how does
//! the commitment scheme combine commitments under a homomorphic action."
//! The semantics depend on the commitment scheme; for Ajtai with the
//! natural homomorphism, both are pure commitment math (the witness array
//! is unused).

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;

/// Π_RLC commitment mixer: `Σρ_i c_i` over the K+k input commitments.
/// User-supplied because the action depends on the commitment scheme; for
/// Ajtai with the natural homomorphism, the mixer ignores the witness array.
pub type RlcMixer = fn(&[Mat<F>], &[Commitment]) -> Commitment;

/// Π_DEC commitment combiner: `Σ b^{i-1} c_i` over the k child commitments.
pub type DecMixer = fn(&[Commitment], u32) -> Commitment;
