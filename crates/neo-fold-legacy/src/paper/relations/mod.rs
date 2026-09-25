//! SuperNeo §7.1 — Relations.
//!
//! Definitions 11–13: structure, CCS, CE. Each one gets its own file with
//! the type alias re-exported here. The user-facing `CcsInstance`
//! constructor lives in `instance.rs`; the closure types Π_RLC and Π_DEC
//! consume from the caller live in `commitment_ops.rs`.
//!
//! ## What this module owns
//!
//! - `Structure`, `CcsClaim`, `CcsWitness`, `CeClaim` — paper-named type
//!   aliases over `neo-ccs` (so paper layer and engine speak one wire format).
//! - `CcsInstance` — the (claim, witness) pair the caller hands to NIFS,
//!   plus its low-norm constructor.
//! - `RlcMixer`, `DecMixer` — low-level homomorphic-action function pointer
//!   types, plus the canonical Ajtai action used by lifecycle preprocessing.
//! - `WitnessMat` — `Mat<F>`, the Z matrix that flows through the protocol.
//!
//! ## What this module does *not* own
//!
//! - Matrix/poly arithmetic (lives in `neo-math`, `neo-ccs`).
//! - The Ajtai homomorphism ℒ (lives in `neo-ajtai`).
//! - The running accumulator U_i (`RunningInstance` lives in
//!   `paper::construction2`; that's an IVC concept, not a paper definition).

pub mod ccs;
pub mod ce;
pub mod commitment_ops;
pub mod instance;
pub mod lanes;
pub mod product_commitment_circuit;

use neo_ccs::Mat;
use neo_math::F;
use thiserror::Error;

// Re-exports — flat surface for consumers, structured layout for auditors.
pub use ccs::{CcsClaim, CcsWitness, Structure};
pub use ce::CeClaim;
pub use commitment_ops::{
    ajtai_dec_mixer, ajtai_rlc_mixer, mix_adv, recompose_adv, validate_adv_shape, AdvPresenceError, DecMixer, RlcMixer,
};
pub use instance::CcsInstance;
pub use lanes::{LaneRanges, LaneScheme, LaneSchemeError};
pub use neo_ccs::superneo_public_x_cols;

/// Witness matrix Z used both as the CCS decomposition and as the carried
/// witness for CE claims after Π_CCS / Π_RLC / Π_DEC.
pub type WitnessMat = Mat<F>;

/// Check the exact SuperNeo coefficient embedding for a public field vector.
pub fn superneo_has_canonical_x_shape(x: &Mat<F>, m_in: usize) -> bool {
    m_in % neo_math::D == 0 && x.rows() == neo_math::D && x.cols() == superneo_public_x_cols(m_in)
}

#[derive(Debug, Error)]
pub enum RelationError {
    #[error("CCS instance: assignment length {got} \u{2260} structure.m {expected}")]
    AssignmentLength { got: usize, expected: usize },
    #[error("CCS instance: m_in {m_in} > assignment length {len}")]
    MInOutOfRange { m_in: usize, len: usize },
    #[error("CCS instance: public-input length m_in={m_in} is not a whole degree-{d} ring block")]
    PublicInputNotWholeRing { m_in: usize, d: usize },
    #[error("CCS instance: \u{2016}z\u{2016}_\u{221E} \u{2265} b at index {idx} (b = {b})")]
    NormBoundViolated { idx: usize, b: u32 },
}
