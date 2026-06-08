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

use neo_ccs::Mat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

// Re-exports — flat surface for consumers, structured layout for auditors.
pub use ccs::{CcsClaim, CcsWitness, Structure};
pub use ce::CeClaim;
pub use commitment_ops::{ajtai_dec_mixer, ajtai_rlc_mixer, DecMixer, RlcMixer};
pub use instance::CcsInstance;

/// Witness matrix Z used both as the CCS decomposition and as the carried
/// witness for CE claims after Π_CCS / Π_RLC / Π_DEC.
pub type WitnessMat = Mat<F>;

/// Number of *active* (non-structurally-zero) ring columns in the SuperNeo-
/// packed `X` matrix of a CCS instance with public-input length `m_in`.
///
/// The instance's full `X` is a `D × m_in` field matrix, but
/// `project_x_from_witness_mat` only populates columns `0..ceil(m_in / D)`
/// — every other column is structurally zero by construction. Hashers and
/// folders that walk `X` can use this helper to skip the inactive columns
/// (and, separately, to constrain them to be zero so a malicious prover
/// can't smuggle data in there).
#[inline]
pub fn superneo_public_x_cols(m_in: usize) -> usize {
    m_in.div_ceil(neo_math::D)
}

/// Soundness guard for `ce_claim_digest`/Π_RLC's "active cols only" shortcut.
///
/// The circuit side enforces `X[r, c] == 0` for `c ∈ [active_cols, x.cols())`
/// and the digest only hashes the active prefix. For native verification to
/// reject the same shapes the circuit rejects, native must check this
/// invariant explicitly before relying on the digest — otherwise an
/// attacker could smuggle non-zero data into inactive columns where it is
/// not transcript-bound.
///
/// Returns `false` if any inactive entry is non-zero or if `active_cols`
/// somehow exceeds `x.cols()`.
pub fn superneo_inactive_x_zero(x: &Mat<F>, m_in: usize) -> bool {
    let active_cols = superneo_public_x_cols(m_in);
    if active_cols > x.cols() {
        return false;
    }
    for r in 0..x.rows() {
        for c in active_cols..x.cols() {
            if x[(r, c)] != F::ZERO {
                return false;
            }
        }
    }
    true
}

#[derive(Debug, Error)]
pub enum RelationError {
    #[error("CCS instance: assignment length {got} \u{2260} structure.m {expected}")]
    AssignmentLength { got: usize, expected: usize },
    #[error("CCS instance: m_in {m_in} > assignment length {len}")]
    MInOutOfRange { m_in: usize, len: usize },
    #[error("CCS instance: \u{2016}z\u{2016}_\u{221E} \u{2265} b at index {idx} (b = {b})")]
    NormBoundViolated { idx: usize, b: u32 },
}
