#![forbid(unsafe_code)]
#![allow(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! neo-math: Fq (Goldilocks), K=F_{q^2}, R_q = F_q[X]/(Phi_eta), cf/cf^{-1}, and S-action.
//!
//! **Normative language:** "MUST", "SHOULD" are used as in BCP‑14 (RFC 2119 / RFC 8174).
//! Violations of **MUST** return errors; **SHOULD** are surfaced as warnings (or errors in strict mode).

pub mod balanced;
pub mod field;
pub mod ring;
pub mod s_action;
pub mod signed_sums;

/// Errors from S-action operations
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SActionError {
    #[error("Dimension mismatch: expected at most {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

pub use field::{from_complex, Fq, KExtensions, K};
pub use ring::{cf, cf_inv, ct, superneo_bar_block, superneo_bar_matrix, superneo_bar_vec, Rq, D, ETA};
pub use s_action::SAction;

/// Short alias used across the workspace (250+ files import `neo_math::F`).
pub use Fq as F;
