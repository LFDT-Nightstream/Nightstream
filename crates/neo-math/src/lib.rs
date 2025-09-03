#![forbid(unsafe_code)]
#![allow(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! neo-math: Fq (Goldilocks), K=F_{q^2}, R_q = F_q[X]/(Phi_eta), cf/cf^{-1}, and S-action.
//!
//! **Normative language:** "MUST", "SHOULD" are used as in BCP‑14 (RFC 2119 / RFC 8174).
//! Violations of **MUST** return errors; **SHOULD** are surfaced as warnings (or errors in strict mode).

pub mod norms;
pub mod field;
pub mod ring;
pub mod s_action;

/// Errors from S-action operations
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SActionError {
    #[error("Dimension mismatch: expected at most {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

pub use field::{Fq, K, GOLDILOCKS_MODULUS, TWO_ADICITY, nonresidue, two_adic_generator, KExtensions, from_base, from_complex, embed_base_to_ext, project_ext_to_base};
pub use ring::{ETA, D, Rq, cf, cf_inv, inf_norm, reduce_mod_phi_81};
pub use s_action::SAction;
pub use norms::{NeoMathError, Norms};

// Import trait for field operations - removed unused import

// Backward compatibility exports for existing crates
pub use Fq as F;  // Field type alias
pub type ExtF = K; // Extension field type alias

// Legacy modules removed as part of codebase cleanup
// Use neo_fold::transcript::FoldTranscript for transcript functionality
// Use neo-ajtai for decomposition functions
// For ModInt/polynomial functionality, use the main field/ring types

// Extension field utility functions moved to field.rs

// Random generation moved to field.rs

// Ring type aliases 
pub type RingElement = Rq;
pub type RotationRing = RingElement;
pub type RotationMatrix = Vec<RingElement>;