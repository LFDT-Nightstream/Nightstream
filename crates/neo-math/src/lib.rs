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

pub use field::{Fq, K, GOLDILOCKS_MODULUS, TWO_ADICITY, nonresidue, two_adic_generator, KExtensions};
pub use ring::{ETA, D, Rq, cf, cf_inv, inf_norm};
pub use s_action::SAction;
pub use norms::{NeoMathError, Norms};

// Import trait for field operations
use p3_field::PrimeCharacteristicRing;

// Backward compatibility exports for existing crates
pub use Fq as F;  // Field type alias
pub type ExtF = K; // Extension field type alias

// Legacy module compatibility - keeping the old modules for now but they won't be actively used
pub mod modint;
pub mod poly;
pub mod decomp;
// transcript module removed - use neo_fold::transcript::FoldTranscript instead

// ModInt and Coeff re-exports from existing modules
pub use modint::{ModInt, Coeff};

// Polynomial re-export
pub use poly::Polynomial;

// Decomposition functions re-export
pub use decomp::{decomp_b, signed_decomp_b};

// Extension field utility functions for backward compatibility
pub fn embed_base_to_ext(base: Fq) -> K {
    K::new_real(base)
}

pub fn from_base(base: Fq) -> K {
    K::new_real(base)
}

pub fn project_ext_to_base(ext: K) -> Option<Fq> {
    // Check if imaginary part is zero
    if ext.imag() == Fq::ZERO {
        Some(ext.real())
    } else {
        None
    }
}

// Generate a random extension field element for backward compatibility
pub fn random_extf() -> K {
    use rand::Rng;
    let mut rng = rand::rng();
    let a = Fq::from_u64(rng.random::<u64>());
    let b = Fq::from_u64(rng.random::<u64>());
    K::new_complex(a, b)
}

// Ring type aliases for backward compatibility
pub type RingElement = Rq;
pub type RotationRing = RingElement;
pub type RotationMatrix = Vec<RingElement>;