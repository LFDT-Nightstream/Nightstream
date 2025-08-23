//! Field utilities wrapping the Goldilocks prime field for Neo lattice-based cryptography.

use p3_field::{extension::BinomialExtensionField, PrimeField64, PrimeCharacteristicRing};

pub use p3_goldilocks::Goldilocks as F;

/// Quadratic extension F[u] / (u^2 - β) with β=7 (non-residue mod p for Goldilocks).
pub type ExtF = BinomialExtensionField<F, 2>;

/// Convert base field element to extension field (embed in constant term)
pub fn embed_base_to_ext(base: F) -> ExtF {
    ExtF::new_real(base)
}

/// Convert extension field element to base field (project to constant term)
pub fn project_ext_to_base(ext: ExtF) -> Option<F> {
    if ext.to_array()[1] == F::ZERO {
        Some(ext.to_array()[0])
    } else {
        None
    }
}

/// Convert base field element to extension field using new_real
pub fn from_base(base: F) -> ExtF {
    ExtF::new_real(base)
}

/// Generate a random extension field element
pub fn random_extf() -> ExtF {
    use rand::Rng;
    let mut rng = rand::rng();
    let a = F::from_u64(rng.random::<u64>());
    let b = F::from_u64(rng.random::<u64>());
    ExtF::new_complex(a, b)
}

/// Extension field norm type for ZK blinding
pub type ExtFieldNorm = u64;

/// Maximum norm bound for ZK blinding
pub const MAX_BLIND_NORM: u64 = 1u64 << 40;

/// Trait for computing extension field norms
pub trait ExtFieldNormTrait {
    fn abs_norm(&self) -> u64;
}

impl ExtFieldNormTrait for ExtF {
    fn abs_norm(&self) -> u64 {
        let arr = self.to_array();
        let a = arr[0].as_canonical_u64();
        let b = arr[1].as_canonical_u64();
        // Simple L∞ norm (max of components)
        a.max(b)
    }
}

// Display implementation removed due to orphan rule - ExtF already has Debug

