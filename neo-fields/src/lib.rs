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

// Spartan2 Engine implementation
pub mod spartan2_engine {
    // Re-export Spartan2's PallasHyraxEngine for Neo integration
    pub use spartan2::provider::PallasHyraxEngine as GoldilocksEngine;
    
    /// Conversion utilities between Neo's Goldilocks field and Spartan2's Pallas field
    pub mod field_conversion {
        use super::super::*;
        use spartan2::provider::pasta::pallas;
        use ff::PrimeField;
        
        /// Convert Goldilocks field element to Pallas base field element
        pub fn goldilocks_to_pallas_base(f: &F) -> pallas::Base {
            // Convert via canonical representation
            let val = f.as_canonical_u64();
            pallas::Base::from(val)
        }
        
        /// Convert Goldilocks field element to Pallas scalar field element
        pub fn goldilocks_to_pallas_scalar(f: &F) -> pallas::Scalar {
            // Convert via canonical representation
            let val = f.as_canonical_u64();
            pallas::Scalar::from(val)
        }
        
        /// Convert Pallas base field element to Goldilocks field element
        pub fn pallas_base_to_goldilocks(f: &pallas::Base) -> F {
            // Extract lower 64 bits and convert to Goldilocks
            let bytes = f.to_repr();
            let mut val = 0u64;
            for (i, &byte) in bytes.as_ref().iter().take(8).enumerate() {
                val |= (byte as u64) << (8 * i);
            }
            F::from_u64(val)
        }
        
        /// Convert Pallas scalar field element to Goldilocks field element
        pub fn pallas_scalar_to_goldilocks(f: &pallas::Scalar) -> F {
            // Extract lower 64 bits and convert to Goldilocks
            let bytes = f.to_repr();
            let mut val = 0u64;
            for (i, &byte) in bytes.as_ref().iter().take(8).enumerate() {
                val |= (byte as u64) << (8 * i);
            }
            F::from_u64(val)
        }
        
        /// Convert vector of Goldilocks elements to Pallas base field
        pub fn goldilocks_vec_to_pallas_base(vec: &[F]) -> Vec<pallas::Base> {
            vec.iter().map(goldilocks_to_pallas_base).collect()
        }
        
        /// Convert vector of Goldilocks elements to Pallas scalar field
        pub fn goldilocks_vec_to_pallas_scalar(vec: &[F]) -> Vec<pallas::Scalar> {
            vec.iter().map(goldilocks_to_pallas_scalar).collect()
        }
        
        /// Convert vector of Pallas base elements to Goldilocks
        pub fn pallas_base_vec_to_goldilocks(vec: &[pallas::Base]) -> Vec<F> {
            vec.iter().map(pallas_base_to_goldilocks).collect()
        }
        
        /// Convert vector of Pallas scalar elements to Goldilocks
        pub fn pallas_scalar_vec_to_goldilocks(vec: &[pallas::Scalar]) -> Vec<F> {
            vec.iter().map(pallas_scalar_to_goldilocks).collect()
        }
    }
}

