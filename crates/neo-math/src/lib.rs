//! # Neo Math: Small Prime Fields and Cyclotomic Rings
//!
//! This crate provides the mathematical foundations for Neo:
//! - Small prime fields F_q (Goldilocks, Mersenne-61, Almost-Goldilocks)  
//! - Quadratic extensions K = F_q^2 for 128-bit security in sum-check
//! - Cyclotomic rings R_q = F_q[X]/(Φ_η) with coefficient maps cf/cf^{-1}
//! - Rotation ring S ≅ R_q for S-module homomorphism

// Re-export the original neo-fields functionality
use p3_field::{extension::BinomialExtensionField, PrimeField64, PrimeCharacteristicRing};

pub mod spartan2_engine; // Spartan2 engine integration for SNARK mode

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
    let mut rng = rand::thread_rng();
    let a = F::from_u64(rng.gen::<u64>());
    let b = F::from_u64(rng.gen::<u64>());
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

// Spartan2 field conversion utilities
#[cfg(feature = "spartan2-compat")]
pub mod spartan2_compat {
    
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
        /// Returns Err if truncation would occur (high bits non-zero)
        pub fn pallas_base_to_goldilocks(f: &pallas::Base) -> Result<F, String> {
            let repr = f.to_repr();
            let bytes = repr.as_ref();
            
            // Check if high bits are zero - enhanced check as recommended
            if bytes[8..].iter().any(|&b| b != 0) {
                return Err("Field truncation: Pallas base value exceeds Goldilocks range".to_string());
            }
            
            // Safe to extract lower 64 bits
            let mut val = 0u64;
            for (i, &byte) in bytes.iter().take(8).enumerate() {
                val |= (byte as u64) << (8 * i);
            }
            
            // Additional safety check: ensure the value is within Goldilocks field
            let goldilocks_val = F::from_u64(val);
            if goldilocks_val.as_canonical_u64() != val {
                return Err("Field conversion: Value not canonical in Goldilocks field".to_string());
            }
            
            Ok(goldilocks_val)
        }
        
        /// Convert Pallas scalar field element to Goldilocks field element
        /// Returns Err if truncation would occur (high bits non-zero)
        pub fn pallas_scalar_to_goldilocks(f: &pallas::Scalar) -> Result<F, String> {
            let repr = f.to_repr();
            let bytes = repr.as_ref();
            
            // Check if high bits are zero - enhanced check as recommended
            if bytes[8..].iter().any(|&b| b != 0) {
                return Err("Field truncation: Pallas scalar value exceeds Goldilocks range".to_string());
            }
            
            // Safe to extract lower 64 bits
            let mut val = 0u64;
            for (i, &byte) in bytes.iter().take(8).enumerate() {
                val |= (byte as u64) << (8 * i);
            }
            
            // Additional safety check: ensure the value is within Goldilocks field
            let goldilocks_val = F::from_u64(val);
            if goldilocks_val.as_canonical_u64() != val {
                return Err("Field conversion: Value not canonical in Goldilocks field".to_string());
            }
            
            Ok(goldilocks_val)
        }
        
        /// Legacy unsafe conversion - use pallas_base_to_goldilocks instead
        #[deprecated(note = "Use pallas_base_to_goldilocks for safe conversion")]
        pub fn pallas_base_to_goldilocks_unsafe(f: &pallas::Base) -> F {
            match pallas_base_to_goldilocks(f) {
                Ok(val) => val,
                Err(e) => panic!("Unsafe conversion failed: {}", e),
            }
        }
        
        /// Legacy unsafe conversion - use pallas_scalar_to_goldilocks instead  
        #[deprecated(note = "Use pallas_scalar_to_goldilocks for safe conversion")]
        pub fn pallas_scalar_to_goldilocks_unsafe(f: &pallas::Scalar) -> F {
            match pallas_scalar_to_goldilocks(f) {
                Ok(val) => val,
                Err(e) => panic!("Unsafe conversion failed: {}", e),
            }
        }
        
        /// Convert vector of Goldilocks elements to Pallas base field
        pub fn goldilocks_vec_to_pallas_base(vec: &[F]) -> Vec<pallas::Base> {
            vec.iter().map(goldilocks_to_pallas_base).collect()
        }
        
        /// Convert vector of Goldilocks elements to Pallas scalar field
        pub fn goldilocks_vec_to_pallas_scalar(vec: &[F]) -> Vec<pallas::Scalar> {
            vec.iter().map(goldilocks_to_pallas_scalar).collect()
        }
        
        /// Convert vector of Pallas base elements to Goldilocks with error handling
        pub fn pallas_base_vec_to_goldilocks(vec: &[pallas::Base]) -> Result<Vec<F>, String> {
            vec.iter().map(pallas_base_to_goldilocks).collect()
        }
        
        /// Convert vector of Pallas scalar elements to Goldilocks with error handling
        pub fn pallas_scalar_vec_to_goldilocks(vec: &[pallas::Scalar]) -> Result<Vec<F>, String> {
            vec.iter().map(pallas_scalar_to_goldilocks).collect()
        }
        
        /// Legacy unsafe vector conversion - use pallas_base_vec_to_goldilocks instead
        #[deprecated(note = "Use pallas_base_vec_to_goldilocks for safe conversion")]
        pub fn pallas_base_vec_to_goldilocks_unsafe(vec: &[pallas::Base]) -> Vec<F> {
            match pallas_base_vec_to_goldilocks(vec) {
                Ok(vals) => vals,
                Err(e) => panic!("Unsafe vector conversion failed: {}", e),
            }
        }
        
        /// Legacy unsafe vector conversion - use pallas_scalar_vec_to_goldilocks instead
        #[deprecated(note = "Use pallas_scalar_to_goldilocks for safe conversion")]
        pub fn pallas_scalar_vec_to_goldilocks_unsafe(vec: &[pallas::Scalar]) -> Vec<F> {
            match pallas_scalar_vec_to_goldilocks(vec) {
                Ok(vals) => vals,
                Err(e) => panic!("Unsafe vector conversion failed: {}", e),
            }
        }
    }
}

// Ring operations - integrate from neo-ring
pub mod ring {
    //! Cyclotomic ring operations from neo-ring
    
    /// Placeholder for ring element - will be replaced with actual implementation
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct RingElementPlaceholder;
    
    impl RingElementPlaceholder {
        pub fn new() -> Self { Self }
    }
    
    // Re-export as RingElement for now
    pub use RingElementPlaceholder as RingElement;
}

// Temporary placeholder - will be replaced
pub use ring::RingElement;
pub type RotationRing = RingElement;
pub type RotationMatrix = Vec<RingElement>;