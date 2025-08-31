#![allow(deprecated)] // Uses legacy MEInstance/MEWitness for bridge compatibility

//! Bridge adapter for converting ME claims to neo-spartan-bridge format
//! 
//! This module provides conversion functions from MeInstance/MeWitness 
//! to the neo-spartan-bridge types for final compression.

use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeField64, PrimeCharacteristicRing};

/// Convert Goldilocks field element to canonical u64 representation
pub fn field_to_u64(f: F) -> u64 {
    f.as_canonical_u64()
}

/// Convert signed i64 to field element (handling negatives correctly) 
pub fn i64_to_field(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        -F::from_u64((-x) as u64)
    }
}

/// Adapter for neo-spartan-bridge BridgePublicIO
#[derive(Clone, Debug)]
pub struct BridgePublicIOAdapter {
    /// Fold header digest
    pub fold_header_digest: [u8; 32],
    /// Commitment coordinates (small field)
    pub c_coords_small: Vec<u64>,
    /// Y outputs (small field)
    pub y_small: Vec<u64>,
    /// Domain tag for verification
    pub domain_tag: Option<[u8; 32]>,
}

/// Adapter for neo-spartan-bridge LinearMeProgram  
#[derive(Clone, Debug)]
pub struct LinearMeProgramAdapter {
    /// Weight vectors (small field)
    pub weights_small: Vec<Vec<u64>>,
    /// Linear rows for Ajtai (small field)
    pub l_rows_small: Option<Vec<Vec<u64>>>,
    /// Whether to check Ajtai commitment
    pub check_ajtai_commitment: bool,
    /// Optional label for debugging
    pub label: Option<String>,
}

/// Adapter for neo-spartan-bridge LinearMeWitness
#[derive(Clone, Debug)]
pub struct LinearMeWitnessAdapter {
    /// Witness digits
    pub z_digits: Vec<i64>,
}

/// Complete adapter that combines all conversions
#[derive(Clone, Debug)]
pub struct MEBridgeAdapter {
    /// Public IO adapter
    pub public_io: BridgePublicIOAdapter,
    /// Program adapter
    pub program: LinearMeProgramAdapter,
    /// Witness adapter
    pub witness: LinearMeWitnessAdapter,
}

impl MEBridgeAdapter {
    /// Create bridge adapter from ME instance and witness
    /// 
    /// NOTE: This is a placeholder implementation. The proper conversion from
    /// the new MEInstance/MEWitness types will need to be implemented once
    /// the bridge adapter structure is finalized.
    pub fn new(_me_instance: &MEInstance, _me_witness: &MEWitness) -> Self
    {
        Self {
            public_io: BridgePublicIOAdapter {
                fold_header_digest: [0u8; 32],
                c_coords_small: vec![],
                y_small: vec![],
                domain_tag: None,
            },
            program: LinearMeProgramAdapter {
                weights_small: vec![],
                l_rows_small: None,
                check_ajtai_commitment: false,
                label: Some("neo-ccs-me-adapter".to_string()),
            },
            witness: LinearMeWitnessAdapter {
                z_digits: vec![],
            },
        }
    }
    
    /// Verify consistency between instance and witness before bridging
    /// 
    /// NOTE: This is a placeholder that always returns true. Proper consistency
    /// checks should be implemented using neo-ccs's check_me_consistency function.
    pub fn verify_consistency(&self, _me_instance: &MEInstance, _me_witness: &MEWitness) -> bool {
        // TODO: Use neo_ccs::check_me_consistency once the bridge adapter
        // structure is properly aligned with the new MEInstance/MEWitness types
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_conversions() {
        let f = F::ONE;
        let u = field_to_u64(f);
        // This is a stub test - will be improved when proper field conversion is implemented
        assert_eq!(u, 1);

        let back = i64_to_field(42);
        assert_eq!(back, F::from_u64(42));
        
        // Test negative numbers
        let neg = i64_to_field(-42);
        assert_eq!(neg, -F::from_u64(42));
    }

    #[test]
    fn test_bridge_adapter_creation() {
        // This test will be expanded once proper MeInstance/MeWitness creation
        // utilities are available in neo-ccs
        
        // For now, just test that the adapter structure can be created
        let adapter = MEBridgeAdapter {
            public_io: BridgePublicIOAdapter {
                fold_header_digest: [1u8; 32],
                c_coords_small: vec![1, 2, 3],
                y_small: vec![4, 5, 6],
                domain_tag: Some([2u8; 32]),
            },
            program: LinearMeProgramAdapter {
                weights_small: vec![vec![1, 2], vec![3, 4]],
                l_rows_small: Some(vec![vec![5, 6]]),
                check_ajtai_commitment: true,
                label: Some("test".to_string()),
            },
            witness: LinearMeWitnessAdapter {
                z_digits: vec![7, 8, 9],
            },
        };

        assert_eq!(adapter.public_io.c_coords_small.len(), 3);
        assert_eq!(adapter.program.weights_small.len(), 2);
        assert_eq!(adapter.witness.z_digits.len(), 3);
    }
}