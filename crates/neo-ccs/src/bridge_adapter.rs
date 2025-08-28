//! Bridge adapter for converting ME claims to neo-spartan-bridge format
//! 
//! This module provides conversion functions from MEInstance/MEWitness 
//! to the neo-spartan-bridge types for final compression.

use crate::{MEInstance, MEWitness};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Convert Goldilocks field element to canonical u64 representation
pub fn field_to_u64(f: F) -> u64 {
    // PrimeCharacteristicRing already imported at module level
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

impl From<&MEInstance> for BridgePublicIOAdapter {
    fn from(me_instance: &MEInstance) -> Self {
        Self {
            fold_header_digest: me_instance.header_digest,
            c_coords_small: me_instance.c_coords.iter().map(|&f| field_to_u64(f)).collect(),
            y_small: me_instance.y_outputs.iter().map(|&f| field_to_u64(f)).collect(),
            domain_tag: None, // Use default bridge domain tag
        }
    }
}

impl From<&MEWitness> for LinearMeProgramAdapter {
    fn from(me_witness: &MEWitness) -> Self {
        let weights_small = me_witness.weight_vectors.iter()
            .map(|weights| weights.iter().map(|&f| field_to_u64(f)).collect())
            .collect();
            
        let l_rows_small = me_witness.ajtai_rows.as_ref().map(|rows| {
            rows.iter()
                .map(|row| row.iter().map(|&f| field_to_u64(f)).collect())
                .collect()
        });
        
        Self {
            weights_small,
            l_rows_small,
            check_ajtai_commitment: me_witness.ajtai_rows.is_some(),
            label: Some("ME(b,L)".into()),
        }
    }
}

impl From<&MEWitness> for LinearMeWitnessAdapter {
    fn from(me_witness: &MEWitness) -> Self {
        Self {
            z_digits: me_witness.z_digits.clone(),
        }
    }
}

/// Complete adapter that combines all conversions
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
    pub fn new(me_instance: &MEInstance, me_witness: &MEWitness) -> Self {
        Self {
            public_io: me_instance.into(),
            program: me_witness.into(),
            witness: me_witness.into(),
        }
    }
    
    /// Verify consistency between instance and witness before bridging
    pub fn verify_consistency(&self, me_instance: &MEInstance, me_witness: &MEWitness) -> bool {
        // Check that dimensions match
        if self.public_io.y_small.len() != self.program.weights_small.len() {
            return false;
        }
        
        if self.witness.z_digits.len() != me_instance.witness_dim() {
            return false;
        }
        
        // Verify witness consistency
        if !me_witness.check_consistency() {
            return false;
        }
        
        // Verify ME equations
        if !me_witness.verify_me_equations(me_instance) {
            return false;
        }
        
        // Verify Ajtai commitment if enabled
        if self.program.check_ajtai_commitment && !me_witness.verify_ajtai_commitment(me_instance) {
            return false;
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // PrimeCharacteristicRing already imported at module level
    
    #[test]
    fn test_field_conversions() {
        // Test positive conversion
        let f_pos = F::from_u64(42);
        assert_eq!(field_to_u64(f_pos), 42);
        assert_eq!(i64_to_field(42), f_pos);
        
        // Test negative conversion
        let f_neg = -F::from_u64(42);
        assert_eq!(i64_to_field(-42), f_neg);
    }
    
    #[test] 
    fn test_me_adapter_basic() {
        // Create simple ME instance
        let me_instance = MEInstance::new(
            vec![F::from_u64(1), F::from_u64(2)], // c_coords
            vec![F::from_u64(0)], // y_outputs: 5*1 + 5*(-1) = 0
            vec![F::from_u64(5), F::from_u64(2)], // r_point
            2, // base_b
            [0u8; 32], // header_digest
        );
        
        // Create simple ME witness  
        let me_witness = MEWitness::new(
            vec![1, -1], // z_digits  
            vec![vec![F::from_u64(5), F::from_u64(5)]], // weight_vectors: <[5,5], [1,-1]> = 5*1 + 5*(-1) = 0
            None, // no ajtai_rows
        );
        
        // Test conversion
        let adapter = MEBridgeAdapter::new(&me_instance, &me_witness);
        
        assert_eq!(adapter.public_io.c_coords_small, vec![1, 2]);
        assert_eq!(adapter.public_io.y_small, vec![0]);
        assert_eq!(adapter.program.weights_small, vec![vec![5, 5]]);
        assert_eq!(adapter.witness.z_digits, vec![1, -1]);
        assert!(!adapter.program.check_ajtai_commitment);
        
        // Verify consistency
        assert!(adapter.verify_consistency(&me_instance, &me_witness));
    }
    
    #[test]
    fn test_me_adapter_with_ajtai() {
        // Create ME instance with Ajtai commitment
        let me_instance = MEInstance::new(
            vec![F::from_u64(3)], // c_coords 
            vec![F::from_u64(0)], // y_outputs (1*2 + (-1)*2 = 0)
            vec![F::from_u64(2), F::from_u64(2)], // r_point
            2, // base_b
            [1u8; 32], // header_digest
        );
        
        // Create ME witness with Ajtai rows
        let me_witness = MEWitness::new(
            vec![1, -1], // z_digits
            vec![vec![F::from_u64(2), F::from_u64(2)]], // weight_vectors  
            Some(vec![vec![F::from_u64(3), F::from_u64(0)]]), // ajtai_rows (3*1 + 0*(-1) = 3)
        );
        
        // Test conversion
        let adapter = MEBridgeAdapter::new(&me_instance, &me_witness);
        
        assert!(adapter.program.check_ajtai_commitment);
        assert_eq!(adapter.program.l_rows_small, Some(vec![vec![3, 0]]));
        
        // Verify consistency  
        assert!(adapter.verify_consistency(&me_instance, &me_witness));
    }
}
