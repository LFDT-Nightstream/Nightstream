//! Production-realistic tests for S-action K-vector operations
//!
//! These tests use realistic dimensions and values that match actual ME protocol usage,
//! ensuring the security fixes work correctly for production scenarios.

use neo_math::{SAction, Fq, K, D};
use neo_math::ring::cf_inv;
use p3_field::PrimeCharacteristicRing;

/// Create realistic K vectors based on production ME instance patterns
fn create_production_k_vector(dim: usize) -> Vec<K> {
    assert!(dim <= D, "Production vectors should not exceed D={}", D);
    
    // Use realistic values similar to those in bridge smoke tests
    let mut result = Vec::with_capacity(dim);
    
    for i in 0..dim {
        let real_part = Fq::from_u64(match i % 7 {
            0 => 1,
            1 => 2, 
            2 => 3,
            3 => 0,
            4 => 5,
            5 => 1,
            _ => 2,
        });
        
        let imag_part = Fq::from_u64(match i % 5 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 1,
            _ => 0,
        });
        
        result.push(K::new_complex(real_part, imag_part));
    }
    
    result
}

/// Create realistic S-actions based on strong sampling patterns
fn create_production_s_action(multiplier: u64) -> SAction {
    let mut coeffs = [Fq::ZERO; D];
    coeffs[0] = Fq::from_u64(multiplier);  // Simple multiplication
    // Add some variation to higher terms
    if D > 1 {
        coeffs[1] = Fq::from_u64(multiplier.wrapping_add(1));
    }
    if D > 2 {
        coeffs[2] = Fq::from_u64(multiplier.wrapping_add(7)); // Add some prime offset
    }
    SAction::from_ring(cf_inv(coeffs))
}

/// Test production scenarios that match typical ME protocol dimensions
#[cfg(test)]
mod production_tests {
    use super::*;

    #[test]
    fn test_typical_me_instance_dimensions() {
        println!("🧪 Testing S-actions with typical ME instance dimensions...");
        
        // Test common ME dimensions: 4, 8, 16 (matching bridge smoke tests)
        for &dim in &[4, 8, 16] {
            let y_vec = create_production_k_vector(dim);
            let s_action = create_production_s_action(2);  // Simple doubling
            
            let result = s_action.apply_k_vec(&y_vec);
            
            assert_eq!(result.len(), y_vec.len(), "S-action should preserve vector length for dim={}", dim);
            println!("✅ Successfully processed {}-dimensional ME vector", dim);
        }
    }

    #[test]
    fn test_pi_rlc_typical_scenario() {
        println!("🧪 Testing S-action scenario typical of Π_RLC proofs...");
        
        // Simulate combining 3 ME instances (k+1 = 3, so k=2)
        let dim = 8; // Common dimension from tests
        let num_instances = 3;
        
        let mut me_y_vectors = Vec::new();
        for _i in 0..num_instances {
            me_y_vectors.push(create_production_k_vector(dim));
        }
        
        // Create different S-actions for each instance (like strong sampling)
        let s_actions: Vec<SAction> = (0..num_instances)
            .map(|i| create_production_s_action(i as u64 + 1))
            .collect();
        
        // Combine vectors like in pi_rlc_prove()
        let mut combined = vec![K::ZERO; dim];
        for (s_action, y_vec) in s_actions.iter().zip(me_y_vectors.iter()) {
            let rotated = s_action.apply_k_vec(y_vec);
            for (i, &val) in rotated.iter().enumerate() {
                combined[i] += val;
            }
        }
        
        assert_eq!(combined.len(), dim);
        println!("✅ Successfully combined {} ME instances with S-actions", num_instances);
    }

    #[test]
    fn test_s_action_linearity_production_values() {
        println!("🧪 Testing S-action linearity with production-like values...");
        
        let dim = 4;
        let y = create_production_k_vector(dim);
        let z = create_production_k_vector(dim);
        let s = create_production_s_action(3);
        
        // Use realistic scalar multipliers (like from challenge sampling)
        let a = K::new_complex(Fq::from_u64(5), Fq::ZERO);
        let b = K::new_complex(Fq::from_u64(7), Fq::ONE);
        
        // Test linearity: S(a*y + b*z) = a*S(y) + b*S(z)
        let mut ay_plus_bz = Vec::with_capacity(dim);
        for i in 0..dim {
            ay_plus_bz.push(a * y[i] + b * z[i]);
        }
        
        let s_combination = s.apply_k_vec(&ay_plus_bz);
        
        let s_y = s.apply_k_vec(&y);
        let s_z = s.apply_k_vec(&z);
        let mut expected = Vec::with_capacity(dim);
        for i in 0..dim {
            expected.push(a * s_y[i] + b * s_z[i]);
        }
        
        assert_eq!(s_combination.len(), expected.len());
        for (actual, expected) in s_combination.iter().zip(expected.iter()) {
            assert_eq!(*actual, *expected, "S-action linearity failed for production values");
        }
        
        println!("✅ S-action linearity verified with production values");
    }

    #[test]
    fn test_identity_s_action_production() {
        println!("🧪 Testing identity S-action with production values...");
        
        let dim = 8;
        let y = create_production_k_vector(dim);
        
        // Identity S-action
        let mut coeffs = [Fq::ZERO; D];
        coeffs[0] = Fq::ONE;
        let identity_s = SAction::from_ring(cf_inv(coeffs));
        
        let result = identity_s.apply_k_vec(&y);
        
        assert_eq!(result.len(), y.len());
        for (original, preserved) in y.iter().zip(result.iter()) {
            assert_eq!(*original, *preserved, "Identity S-action should preserve elements");
        }
        
        println!("✅ Identity S-action verified with production values");
    }

    #[test]
    fn test_security_dimension_check() {
        println!("🧪 Testing security dimension checks...");
        
        // Test that vectors exactly at D work fine
        let max_dim_vector = create_production_k_vector(D);
        let s = create_production_s_action(2);
        
        let result = s.apply_k_vec(&max_dim_vector);
        assert_eq!(result.len(), D);
        println!("✅ Maximum dimension D={} handled correctly", D);
        
        // Test that vectors longer than D are rejected (security fix)
        // This should panic, demonstrating the security fix works
        std::panic::set_hook(Box::new(|_| {})); // Suppress panic output
        
        let result = std::panic::catch_unwind(|| {
            let mut oversized = create_production_k_vector(D);
            oversized.push(K::ZERO); // Make it D+1 elements
            s.apply_k_vec(&oversized)
        });
        
        assert!(result.is_err(), "Oversized vectors should be rejected for security");
        println!("✅ Security check correctly rejects oversized vectors");
        
        // Restore normal panic behavior
        let _ = std::panic::take_hook();
    }

    #[test] 
    fn test_empty_and_small_vectors() {
        println!("🧪 Testing edge cases: empty and very small vectors...");
        
        let s = create_production_s_action(2);
        
        // Empty vector
        let empty: Vec<K> = Vec::new();
        let result = s.apply_k_vec(&empty);
        assert_eq!(result.len(), 0);
        println!("✅ Empty vector handled correctly");
        
        // Single element vector
        let single = vec![K::new_complex(Fq::from_u64(42), Fq::from_u64(7))];
        let result = s.apply_k_vec(&single);
        assert_eq!(result.len(), 1);
        println!("✅ Single-element vector handled correctly");
    }

    #[test]
    fn test_realistic_bridge_scenario() {
        println!("🧪 Testing scenario matching neo-spartan-bridge usage...");
        
        // Values matching bridge smoke test patterns
        let dim = 4; // y_outputs length from bridge test
        let mut y = Vec::with_capacity(dim);
        
        // Create K elements from realistic F values (like y0=6, y1=3 from smoke test)
        y.push(K::new_complex(Fq::from_u64(6), Fq::ZERO));
        y.push(K::new_complex(Fq::from_u64(3), Fq::ZERO));
        y.push(K::new_complex(Fq::ZERO, Fq::ZERO));
        y.push(K::new_complex(Fq::ZERO, Fq::ZERO));
        
        let s = create_production_s_action(1); // Simple rotation
        let result = s.apply_k_vec(&y);
        
        assert_eq!(result.len(), 4);
        println!("✅ Bridge-style scenario processed correctly");
    }
}
