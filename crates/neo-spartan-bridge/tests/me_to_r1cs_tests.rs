
// Integration tests for me_to_r1cs module
// These tests are automatically run by `cargo test`

use neo_spartan_bridge::me_to_r1cs::{MeCircuit, CircuitKey};
#[allow(deprecated)] // Using legacy types for bridge testing  
use neo_ccs::{MEInstance, MEWitness};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[allow(deprecated)]
fn create_test_instance() -> (MEInstance, MEWitness) {
    // Create a minimal test instance for circuit key testing
    let z_digits = vec![1, 2, -1, 3]; // Some test witness digits
    let ajtai_rows = vec![
        vec![F::from_u64(1), F::from_u64(2), F::ZERO, F::ONE], // Row 0
        vec![F::ONE, F::ZERO, F::from_u64(3), F::from_u64(2)], // Row 1
    ];
    
    // Compute consistent c_coords
    let mut c_coords = vec![];
    for row in &ajtai_rows {
        let mut sum = F::ZERO;
        for (j, &coeff) in row.iter().enumerate() {
            if j < z_digits.len() {
                let z = z_digits[j];
                let zf = if z >= 0 { F::from_u64(z as u64) } else { F::ZERO - F::from_u64((-z) as u64) };
                sum += coeff * zf;
            }
        }
        c_coords.push(sum);
    }
    
    let me = MEInstance {
        c_coords,
        y_outputs: vec![F::from_u64(42), F::from_u64(84)],
        r_point: vec![F::from_u64(123)],
        base_b: 4,
        header_digest: [42; 32],
    };
    
    let wit = MEWitness {
        z_digits,
        ajtai_rows: Some(ajtai_rows),
        weight_vectors: vec![
            vec![F::ONE, F::from_u64(2)],
            vec![F::from_u64(3), F::ONE],
        ],
    };
    
    (me, wit)
}

#[allow(deprecated)]
#[test]
    fn circuit_key_deterministic() {
    let (me, wit) = create_test_instance();
    let circuit1 = MeCircuit::new(me.clone(), wit.clone(), None, [42; 32]);
    let circuit2 = MeCircuit::new(me, wit, None, [42; 32]);
    
    let key1 = CircuitKey::from_circuit(&circuit1);
    let key2 = CircuitKey::from_circuit(&circuit2);
    
    assert_eq!(key1, key2, "Same circuit should produce same key");
    }
    
#[allow(deprecated)]
#[test]
    fn circuit_key_sensitivity_ajtai_rows() {
    // Test that different Ajtai rows produce different circuit fingerprints
    // This ensures secure caching: we can't reuse SNARK keys across semantically different circuits
    let (mut me, mut wit) = create_test_instance();
    
    // Create first circuit with original Ajtai rows
    let circuit1 = MeCircuit::new(me.clone(), wit.clone(), None, [42; 32]);
    let key1 = CircuitKey::from_circuit(&circuit1);
    
    // Modify one coefficient in the first Ajtai row
    if let Some(ajtai_rows) = &mut wit.ajtai_rows {
        if !ajtai_rows.is_empty() && !ajtai_rows[0].is_empty() {
            ajtai_rows[0][0] = ajtai_rows[0][0] + neo_math::F::ONE;
            
            // Recompute the corresponding c_coord to maintain consistency
            let mut sum = neo_math::F::ZERO;
            for (j, &a) in ajtai_rows[0].iter().enumerate() {
                if j < wit.z_digits.len() {
                    let z = wit.z_digits[j];
                    let zf = if z >= 0 {
                        neo_math::F::from_u64(z as u64)
                    } else {
                        neo_math::F::ZERO - neo_math::F::from_u64((-z) as u64)
                    };
                    sum += a * zf;
                }
            }
            me.c_coords[0] = sum;
        }
    }
    
    // Create second circuit with modified Ajtai rows
    let circuit2 = MeCircuit::new(me, wit, None, [42; 32]);
    let key2 = CircuitKey::from_circuit(&circuit2);
    
    assert_ne!(key1, key2, 
        "Different Ajtai rows must produce different circuit keys (prevents key reuse across circuits)");
    }
    
#[allow(deprecated)]
#[test] 
    fn circuit_key_sensitivity_weight_vectors() {
    // Test that different weight vectors produce different circuit fingerprints
    let (me, mut wit) = create_test_instance();
    
    let circuit1 = MeCircuit::new(me.clone(), wit.clone(), None, [42; 32]);
    let key1 = CircuitKey::from_circuit(&circuit1);
    
    // Modify one weight vector coefficient
    if !wit.weight_vectors.is_empty() && !wit.weight_vectors[0].is_empty() {
        wit.weight_vectors[0][0] = wit.weight_vectors[0][0] + neo_math::F::ONE;
    }
    
    let circuit2 = MeCircuit::new(me, wit, None, [42; 32]);
    let key2 = CircuitKey::from_circuit(&circuit2);
    
    assert_ne!(key1, key2,
        "Different weight vectors must produce different circuit keys");
    }
    
#[allow(deprecated)]
#[test]
    fn circuit_key_ignores_fold_digest() {
    // Test that different fold digests produce the SAME circuit key
    // (fold digest is per-instance randomness, not circuit structure)
    let (me, wit) = create_test_instance();
    
    let circuit1 = MeCircuit::new(me.clone(), wit.clone(), None, [42; 32]);
    let key1 = CircuitKey::from_circuit(&circuit1);
    
    let circuit2 = MeCircuit::new(me, wit, None, [255; 32]); // Different digest
    let key2 = CircuitKey::from_circuit(&circuit2);
    
    assert_eq!(key1, key2,
        "Different fold digests must produce SAME circuit keys (digest is per-instance data)");
    }
    
#[allow(deprecated)]
#[test]
    fn singleton_poseidon2_stability() {
    // Test that the singleton Poseidon2 produces consistent results
    let (me, wit) = create_test_instance();
    let circuit = MeCircuit::new(me, wit, None, [42; 32]);
    
    // Get keys multiple times - should be identical due to singleton
    let key1 = CircuitKey::from_circuit(&circuit);
    let key2 = CircuitKey::from_circuit(&circuit);
    let key3 = CircuitKey::from_circuit(&circuit);
    
    assert_eq!(key1, key2);
    assert_eq!(key2, key3);
    assert_eq!(key1, key3);
    }
    
#[allow(deprecated)]
#[test]
    fn circuit_key_enables_cache_reuse() {
    // CRITICAL: Test that different instances of the same program produce the same key,
    // enabling PK/VK cache reuse across multiple proofs.
    let (me1, wit1) = create_test_instance();
    let (mut me2, wit2) = create_test_instance();
    
    // Change public input values (should NOT affect cache key)
    me2.c_coords[0] = me2.c_coords[0] + neo_math::F::from_u64(42);
    me2.y_outputs[0] = me2.y_outputs[0] + neo_math::F::from_u64(17);
    me2.r_point[0] = me2.r_point[0] + neo_math::F::from_u64(99);
    
    // Use different fold digests (per-instance randomness)
    let circuit1 = MeCircuit::new(me1, wit1, None, [1; 32]);
    let circuit2 = MeCircuit::new(me2, wit2, None, [2; 32]);
    
    let key1 = CircuitKey::from_circuit(&circuit1);
    let key2 = CircuitKey::from_circuit(&circuit2);
    
    assert_eq!(key1, key2, 
        "Different instances with same program structure must produce same cache key");
    }
    
#[allow(deprecated)]
#[test]
    fn circuit_key_prevents_cache_collisions() {
    // Test that changing program structure produces different keys
    let (me, wit1) = create_test_instance();
    let mut wit2 = wit1.clone();
    
    // Change program structure: different z_digits length (affects constraint count)
    wit2.z_digits.push(42);
    
    let circuit1 = MeCircuit::new(me.clone(), wit1, None, [42; 32]);
    let circuit2 = MeCircuit::new(me, wit2, None, [42; 32]);
    
    let key1 = CircuitKey::from_circuit(&circuit1);
    let key2 = CircuitKey::from_circuit(&circuit2);
    
    assert_ne!(key1, key2,
        "Different program structures must produce different cache keys");
}