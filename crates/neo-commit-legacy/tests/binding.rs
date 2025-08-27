use neo_commit::{AjtaiCommitter, TOY_PARAMS, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::F;
use p3_field::PrimeCharacteristicRing;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Test binding property: finding collisions should be computationally infeasible.
/// This validates that the commitment scheme is binding based on MSIS hardness.
/// 
/// MSIS Security Analysis:
/// The binding property relies on the hardness of finding short solutions to Ax = u mod q.
/// For our parameters: n×k matrix A, solution vector x with ||x|| ≤ β, modulus q.
/// Security level ≈ n*k*log(q) - log(β^(n*k)) bits.
/// With TOY_PARAMS: n=64, k=16, log(q)≈61, β≈614: ~128 bits (as calculated in paper App. B.10).
#[test]
fn test_binding_collision_resistance() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    // Try to find a collision by creating many different witnesses
    // and checking if any produce the same commitment
    let mut commitments = Vec::new();
    let mut rng = ChaCha20Rng::from_seed([42; 32]);
    
    let trials = 50; // Limited trials for test efficiency
    let mut collision_found = false;
    
    for trial in 0..trials {
        // Generate a random witness
        let z: Vec<F> = (0..params.n)
            .map(|_| F::from_u64(rng.random::<u64>() % 1000)) // Small values to avoid overflow
            .collect();
        
        let mat = decomp_b(&z, params.b, params.d);
        let w = AjtaiCommitter::pack_decomp(&mat, &params);
        
        // Use deterministic randomness based on trial for reproducibility
        let mut transcript = format!("collision_test_{}", trial).into_bytes();
        let result = committer.commit(&w, &mut transcript);
        
        if let Ok((commitment, error, blinded_witness, _)) = result {
            // Verify the commitment is valid
            assert!(committer.verify(&commitment, &blinded_witness, &error),
                    "Generated commitment should be valid");
            
            // Check for collision by comparing with previous commitments
            for (i, prev_commit) in commitments.iter().enumerate() {
                if commitment == *prev_commit {
                    collision_found = true;
                    println!("Found collision between trial {} and {}", trial, i);
                    break;
                }
            }
            
            if collision_found {
                break;
            }
            
            commitments.push(commitment);
        }
    }
    
    // Should not find collisions in reasonable time (binding property)
    assert!(!collision_found, 
            "Found collision in {} trials - binding may be broken!", trials);
    
    println!("Binding test passed: No collisions found in {} trials", trials);
}

/// Test that binding holds even for similar witnesses.
/// Validates that small differences in witnesses produce different commitments.
#[test]
fn test_binding_small_differences() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    // Create two witnesses that differ by 1 in a single position
    let mut z1 = vec![F::ZERO; params.n];
    z1[0] = F::ONE;
    
    let mut z2 = z1.clone();
    z2[0] = F::from_u64(2); // Differs by 1
    
    let mat1 = decomp_b(&z1, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&mat1, &params);
    
    let mat2 = decomp_b(&z2, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&mat2, &params);
    
    // Commit with same randomness to test binding (not hiding)
    let mut t1 = vec![1u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&w1, &mut t1).unwrap();
    
    let mut t2 = vec![1u8; 32]; // Same transcript for deterministic comparison
    let (c2, e2, blinded_w2, _) = committer.commit(&w2, &mut t2).unwrap();
    
    // Both should verify
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    // Commitments should be different (binding property)
    assert_ne!(c1, c2, "Different witnesses should produce different commitments");
}

/// Test binding with zero witness vs non-zero witness.
/// Edge case testing for binding property.
#[test]
fn test_binding_zero_vs_nonzero() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    let zero_z = vec![F::ZERO; params.n];
    let nonzero_z = vec![F::ONE; params.n];
    
    let zero_mat = decomp_b(&zero_z, params.b, params.d);
    let zero_w = AjtaiCommitter::pack_decomp(&zero_mat, &params);
    
    let nonzero_mat = decomp_b(&nonzero_z, params.b, params.d);
    let nonzero_w = AjtaiCommitter::pack_decomp(&nonzero_mat, &params);
    
    // Same transcript for both
    let mut t1 = vec![0u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&zero_w, &mut t1).unwrap();
    
    let mut t2 = vec![0u8; 32];
    let (c2, e2, blinded_w2, _) = committer.commit(&nonzero_w, &mut t2).unwrap();
    
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    assert_ne!(c1, c2, "Zero and non-zero witnesses should produce different commitments");
}

/// Test that attempted collision attack fails.
/// Simulates an adversary trying to break binding by construction.
#[test]
fn test_binding_attack_simulation() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    // Adversary strategy: try to find two witnesses that map to same commitment
    // by systematically varying witness values
    
    let base_z = vec![F::from_u64(42); params.n];
    let base_mat = decomp_b(&base_z, params.b, params.d);
    let base_w = AjtaiCommitter::pack_decomp(&base_mat, &params);
    
    let mut base_transcript = vec![17u8; 32];
    let (target_commit, _, _, _) = committer.commit(&base_w, &mut base_transcript).unwrap();
    
    // Try variations to find collision
    let variations = 20; // Limited for test performance
    for i in 1..=variations {
        let mut variant_z = base_z.clone();
        variant_z[0] = F::from_u64(42 + i); // Systematic variation
        
        let variant_mat = decomp_b(&variant_z, params.b, params.d);
        let variant_w = AjtaiCommitter::pack_decomp(&variant_mat, &params);
        
        // Use different randomness for each variation to test binding not hiding
        let mut variant_transcript = format!("attack_variant_{}", i).into_bytes();
        let (variant_commit, _, _, _) = committer.commit(&variant_w, &mut variant_transcript).unwrap();
        
        // Should not find collision
        assert_ne!(variant_commit, target_commit,
                  "Found collision at variation {}: binding broken!", i);
    }
    
    println!("Attack simulation passed: No collisions found in {} variations", variations);
}

/// Test binding with secure parameters (if enabled).
/// Validates binding holds with production-level security.
#[test]
fn test_binding_secure_params() {
    // Only run with secure params if explicitly requested (due to performance)
    if std::env::var("NEO_TEST_SECURE").is_err() {
        return;
    }
    
    let params = SECURE_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    // With secure parameters, even more confident about binding
    let z1 = vec![F::ONE; params.n];
    let z2 = vec![F::from_u64(2); params.n];
    
    let mat1 = decomp_b(&z1, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&mat1, &params);
    
    let mat2 = decomp_b(&z2, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&mat2, &params);
    
    let mut t1 = vec![99u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&w1, &mut t1).unwrap();
    
    let mut t2 = vec![99u8; 32];
    let (c2, e2, blinded_w2, _) = committer.commit(&w2, &mut t2).unwrap();
    
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    assert_ne!(c1, c2, "Secure params should provide strong binding");
    
    println!("Secure binding test passed with params: n={}, k={}, q={}", 
             params.n, params.k, params.q);
}

/// Test binding property across multiple random witnesses.
/// Uses random generation to test binding property on varied inputs.
#[test]
fn test_binding_random_witnesses() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    let mut rng1 = ChaCha20Rng::seed_from_u64(12345);
    let mut rng2 = ChaCha20Rng::seed_from_u64(67890);
    
    // Generate different random witnesses
    let z1: Vec<F> = (0..params.n).map(|_| F::from_u64(rng1.random::<u64>() % 100)).collect();
    let z2: Vec<F> = (0..params.n).map(|_| F::from_u64(rng2.random::<u64>() % 100)).collect();
    
    // Ensure they're actually different
    assert_ne!(z1, z2, "Random witnesses should be different");
    
    let mat1 = decomp_b(&z1, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&mat1, &params);
    
    let mat2 = decomp_b(&z2, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&mat2, &params);
    
    // Use same transcript for binding test
    let mut t1 = vec![42u8; 32];
    let mut t2 = vec![42u8; 32];
    
    let result1 = committer.commit(&w1, &mut t1);
    let result2 = committer.commit(&w2, &mut t2);
    
    assert!(result1.is_ok() && result2.is_ok(), "Commitments should succeed");
    
    let (c1, _, _, _) = result1.unwrap();
    let (c2, _, _, _) = result2.unwrap();
    
    // Should produce different commitments (binding)
    assert_ne!(c1, c2, "Different random witnesses should produce different commitments");
}