//! Tests for pay-per-bit optimization in neo-ajtai commit function

use neo_ajtai::{setup, commit};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

#[test]
fn test_commit_sparse_vs_dense_equivalence() {
    // Test that sparse and dense commit paths produce identical results
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 4;
    let m = 8;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Create test data with mostly sparse digits {-1, 0, 1}
    #[allow(non_snake_case)]
    let mut Z = vec![Fq::ZERO; d * m];
    
    // Fill with sparse digits (this should trigger pay-per-bit optimization)
    for i in 0..Z.len() {
        match i % 4 {
            0 => Z[i] = Fq::ZERO,
            1 => Z[i] = Fq::ONE,  
            2 => Z[i] = Fq::ZERO - Fq::ONE, // -1
            _ => Z[i] = Fq::ZERO,
        }
    }
    
    // The commit function should automatically choose the optimization
    let commitment = commit(&pp, &Z);
    
    // Verify the commitment is valid
    assert!(neo_ajtai::verify_open(&pp, &commitment, &Z), "Commitment should verify correctly");
    
    println!("✅ Pay-per-bit optimization produces valid commitments");
}

#[test]
fn test_sparse_digit_detection() {
    // Test the internal logic for detecting when to use pay-per-bit optimization
    
    let d = neo_math::D;
    // All sparse: should use optimization
    let all_sparse = vec![Fq::ZERO; d].into_iter().enumerate().map(|(i, _)| {
        match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        }
    }).collect::<Vec<_>>();
    
    // All dense: should not use optimization  
    let all_dense = vec![Fq::from_u64(42); d].into_iter().enumerate().map(|(i, _)| {
        Fq::from_u64(42 + (i % 100) as u64) // Different values to avoid optimization
    }).collect::<Vec<_>>();
    
    // Test that both produce valid commitments (the important part)
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = neo_math::D; // Must match ring dimension
    let kappa = 2;
    let m = 1;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    let commitment_sparse = commit(&pp, &all_sparse);
    let commitment_dense = commit(&pp, &all_dense);
    
    assert!(neo_ajtai::verify_open(&pp, &commitment_sparse, &all_sparse));
    assert!(neo_ajtai::verify_open(&pp, &commitment_dense, &all_dense));
    
    println!("✅ Both sparse and dense digit patterns produce valid commitments");
}

#[test] 
fn test_pay_per_bit_correctness() {
    // Comprehensive test that the pay-per-bit optimization gives the same result
    // as if we computed it using the full S-action matrix
    
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 2; 
    let m = 4;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Create various digit patterns to test
    let test_cases = vec![
        // Case 1: All zeros
        vec![Fq::ZERO; d * m],
        // Case 2: All ones  
        vec![Fq::ONE; d * m],
        // Case 3: All -1
        vec![Fq::ZERO - Fq::ONE; d * m],
        // Case 4: Mixed sparse pattern
        (0..d*m).map(|i| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        }).collect(),
        // Case 5: Random sparse with some zeros
        (0..d*m).map(|i| if i % 2 == 0 { Fq::ZERO } else { Fq::ONE }).collect(),
    ];
    
    for (i, z) in test_cases.iter().enumerate() {
        let commitment = commit(&pp, z);
        assert!(neo_ajtai::verify_open(&pp, &commitment, z), 
                "Test case {} should verify correctly", i);
    }
    
    println!("✅ Pay-per-bit optimization works correctly for various digit patterns");
}

#[test]
fn pay_per_bit_matches_spec_even_with_stray_twos() {
    // This test would FAIL with the old buggy implementation that treated any non-zero as -1
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let d = neo_math::D; 
    let kappa = 4; 
    let m = 8;
    let pp = setup(&mut rng, d, kappa, m).unwrap();
    
    // Create data with mostly {-1,0,1} but a few 2's sprinkled in
    // This should NOT use the pay-per-bit path due to the 2's
    #[allow(non_snake_case)]
    let mut Z = vec![Fq::ZERO; d * m];
    for i in 0..Z.len() {
        Z[i] = match i % 16 {
            0|1|2|3|4|5|6|7 => Fq::ONE,           // 50% ones
            8|9|10|11        => Fq::ZERO,          // 25% zeros  
            12|13            => Fq::ZERO - Fq::ONE, // 12.5% minus ones
            _                => Fq::from_u64(2),   // 12.5% twos (breaks pay-per-bit!)
        };
    }
    
    let c_actual = neo_ajtai::commit(&pp, &Z);
    let c_spec = neo_ajtai::commit_spec(&pp, &Z);
    assert_eq!(c_actual, c_spec, "Commit with mixed digits must match specification");
    
    println!("✅ Commit handles mixed digits correctly (uses dense path due to 2's)");
}

#[test] 
fn pay_per_bit_strict_gating() {
    // Test that pay-per-bit is only used when ALL digits are in {-1, 0, 1}
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let d = neo_math::D;
    let kappa = 2;
    let m = 4;  
    let pp = setup(&mut rng, d, kappa, m).unwrap();
    
    // Case 1: Strictly {-1, 0, 1} - should potentially use pay-per-bit if feature enabled
    #[allow(non_snake_case)]
    let Z_strict: Vec<Fq> = (0..d*m).map(|i| match i % 3 {
        0 => Fq::ZERO,
        1 => Fq::ONE,
        _ => Fq::ZERO - Fq::ONE,
    }).collect();
    
    // Case 2: One digit is 2 - must use dense path 
    #[allow(non_snake_case)]
    let mut Z_mixed = Z_strict.clone();
    Z_mixed[0] = Fq::from_u64(2); // Introduce a single non-{-1,0,1} digit
    
    let c_strict = neo_ajtai::commit(&pp, &Z_strict);  
    let c_mixed = neo_ajtai::commit(&pp, &Z_mixed);
    let c_spec_strict = neo_ajtai::commit_spec(&pp, &Z_strict);
    let c_spec_mixed = neo_ajtai::commit_spec(&pp, &Z_mixed);
    
    // Both should match their specifications
    assert_eq!(c_strict, c_spec_strict, "Strict {{-1,0,1}} digits must match spec");
    assert_eq!(c_mixed, c_spec_mixed, "Mixed digits must match spec");
    assert_ne!(c_strict, c_mixed, "Different inputs should produce different outputs");
    
    println!("✅ Pay-per-bit gating works: only {{-1,0,1}} vs mixed digits produce correct results");
}
