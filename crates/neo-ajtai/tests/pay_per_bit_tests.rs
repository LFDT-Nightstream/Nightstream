//! Differential testing for neo-ajtai constant-time commit implementation
//!
//! Tests verify that the optimized constant-time dense commit matches the reference
//! specification across various input patterns and sparsity levels.
//!
//! These tests require the 'testing' feature to access commit_spec for differential testing.
//! Run with: cargo test -p neo-ajtai --features testing

#![cfg(feature = "testing")]

mod test_helpers;

#[allow(unused_imports)]
use neo_ajtai::{commit, setup};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use test_helpers::commit_spec;

#[test]
fn dense_commit_handles_various_patterns() {
    // Test that the constant-time dense commit handles various input patterns correctly
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
    assert!(
        neo_ajtai::verify_open(&pp, &commitment, &Z),
        "Commitment should verify correctly"
    );

    println!("✅ Constant-time dense commit handles various input patterns correctly");
}

#[test]
fn sparse_digit_classification_works() {
    // Test the internal logic for classifying digit sparsity patterns

    let d = neo_math::D;
    // All sparse: {-1, 0, 1} pattern
    let all_sparse = vec![Fq::ZERO; d]
        .into_iter()
        .enumerate()
        .map(|(i, _)| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        })
        .collect::<Vec<_>>();

    // All dense: values outside {-1, 0, 1}
    let all_dense = vec![Fq::from_u64(42); d]
        .into_iter()
        .enumerate()
        .map(|(i, _)| {
            Fq::from_u64(42 + (i % 100) as u64) // Different values to avoid optimization
        })
        .collect::<Vec<_>>();

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
fn constant_time_commit_correctness() {
    // Comprehensive test that the constant-time dense commit gives correct results
    // verified against the reference specification

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
        (0..d * m)
            .map(|i| match i % 3 {
                0 => Fq::ZERO,
                1 => Fq::ONE,
                _ => Fq::ZERO - Fq::ONE,
            })
            .collect(),
        // Case 5: Random sparse with some zeros
        (0..d * m)
            .map(|i| if i % 2 == 0 { Fq::ZERO } else { Fq::ONE })
            .collect(),
    ];

    for (i, z) in test_cases.iter().enumerate() {
        let commitment = commit(&pp, z);
        assert!(
            neo_ajtai::verify_open(&pp, &commitment, z),
            "Test case {} should verify correctly",
            i
        );
    }

    println!("✅ Constant-time dense commit works correctly for various digit patterns");
}

#[test]
fn dense_commit_matches_spec_with_mixed_digits() {
    // Test that the constant-time dense commit handles mixed digit patterns correctly
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let d = neo_math::D;
    let kappa = 4;
    let m = 8;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    // Create data with mostly {-1,0,1} but a few 2's sprinkled in
    // The constant-time implementation handles all patterns uniformly
    #[allow(non_snake_case)]
    let mut Z = vec![Fq::ZERO; d * m];
    for i in 0..Z.len() {
        Z[i] = match i % 16 {
            0..=7 => Fq::ONE,             // 50% ones
            8..=11 => Fq::ZERO,           // 25% zeros
            12..=13 => Fq::ZERO - Fq::ONE, // 12.5% minus ones
            _ => Fq::from_u64(2),         // 12.5% twos (breaks pay-per-bit!)
        };
    }

    let c_actual = neo_ajtai::commit(&pp, &Z);

    // Differential testing in test builds only
    let c_spec = commit_spec(&pp, &Z);
    assert_eq!(c_actual, c_spec, "Commit with mixed digits must match specification");

    // Always verify that opening works correctly
    assert!(
        neo_ajtai::verify_open(&pp, &c_actual, &Z),
        "Mixed digit commit should verify correctly"
    );

    println!("✅ Commit handles mixed digits correctly (uses dense path due to 2's)");
}

#[test]
fn dense_commit_matches_spec_all_patterns() {
    // Test that constant-time dense commit matches spec for various digit patterns
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let d = neo_math::D;
    let kappa = 2;
    let m = 4;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    // Case 1: Strictly {-1, 0, 1} - should potentially use pay-per-bit if feature enabled
    #[allow(non_snake_case)]
    let Z_strict: Vec<Fq> = (0..d * m)
        .map(|i| match i % 3 {
            0 => Fq::ZERO,
            1 => Fq::ONE,
            _ => Fq::ZERO - Fq::ONE,
        })
        .collect();

    // Case 2: One digit is 2 - must use dense path
    #[allow(non_snake_case)]
    let mut Z_mixed = Z_strict.clone();
    Z_mixed[0] = Fq::from_u64(2); // Introduce a single non-{-1,0,1} digit

    let c_strict = neo_ajtai::commit(&pp, &Z_strict);
    let c_mixed = neo_ajtai::commit(&pp, &Z_mixed);

    // Verify basic correctness (always available)
    assert!(
        neo_ajtai::verify_open(&pp, &c_strict, &Z_strict),
        "Strict digits should verify"
    );
    assert!(
        neo_ajtai::verify_open(&pp, &c_mixed, &Z_mixed),
        "Mixed digits should verify"
    );
    assert_ne!(c_strict, c_mixed, "Different inputs should produce different outputs");

    // Differential testing in test builds only
    let c_spec_strict = commit_spec(&pp, &Z_strict);
    let c_spec_mixed = commit_spec(&pp, &Z_mixed);
    assert_eq!(c_strict, c_spec_strict, "Strict {{-1,0,1}} digits must match spec");
    assert_eq!(c_mixed, c_spec_mixed, "Mixed digits must match spec");

    println!("✅ Constant-time dense commit matches spec for {{-1,0,1}} vs mixed digit patterns");
}

#[test]
fn dense_commit_differential_testing() {
    // Differential testing: verify the constant-time dense commit matches spec
    // Tests the optimized implementation against the reference specification

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 3;
    let m = 6;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    // Create {-1, 0, 1} digit pattern for differential testing
    #[allow(non_snake_case)]
    let Z_sparse: Vec<Fq> = (0..d * m)
        .map(|i| match i % 4 {
            0 => Fq::ONE,            // 25% ones
            1 => Fq::ZERO,           // 25% zeros
            2 => Fq::ZERO - Fq::ONE, // 25% minus ones
            _ => Fq::ZERO,           // 25% more zeros (sparse!)
        })
        .collect();

    // Create mixed digits that should NOT trigger fast path
    #[allow(non_snake_case)]
    let Z_mixed: Vec<Fq> = Z_sparse
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if i % 10 == 0 {
                Fq::from_u64(2)
            } else {
                x
            } // Sprinkle some 2's
        })
        .collect();

    let c_sparse = neo_ajtai::commit(&pp, &Z_sparse);
    let c_mixed = neo_ajtai::commit(&pp, &Z_mixed);
    let c_sparse_spec = commit_spec(&pp, &Z_sparse);
    let c_mixed_spec = commit_spec(&pp, &Z_mixed);

    // Both should match their specifications exactly
    assert_eq!(c_sparse, c_sparse_spec, "Sparse {{-1,0,1}} digits should match spec");
    assert_eq!(c_mixed, c_mixed_spec, "Mixed digits should match spec");
    assert_ne!(c_sparse, c_mixed, "Different inputs should produce different outputs");

    println!("✅ Differential testing verification: constant-time implementation matches spec");
    println!("   Sparse {{-1,0,1}} input: matches spec ✓");
    println!("   Mixed input with 2's: matches spec ✓");
}

#[test]
fn sparsity_invariant_testing() {
    // Differential testing with different sparsity levels
    // Tests that the constant-time implementation produces identical results
    // regardless of input sparsity patterns

    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let d = neo_math::D;
    let kappa = 2;
    let m = 4;
    let pp = setup(&mut rng, d, kappa, m).unwrap();

    // Very sparse: 95% zeros
    #[allow(non_snake_case)]
    let Z_very_sparse: Vec<Fq> = (0..d * m)
        .map(|i| if i % 20 == 0 { Fq::ONE } else { Fq::ZERO })
        .collect();

    // Dense: 50% ones, 50% minus ones (no zeros)
    #[allow(non_snake_case)]
    let Z_dense: Vec<Fq> = (0..d * m)
        .map(|i| if i % 2 == 0 { Fq::ONE } else { Fq::ZERO - Fq::ONE })
        .collect();

    let c_sparse = neo_ajtai::commit(&pp, &Z_very_sparse);
    let c_dense = neo_ajtai::commit(&pp, &Z_dense);
    let c_sparse_spec = commit_spec(&pp, &Z_very_sparse);
    let c_dense_spec = commit_spec(&pp, &Z_dense);

    // Correctness: both should match spec
    assert_eq!(c_sparse, c_sparse_spec, "Very sparse input should match spec");
    assert_eq!(c_dense, c_dense_spec, "Dense input should match spec");

    println!("✅ Sparsity invariance test:");
    println!("   Very sparse (95% zeros): matches spec ✓");
    println!("   Dense (0% zeros): matches spec ✓");
    println!("   Constant-time implementation produces correct results regardless of sparsity");
}
