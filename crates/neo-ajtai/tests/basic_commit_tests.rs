//! Basic functionality tests for neo-ajtai commit operations
//! 
//! These tests verify core commit/verify functionality without requiring differential
//! testing against the specification. They run with default features and catch 
//! regressions in fundamental operations.

use neo_ajtai::{setup, commit, verify_open, commit_masked_ct, commit_precomp_ct, rows_for_coords, compute_single_ajtai_row};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

#[test]
fn commit_and_verify_basic_patterns() {
    // Test basic commit/verify cycle with various input patterns
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 3;
    let m = 4;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Test case 1: All zeros
    let z_zeros = vec![Fq::ZERO; d * m];
    let c_zeros = commit(&pp, &z_zeros);
    assert!(verify_open(&pp, &c_zeros, &z_zeros), "All-zeros should verify");
    
    // Test case 2: All ones  
    let z_ones = vec![Fq::ONE; d * m];
    let c_ones = commit(&pp, &z_ones);
    assert!(verify_open(&pp, &c_ones, &z_ones), "All-ones should verify");
    
    // Test case 3: Sparse pattern {-1, 0, 1}
    let z_sparse: Vec<Fq> = (0..d*m).map(|i| match i % 3 {
        0 => Fq::ZERO,
        1 => Fq::ONE,
        _ => Fq::ZERO - Fq::ONE, // -1
    }).collect();
    let c_sparse = commit(&pp, &z_sparse);
    assert!(verify_open(&pp, &c_sparse, &z_sparse), "Sparse pattern should verify");
    
    // Test case 4: Dense random pattern
    let z_dense: Vec<Fq> = (0..d*m).map(|i| Fq::from_u64((i * 17 + 42) as u64)).collect();
    let c_dense = commit(&pp, &z_dense);
    assert!(verify_open(&pp, &c_dense, &z_dense), "Dense pattern should verify");
    
    // Test case 5: Mixed values including 2's
    let mut z_mixed = z_sparse.clone();
    for i in (0..z_mixed.len()).step_by(10) {
        z_mixed[i] = Fq::from_u64(2);
    }
    let c_mixed = commit(&pp, &z_mixed);
    assert!(verify_open(&pp, &c_mixed, &z_mixed), "Mixed pattern should verify");
    
    println!("✅ Basic commit/verify functionality works for all test patterns");
}

#[test]
fn commit_deterministic() {
    // Test that commit is deterministic - same input produces same output
    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let d = neo_math::D;
    let kappa = 2;
    let m = 3;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Create test input
    let z: Vec<Fq> = (0..d*m).map(|i| Fq::from_u64((i % 100) as u64)).collect();
    
    // Commit twice
    let c1 = commit(&pp, &z);
    let c2 = commit(&pp, &z);
    
    // Should be identical
    assert_eq!(c1, c2, "Commit should be deterministic");
    
    // Both should verify
    assert!(verify_open(&pp, &c1, &z), "First commit should verify");
    assert!(verify_open(&pp, &c2, &z), "Second commit should verify");
    
    println!("✅ Commit is deterministic");
}

#[test]
fn verify_rejects_wrong_opening() {
    // Test that verification properly rejects incorrect openings
    let mut rng = ChaCha20Rng::seed_from_u64(456);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Create correct commitment
    let z_correct: Vec<Fq> = (0..d*m).map(|i| Fq::from_u64(i as u64)).collect();
    let commitment = commit(&pp, &z_correct);
    
    // Verify correct opening works
    assert!(verify_open(&pp, &commitment, &z_correct), "Correct opening should verify");
    
    // Create wrong opening (modify one element)
    let mut z_wrong = z_correct.clone();
    z_wrong[0] = z_wrong[0] + Fq::ONE;
    
    // Verify wrong opening is rejected
    assert!(!verify_open(&pp, &commitment, &z_wrong), "Wrong opening should be rejected");
    
    println!("✅ Verification correctly rejects tampered openings");
}

#[test]
fn different_inputs_produce_different_commitments() {
    // Test that different inputs produce different commitments (binding property)
    let mut rng = ChaCha20Rng::seed_from_u64(789);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Create two different inputs
    let z1: Vec<Fq> = (0..d*m).map(|i| Fq::from_u64(i as u64)).collect();
    let mut z2 = z1.clone();
    z2[0] = z2[0] + Fq::ONE; // Change one element
    
    // Commit both
    let c1 = commit(&pp, &z1);
    let c2 = commit(&pp, &z2);
    
    // Should be different (binding property)
    assert_ne!(c1, c2, "Different inputs should produce different commitments");
    
    // Each should verify with its correct opening
    assert!(verify_open(&pp, &c1, &z1), "First commitment should verify with first opening");
    assert!(verify_open(&pp, &c2, &z2), "Second commitment should verify with second opening");
    
    // Cross-verification should fail
    assert!(!verify_open(&pp, &c1, &z2), "First commitment should not verify with second opening");
    assert!(!verify_open(&pp, &c2, &z1), "Second commitment should not verify with first opening");
    
    println!("✅ Different inputs produce different commitments (binding property)");
}

#[test]
fn try_commit_error_handling() {
    // Test that try_commit properly handles dimension mismatches
    let mut rng = ChaCha20Rng::seed_from_u64(999);
    let d = neo_math::D;
    let kappa = 2;
    let m = 3;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Correct size should work
    let z_correct = vec![Fq::ZERO; d * m];
    assert!(neo_ajtai::try_commit(&pp, &z_correct).is_ok(), "Correct size should work");
    
    // Too short should fail
    let z_short = vec![Fq::ZERO; d * m - 1];
    let result_short = neo_ajtai::try_commit(&pp, &z_short);
    assert!(result_short.is_err(), "Too short should fail");
    
    // Too long should fail  
    let z_long = vec![Fq::ZERO; d * m + 1];
    let result_long = neo_ajtai::try_commit(&pp, &z_long);
    assert!(result_long.is_err(), "Too long should fail");
    
    println!("✅ try_commit properly handles dimension errors");
}

#[test]
fn constant_time_variants_basic_functionality() {
    // Test that the constant-time variants work correctly without needing spec comparison
    let mut rng = ChaCha20Rng::seed_from_u64(2023);
    let d = neo_math::D;
    let kappa = 2;
    let m = 2;
    
    let pp = setup(&mut rng, d, kappa, m).expect("setup should work");
    
    // Create test input with various patterns
    let z_test: Vec<Fq> = (0..d*m).map(|i| match i % 4 {
        0 => Fq::ZERO,
        1 => Fq::ONE,
        2 => Fq::ZERO - Fq::ONE, // -1  
        _ => Fq::from_u64(2),
    }).collect();
    
    // Test masked constant-time commit
    let c_masked = commit_masked_ct(&pp, &z_test);
    assert!(verify_open(&pp, &c_masked, &z_test), "Masked CT commit should verify");
    
    // Test precomputed constant-time commit
    let c_precomp = commit_precomp_ct(&pp, &z_test);
    assert!(verify_open(&pp, &c_precomp, &z_test), "Precomp CT commit should verify");
    
    // Test that both are deterministic
    let c_masked_2 = commit_masked_ct(&pp, &z_test);
    let c_precomp_2 = commit_precomp_ct(&pp, &z_test);
    assert_eq!(c_masked, c_masked_2, "Masked CT should be deterministic");
    assert_eq!(c_precomp, c_precomp_2, "Precomp CT should be deterministic");
    
    // Test that different inputs produce different outputs
    let mut z_different = z_test.clone();
    z_different[0] = z_different[0] + Fq::ONE;
    
    let c_masked_diff = commit_masked_ct(&pp, &z_different);
    let c_precomp_diff = commit_precomp_ct(&pp, &z_different);
    
    assert_ne!(c_masked, c_masked_diff, "Masked CT: different inputs should produce different outputs");
    assert_ne!(c_precomp, c_precomp_diff, "Precomp CT: different inputs should produce different outputs");
    
    // Both different outputs should still verify
    assert!(verify_open(&pp, &c_masked_diff, &z_different), "Masked CT different input should verify");
    assert!(verify_open(&pp, &c_precomp_diff, &z_different), "Precomp CT different input should verify");
    
    println!("✅ Constant-time commit variants work correctly and maintain basic security properties");
}

#[test]
fn streaming_matches_batch_computation() {
    // Test that the streaming single-row computation matches the batch computation
    use rand::{SeedableRng, rngs::StdRng};
    
    let mut rng = StdRng::seed_from_u64(12345);
    let d = neo_math::D;
    let kappa = 4;
    let m = 3;

    let pp = setup(&mut rng, d, kappa, m).expect("setup ok");
    let z_len = d * m;
    let num_coords = d * kappa;

    // Compute all rows using the original batch method
    let batch_rows = rows_for_coords(&pp, z_len, num_coords).expect("batch rows ok");

    // Compute the same rows using the streaming method
    let mut streaming_rows = Vec::new();
    for coord_idx in 0..num_coords {
        let row = compute_single_ajtai_row(&pp, coord_idx, z_len, num_coords)
            .expect("streaming row ok");
        streaming_rows.push(row);
    }

    // Compare results
    assert_eq!(batch_rows.len(), streaming_rows.len(), "Row count mismatch");
    
    for (i, (batch_row, streaming_row)) in batch_rows.iter().zip(streaming_rows.iter()).enumerate() {
        assert_eq!(batch_row.len(), streaming_row.len(), "Row {} length mismatch", i);
        
        for (j, (&batch_val, &streaming_val)) in batch_row.iter().zip(streaming_row.iter()).enumerate() {
            assert_eq!(batch_val, streaming_val, 
                "Mismatch at row {} position {}: batch={:?} vs streaming={:?}", 
                i, j, batch_val, streaming_val);
        }
    }
    
    println!("✅ Streaming computation matches batch computation for {} rows", num_coords);
}

#[test]
fn rows_for_coords_matches_commit() {
    // Test that rows_for_coords produces the correct linear algebra representation
    use rand::{SeedableRng, rngs::StdRng, Rng};
    
    let mut rng = StdRng::seed_from_u64(42);
    let d = neo_math::D;
    let kappa = 4;
    let m = 3;

    let pp = setup(&mut rng, d, kappa, m).expect("setup ok");

    // Create random Z vector
    let mut z = vec![Fq::ZERO; d*m];
    for x in &mut z { 
        // Generate random field element (equivalent to sample_uniform_fq)
        *x = Fq::from_u64(rng.random::<u64>());
    }

    // Build rows using rows_for_coords
    let z_len = d*m;
    let num_coords = d*kappa;
    let rows = rows_for_coords(&pp, z_len, num_coords).expect("rows ok");

    // Compute L·z (matrix-vector product)
    let mut c_flat = vec![Fq::ZERO; num_coords];
    for (row_i, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), z_len);
        let mut acc = Fq::ZERO;
        for (a, &b) in row.iter().zip(&z) { 
            acc += (*a) * b; 
        }
        c_flat[row_i] = acc;
    }

    // Compute commitment and flatten to column-major order
    let c = commit_masked_ct(&pp, &z);
    let mut c_flat_expected = vec![Fq::ZERO; num_coords];
    for i in 0..kappa {
        for r in 0..d {
            c_flat_expected[i*d + r] = c.data[i * d + r];
        }
    }
    
    // The linear algebra representation should match the commitment
    assert_eq!(c_flat, c_flat_expected, "L·z must equal vec(commit(pp, Z))");
    
    println!("✅ rows_for_coords produces correct linear algebra representation");
}
