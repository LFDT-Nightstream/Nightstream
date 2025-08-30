#[allow(clippy::uninlined_format_args, non_snake_case, clippy::identity_op, clippy::useless_vec)]
use neo_ajtai::{setup, commit, verify_open, verify_split_open, decomp_b, split_b, assert_range_b, DecompStyle};
use neo_ajtai::util::to_balanced_i128;
use p3_goldilocks::Goldilocks as Fq;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use neo_math::ring::D;

fn rng() -> ChaCha20Rng { ChaCha20Rng::from_seed([7u8;32]) }

#[test]
fn tampering_a_digit_breaks_opening() {
    let mut r = rng();
    let d = D; let kappa = 3usize; let m = 4usize;
    let pp = setup(&mut r, d, kappa, m);

    // Build small Z (d×m), here just zeros except a couple of ones
    let mut Z = vec![Fq::ZERO; d*m];
    Z[0] = Fq::ONE; 
    Z[d+1] = Fq::from_u64(5);

    let c = commit(&pp, &Z);

    // Tamper one digit
    let mut Z_bad = Z.clone();
    Z_bad[d+1] += Fq::ONE;

    assert!(verify_open(&pp, &c, &Z));
    assert!(!verify_open(&pp, &c, &Z_bad));
    println!("✅ RED TEAM: Commitment opening tampering correctly detected");
}

#[test]
fn split_recomposition_rejects_tampered_piece() {
    let mut r = rng();
    let d = D; let kappa = 2usize; let m = 3usize; let b=2u32; let k=4usize;
    let pp = setup(&mut r, d, kappa, m);

    // Simple column vector then decomposed & split
    let z = vec![Fq::from_u64(3); m]; // original vector to decompose
    let Z = decomp_b(&z, b, d, DecompStyle::Balanced);
    let parts = split_b(&Z, b, d, m, k, DecompStyle::Balanced);
    let c_parts: Vec<_> = parts.iter().map(|Zi| commit(&pp, Zi)).collect();
    let c = commit(&pp, &Z);

    // Good split is accepted
    assert!(verify_split_open(&pp, &c, b, &c_parts, &parts));

    // Tamper one piece
    let mut c_parts_bad = c_parts.clone();
    c_parts_bad[0].data[0] += Fq::ONE;
    assert!(!verify_split_open(&pp, &c, b, &c_parts_bad, &parts));
    println!("✅ RED TEAM: Split recomposition tampering correctly rejected");
}

#[test]
fn range_assertion_catches_out_of_range_coeff() {
    let b = 2u32;
    let mut Z = vec![Fq::ZERO; D*2];
    Z[0] = Fq::from_u64(b as u64); // equals b, should violate |.| < b
    let res = std::panic::catch_unwind(|| assert_range_b(&Z, b));
    assert!(res.is_err(), "range check should panic on |x| >= b");
    println!("✅ RED TEAM: Range assertion correctly catches out-of-range coefficients");
}

#[test]
fn balanced_map_is_centered() {
    // sanity: to_balanced_i128 respects symmetry for small values
    for v in [-3i64, -1, 0, 1, 3] {
        let x = if v>=0 { 
            Fq::from_u64(v as u64) 
        } else { 
            Fq::ZERO - Fq::from_u64((-v) as u64) 
        };
        let m = to_balanced_i128(x);
        assert!(m == v as i128);
    }
    println!("✅ RED TEAM: Balanced mapping preserves signed values correctly");
}

#[test]
fn commitment_linearity_under_tampering() {
    let mut r = rng();
    let d = D; let kappa = 2usize; let m = 3usize;
    let pp = setup(&mut r, d, kappa, m);

    // Create two different Z matrices
    let Z1 = vec![Fq::from_u64(1); d*m];
    let Z2 = vec![Fq::from_u64(2); d*m];
    
    let c1 = commit(&pp, &Z1);
    let c2 = commit(&pp, &Z2);
    
    // Z1 + Z2 = Z3
    let mut Z3 = vec![Fq::ZERO; d*m];
    for i in 0..d*m {
        Z3[i] = Z1[i] + Z2[i];
    }
    let c3 = commit(&pp, &Z3);
    
    // c1 + c2 should equal c3 (linearity)
    let mut c1_plus_c2 = c1.clone();
    c1_plus_c2.add_inplace(&c2);
    
    assert_eq!(c1_plus_c2, c3);
    
    // Now tamper Z3 and verify linearity breaks
    let mut Z3_bad = Z3.clone();
    Z3_bad[0] += Fq::ONE;
    let c3_bad = commit(&pp, &Z3_bad);
    
    assert_ne!(c1_plus_c2, c3_bad);
    println!("✅ RED TEAM: Commitment linearity verified and tampering detection works");
}

#[test]
fn decomposition_recomposition_consistency() {
    let b = 4u32;
    let d = D;
    let m = 5usize;
    
    // Test several values including edge cases
    let test_values = vec![
        Fq::ZERO,
        Fq::ONE,
        Fq::from_u64(b as u64 - 1),
        Fq::from_u64((b * b) as u64 - 1),
        Fq::ZERO - Fq::ONE, // -1
        Fq::ZERO - Fq::from_u64(b as u64 - 1), // -(b-1)
    ];
    
    for &val in &test_values {
        let z = vec![val; m];
        let Z = decomp_b(&z, b, d, DecompStyle::Balanced);
        
        // Range check
        assert_range_b(&Z, b);
        
        // Recompose and verify
        let mut recomposed = vec![Fq::ZERO; m];
        let mut power = Fq::ONE;
        let b_f = Fq::from_u64(b as u64);
        
        for digit_idx in 0..d {
            for col in 0..m {
                recomposed[col] += Z[col * d + digit_idx] * power;
            }
            power *= b_f;
        }
        
        for j in 0..m {
            assert_eq!(recomposed[j], z[j], "Recomposition failed for value {:?}", val);
        }
    }
    
    println!("✅ RED TEAM: Decomposition-recomposition consistency verified for edge cases");
}

#[test]
fn split_recomposition_edge_cases() {
    let b = 2u32;
    let d = D;
    let m = 2usize;
    let k = 8usize; // enough digits for large values
    
    // Create a Z matrix with large values that require splitting
    let mut Z = vec![Fq::ZERO; d*m];
    Z[0] = Fq::from_u64((1u64 << 10) - 1); // Large value requiring multiple digits
    Z[d + 0] = Fq::from_u64((1u64 << 8) - 1);  // Another large value
    
    let parts = split_b(&Z, b, d, m, k, DecompStyle::Balanced);
    
    // Verify each part satisfies range
    for (i, part) in parts.iter().enumerate() {
        assert_range_b(part, b);
        println!("Part {} satisfies range constraint", i);
    }
    
    // Recompose and verify
    let mut recomposed = vec![Fq::ZERO; d*m];
    let mut power = Fq::ONE;
    let b_f = Fq::from_u64(b as u64);
    
    for part in &parts {
        for i in 0..d*m {
            recomposed[i] += part[i] * power;
        }
        power *= b_f;
    }
    
    for i in 0..d*m {
        assert_eq!(recomposed[i], Z[i], "Split-recomposition failed at index {}", i);
    }
    
    println!("✅ RED TEAM: Split-recomposition works correctly for large values");
}

#[test]
fn commitment_binding_under_different_inputs() {
    let mut r = rng();
    let d = D; let kappa = 3usize; let m = 4usize;
    let pp = setup(&mut r, d, kappa, m);
    
    // Generate several different Z matrices
    let test_matrices = vec![
        vec![Fq::ZERO; d*m],
        vec![Fq::ONE; d*m],
        (0..d*m).map(|i| Fq::from_u64(i as u64)).collect::<Vec<_>>(),
        (0..d*m).map(|i| Fq::from_u64((i * 7) as u64)).collect::<Vec<_>>(),
    ];
    
    let commitments: Vec<_> = test_matrices.iter().map(|z| commit(&pp, z)).collect();
    
    // All commitments should be different (binding property)
    for i in 0..commitments.len() {
        for j in (i+1)..commitments.len() {
            assert_ne!(commitments[i], commitments[j], 
                      "Commitments {} and {} should be different", i, j);
        }
    }
    
    // Verify each opens correctly to its own matrix
    for (z, c) in test_matrices.iter().zip(commitments.iter()) {
        assert!(verify_open(&pp, c, z));
    }
    
    // Verify no commitment opens to a different matrix
    for (i, c) in commitments.iter().enumerate() {
        for (j, z) in test_matrices.iter().enumerate() {
            if i != j {
                assert!(!verify_open(&pp, c, z), 
                       "Commitment {} should not open to matrix {}", i, j);
            }
        }
    }
    
    println!("✅ RED TEAM: Commitment binding property verified across different inputs");
}

#[test]
fn zero_knowledge_like_properties() {
    let mut r = rng();
    let d = D; let kappa = 2usize; let m = 3usize;
    let pp = setup(&mut r, d, kappa, m);
    
    // Same value, different randomness (via different setup)
    let mut r2 = ChaCha20Rng::from_seed([42u8;32]);
    let pp2 = setup(&mut r2, d, kappa, m);
    
    let Z = vec![Fq::from_u64(42); d*m];
    let c1 = commit(&pp, &Z);
    let c2 = commit(&pp2, &Z);
    
    // Different setups should give different commitments even for same Z
    assert_ne!(c1, c2);
    
    // But each should verify correctly under its own setup
    assert!(verify_open(&pp, &c1, &Z));
    assert!(verify_open(&pp2, &c2, &Z));
    
    // And not verify under the wrong setup
    assert!(!verify_open(&pp, &c2, &Z));
    assert!(!verify_open(&pp2, &c1, &Z));
    
    println!("✅ RED TEAM: Different setups produce different commitments (setup binding)");
}
