//! Security fixes validation tests
//! 
//! Tests for the critical security fixes implemented in response to the comprehensive audit.
//! These tests ensure that the fixes work correctly and prevent regressions.

use neo_fold::bridge_adapter::{modern_to_legacy_witness, modern_to_legacy_instance};
use neo_transcript::{Poseidon2Transcript, Transcript, labels as tr_labels};
use neo_ccs::{MeInstance, MeWitness, Mat};
use neo_ajtai::setup;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

#[allow(deprecated)]
#[allow(non_snake_case)]

/// Test bridge witness layout: ensure column-major flattening is consistent
#[test]
fn test_bridge_witness_layout_consistency() {
    // Create a small test matrix with distinct values to detect layout swaps
    let d = 3;
    let m = 2;
    let params = neo_params::NeoParams::goldilocks_127();
    
    // Create witness matrix with small sentinel values (within range for b=2)
    // Z = [[0, 1],    <- row 0 (values 0,1 are within range for b=2)
    //      [1, 0],    <- row 1  
    //      [0, 1]]    <- row 2
    let mut witness_data = Vec::with_capacity(d * m);
    for row in 0..d {
        for col in 0..m {
            let val = if (row + col) % 2 == 0 { 0 } else { 1 };
            witness_data.push(F::from_u64(val));
        }
    }
    let witness_matrix = Mat::from_row_major(d, m, witness_data.clone());
    let witness = MeWitness { Z: witness_matrix };
    
    // Convert to legacy format (handling Result type)
    let legacy_witness = modern_to_legacy_witness(&witness, &params)
        .expect("witness conversion should succeed with valid data");
    
    // Verify column-major flattening on the *prefix* (un-padded part): [0,1,0, 1,0,1]
    // Column 0: [0,1,0]; Column 1: [1,0,1]
    let expected_order = [0i64, 1, 0, 1, 0, 1];
    let dm = d * m; // 3*2 = 6

    // Since the bridge now pads z_digits to a power of two for Hash-MLE,
    // the total length should be next_power_of_two(dm) and padding must be zeros.
    let expected_len = dm.next_power_of_two();
    assert_eq!(
        legacy_witness.z_digits.len(), expected_len,
        "z_digits must be padded to next power-of-two ({} -> {})",
        dm, expected_len
    );

    // Check the unpadded prefix matches the intended column-major layout
    assert_eq!(
        &legacy_witness.z_digits[..dm], expected_order.as_slice(),
        "prefix (un-padded) must match column-major flattening"
    );

    // Check the padding is all zeros
    assert!(legacy_witness.z_digits[dm..].iter().all(|&x| x == 0),
        "padding must be zeros");
}

/// Test Π_DEC recomposition: ensure digit recomposition is inverse of decomposition  
#[test]
#[allow(non_snake_case)]
fn test_pi_dec_recomposition_correctness() {
    use neo_ajtai::{decomp_b, DecompStyle};
    let b = 3u32;
    let d = 4; // Enough digits for test values
    
    // Test various values to ensure recomposition is correct
    let test_values = [F::ZERO, F::ONE, F::from_u64(2), F::from_u64(8), F::from_u64(26)];
    
    for &original_value in &test_values {
        // Step 1: Decompose using neo-ajtai
        let z = vec![original_value];
        let Z_col_major = decomp_b(&z, b, d, DecompStyle::Balanced);
        
        // Step 2: Recompose using the same formula as pi_dec.rs
        let base_f = F::from_u64(b as u64);
        let mut recomposed = F::ZERO;
        let mut pow = F::ONE;
        
        for digit_idx in 0..d {
            // Z_col_major is column-major: Z[col * d + row] = Z[0 * d + digit_idx]
            let digit = Z_col_major[digit_idx];
            recomposed += digit * pow;
            pow *= base_f;
        }
        
        // Step 3: Verify recomposition equals original
        assert_eq!(recomposed, original_value,
            "Recomposition failed for value {}: got {} (digits: {:?})",
            original_value.as_canonical_u64(), recomposed.as_canonical_u64(),
            &Z_col_major[..d]);
    }
}

/// Test transcript reproducibility: ensure Poseidon2 parameters are deterministic
#[test] 
fn test_transcript_deterministic_parameters() {
    // Create multiple transcripts with the same seed
    let transcript1 = Poseidon2Transcript::new(b"test_seed");
    let transcript2 = Poseidon2Transcript::new(b"test_seed");
    
    // Clone for separate operations
    let mut tr1 = transcript1;
    let mut tr2 = transcript2;
    
    // Apply identical operations
    tr1.append_message(tr_labels::PI_CCS, b"");
    tr1.append_message(b"bytes", b"test_data");
    tr1.append_fields(b"F", &[F::ONE, F::from_u64(42)]);
    
    tr2.append_message(tr_labels::PI_CCS, b"");
    tr2.append_message(b"bytes", b"test_data");
    tr2.append_fields(b"F", &[F::ONE, F::from_u64(42)]);
    
    // Get state digests - should be identical
    let digest1 = tr1.digest32();
    let digest2 = tr2.digest32();
    
    assert_eq!(digest1, digest2, "Deterministic transcript parameters failed: different digests");
    
    // Test that different seeds produce different results  
    let mut tr3 = Poseidon2Transcript::new(b"different_seed");
    tr3.append_message(tr_labels::PI_CCS, b"");
    tr3.append_message(b"bytes", b"test_data");
    tr3.append_fields(b"F", &[F::ONE, F::from_u64(42)]);
    let digest3 = tr3.digest32();
    
    assert_ne!(digest1, digest3, "Different seeds should produce different digests");
}

/// Test that polynomial absorption is deterministic and affects challenges
#[test]
fn test_polynomial_absorption_affects_challenges() {
    use neo_ccs::{SparsePoly, Term};
    
    // Create two different polynomials
    let poly1 = SparsePoly::new(2, vec![
        Term { coeff: F::ONE, exps: vec![1, 0] },  // x
        Term { coeff: F::from_u64(2), exps: vec![0, 1] }, // 2y  
    ]);
    
    let poly2 = SparsePoly::new(2, vec![
        Term { coeff: F::ONE, exps: vec![1, 0] },  // x
        Term { coeff: F::from_u64(3), exps: vec![0, 1] }, // 3y (different coefficient)
    ]);
    
    // Create transcripts and absorb different polynomials
    let mut tr1 = Poseidon2Transcript::new(b"poly_test");
    tr1.append_message(b"neo/ccs/poly", b"");
    tr1.append_u64s(b"arity", &[poly1.arity() as u64]);
    tr1.append_u64s(b"terms_len", &[poly1.terms().len() as u64]);
    for term in poly1.terms() {
        tr1.append_fields(b"coeff", &[term.coeff]);
        let exps: Vec<u64> = term.exps.iter().map(|&e| e as u64).collect();
        tr1.append_u64s(b"exps", &exps);
    }
    
    let mut tr2 = Poseidon2Transcript::new(b"poly_test");
    tr2.append_message(b"neo/ccs/poly", b"");
    tr2.append_u64s(b"arity", &[poly2.arity() as u64]);
    tr2.append_u64s(b"terms_len", &[poly2.terms().len() as u64]);
    for term in poly2.terms() {
        tr2.append_fields(b"coeff", &[term.coeff]);
        let exps2: Vec<u64> = term.exps.iter().map(|&e| e as u64).collect();
        tr2.append_u64s(b"exps", &exps2);
    }
    
    // Sample challenges - should be different
    let ch1 = tr1.challenge_fields(b"chal/k", 2);
    let ch2 = tr2.challenge_fields(b"chal/k", 2);
    let challenge1 = neo_math::from_complex(ch1[0], ch1[1]);
    let challenge2 = neo_math::from_complex(ch2[0], ch2[1]);
    
    assert_ne!(challenge1, challenge2, 
        "Different polynomials should produce different challenges");
}

/// Test bridge header digest binding  
#[test]
#[allow(deprecated)]
fn test_bridge_header_digest_binding() {
    let mut rng = ChaCha20Rng::seed_from_u64(9999);
    let params = neo_params::NeoParams::goldilocks_127();
    
    // Create two ME instances that differ only in r values  
    // Use correct dimensions: d must match ring dimension (54 for Goldilocks)
    let d = neo_math::D;  // 54 for Goldilocks 
    let m = 4;
    let kappa = 2;
    let pp = setup(&mut rng, d, kappa, m).expect("setup should succeed");
    let witness_data = vec![F::ZERO; d * m]; // Use zeros to stay in range
    let commitment = neo_ajtai::commit(&pp, &witness_data);
    
    let m_in = 2;
    let instance1 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: commitment.clone(),
        X: Mat::from_row_major(d, m_in, vec![F::ZERO; d * m_in]),
        r: vec![K::ONE, K::ZERO],  // Different r values
        y: vec![vec![K::ZERO; d]; 2],
        y_scalars: vec![K::ONE, K::ZERO],
        m_in,
        fold_digest: [0u8; 32],
    };
    
    let instance2 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: commitment,
        X: Mat::from_row_major(d, m_in, vec![F::ZERO; d * m_in]),
        r: vec![K::ZERO, K::ONE], // Different r values  
        y: vec![vec![K::ZERO; d]; 2],
        y_scalars: vec![K::ONE, K::ZERO],
        m_in,
        fold_digest: [0u8; 32],
    };
    
    // Convert to legacy format - should have different header digests
    let legacy1 = modern_to_legacy_instance(&instance1, &params);
    let legacy2 = modern_to_legacy_instance(&instance2, &params);
    
    // Test that conversion succeeds and instances have different header digests
    assert_ne!(legacy1.header_digest, legacy2.header_digest,
        "Different r values should produce different header digests");
}
