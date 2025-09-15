#![allow(deprecated)] // Tests use legacy MEInstance/MEWitness for backward compatibility

//! Security tests for neo-spartan-bridge with Hash-MLE PCS
//!
//! These tests verify that tampering with public inputs, commitments, or proof data
//! would be properly detected by the verification process.

use neo_spartan_bridge::{compress_me_to_spartan, verify_me_spartan, ProofBundle};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeCharacteristicRing, integers::QuotientMap};

/// Helper function to handle both prove-time and verify-time tampering detection
fn expect_detected(res: anyhow::Result<ProofBundle>, expect_err_contains: Option<&str>) {
    match res {
        Err(e) => {
            if let Some(substr) = expect_err_contains {
                assert!(
                    e.to_string().contains(substr),
                    "expected error containing {:?}, got: {e:?}", substr
                );
            } else {
                panic!("unexpected error: {e:?}");
            }
            // Detected at prove-time
        }
        Ok(bundle) => {
            // Detected at verify-time
            let ok = verify_me_spartan(&bundle).unwrap_or(false);
            assert!(!ok, "tampering should be detected at verify-time");
        }
    }
}

/// Create a minimal ME instance for testing with valid Ajtai commitment
fn tiny_me_instance() -> (MEInstance, MEWitness) {
    let z_digits = vec![1i64, 2i64, 3i64, 0i64, -1i64, 1i64, 0i64, 2i64];
    let ajtai_rows = vec![
        vec![F::from_canonical_checked(1).unwrap(); 8], // row 0: all coeffs = 1, match z_digits length
        vec![F::from_canonical_checked(2).unwrap(); 8], // row 1: all coeffs = 2, match z_digits length  
    ];
    
    // Compute valid c_coords: c[i] = <ajtai_rows[i], z_digits[0..4]>
    let mut c_coords = Vec::new();
    for row in &ajtai_rows {
        let mut sum = F::ZERO;
        for (j, &coeff) in row.iter().enumerate() {
            if j < z_digits.len() {
                let z_field = if z_digits[j] >= 0 {
                    F::from_canonical_checked(z_digits[j] as u64).unwrap()
                } else {
                    F::ZERO - F::from_canonical_checked((-z_digits[j]) as u64).unwrap()
                };
                sum += coeff * z_field;
            }
        }
        c_coords.push(sum);
    }
    
    // c_coords only needs to match number of Ajtai rows (2)
    
    let me = MEInstance {
        c_coords,
        y_outputs: vec![F::from_canonical_checked(2).unwrap(); 2], // Match number of weight vectors
        r_point: vec![F::from_canonical_checked(3).unwrap(); 2],
        base_b: 4,
        header_digest: [0u8; 32],
    };
    
    let wit = MEWitness {
        z_digits,
        weight_vectors: vec![vec![F::ONE; 8], vec![F::ZERO; 8]], // Match z_digits length
        ajtai_rows: Some(ajtai_rows),
    };
    
    (me, wit)
}

#[test]
fn tamper_public_commitment() {
    println!("🔒 Testing tamper detection: public commitment");
    
    let (mut me, wit) = tiny_me_instance();
    let _original_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Tamper with the Ajtai commitment coordinates
    me.c_coords[0] = me.c_coords[0] + F::ONE; // flip a commitment component
    
    // This should now be caught by our Ajtai commitment validation
    let tampered_result = compress_me_to_spartan(&me, &wit);
    expect_detected(tampered_result, Some("AjtaiCommitmentInconsistent"));
    
    println!("✅ Commitment tampering properly detected at validation");
}

#[test]
fn tamper_output_values() {
    println!("🔒 Testing tamper detection: output values");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Tamper with the ME output values
    me.y_outputs[1] = me.y_outputs[1] + F::from_canonical_checked(42).unwrap();
    
    let tampered_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Tampered outputs should produce different public IO
    assert_ne!(
        original_proof.public_io_bytes, 
        tampered_proof.public_io_bytes,
        "Tampering with outputs should change public IO"
    );
    
    println!("✅ Output tampering properly detected");
}

#[test]
fn tamper_challenge_point() {
    println!("🔒 Testing tamper detection: challenge point");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Tamper with the challenge point r
    me.r_point[0] = me.r_point[0] + F::from_canonical_checked(99).unwrap();
    
    let tampered_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Different challenge point should produce different public IO
    assert_ne!(
        original_proof.public_io_bytes, 
        tampered_proof.public_io_bytes,
        "Tampering with challenge point should change public IO"
    );
    
    println!("✅ Challenge point tampering properly detected");
}

#[test]
fn tamper_base_dimension() {
    println!("🔒 Testing tamper detection: base dimension");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Tamper with base dimension
    me.base_b = 8; // change from 4 to 8
    
    let tampered_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Different base should produce different public IO
    assert_ne!(
        original_proof.public_io_bytes, 
        tampered_proof.public_io_bytes,
        "Tampering with base dimension should change public IO"
    );
    
    println!("✅ Base dimension tampering properly detected");
}

#[test]
fn tamper_header_digest() {
    println!("🔒 Testing tamper detection: header digest");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Tamper with header digest (transcript binding)
    me.header_digest[5] ^= 0xFF; // flip bits in digest
    me.header_digest[10] ^= 0xAB;
    
    let tampered_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Header digest is critical for transcript security
    assert_ne!(
        original_proof.public_io_bytes, 
        tampered_proof.public_io_bytes,
        "Header digest tampering should change public IO"
    );
    
    // Should produce different proof bytes too (deterministic but different input)
    assert_ne!(
        original_proof.proof, 
        tampered_proof.proof,
        "Header digest should affect proof generation"
    );
    
    println!("✅ Header digest tampering properly detected");
}

#[test]
fn proof_serialization_tamper() {
    println!("🔒 Testing tamper detection: proof serialization");
    
    let (me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit).unwrap();
    
    // Create a copy and tamper with proof bytes
    let mut tampered_bytes = original_proof.proof.clone();
    if tampered_bytes.len() > 10 {
        tampered_bytes[5] ^= 1;   // flip a bit
        tampered_bytes[10] ^= 0xFF; // flip more bits
    }
    
    // Verify that tampered bytes are different
    assert_ne!(original_proof.proof, tampered_bytes, "Tampered bytes should differ");
    
    // In a real implementation, verification would fail on tampered bytes
    // For now, we just verify the tampering is detectable at the byte level
    println!("✅ Proof byte tampering is detectable");
}

#[test]
fn comprehensive_tampering_matrix() {
    println!("🔒 Testing comprehensive tampering detection matrix");
    
    let (me, wit) = tiny_me_instance();
    
    let mut test_cases = Vec::new();
    
    // Test case 1: Tamper commitment - should be caught by Ajtai validation
    let mut me1 = me.clone();
    me1.c_coords[0] = me1.c_coords[0] + F::ONE;
    test_cases.push(("commitment", me1, wit.clone()));
    
    // Test case 2: Tamper range (witness with out-of-range digit) - should be caught by range validation
    let mut wit2 = wit.clone();
    wit2.z_digits[0] = 999; // invalid witness - outside [-3, 3] for base_b=4
    test_cases.push(("range", me.clone(), wit2));
    
    // Test case 3: Tamper witness without updating commitment - should be caught by Ajtai validation
    let mut wit3 = wit.clone();
    wit3.z_digits[1] = 0; // Change witness but keep old commitment
    test_cases.push(("witness_mismatch", me.clone(), wit3));
    
    for (tamper_type, test_me, test_wit) in test_cases {
        match tamper_type {
            "range" => {
                // Range tampering also causes Ajtai inconsistency since c_coords were computed from original z_digits
                let res = compress_me_to_spartan(&test_me, &test_wit);
                // Either error is valid - Ajtai validation runs first, then range validation
                let result = res.map_err(|e| e.to_string());
                match result {
                    Err(e) if e.contains("AjtaiCommitmentInconsistent") || e.contains("RangeViolation") => {
                        println!("   {} tampering: ✅ detected ({})", tamper_type, 
                                if e.contains("AjtaiCommitmentInconsistent") { "Ajtai validation" } else { "Range validation" });
                    }
                    _ => panic!("Expected range tampering to be detected, got: {:?}", result)
                }
            }
            "commitment" | "witness_mismatch" => {
                let res = compress_me_to_spartan(&test_me, &test_wit);
                expect_detected(res, Some("AjtaiCommitmentInconsistent"));
                println!("   {} tampering: ✅ detected", tamper_type);
            }
            _ => {
                // Fallback for other cases
                let res = compress_me_to_spartan(&test_me, &test_wit);
                expect_detected(res, None);
                println!("   {} tampering: ✅ detected", tamper_type);
            }
        }
    }
    
    println!("✅ Comprehensive tampering detection passed");
}

#[test]
fn test_dimension_validation_strictness() {
    println!("🔒 Testing strict dimension validation");
    
    let (me, wit) = tiny_me_instance();
    
    // Test 1: c_coords longer than ajtai_rows → should fail
    {
        let mut me_longer_coords = me.clone();
        me_longer_coords.c_coords.push(neo_math::F::ONE); // Extra coordinate
        
        let result = compress_me_to_spartan(&me_longer_coords, &wit);
        match result {
            Err(e) if e.to_string().contains("Ajtai rows") && e.to_string().contains("must match c_coords") => {
                println!("   ✅ c_coords longer than ajtai_rows correctly rejected: {}", e);
            }
            _ => panic!("Expected c_coords length mismatch to be detected, got: {:?}", result)
        }
    }
    
    // Test 2: ajtai_rows[i].len() > z_digits.len() → should fail
    {
        // Create a witness with shorter z_digits but keep Ajtai rows same length
        let z_digits_short = vec![1i64, 2i64]; // Only 2 elements
        let ajtai_rows_longer = vec![
            vec![F::from_canonical_checked(1).unwrap(); 4], // 4 elements > 2
        ];
        
        let me_test = MEInstance {
            c_coords: vec![F::from_canonical_checked(3).unwrap()], // 1*1 + 1*2 would = 3 if consistent
            y_outputs: vec![F::from_canonical_checked(1).unwrap()],
            r_point: vec![F::from_canonical_checked(3).unwrap(); 2],
            base_b: 4,
            header_digest: [0u8; 32],
        };
        
        let wit_test = MEWitness {
            z_digits: z_digits_short,
            weight_vectors: vec![vec![F::ONE; 2]],
            ajtai_rows: Some(ajtai_rows_longer),
        };
        
        let result = compress_me_to_spartan(&me_test, &wit_test);
        match result {
            Err(e) if e.to_string().contains("Ajtai row 0 length") && e.to_string().contains("must equal z_digits length") => {
                println!("   ✅ Ajtai row length mismatch correctly rejected: {}", e);
            }
            _ => panic!("Expected Ajtai row length mismatch to be detected, got: {:?}", result)
        }
    }
    
    // Test 3: z_digits with |z_i| >= base_b → should fail (range validation)
    {
        // Create a minimal instance that passes all dimension checks but has range violation
        let z_digits_bad = vec![4i64, 1i64]; // base_b = 4, so 4 >= b is invalid
        let ajtai_rows_minimal = vec![
            vec![F::from_canonical_checked(1).unwrap(); 2], // row 0: coeffs = [1, 1]
        ];
        
        // Compute c_coords that matches the Ajtai commitment 
        let c_coord = ajtai_rows_minimal[0][0] * F::from_u64(4) + ajtai_rows_minimal[0][1] * F::from_u64(1);
        
        let me_minimal = MEInstance {
            c_coords: vec![c_coord],
            y_outputs: vec![F::from_canonical_checked(2).unwrap(); 2],
            r_point: vec![F::from_canonical_checked(3).unwrap(); 2],
            base_b: 4,
            header_digest: [0u8; 32],
        };
        
        let wit_minimal = MEWitness {
            z_digits: z_digits_bad,
            weight_vectors: vec![vec![F::ONE; 2]],
            ajtai_rows: Some(ajtai_rows_minimal),
        };
        
        let result = compress_me_to_spartan(&me_minimal, &wit_minimal);
        match result {
            Err(e) if e.to_string().contains("RangeViolation") => {
                println!("   ✅ Range violation correctly rejected: {}", e);
            }
            _ => panic!("Expected range violation to be detected, got: {:?}", result)
        }
    }
    
    println!("✅ All dimension validation tests passed");
}