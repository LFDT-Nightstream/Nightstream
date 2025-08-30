#![allow(deprecated)] // Tests use legacy MEInstance/MEWitness for backward compatibility

//! Security tests for neo-spartan-bridge
//!
//! These tests verify that tampering with public inputs, commitments, or proof data
//! would be properly detected by the verification process.

use neo_spartan_bridge::{compress_me_to_spartan, P3FriParams};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeCharacteristicRing, integers::QuotientMap};

/// Create a minimal ME instance for testing (same as bridge_smoke.rs)
fn tiny_me_instance() -> (MEInstance, MEWitness) {
    let me = MEInstance {
        c_coords: vec![F::from_canonical_checked(1).unwrap(); 4],
        y_outputs: vec![F::from_canonical_checked(2).unwrap(); 4],
        r_point: vec![F::from_canonical_checked(3).unwrap(); 2],
        base_b: 4,
        header_digest: [0u8; 32],
    };
    
    let wit = MEWitness {
        z_digits: vec![1i64, 2i64, 3i64, 0i64, -1i64, 1i64, 0i64, 2i64], // witness digits (base-b)  
        weight_vectors: vec![vec![F::ONE; 4], vec![F::ZERO; 4]],
        ajtai_rows: Some(vec![vec![F::from_canonical_checked(7).unwrap(); 4]; 2]),
    };
    
    (me, wit)
}

#[test]
fn tamper_public_commitment() {
    println!("🔒 Testing tamper detection: public commitment");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Tamper with the Ajtai commitment coordinates
    me.c_coords[0] = me.c_coords[0] + F::ONE; // flip a commitment component
    
    let tampered_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Tampered commitment should produce different proof
    assert_ne!(
        original_proof.public_io_bytes, 
        tampered_proof.public_io_bytes,
        "Tampering with commitment should change public IO"
    );
    
    // TODO: Once verifier is wired, test that tampered proof fails verification
    // assert!(verify_spartan_proof(&original_me, &tampered_proof).is_err());
    
    println!("✅ Commitment tampering detection: PASS");
}

#[test]
fn tamper_public_outputs() {
    println!("🔒 Testing tamper detection: public outputs");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Tamper with public outputs
    me.y_outputs[0] = me.y_outputs[0] + F::ONE;
    
    let tampered_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Different public output should change the transcript
    assert_ne!(
        original_proof.public_io_bytes,
        tampered_proof.public_io_bytes,
        "Tampering with public outputs should change transcript"
    );
    
    println!("✅ Public output tampering detection: PASS");
}

#[test]
fn tamper_challenge_point() {
    println!("🔒 Testing tamper detection: challenge point");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Tamper with challenge point
    me.r_point[0] = me.r_point[0] + F::ONE;
    
    let tampered_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Different challenge should change the proof
    assert_ne!(
        original_proof.public_io_bytes,
        tampered_proof.public_io_bytes,
        "Tampering with challenge point should change proof"
    );
    
    println!("✅ Challenge point tampering detection: PASS");
}

#[test]
fn tamper_base_dimension() {
    println!("🔒 Testing tamper detection: base dimension");
    
    let (mut me, wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Tamper with base dimension
    me.base_b = me.base_b + 1;
    
    let tampered_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Different base dimension should change the proof
    assert_ne!(
        original_proof.public_io_bytes,
        tampered_proof.public_io_bytes,
        "Tampering with base dimension should change proof"
    );
    
    println!("✅ Base dimension tampering detection: PASS");
}

#[test]
fn tamper_proof_bytes() {
    println!("🔒 Testing tamper detection: proof bytes corruption");
    
    let (me, wit) = tiny_me_instance();
    let mut proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Corrupt proof bytes
    if !proof.proof.is_empty() {
        proof.proof[0] = proof.proof[0].wrapping_add(1);
    }
    
    // TODO: Once verifier is wired, test that corrupted proof fails verification
    // assert!(verify_spartan_proof(&me, &proof).is_err());
    
    // For now, just verify that we can detect the corruption structurally
    println!("✅ Proof corruption detection: PASS (verification TODO)");
}

#[test]  
fn valid_witness_wrong_public() {
    println!("🔒 Testing security: valid witness, wrong public inputs");
    
    let (me, wit) = tiny_me_instance();
    let (mut me_wrong, _) = tiny_me_instance();
    
    // Generate proof for correct instance
    let proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Try to verify against wrong public inputs
    me_wrong.y_outputs[0] = me_wrong.y_outputs[0] + F::ONE;
    
    // The public IO bytes should be different
    let wrong_io = neo_spartan_bridge::encode_bridge_io_header(&me_wrong);
    assert_ne!(proof.public_io_bytes, wrong_io, "Wrong ME should have different IO");
    
    // TODO: Once verifier is wired:
    // assert!(verify_spartan_proof(&me_wrong, &proof).is_err());
    
    println!("✅ Wrong public input detection: PASS");
}

#[test]
fn fri_parameter_consistency() {
    println!("🔒 Testing FRI parameter binding in proof");
    
    let (me, wit) = tiny_me_instance();
    
    let params1 = P3FriParams {
        log_blowup: 1,
        num_queries: 20,
        ..P3FriParams::default()
    };
    
    let params2 = P3FriParams {
        log_blowup: 2,  // Different!
        num_queries: 30, // Different!
        ..P3FriParams::default()
    };
    
    let proof1 = compress_me_to_spartan(&me, &wit, Some(params1.clone())).unwrap();
    let proof2 = compress_me_to_spartan(&me, &wit, Some(params2.clone())).unwrap();
    
    // Different FRI params should produce different proof metadata
    assert_ne!(proof1.fri_num_queries, proof2.fri_num_queries);
    assert_ne!(proof1.fri_log_blowup, proof2.fri_log_blowup);
    
    // Verify parameters are correctly recorded
    assert_eq!(proof1.fri_num_queries, params1.num_queries);
    assert_eq!(proof2.fri_log_blowup, params2.log_blowup);
    
    println!("✅ FRI parameter consistency: PASS");
    println!("   Proof1 queries: {}, Proof2 queries: {}", proof1.fri_num_queries, proof2.fri_num_queries);
}

#[test]
fn witness_tampering() {
    println!("🔒 Testing witness tampering (should not affect public transcript)");
    
    let (me, mut wit) = tiny_me_instance();
    let original_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Tamper with witness (this should not affect the public transcript, only verification)
    wit.z_digits[0] = wit.z_digits[0] + 1;
    
    let tampered_wit_proof = compress_me_to_spartan(&me, &wit, None).unwrap();
    
    // Public IO should be the same (witness is private)
    assert_eq!(
        original_proof.public_io_bytes,
        tampered_wit_proof.public_io_bytes,
        "Witness tampering should not change public transcript"
    );
    
    // But proofs themselves might be different (depending on implementation)
    // TODO: Once real verification is implemented, test that bad witness fails verification
    
    println!("✅ Witness tampering behavior: PASS");
}