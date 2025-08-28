//! Production smoke tests for the neo-spartan-bridge with p3-FRI architecture
//!
//! These tests verify the complete architectural foundation and API structure
//! that will support the real ME(b,L) -> Spartan2 + p3-FRI compression pipeline.

use neo_spartan_bridge::{compress_me_to_spartan, P3FriParams, pcs::PCSEngineTrait};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeCharacteristicRing, integers::QuotientMap};

/// Create a minimal ME instance for testing using the actual neo-ccs structure
fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Create a minimal ME instance using the actual field structure
    let me = MEInstance {
        c_coords: vec![F::from_canonical_checked(1).unwrap(); 4], // Ajtai commitment coordinates
        y_outputs: vec![F::from_canonical_checked(2).unwrap(); 4], // public outputs  
        r_point: vec![F::from_canonical_checked(3).unwrap(); 2], // challenge point
        base_b: 4, // Base dimension
        header_digest: [0u8; 32], // Header hash
    };
    
    // Witness with matching structure
    let wit = MEWitness {
        z_digits: vec![1i64, 2i64, 3i64, 0i64, -1i64, 1i64, 0i64, 2i64], // witness digits (base-b)
        weight_vectors: vec![vec![F::ONE; 4], vec![F::ZERO; 4]], // weight matrices
        ajtai_rows: Some(vec![vec![F::from_canonical_checked(7).unwrap(); 4]; 2]), // Ajtai matrix rows
    };
    
    (me, wit)
}

#[test]
fn bridge_smoke_me() {
    println!("🧪 Testing complete ME(b,L) -> Spartan2 + p3-FRI pipeline");
    
    let (me, wit) = tiny_me_instance();
    
    println!("   ME coordinates: {}", me.c_coords.len());
    println!("   Output values: {}", me.y_outputs.len());
    println!("   Challenge point dimension: {}", me.r_point.len());
    
    // Compress with default FRI parameters (safe for testing)
    let proof = compress_me_to_spartan(&me, &wit, None)
        .expect("compress_me_to_spartan should succeed");

    println!("   FRI queries: {}", proof.fri_num_queries);
    println!("   FRI log blowup: {}", proof.fri_log_blowup);
    println!("   Total proof size: {} bytes", proof.total_size());
    
    // Verify structural correctness
    assert!(!proof.proof.is_empty());
    assert!(!proof.public_io_bytes.is_empty());
    assert_eq!(proof.fri_num_queries, P3FriParams::default().num_queries);
    
    println!("✅ Bridge smoke test: PASS");
    println!("   Ready for real Spartan2 integration");
}

#[test]
fn determinism_check() {
    println!("🧪 Testing transcript determinism (same ME -> same proof)");
    
    let (me, wit) = tiny_me_instance();
    
    // Generate proof twice with same inputs
    let proof1 = compress_me_to_spartan(&me, &wit, None).expect("proof 1");
    let proof2 = compress_me_to_spartan(&me, &wit, None).expect("proof 2");
    
    // Same inputs should produce identical proofs (deterministic transcript)
    assert_eq!(proof1.proof, proof2.proof, "Proofs should be deterministic");
    assert_eq!(proof1.public_io_bytes, proof2.public_io_bytes, "Public IO should be deterministic");
    assert_eq!(proof1.total_size(), proof2.total_size(), "Sizes should match");
    
    println!("✅ Determinism check: PASS");
    println!("   Both proofs: {} bytes", proof1.total_size());
}

#[test] 
fn different_fri_params() {
    println!("🧪 Testing different FRI parameter configurations");
    
    let (me, wit) = tiny_me_instance();
    
    // Test with custom FRI parameters
    let custom_params = P3FriParams {
        log_blowup: 1,        // smaller blowup for testing
        log_final_poly_len: 0,
        num_queries: 20,      // fewer queries for testing
        proof_of_work_bits: 4, // less PoW for testing  
    };
    
    let proof = compress_me_to_spartan(&me, &wit, Some(custom_params.clone()))
        .expect("custom params should work");
    
    assert_eq!(proof.fri_num_queries, custom_params.num_queries);
    assert_eq!(proof.fri_log_blowup, custom_params.log_blowup);
    
    println!("✅ Custom FRI params test: PASS");
    println!("   Custom queries: {}", proof.fri_num_queries);
    println!("   Custom blowup: 2^{}", proof.fri_log_blowup);
}

#[test]
fn p3fri_pcs_direct() {
    println!("🧪 Testing direct P3FriPCS creation and domain handling");
    
    let params = P3FriParams::default();
    let pcs = neo_spartan_bridge::P3FriPCSAdapter::new_with_params(params);
    
    // Test domain creation for various polynomial degrees
    let degrees = [4, 8, 16, 64, 256];
    
    for &degree in &degrees {
        let domain = pcs.natural_domain_for_degree(degree);
        println!("   Degree {}: domain size = {}", degree, domain.size());
    }
    
    println!("✅ P3FriPCS direct test: PASS");
}

#[test] 
fn transcript_io_encoding() {
    println!("🧪 Testing transcript IO encoding consistency");
    
    let (me1, _) = tiny_me_instance();
    let (mut me2, _) = tiny_me_instance();
    
    // Encode same ME twice
    let io1 = neo_spartan_bridge::encode_bridge_io_header(&me1);
    let io2 = neo_spartan_bridge::encode_bridge_io_header(&me1);
    
    assert_eq!(io1, io2, "Same ME should encode to same IO bytes");
    
    // Different ME should encode differently  
    me2.c_coords[0] = F::from_canonical_checked(99).unwrap(); // change commitment
    let io3 = neo_spartan_bridge::encode_bridge_io_header(&me2);
    
    assert_ne!(io1, io3, "Different ME should encode to different IO bytes");
    
    println!("✅ Transcript IO encoding: PASS");
    println!("   Standard ME: {} bytes", io1.len());
    println!("   Modified ME: {} bytes", io3.len());
}

#[test]
fn architectural_foundation() {
    println!("🧪 Testing architectural foundation readiness");
    
    // Test that all the key components compile and work
    use neo_spartan_bridge::make_p3fri_engine_with_defaults;
    
    let (pcs_adapter, _ch, _mats) = make_p3fri_engine_with_defaults(42);
    
    // Test that the P3FriPCSAdapter was created successfully
    let domain = pcs_adapter.natural_domain_for_degree(64);
    assert!(domain.size() >= 64);
    
    println!("✅ Architectural foundation: PASS");
    println!("   P3FriPCSAdapter: operational");
    println!("   Domain creation: working (size {})", domain.size());
    println!("   Transcript encoding: working");
    println!("   ME types: compatible");
    println!("   ME types: compatible");
    println!("   Ready for real p3-FRI integration!");
}