#![allow(deprecated)] // Tests use legacy MEInstance/MEWitness for backward compatibility

//! Smoke tests for the neo-spartan-bridge with Hash-MLE PCS (Poseidon2-only)
//!
//! These tests verify the complete architectural foundation and API structure
//! for ME(b,L) -> Spartan2 + Hash-MLE compression pipeline.

use neo_spartan_bridge::compress_me_to_spartan;
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

/// Helper to compute dot product of field vector and i64 witness
fn dot_f_z(row: &[F], z: &[i64]) -> F {
    row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
        let zi_f = if zi >= 0 { F::from_u64(zi as u64) }
                   else         { -F::from_u64((-zi) as u64) };
        acc + (*a) * zi_f
    })
}

/// Create a mathematically consistent ME instance for testing
fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Witness digits (len = 8 = 2^3, perfect for Hash-MLE)
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];

    // Ajtai rows: use simple unit vectors so constraints become z0=c0 and z1=c1
    let ajtai_rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
    ];
    let c0 = dot_f_z(&ajtai_rows[0], &z); // = z0 = 1
    let c1 = dot_f_z(&ajtai_rows[1], &z); // = z1 = 2
    let c_coords = vec![c0, c1, F::ZERO, F::ZERO]; // keep 4 publics; last two unused

    // ME weights: first sums z0..z3; second sums z5+z7 to get non-zero result  
    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE]; // z5 + z7 = 1 + 2 = 3
    let y0 = dot_f_z(&w0, &z); // 1+2+3+0 = 6
    let y1 = dot_f_z(&w1, &z); // z5 + z7 = 1 + 2 = 3
    let y_outputs = vec![y0, y1, F::ZERO, F::ZERO]; // length 4 for stability

    let me = MEInstance {
        c_coords,
        y_outputs,
        r_point: vec![F::from_u64(3); 2], // unused by constraints here
        base_b: 4,
        header_digest: [0u8; 32],
    };

    let wit = MEWitness {
        z_digits: z,
        weight_vectors: vec![w0, w1],
        ajtai_rows: Some(ajtai_rows),
    };

    (me, wit)
}

#[test]
fn bridge_smoke_me_hash_mle() {
    println!("🧪 Testing complete ME(b,L) -> Spartan2 + Hash-MLE pipeline");
    
    let (me, wit) = tiny_me_instance();
    
    println!("   ME coordinates: {}", me.c_coords.len());
    println!("   Output values: {}", me.y_outputs.len());
    println!("   Challenge point dimension: {}", me.r_point.len());
    
    // Compress using Hash-MLE PCS (no FRI parameters needed)
    let proof = match compress_me_to_spartan(&me, &wit) {
        Ok(proof) => proof,
        Err(e) => {
            println!("🚨 Spartan2 API diagnostic:");
            println!("{:?}", e);
            println!("Full error chain:");
            let mut current_error: &dyn std::error::Error = &*e;
            loop {
                println!("  {}", current_error);
                match current_error.source() {
                    Some(source) => current_error = source,
                    None => break,
                }
            }
            panic!("compress_me_to_spartan failed - see diagnostic above");
        }
    };

    println!("   Total proof size: {} bytes", proof.total_size());
    
    // Verify proof structure
    assert!(!proof.proof.is_empty(), "Proof bytes should not be empty");
    assert!(!proof.vk.is_empty(), "Verifier key should not be empty");
    assert!(!proof.public_io_bytes.is_empty(), "Public IO should not be empty");
    
    // Real verification
    let ok = neo_spartan_bridge::verify_me_spartan(&proof).expect("verify should run");
    assert!(ok);
    
    println!("✅ ME(b,L) -> Spartan2 compression succeeded");
}

#[test]
fn determinism_check() {
    println!("🧪 Testing real SNARK proof generation");
    
    let (me, wit) = tiny_me_instance();
    
    // Generate two proofs with identical inputs – proofs are randomized
    let proof1 = compress_me_to_spartan(&me, &wit).expect("first proof");
    let proof2 = compress_me_to_spartan(&me, &wit).expect("second proof");
    
    // Real SNARKs are randomized: proofs usually differ, VKs match, IO matches
    assert_ne!(proof1.proof, proof2.proof, "Proofs need not be bit-equal");
    assert_eq!(proof1.vk, proof2.vk, "VK should be deterministic");
    assert_eq!(proof1.public_io_bytes, proof2.public_io_bytes, "Public IO should be stable");
    assert!(neo_spartan_bridge::verify_me_spartan(&proof1).unwrap());
    assert!(neo_spartan_bridge::verify_me_spartan(&proof2).unwrap());
    
    println!("✅ Real SNARK proof generation and verification succeeded");
}

#[test]
fn header_binding_consistency() {
    println!("🧪 Testing header digest binding");
    
    let (mut me, wit) = tiny_me_instance();
    
    // Generate proof with original header
    let proof1 = compress_me_to_spartan(&me, &wit).expect("original proof");
    
    // Change header digest
    me.header_digest[0] ^= 1; // flip one bit
    let proof2 = compress_me_to_spartan(&me, &wit).expect("modified proof");
    
    // Header digest change should affect public IO
    assert_ne!(
        proof1.public_io_bytes, 
        proof2.public_io_bytes,
        "Header digest must bind transcript"
    );
    
    println!("✅ Header digest properly binds to transcript");
}

#[test]
fn proof_bundle_structure() {
    println!("🧪 Testing ProofBundle structure");
    
    let (me, wit) = tiny_me_instance();
    let bundle = compress_me_to_spartan(&me, &wit).expect("proof bundle");
    
    // Verify bundle contains expected components
    assert!(bundle.proof.len() > 0, "Proof component should have content");
    assert!(bundle.vk.len() > 0, "Verifier key should have content");
    assert!(bundle.public_io_bytes.len() > 0, "Public IO should have content");
    
    // Verify size calculation
    let expected_size = bundle.proof.len() + bundle.vk.len() + bundle.public_io_bytes.len();
    assert_eq!(bundle.total_size(), expected_size, "Size calculation should be correct");
    
    println!("   Proof size: {} bytes", bundle.proof.len());
    println!("   VK size: {} bytes", bundle.vk.len());
    println!("   Public IO size: {} bytes", bundle.public_io_bytes.len());
    println!("✅ ProofBundle structure is valid");
}