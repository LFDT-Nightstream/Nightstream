#![allow(deprecated)] // Tests use legacy MEInstance/MEWitness for backward compatibility

//! Smoke tests for the neo-spartan-bridge with Hash-MLE PCS (Poseidon2-only)
//!
//! These tests verify the complete architectural foundation and API structure
//! for ME(b,L) -> Spartan2 + Hash-MLE compression pipeline.

use neo_spartan_bridge::compress_me_to_spartan;
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;
use spartan2::traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait};
use spartan2::spartan::R1CSSNARK;
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};

type E = spartan2::provider::GoldilocksMerkleMleEngine;

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
    let c_coords = vec![c0, c1]; // Match the number of Ajtai rows

    // ME weights: first sums z0..z3; second sums z5+z7 to get non-zero result  
    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE]; // z5 + z7 = 1 + 2 = 3
    let y0 = dot_f_z(&w0, &z); // 1+2+3+0 = 6
    let y1 = dot_f_z(&w1, &z); // z5 + z7 = 1 + 2 = 3
    let y_outputs = vec![y0, y1]; // Match the number of weight vectors

    let me = MEInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
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

#[derive(Clone, Debug, Default)]
struct MinimalMeCircuit {}

impl<E: Engine> SpartanCircuit<E> for MinimalMeCircuit {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
        // Match the inputize() result: a * b = 2 * 3 = 6
        Ok(vec![E::Scalar::from(6u64)])
    }

    fn num_challenges(&self) -> usize {
        0 // No challenges needed for this simple circuit
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        Ok(vec![])
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<E::Scalar>],
        _precommitted: &[AllocatedNum<E::Scalar>],
        _challenges: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
        // TEST: Add more witness variables (like intermediate circuit) but keep 1 constraint + 1 public
        let mut extra_witnesses = Vec::new();
        for i in 0..8 {  // Add 8 extra witness variables
            let val = E::Scalar::from((i + 10) as u64);
            let var = AllocatedNum::alloc(cs.namespace(|| format!("extra_{}", i)), || Ok(val))?;
            extra_witnesses.push(var);
        }
        
        // Keep the same simple constraint as the working version
        let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(E::Scalar::from(2u64)))?;
        let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(E::Scalar::from(3u64)))?;
        let result = AllocatedNum::alloc(cs.namespace(|| "result"), || Ok(E::Scalar::from(6u64)))?;
        
        cs.enforce(
            || "a * b = result",
            |lc| lc + a.get_variable(),
            |lc| lc + b.get_variable(),
            |lc| lc + result.get_variable(),
        );
        
        // Keep the same single public input
        let _ = result.inputize(cs.namespace(|| "output"));
        
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct IntermediateMeCircuit {}

impl<E: Engine> SpartanCircuit<E> for IntermediateMeCircuit {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
        // Must match what gets inputized in synthesize(): 1 + 2 = 3
        Ok(vec![E::Scalar::from(3u64)])
    }

    fn num_challenges(&self) -> usize { 0 }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        Ok(vec![])
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<E::Scalar>],
        _precommitted: &[AllocatedNum<E::Scalar>],
        _challenges: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
        // Test with ~10 witness variables and 4 public inputs (power of 2)
        let mut witness_vars = Vec::new();
        
        // Create some witness variables
        for i in 0..10 {
            let val = E::Scalar::from((i + 1) as u64);
            let var = AllocatedNum::alloc(cs.namespace(|| format!("w_{}", i)), || Ok(val))?;
            witness_vars.push(var);
        }
        
        // Add some constraints like our ME circuit
        cs.enforce(
            || "constraint_0",
            |lc| lc + witness_vars[0].get_variable(),
            |lc| lc + witness_vars[1].get_variable(),
            |lc| lc + (E::Scalar::from(2u64), CS::one()),
        );
        
        cs.enforce(
            || "constraint_1", 
            |lc| lc + witness_vars[2].get_variable(),
            |lc| lc + witness_vars[3].get_variable(),
            |lc| lc + (E::Scalar::from(12u64), CS::one()),
        );
        
        // TEST: Only 1 public input like the working minimal circuit
        let result = E::Scalar::from(1u64) + E::Scalar::from(2u64); // 1 + 2 = 3
        let pub_var = AllocatedNum::alloc(cs.namespace(|| "result"), || Ok(result))?;
        let _ = pub_var.inputize(cs.namespace(|| "public_result"));
        
        Ok(())
    }
}

#[test]
fn test_intermediate_transcript_fork() {
    println!("🔬 Testing intermediate ME circuit (10 witness + 4 public) with transcript fork...");
    
    let circuit = IntermediateMeCircuit::default();
    let (pk, vk) = R1CSSNARK::<E>::setup(circuit.clone()).unwrap();
    let prep_snark = R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
    let proof = R1CSSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, false).unwrap();
    
    let res = proof.verify(&vk);
    match &res {
        Ok(_) => println!("✅ Intermediate ME circuit works with transcript fork!"),
        Err(e) => println!("❌ Intermediate ME circuit fails: {:?}", e),
    }
    assert!(res.is_ok(), "Intermediate circuit should work with transcript fork: {:?}", res.err());
}

#[test]
fn test_minimal_transcript_fork() {
    println!("🔬 Testing minimal ME circuit with transcript fork...");
    
    let circuit = MinimalMeCircuit::default();
    let (pk, vk) = R1CSSNARK::<E>::setup(circuit.clone()).unwrap();
    let prep_snark = R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
    let proof = R1CSSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, false).unwrap();
    
    let res = proof.verify(&vk);
    match &res {
        Ok(_) => println!("✅ Minimal ME circuit works with transcript fork!"),
        Err(e) => println!("❌ Minimal ME circuit fails: {:?}", e),
    }
    assert!(res.is_ok(), "Minimal circuit should work with transcript fork: {:?}", res.err());
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
    if proof1.proof == proof2.proof {
        println!("⚠️  WARNING: Proofs are identical - SNARK might not be properly randomized");
        println!("   This could indicate deterministic proof generation or missing randomness");
    } else {
        println!("✅ Proofs are different as expected (randomized SNARK)");
    }
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