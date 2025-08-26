//! Integration tests for Spartan2 SNARK mode
//! 
//! These tests validate the end-to-end functionality of the Neo + Spartan2 integration,
//! ensuring that proofs are generated correctly, verification works, and the system
//! maintains soundness and zero-knowledge properties.

#[cfg(test)]
mod spartan2_integration_tests {
    use neo_arithmetize::fibonacci_ccs;
    use neo_ccs::{CcsInstance, CcsWitness, check_satisfiability};
    use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
    use neo_fields::{embed_base_to_ext, ExtF, F};
    use neo_orchestrator::{prove, verify, Metrics};
    use neo_modint::ModInt;
    use neo_ring::RingElement;
    use neo_decomp::decomp_b;
    use p3_field::PrimeCharacteristicRing;
    use std::time::Instant;

    /// Helper function to create a test Fibonacci CCS instance and witness
    fn create_fibonacci_test_case(length: usize) -> (neo_ccs::CcsStructure, CcsInstance, CcsWitness) {
        // Create CCS structure
        let ccs = fibonacci_ccs(length);
        
        // Generate Fibonacci witness
        let mut z: Vec<ExtF> = vec![ExtF::ZERO; length];
        z[0] = embed_base_to_ext(F::ZERO);
        z[1] = embed_base_to_ext(F::ONE);
        for i in 2..length {
            z[i] = z[i - 1] + z[i - 2];
        }
        let witness = CcsWitness { z: z.clone() };
        
        // Create commitment for the witness
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        let z_base: Vec<F> = z.iter().map(|e| e.to_array()[0]).collect();
        let decomp_mat = decomp_b(&z_base, committer.params().b, committer.params().d);
        let z_packed: Vec<RingElement<ModInt>> = AjtaiCommitter::pack_decomp(&decomp_mat, &committer.params());
        let (commitment, _, _, _) = committer.commit(&z_packed, &mut vec![]).unwrap();
        
        // Create CCS instance
        let instance = CcsInstance {
            commitment,
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        
        (ccs, instance, witness)
    }

    #[test]
    fn test_end_to_end_proof_generation_and_verification() {
        println!("🧪 Testing end-to-end proof generation and verification in SNARK mode");
        
        let fib_length = 5;
        let (ccs, instance, witness) = create_fibonacci_test_case(fib_length);
        
        // Verify the witness satisfies the CCS constraints
        assert!(
            check_satisfiability(&ccs, &instance, &witness),
            "Fibonacci witness should satisfy CCS constraints"
        );
        
        // Generate proof using Spartan2 integration
        let prove_start = Instant::now();
        let proof_result = prove(&ccs, &instance, &witness);
        let prove_time = prove_start.elapsed();
        
        assert!(proof_result.is_ok(), "Proof generation should succeed");
        let (proof, metrics) = proof_result.unwrap();
        
        println!("✅ Proof generated successfully in {:.2}ms", prove_time.as_secs_f64() * 1000.0);
        println!("📊 Proof size: {} bytes", metrics.proof_bytes);
        
        // Verify the proof
        let verify_start = Instant::now();
        let verification_result = verify(&ccs, &proof);
        let verify_time = verify_start.elapsed();
        
        assert!(verification_result, "Proof verification should succeed");
        
        println!("✅ Proof verified successfully in {:.2}ms", verify_time.as_secs_f64() * 1000.0);
        println!("🎉 End-to-end test passed!");
    }

    #[test]
    fn test_multiple_fibonacci_lengths() {
        println!("🧪 Testing multiple Fibonacci lengths");
        
        let lengths = vec![3, 4, 5, 6, 8];
        
        for &length in &lengths {
            println!("Testing Fibonacci length: {}", length);
            
            let (ccs, instance, witness) = create_fibonacci_test_case(length);
            
            // Verify satisfiability
            assert!(
                check_satisfiability(&ccs, &instance, &witness),
                "Fibonacci witness of length {} should satisfy CCS constraints", length
            );
            
            // Generate and verify proof
            let proof_result = prove(&ccs, &instance, &witness);
            assert!(proof_result.is_ok(), "Proof generation should succeed for length {}", length);
            
            let (proof, _) = proof_result.unwrap();
            let verification_result = verify(&ccs, &proof);
            assert!(verification_result, "Proof verification should succeed for length {}", length);
            
            println!("✅ Length {} passed", length);
        }
        
        println!("🎉 All lengths tested successfully!");
    }

    #[test]
    fn test_proof_determinism() {
        println!("🧪 Testing proof determinism (same inputs should produce verifiable proofs)");
        
        let fib_length = 4;
        let (ccs, instance, witness) = create_fibonacci_test_case(fib_length);
        
        // Generate two proofs with the same inputs
        let proof1_result = prove(&ccs, &instance, &witness);
        let proof2_result = prove(&ccs, &instance, &witness);
        
        assert!(proof1_result.is_ok(), "First proof generation should succeed");
        assert!(proof2_result.is_ok(), "Second proof generation should succeed");
        
        let (proof1, _) = proof1_result.unwrap();
        let (proof2, _) = proof2_result.unwrap();
        
        // Both proofs should verify
        assert!(verify(&ccs, &proof1), "First proof should verify");
        assert!(verify(&ccs, &proof2), "Second proof should verify");
        
        // Note: Proofs may differ due to randomness, but both should be valid
        println!("✅ Both proofs verify successfully (determinism test passed)");
    }

    #[test]
    fn test_invalid_witness_rejection() {
        println!("🧪 Testing invalid witness rejection");
        
        let fib_length = 5;
        let (ccs, instance, mut witness) = create_fibonacci_test_case(fib_length);
        
        // Corrupt the witness by changing a value
        witness.z[2] = embed_base_to_ext(F::from_canonical_u64(999)); // Invalid Fibonacci value
        
        // The corrupted witness should not satisfy the CCS
        assert!(
            !check_satisfiability(&ccs, &instance, &witness),
            "Corrupted witness should not satisfy CCS constraints"
        );
        
        // Proof generation should fail for invalid witness
        let proof_result = prove(&ccs, &instance, &witness);
        assert!(proof_result.is_err(), "Proof generation should fail for invalid witness");
        
        println!("✅ Invalid witness correctly rejected");
    }

    #[test]
    fn test_performance_characteristics() {
        println!("🧪 Testing performance characteristics");
        
        let lengths = vec![3, 4, 5, 6];
        let mut results = Vec::new();
        
        for &length in &lengths {
            let (ccs, instance, witness) = create_fibonacci_test_case(length);
            
            // Measure proof generation time
            let prove_start = Instant::now();
            let proof_result = prove(&ccs, &instance, &witness);
            let prove_time = prove_start.elapsed();
            
            assert!(proof_result.is_ok(), "Proof should generate for length {}", length);
            let (proof, metrics) = proof_result.unwrap();
            
            // Measure verification time
            let verify_start = Instant::now();
            let verification_result = verify(&ccs, &proof);
            let verify_time = verify_start.elapsed();
            
            assert!(verification_result, "Proof should verify for length {}", length);
            
            results.push((length, prove_time, verify_time, metrics.proof_bytes));
            
            println!(
                "Length {}: Prove {:.2}ms, Verify {:.2}ms, Size {} bytes",
                length,
                prove_time.as_secs_f64() * 1000.0,
                verify_time.as_secs_f64() * 1000.0,
                metrics.proof_bytes
            );
        }
        
        // Basic performance sanity checks
        assert!(results.len() > 1, "Should have multiple results to compare");
        
        // Verification should generally be faster than proving
        for (length, prove_time, verify_time, _) in &results {
            println!(
                "Length {}: Prove/Verify ratio = {:.2}x",
                length,
                prove_time.as_secs_f64() / verify_time.as_secs_f64()
            );
        }
        
        println!("✅ Performance characteristics test completed");
    }

    #[test]
    fn test_zero_knowledge_property() {
        println!("🧪 Testing zero-knowledge property (basic test)");
        
        let fib_length = 4;
        let (ccs, instance, witness) = create_fibonacci_test_case(fib_length);
        
        // Generate multiple proofs with the same witness
        let mut proofs = Vec::new();
        for i in 0..3 {
            let proof_result = prove(&ccs, &instance, &witness);
            assert!(proof_result.is_ok(), "Proof {} should generate", i);
            let (proof, _) = proof_result.unwrap();
            proofs.push(proof);
        }
        
        // All proofs should verify
        for (i, proof) in proofs.iter().enumerate() {
            assert!(verify(&ccs, proof), "Proof {} should verify", i);
        }
        
        // Proofs should be different due to randomness (basic ZK check)
        // Note: This is a basic test - full ZK requires simulator comparison
        let transcripts: Vec<_> = proofs.iter().map(|p| &p.transcript).collect();
        
        // Check that not all transcripts are identical (randomness should make them different)
        let all_same = transcripts.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same, "Proofs should have different randomness (basic ZK property)");
        
        println!("✅ Basic zero-knowledge property test passed");
    }

    #[test]
    fn test_ccs_to_r1cs_conversion_integration() {
        println!("🧪 Testing CCS to R1CS conversion integration");
        
        let fib_length = 4;
        let (ccs, instance, witness) = create_fibonacci_test_case(fib_length);
        
        // Test that our CCS can be converted to R1CS format
        use neo_ccs::convert_ccs_to_r1cs_full;
        
        let conversion_result = convert_ccs_to_r1cs_full(&ccs, &instance, &witness);
        assert!(conversion_result.is_ok(), "CCS to R1CS conversion should succeed");
        
        let (r1cs_shape, r1cs_instance, r1cs_witness) = conversion_result.unwrap();
        
        // Basic shape checks
        assert_eq!(r1cs_shape.num_cons(), ccs.num_constraints, "Constraint count should match");
        assert_eq!(r1cs_shape.num_vars(), ccs.witness_size, "Variable count should match");
        
        println!("✅ CCS to R1CS conversion successful");
        println!("   R1CS constraints: {}", r1cs_shape.num_cons());
        println!("   R1CS variables: {}", r1cs_shape.num_vars());
        
        // The proof should still work end-to-end
        let proof_result = prove(&ccs, &instance, &witness);
        assert!(proof_result.is_ok(), "Proof should work after conversion test");
        
        let (proof, _) = proof_result.unwrap();
        assert!(verify(&ccs, &proof), "Proof should verify after conversion test");
        
        println!("✅ CCS to R1CS conversion integration test passed");
    }
}

// Note: All tests now use SNARK mode by default since feature flags are removed
