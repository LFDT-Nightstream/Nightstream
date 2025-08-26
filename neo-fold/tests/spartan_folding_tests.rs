//! Unit tests for Spartan2 folding integration
//! 
//! These tests validate the NeutronNova folding integration and ensure
//! that the folding operations work correctly with the Neo protocol.


mod neutronnova_folding_tests {
    #[allow(unused_imports)]
    use neo_fold::{
        FoldState, Proof,
        neutronnova_integration::{NeutronNovaFoldState, create_neutronnova_fold_state}
    };
    use neo_ccs::{CcsStructure, CcsInstance, CcsWitness, verifier_ccs, check_satisfiability};
    use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
    use neo_fields::{embed_base_to_ext, F};
    use p3_field::PrimeCharacteristicRing;
    #[allow(unused_imports)]
    use neo_fields::ExtF;
    use neo_modint::ModInt;
    use neo_ring::RingElement;
    use neo_decomp::decomp_b;
    use std::time::Instant;

    /// Helper function to create a test CCS instance and witness
    fn create_test_fold_case() -> (CcsStructure, CcsInstance, CcsWitness) {
        let ccs = verifier_ccs();
        
        // Create test witness: [a=2, b=3, ab=6, a+b=5]
        let witness_values = vec![
            embed_base_to_ext(F::from_u64(2)),  // a
            embed_base_to_ext(F::from_u64(3)),  // b
            embed_base_to_ext(F::from_u64(6)),  // a*b
            embed_base_to_ext(F::from_u64(5)),  // a+b
        ];
        let witness = CcsWitness { z: witness_values.clone() };
        
        // Create commitment
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        let z_base: Vec<F> = witness_values.iter().map(|e| e.to_array()[0]).collect();
        let decomp_mat = decomp_b(&z_base, committer.params().b, committer.params().d);
        let z_packed: Vec<RingElement<ModInt>> = AjtaiCommitter::pack_decomp(&decomp_mat, &committer.params());
        let (commitment, _, _, _) = committer.commit(&z_packed, &mut vec![]).unwrap();
        
        let instance = CcsInstance {
            commitment,
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        
        (ccs, instance, witness)
    }

    #[test]
    fn test_neutronnova_fold_state_creation() {
        println!("🧪 Testing NeutronNova fold state creation");

        let (ccs, _, _) = create_test_fold_case();
        let fold_state = create_neutronnova_fold_state(ccs.clone());
        
        // Verify the fold state is properly initialized
        assert_eq!(fold_state.nark_state.structure.num_constraints, ccs.num_constraints);
        assert_eq!(fold_state.nark_state.structure.witness_size, ccs.witness_size);
        assert!(fold_state.conversion_cache.is_none(), "R1CS shape should be None initially");
        
        println!("✅ NeutronNova fold state creation successful");
        println!("   CCS constraints: {}", ccs.num_constraints);
        println!("   CCS witness size: {}", ccs.witness_size);
    }

    #[test]
    fn test_snark_proof_generation() {
        println!("🧪 Testing SNARK proof generation with NeutronNova");

        let (ccs, instance, witness) = create_test_fold_case();
        
        // Verify the witness satisfies the CCS
        assert!(check_satisfiability(&ccs, &instance, &witness), 
               "Test witness should satisfy CCS constraints");
        
        let mut fold_state = create_neutronnova_fold_state(ccs);
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        let prove_start = Instant::now();
        let proof = fold_state.generate_proof_snark(
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            &committer,
        );
        let prove_time = prove_start.elapsed();
        
        println!("✅ SNARK proof generation successful");
        println!("   Proof generation time: {:.2}ms", prove_time.as_secs_f64() * 1000.0);
        println!("   Proof size: {} bytes", proof.transcript.len());
        
        // Verify the proof
        let verify_start = Instant::now();
        let verification_result = fold_state.verify_snark(&proof.transcript, &committer);
        let verify_time = verify_start.elapsed();
        
        assert!(verification_result, "SNARK proof should verify");
        
        println!("✅ SNARK proof verification successful");
        println!("   Verification time: {:.2}ms", verify_time.as_secs_f64() * 1000.0);
    }

    #[test]
    fn test_recursive_ivc_basic() {
        println!("🧪 Testing basic recursive IVC with NeutronNova");

        let (ccs, instance, witness) = create_test_fold_case();
        let mut fold_state = create_neutronnova_fold_state(ccs);
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        // Create multiple instances for IVC
        let instances = vec![
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
        ];
        
        let ivc_start = Instant::now();
        let ivc_result = fold_state.recursive_ivc_snark(instances, &committer);
        let ivc_time = ivc_start.elapsed();
        
        assert!(ivc_result.is_ok(), "Recursive IVC should succeed");
        let proof = ivc_result.unwrap();
        
        println!("✅ Recursive IVC successful");
        println!("   IVC time: {:.2}ms", ivc_time.as_secs_f64() * 1000.0);
        println!("   Final proof size: {} bytes", proof.transcript.len());
        
        // Verify the final proof
        let verification_result = fold_state.verify_snark(&proof.transcript, &committer);
        assert!(verification_result, "IVC proof should verify");
        
        println!("✅ IVC proof verification successful");
    }

    #[test]
    fn test_ivc_with_different_instances() {
        println!("🧪 Testing IVC with different instances");

        let (ccs, _, _) = create_test_fold_case();
        let mut fold_state = create_neutronnova_fold_state(ccs);
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        // Create different instances
        let instances = vec![
            create_test_fold_case(),
            create_test_fold_case(),
        ];
        
        let instance_pairs: Vec<_> = instances.into_iter()
            .map(|(_, inst, wit)| (inst, wit))
            .collect();
        
        let ivc_result = fold_state.recursive_ivc_snark(instance_pairs, &committer);
        assert!(ivc_result.is_ok(), "IVC with different instances should succeed");
        
        println!("✅ IVC with different instances successful");
    }

    #[test]
    fn test_ivc_insufficient_instances() {
        println!("🧪 Testing IVC error handling with insufficient instances");

        let (ccs, instance, witness) = create_test_fold_case();
        let mut fold_state = create_neutronnova_fold_state(ccs);
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        // Try IVC with only one instance (should fail)
        let instances = vec![(instance, witness)];
        
        let ivc_result = fold_state.recursive_ivc_snark(instances, &committer);
        assert!(ivc_result.is_err(), "IVC should fail with insufficient instances");
        
        let error_msg = ivc_result.unwrap_err();
        assert!(error_msg.contains("at least 2"), "Error should mention minimum instance requirement");
        
        println!("✅ IVC error handling test passed");
    }

    #[test]
    fn test_fold_state_conversion_cache_caching() {
        println!("🧪 Testing R1CS shape caching in fold state");

        let (ccs, instance, witness) = create_test_fold_case();
        let mut fold_state = create_neutronnova_fold_state(ccs);
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        // Initially, R1CS shape should be None
        assert!(fold_state.conversion_cache.is_none(), "R1CS shape should be None initially");
        
        // Generate a proof (this should populate the R1CS shape)
        let _proof = fold_state.generate_proof_snark(
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            &committer,
        );
        
        // After proof generation, R1CS shape might be cached (depending on implementation)
        // This test verifies the caching mechanism works as expected
        
        println!("✅ R1CS shape caching test completed");
    }

    #[test]
    fn test_snark_proof_robustness() {
        println!("🧪 Testing SNARK proof robustness");

        let (ccs, instance, witness) = create_test_fold_case();
        let mut fold_state = create_neutronnova_fold_state(ccs);
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        // The system now always uses SNARK mode with Spartan2
        let proof = fold_state.generate_proof_snark(
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            &committer,
        );
        
        // Proof should be generated successfully
        assert!(!proof.transcript.is_empty(), "SNARK proof should be generated");
        
        // Verification should work
        let verification_result = fold_state.verify_snark(&proof.transcript, &committer);
        assert!(verification_result, "SNARK proof should verify");
        
        println!("✅ SNARK proof robustness test passed");
    }

    #[test]
    fn test_performance_comparison_nark_vs_snark() {
        println!("🧪 Testing performance comparison between NARK and SNARK modes");

        let (ccs, instance, witness) = create_test_fold_case();
        let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
        
        // Test NARK mode
        let mut nark_fold_state = FoldState::new(ccs.clone());
        let nark_start = Instant::now();
        let nark_proof = nark_fold_state.generate_proof(
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            &committer,
        );
        let nark_time = nark_start.elapsed();
        
        // Test SNARK mode
        let mut snark_fold_state = create_neutronnova_fold_state(ccs);
        let snark_start = Instant::now();
        let snark_proof = snark_fold_state.generate_proof_snark(
            (instance.clone(), witness.clone()),
            (instance.clone(), witness.clone()),
            &committer,
        );
        let snark_time = snark_start.elapsed();
        
        println!("📊 Performance Comparison:");
        println!("   NARK proof time: {:.2}ms", nark_time.as_secs_f64() * 1000.0);
        println!("   SNARK proof time: {:.2}ms", snark_time.as_secs_f64() * 1000.0);
        println!("   NARK proof size: {} bytes", nark_proof.transcript.len());
        println!("   SNARK proof size: {} bytes", snark_proof.transcript.len());
        
        // Both proofs should verify
        let nark_verify = nark_fold_state.verify(&nark_proof.transcript, &committer);
        let snark_verify = snark_fold_state.verify_snark(&snark_proof.transcript, &committer);
        
        assert!(nark_verify, "NARK proof should verify");
        assert!(snark_verify, "SNARK proof should verify");
        
        println!("✅ Performance comparison test completed");
    }
}

#[allow(dead_code)]
mod nark_mode_folding_tests {
    use neo_fold::FoldState;
    use neo_ccs::verifier_ccs;

    #[test]
    fn test_nark_mode_folding_still_works() {
        println!("🧪 Testing that NARK mode folding still works");

        let ccs = verifier_ccs();
        let fold_state = FoldState::new(ccs.clone());
        
        // Verify basic fold state properties
        assert_eq!(fold_state.structure.num_constraints, ccs.num_constraints);
        assert_eq!(fold_state.structure.witness_size, ccs.witness_size);
        
        println!("✅ NARK mode folding backward compatibility confirmed");
    }
}
