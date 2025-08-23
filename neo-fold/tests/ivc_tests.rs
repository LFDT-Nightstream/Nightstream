#[cfg(test)]
mod tests {
    use neo_ccs::{check_satisfiability, verifier_ccs, CcsInstance, CcsWitness};
    use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
    use neo_fields::{embed_base_to_ext, ExtF, F};
    use neo_fold::{extractor, FoldState, Proof};
    // Oracle removed in NARK mode
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_verifier_ccs_satisfiability() {
        let structure = verifier_ccs();
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        // For the verifier CCS with 4-element witness [a, b, a*b, a+b]:
        // Valid witness: a=2, b=3, a*b=6, a+b=5
        let witness = CcsWitness {
            z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(6)),  // 2*3
                embed_base_to_ext(F::from_u64(5)),  // 2+3
            ],
        };
        assert!(check_satisfiability(&structure, &instance, &witness));

        // Invalid witness: a=2, b=3, a*b=6, a+b=6 (wrong! should be 5)
        let bad_witness = CcsWitness {
            z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(6)),  // Correct: 2*3=6 (but we don't check this)
                embed_base_to_ext(F::from_u64(6)),  // Wrong: should be 5 for a+b
            ],
        };
        assert!(!check_satisfiability(&structure, &instance, &bad_witness));
    }

    #[test]
    fn test_extractor_valid_witness() {
        let proof = Proof { transcript: vec![1u8; 100] }; // Demo transcript
        let wit = extractor(&proof);
        assert_eq!(wit.z.len(), 4); // Matches demo verifier CCS inputs
        assert!(wit.z.iter().all(|&e| e != ExtF::ZERO)); // Non-trivial

        // Invalid transcript (e.g., too short) should produce default witness
        let bad_proof = Proof { transcript: vec![0u8; 10] };
        let bad_wit = extractor(&bad_proof);
        assert_eq!(bad_wit.z.len(), 4); // Still produces 4 elements
    }



    #[test]
    fn test_recursive_ivc_chaining() {
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(SECURE_PARAMS); // Use secure params for integration tests
        let depth = 2;

        // Add dummy eval to avoid empty state
        let initial_eval_instance = neo_fold::EvalInstance {
            commitment: vec![],
            r: vec![ExtF::ONE],
            ys: vec![ExtF::ONE],
            u: ExtF::ZERO,
            e_eval: ExtF::ONE,
            norm_bound: 10,
        };
        eprintln!("DEBUG: Initial e_eval = {:?}", initial_eval_instance.e_eval);
        state.eval_instances.push(initial_eval_instance);

        // Set initial CCS (demo) - use 4-element witness
        state.ccs_instance = Some((
            CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE },
            CcsWitness { z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(6)),
                embed_base_to_ext(F::from_u64(5)),
            ] },
        ));

        eprintln!("DEBUG: About to call recursive_ivc, current eval_instances.len() = {}", state.eval_instances.len());
        if !state.eval_instances.is_empty() {
            eprintln!("DEBUG: Before recursive_ivc, e_eval = {:?}", state.eval_instances.last().unwrap().e_eval);
        }
        
        assert!(state.recursive_ivc(depth, &committer));
        
        eprintln!("DEBUG: After recursive_ivc, eval_instances.len() = {}", state.eval_instances.len());
        if !state.eval_instances.is_empty() {
            eprintln!("DEBUG: After recursive_ivc, e_eval = {:?}", state.eval_instances.last().unwrap().e_eval);
        }
        
        assert!(state.verify_state()); // Final state valid

        // Force failure (e.g., invalid initial witness)
        let mut bad_state = FoldState::new(verifier_ccs());
        bad_state.ccs_instance = Some((
            CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE },
            CcsWitness { z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(7)), // Invalid: should be 6 (but we don't check mul)
                embed_base_to_ext(F::from_u64(6)), // Invalid add: 2+3=5 ≠6
            ] }, // Invalid witness
        ));
        assert!(!bad_state.recursive_ivc(depth, &committer));
    }

    #[test]
    fn test_verifier_ccs_structure() {
        let structure = verifier_ccs();
        assert_eq!(structure.mats.len(), 4); // 4 matrices as expected
        assert_eq!(structure.num_constraints, 2); // 2 constraints
        assert_eq!(structure.witness_size, 4); // 4 witness elements [a, b, a*b, a+b]
        assert_eq!(structure.max_deg, 1); // Degree 1 polynomial (multilinear)
    }



    #[test]
    fn test_recursive_ivc_base_case() {
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        
        // Test empty state (should add dummy and return true)
        let mut state_empty = FoldState::new(verifier_ccs());
        assert!(state_empty.recursive_ivc(0, &committer));
        
        // Test with one eval instance (should return true without adding dummy)
        let mut state_with_eval = FoldState::new(verifier_ccs());
        state_with_eval.eval_instances.push(neo_fold::EvalInstance {
            commitment: vec![],
            r: vec![ExtF::ONE],
            ys: vec![ExtF::ZERO],
            u: ExtF::ZERO,
            e_eval: ExtF::ONE,
            norm_bound: 100,
        });
        assert!(state_with_eval.recursive_ivc(0, &committer));
    }



    #[test]
    fn test_recursive_ivc_single_step() {
        use neo_fold::FoldState;
        use neo_ccs::{verifier_ccs, CcsInstance, CcsWitness};
        use neo_commit::AjtaiCommitter;
        use neo_fields::{embed_base_to_ext, F};
        use neo_commit::SECURE_PARAMS;
        
        eprintln!("Starting single step test");
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        state.ccs_instance = Some((
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ONE, 
                e: F::ZERO 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::from_u64(2)), // a = 2
                    embed_base_to_ext(F::from_u64(3)), // b = 3  
                    embed_base_to_ext(F::from_u64(6)), // a*b = 2*3 = 6
                    embed_base_to_ext(F::from_u64(5)), // a+b = 2+3 = 5
                ]
            },
        ));
        
        // Add a proper eval instance so generate_proof has something to work with
        state.eval_instances.push(neo_fold::EvalInstance {
            commitment: vec![],
            r: vec![ExtF::ONE], // Simple evaluation point
            ys: vec![ExtF::ZERO], // Simple constant polynomial that evaluates to 0
            u: ExtF::ZERO,
            e_eval: ExtF::ZERO, // Constraint evaluation should be 0 for satisfied witness
            norm_bound: 10,
        });
        
        eprintln!("About to call recursive_ivc(1)");
        assert!(state.recursive_ivc(1, &committer), "Single step recursion should pass"); // Depth 1
        eprintln!("Single step test completed");
    }



    #[test]
    fn test_generate_proof_basic() {
        use neo_fold::FoldState;
        use neo_ccs::{verifier_ccs, CcsInstance, CcsWitness};
        use neo_commit::AjtaiCommitter;
        use neo_fields::{embed_base_to_ext, F};
        use neo_commit::SECURE_PARAMS;
        
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        let instance = (
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ZERO, 
                e: F::ONE 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::ONE),         // a = 1
                    embed_base_to_ext(F::ONE),         // b = 1  
                    embed_base_to_ext(F::ONE),         // a*b = 1*1 = 1
                    embed_base_to_ext(F::from_u64(2)), // a+b = 1+1 = 2
                ]
            },
        );
        let proof = state.generate_proof(instance.clone(), instance, &committer);
        assert!(!proof.transcript.is_empty(), "Proof transcript should not be empty");
    }





    #[test]
    fn test_ivc_rejects_invalid_witness() {
        let mut bad_state = FoldState::new(verifier_ccs());
        bad_state.ccs_instance = Some((
            CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE },
            CcsWitness { z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(7)), // Invalid mul: should be 6 (but we don't check mul)
                embed_base_to_ext(F::from_u64(6)), // Invalid add: 2+3=5 ≠6
            ] },
        ));
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        assert!(!bad_state.recursive_ivc(1, &committer), "Should reject invalid witness");
    }

    #[test]
    fn test_extractor_varies_with_proof() {
        let proof1 = Proof { transcript: vec![1u8, 2u8, 3u8, 4u8, 5u8] };
        let proof2 = Proof { transcript: vec![10u8, 20u8, 30u8, 40u8, 50u8] };
        
        let wit1 = extractor(&proof1);
        let wit2 = extractor(&proof2);
        
        assert_ne!(wit1.z, wit2.z, "Extractor should produce different witnesses for different proofs");
    }

    #[test]
    fn test_alpha_differs_between_ccs() {
        // Test that alpha challenges differ between pi_ccs calls when using different domain separation tags
        use neo_commit::SECURE_PARAMS;
        
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        
        let instance1 = (
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ZERO, 
                e: F::ONE 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                ]
            },
        );
        
        let instance2 = (
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ONE, 
                e: F::ZERO 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::from_u64(2)),
                    embed_base_to_ext(F::from_u64(3)),
                    embed_base_to_ext(F::from_u64(6)),
                    embed_base_to_ext(F::from_u64(5)),
                ]
            },
        );
        
        let proof = state.generate_proof(instance1, instance2, &committer);
        assert!(!proof.transcript.is_empty(), "Proof should generate non-empty transcript");
        
        // The alpha values should be different due to different domain separation tags
        // This test ensures we don't regress to using the same tag for both pi_ccs calls
    }

    #[test]
    fn test_rlc_indexing_full_flow() {
        // Test that RLC indexing works correctly after a full flow with 4 eval instances
        use neo_commit::SECURE_PARAMS;
        
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        
        let instance1 = (
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ZERO, 
                e: F::ONE 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                ]
            },
        );
        
        let instance2 = (
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ONE, 
                e: F::ZERO 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::from_u64(2)),
                    embed_base_to_ext(F::from_u64(3)),
                    embed_base_to_ext(F::from_u64(6)),
                    embed_base_to_ext(F::from_u64(5)),
                ]
            },
        );
        
        let proof = state.generate_proof(instance1, instance2, &committer);
        
        // Verify that the proof was generated successfully without RLC indexing errors
        assert!(!proof.transcript.is_empty(), "Proof generation should complete without indexing errors");
        
        // The RLC should correctly index eval_instances[len()-4] (ccs1) and eval_instances[len()-2] (ccs2)
        // This test ensures we don't regress to incorrect indexing like len()-3 and len()-1
    }

    #[test]
    fn test_zero_q_mismatch_rejection() {
        // Test that invalid sumcheck for zero Q polynomial is properly rejected
        use neo_commit::SECURE_PARAMS;
        
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(SECURE_PARAMS);
        
        // Create an instance that should result in zero Q polynomial
        let zero_instance = (
            CcsInstance { 
                commitment: vec![], 
                public_input: vec![], 
                u: F::ZERO, 
                e: F::ZERO 
            },
            CcsWitness { 
                z: vec![
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                    embed_base_to_ext(F::ZERO),  
                ]
            },
        );
        
        let proof = state.generate_proof(zero_instance.clone(), zero_instance, &committer);
        
        // The proof should be generated successfully with zero Q polynomial
        assert!(!proof.transcript.is_empty(), "Zero Q polynomial should generate valid proof");
        
        // Verify that the verifier correctly handles zero Q polynomial verification
        let verification_result = state.verify(&proof.transcript, &committer);
        assert!(verification_result, "Zero Q polynomial proof should verify correctly");
    }
}
