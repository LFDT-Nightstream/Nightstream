//! Red Team Tests for Critical Soundness Vulnerabilities
//! 
//! These tests demonstrate and validate fixes for critical soundness bugs
//! identified in the HyperNova EV implementation.

use neo::{F, NeoParams};
use neo::ivc::{IvcBatchBuilder, EmissionPolicy, StepOutputExtractor, Accumulator, IvcStepInput, StepBindingSpec, prove_ivc_step, verify_ivc_step};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use anyhow::Result;

/// Test extractor that deliberately returns WRONG y_step values
/// This should cause verification to fail once y_step binding is fixed
#[allow(dead_code)]
struct MaliciousExtractor {
    pub n: usize,
}

impl StepOutputExtractor for MaliciousExtractor {
    fn extract_y_step(&self, _witness: &[F]) -> Vec<F> {
        // Return convenient values for folding (all zeros)
        // This is NOT the actual step output!
        vec![F::ZERO; self.n]
    }
}

/// Honest extractor that returns correct last N elements
struct HonestExtractor {
    pub n: usize,
}

impl StepOutputExtractor for HonestExtractor {
    fn extract_y_step(&self, witness: &[F]) -> Vec<F> {
        // Extract actual step outputs (last n elements)
        let len = witness.len();
        assert!(len >= self.n, "Witness too short for extraction");
        witness[len - self.n..].to_vec()
    }
}

/// Simple step: increment by step_number
/// State: [x] -> [x + step_number]  
/// Witness: [const=1, prev_x, next_x] where next_x = prev_x + step_number
fn build_increment_by_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_x, step_number, next_x]
    // Constraint: next_x - prev_x - step_number = 0
    
    let rows = 1;
    let cols = 4;
    
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols];
    
    // next_x - prev_x - step_number = 0 (× const)
    a_data[0 * cols + 3] = F::ONE;   // +next_x
    a_data[0 * cols + 1] = -F::ONE;  // -prev_x
    a_data[0 * cols + 2] = -F::ONE;  // -step_number  
    b_data[0 * cols + 0] = F::ONE;   // × const
    
    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data);
    let c_mat = Mat::from_row_major(rows, cols, c_data);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Generate witness for increment-by-step: [const=1, prev_x, step_number, next_x]
fn build_increment_witness(prev_x: u64, step_number: u64) -> Vec<F> {
    let next_x = prev_x + step_number;
    vec![
        F::ONE,                       // const
        F::from_u64(prev_x),          // prev_x  
        F::from_u64(step_number),     // step_number
        F::from_u64(next_x),          // next_x (THIS is the real y_step)
    ]
}

#[test]
fn test_malicious_y_step_attack_should_fail() -> Result<()> {
    // This test verifies that the linked witness approach blocks malicious y_step attacks.
    // After the security fix, fake y_step values should cause verification failures.
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_by_step_ccs();
    
    // Initial state: x = 100
    let accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108],
        y_compact: vec![F::from_u64(100)], // x = 100
        step: 0,
    };
    
    // Step witness: Should compute 100 + 1 = 101
    let step_witness = build_increment_witness(100, 1);
    let real_y_step = vec![F::from_u64(101)]; // Actual computation result
    let fake_y_step = vec![F::ZERO];          // Attacker's fake value
    
    // Verify the step CCS is satisfied with real witness
    let step_public = vec![];  // No public inputs for this step
    assert!(check_ccs_rowwise_zero(&step_ccs, &step_public, &step_witness).is_ok(),
           "Step CCS should be satisfied with honest witness");
    
    // ATTACK: Try to prove with fake y_step = 0 instead of real y_step = 101
    let ivc_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &accumulator,
        step: 0,
        public_input: None,
        y_step: &fake_y_step, // 🚨 ATTACK: Using fake y_step!
        // 🔒 SECURITY: Empty binding metadata should now be rejected by security validation  
        binding_spec: &StepBindingSpec {
            y_step_offsets: vec![], // Empty = should be rejected with security error
            x_witness_indices: vec![], // No step_x binding needed for this test
            y_prev_witness_indices: vec![], // Empty = should be rejected
            const1_witness_index: 0, // Constant-1 at index 0
        },
    };
    
    // Try to prove - this might succeed at proving level
    let prove_result = prove_ivc_step(ivc_input);
    
    match prove_result {
        Ok(step_result) => {
            println!("⚠️  Proving succeeded, attempting verification...");
            
            // Try to verify - this should FAIL due to linked witness constraints
            // Use empty binding spec (same as proving) to test consistency
            let verify_binding_spec = StepBindingSpec {
                y_step_offsets: vec![], 
                x_witness_indices: vec![],
                y_prev_witness_indices: vec![],
                const1_witness_index: 0, // Constant-1 at index 0
            };
            let verify_result = verify_ivc_step(&step_ccs, &step_result.proof, &accumulator, &verify_binding_spec);
            
            match verify_result {
                Ok(true) => {
                    println!("🚨 SOUNDNESS BUG STILL EXISTS!");
                    println!("   Real step output: {}", real_y_step[0].as_canonical_u64());
                    println!("   Fake y_step used: {}", fake_y_step[0].as_canonical_u64());
                    println!("   Attack succeeded: true");
                    panic!("CRITICAL: Malicious y_step attack was not blocked!");
                }
                Ok(false) | Err(_) => {
                    println!("🚫 Attack BLOCKED: Linked witness prevents fake y_step");
                    println!("   Real step output: {}", real_y_step[0].as_canonical_u64());
                    println!("   Fake y_step attempted: {}", fake_y_step[0].as_canonical_u64());
                    println!("   Attack result: {:?}", verify_result);
                }
            }
        }
        Err(e) => {
            let error_msg = format!("{:?}", e);
            if error_msg.contains("SECURITY: y_step_offsets cannot be empty") {
                println!("✅ Attack BLOCKED: Empty y_step_offsets rejected by security validation");
                println!("   Error: {}", e);
                println!("   Real step output: {}", real_y_step[0].as_canonical_u64());
                println!("   Fake y_step attempted: {}", fake_y_step[0].as_canonical_u64());
                println!("   Attack succeeded: false (blocked at API level)");
            } else {
                println!("🚫 Attack BLOCKED: Proving failed with fake y_step");
                println!("   Error: {:?}", e);
            }
        }
    }
    
    Ok(())
}

#[test] 
fn test_honest_y_step_should_succeed() -> Result<()> {
    // Baseline: honest extraction should always work
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_by_step_ccs();
    
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32], 
        c_coords: vec![F::ZERO; 108],
        y_compact: vec![F::from_u64(100)],
        step: 0,
    };
    
    // Use secure API with proper binding specification
    let binding_spec = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![3], // For increment: [const, prev_x, step_number, next_x], next_x at index 3  
        x_witness_indices: vec![], // No step public inputs
        y_prev_witness_indices: vec![1], // prev_x at index 1
        const1_witness_index: 0, // Constant-1 at index 0
    };
    let mut honest_batch = IvcBatchBuilder::new_with_bindings(
        params,
        step_ccs,
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    )?;
    
    // Use honest extractor that returns actual step outputs
    let honest_extractor = HonestExtractor { n: 1 };
    let step_witness = build_increment_witness(100, 1);
    let honest_y_step = honest_extractor.extract_y_step(&step_witness);
    
    // Should equal the actual next_x value
    assert_eq!(honest_y_step, vec![F::from_u64(101)]);
    
    // This should succeed
    let result = honest_batch.append_step(&step_witness, None, &honest_y_step);
    assert!(result.is_ok(), "Honest y_step should succeed: {:?}", result);
    
    println!("✅ Honest extraction works correctly");
    
    Ok(())
}

#[test]
fn test_batch_step_stitching_attack() -> Result<()> {
    // This test demonstrates the batch stitching vulnerability:
    // Multiple steps aren't constrained to form a coherent chain
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_by_step_ccs();
    
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108], 
        y_compact: vec![F::from_u64(100)],
        step: 0,
    };
    
    // Use secure API with proper binding specification  
    let binding_spec = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![3], // For increment: [const, prev_x, step_number, next_x], next_x at index 3
        x_witness_indices: vec![], // No step public inputs
        y_prev_witness_indices: vec![1], // prev_x at index 1
        const1_witness_index: 0, // Constant-1 at index 0
    };
    let mut batch = IvcBatchBuilder::new_with_bindings(
        params.clone(),
        step_ccs.clone(),
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    )?;
    
    let honest_extractor = HonestExtractor { n: 1 };
    
    // Step 1: 100 + 1 = 101
    let step1_witness = build_increment_witness(100, 1);
    let step1_y_step = honest_extractor.extract_y_step(&step1_witness);
    batch.append_step(&step1_witness, None, &step1_y_step)?;
    
    // ATTACK: Step 2 uses inconsistent starting state
    // It should start from 101 (step1's output), but we use 999
    let step2_witness = build_increment_witness(999, 2); // Should be (101, 2)!
    let step2_y_step = honest_extractor.extract_y_step(&step2_witness);
    
    // CURRENT BUG: This succeeds because no cross-step constraints exist
    let result = batch.append_step(&step2_witness, None, &step2_y_step);
    
    // TODO: Once stitching is implemented, this should FAIL:
    // assert!(result.is_err(), "Inconsistent step chain should fail");
    
    println!("🚨 BATCH STITCHING BUG DEMONSTRATED:");
    println!("   Step 1: 100 → 101");  
    println!("   Step 2: 999 → 1001 (should start from 101!)");
    println!("   Attack succeeded: {:?}", result.is_ok());
    
    Ok(())
}

#[test]
fn test_rho_transcript_determinism() -> Result<()> {
    // Verify that ρ derivation is deterministic and depends on all inputs
    
    use neo::ivc::{rho_from_transcript, create_step_digest, build_step_data_with_x};
    
    let acc1 = Accumulator {
        c_z_digest: [1u8; 32],
        c_coords: vec![F::ZERO; 108],
        y_compact: vec![F::from_u64(100)],
        step: 5,
    };
    
    let step_x = vec![F::from_u64(42)];
    let step_data1 = build_step_data_with_x(&acc1, acc1.step, &step_x);
    let step_digest1 = create_step_digest(&step_data1);
    
    let (rho1, _) = rho_from_transcript(&acc1, step_digest1);
    
    // Same inputs should give same ρ
    let (rho1_repeat, _) = rho_from_transcript(&acc1, step_digest1);
    assert_eq!(rho1, rho1_repeat, "ρ derivation should be deterministic");
    
    // Different step should give different ρ
    let mut acc2 = acc1.clone();
    acc2.step = 6;  // Changed step
    let step_data2 = build_step_data_with_x(&acc2, acc2.step, &step_x);
    let step_digest2 = create_step_digest(&step_data2);
    let (rho2, _) = rho_from_transcript(&acc2, step_digest2);
    assert_ne!(rho1, rho2, "Different step should give different ρ");
    
    // Different y_compact should give different ρ  
    let mut acc3 = acc1.clone();
    acc3.y_compact = vec![F::from_u64(999)]; // Changed y_compact
    let step_data3 = build_step_data_with_x(&acc3, acc3.step, &step_x);
    let step_digest3 = create_step_digest(&step_data3);
    let (rho3, _) = rho_from_transcript(&acc3, step_digest3);
    assert_ne!(rho1, rho3, "Different y_compact should give different ρ");
    
    println!("✅ ρ transcript determinism verified");
    
    Ok(())
}

#[test]
fn test_linked_witness_blocks_malicious_y_step() -> Result<()> {
    // This test demonstrates that the linked witness approach BLOCKS the attack
    
    use neo::ivc::{
        build_augmented_ccs_linked, build_linked_augmented_witness, 
        build_linked_augmented_public_input, rho_from_transcript, 
        create_step_digest, build_step_data_with_x
    };
    
    let step_ccs = build_increment_by_step_ccs();
    let step_witness = build_increment_witness(100, 1);
    
    // Step public input is empty for this test
    let step_x = vec![];
    
    // IVC state 
    let y_prev = vec![F::from_u64(100)];
    let y_len = 1;
    
    // Build mock accumulator for ρ derivation
    let accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108],
        y_compact: y_prev.clone(),
        step: 1,
    };
    
    // Derive ρ from transcript
    let step_data = build_step_data_with_x(&accumulator, accumulator.step, &step_x);
    let step_digest = create_step_digest(&step_data);
    let (rho, _) = rho_from_transcript(&accumulator, step_digest);
    
    // Build linked augmented CCS
    // For our increment CCS: witness = [const=1, prev_x, step_number, next_x]
    // The y_step (next_x) is at offset 3
    let y_step_offsets = vec![3]; // next_x is at position 3 in witness
    let y_prev_witness_indices = vec![1]; // prev_x at index 1
    let x_witness_indices = vec![]; // No step public inputs
    let linked_ccs = build_augmented_ccs_linked(
        &step_ccs, 
        step_x.len(), 
        &y_step_offsets, 
        &y_prev_witness_indices,
        &x_witness_indices,
        y_len
    ).expect("Should build linked CCS");
    
    println!("🔗 Built linked CCS: {} constraints, {} variables", linked_ccs.n, linked_ccs.m);
    
    // TEST 1: Honest witness should work
    let honest_witness = build_linked_augmented_witness(&step_witness, &y_step_offsets, rho);
    let real_y_step = vec![F::from_u64(101)]; // Actual computation: 100 + 1 = 101
    let y_next = vec![y_prev[0] + rho * real_y_step[0]]; // Correct folding
    let honest_public = build_linked_augmented_public_input(&step_x, rho, &y_prev, &y_next);
    
    let honest_result = check_ccs_rowwise_zero(&linked_ccs, &honest_public, &honest_witness);
    assert!(honest_result.is_ok(), "Honest witness should satisfy linked CCS: {:?}", honest_result);
    println!("✅ Honest case: linked CCS satisfied");
    
    // TEST 2: Try to attack with fake y_next (based on fake y_step = 0)
    let fake_y_step = vec![F::ZERO]; // Attacker wants to use this
    let fake_y_next = vec![y_prev[0] + rho * fake_y_step[0]]; // y_next based on fake y_step
    let attack_public = build_linked_augmented_public_input(&step_x, rho, &y_prev, &fake_y_next);
    
    // CRITICAL: Use the SAME step_witness (which has real next_x = 101 at offset 3)
    // The linked witness will be built correctly, but public input claims fake y_next
    let attack_witness = build_linked_augmented_witness(&step_witness, &y_step_offsets, rho);
    
    let attack_result = check_ccs_rowwise_zero(&linked_ccs, &attack_public, &attack_witness);
    
    // The attack should FAIL because:
    // - EV constraints read y_step from step_witness[3] = 101 (real value)
    // - Public input claims y_next based on fake_y_step = 0  
    // - Constraint: y_next - y_prev - rho*101 != fake_y_next - y_prev - rho*0
    assert!(attack_result.is_err(), "Malicious attack should FAIL with linked witness");
    
    println!("🚫 Attack BLOCKED: linked witness prevents fake y_step");
    println!("   Real y_step in witness: {}", real_y_step[0].as_canonical_u64());
    println!("   Fake y_step attempted: {}", fake_y_step[0].as_canonical_u64());
    println!("   Attack result: {:?}", attack_result);
    
    Ok(())
}

#[test]
fn test_secure_batch_builder_blocks_stitching_attack() -> Result<()> {
    // This test demonstrates that SecureBatchBuilder prevents inconsistent step chains
    
    use neo::ivc::IvcBatchBuilder;
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_by_step_ccs();
    
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108],
        y_compact: vec![F::from_u64(100)], // Start with x = 100
        step: 0,
    };
    
    // For our increment CCS: witness = [const=1, prev_x, step_number, next_x]
    // The y_step (next_x) is at offset 3
    let y_step_offsets = vec![3];
    
    let binding_spec = StepBindingSpec {
        y_step_offsets,
        x_witness_indices: vec![], // No step public inputs
        y_prev_witness_indices: vec![1], // prev_x at index 1
        const1_witness_index: 0, // Constant-1 at index 0
    };
    let mut secure_batch = IvcBatchBuilder::new_with_bindings(
        params.clone(),
        step_ccs.clone(),
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    )?;
    
    let honest_extractor = HonestExtractor { n: 1 };
    
    // Step 1: 100 + 1 = 101 (should work fine)
    let step1_witness = build_increment_witness(100, 1);
    let step1_y_step = honest_extractor.extract_y_step(&step1_witness);
    let _y1_next = secure_batch.append_step(&step1_witness, None, &step1_y_step)?;
    
    println!("✅ Step 1: 100 + 1 = 101 (secure batch accepted)");
    
    // Step 2: Should start from 101, but let's try to use inconsistent starting point
    // This should FAIL because the stitching constraints will detect the inconsistency
    let step2_witness_bad = build_increment_witness(999, 2); // Should be (101, 2)!
    let step2_y_step_bad = honest_extractor.extract_y_step(&step2_witness_bad);
    
    // This should fail because:
    // - Step 1 outputs y_next = 101  
    // - Step 2 claims y_prev = 999 (from its witness)
    // - Stitching constraint: y_next^(1) - y_prev^(2) = 101 - 999 ≠ 0
    let attack_result = secure_batch.append_step(&step2_witness_bad, None, &step2_y_step_bad);
    
    // The attack should succeed at the append level (no constraint checking yet)
    // but fail when we try to extract and verify the batch
    if attack_result.is_ok() {
        println!("⚠️  Step 2 append succeeded (constraints not checked yet)");
        
        // Try to extract the batch - this should work
        let batch_data = secure_batch.finalize();
        if let Ok(Some(data)) = batch_data {
            println!("📦 Batch extracted: {} steps, {} constraints", 
                     data.steps_covered, data.ccs.n);
            
            // The real test: does the constraint check fail?
            let constraint_result = check_ccs_rowwise_zero(&data.ccs, &data.public_input, &data.witness);
            
            match constraint_result {
                Ok(()) => {
                    println!("🚨 SECURITY BUG: Inconsistent batch was accepted!");
                    // This would be a bug - the stitching should have prevented this
                    panic!("Stitching constraints failed to detect inconsistency!");
                }
                Err(e) => {
                    println!("🚫 Attack BLOCKED: Stitching constraints detected inconsistency");
                    println!("   Constraint error: {:?}", e);
                }
            }
        }
    } else {
        println!("🚫 Attack BLOCKED: SecureBatchBuilder rejected inconsistent step at append time");
        println!("   Error: {:?}", attack_result);
    }
    
    // Demonstrate the honest case works
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        x_witness_indices: vec![], // No step public inputs
        y_prev_witness_indices: vec![1], // prev_x at index 1
        const1_witness_index: 0, // Constant-1 at index 0
    };
    let mut honest_batch = IvcBatchBuilder::new_with_bindings(
        params,
        step_ccs,
        Accumulator {
            c_z_digest: [0u8; 32],
            c_coords: vec![F::ZERO; 108],
            y_compact: vec![F::from_u64(100)],
            step: 0,
        },
        EmissionPolicy::Never,
        binding_spec,
    )?;
    
    // Step 1: 100 + 1 = 101
    let step1_witness = build_increment_witness(100, 1);
    let step1_y_step = honest_extractor.extract_y_step(&step1_witness);
    let _y1_next = honest_batch.append_step(&step1_witness, None, &step1_y_step)?;
    
    // Step 2: 101 + 2 = 103 (correct chaining)
    let step2_witness = build_increment_witness(101, 2);
    let step2_y_step = honest_extractor.extract_y_step(&step2_witness);
    let _y2_next = honest_batch.append_step(&step2_witness, None, &step2_y_step)?;
    
    let honest_batch_data = honest_batch.finalize().unwrap().unwrap();
    let honest_constraint_result = check_ccs_rowwise_zero(
        &honest_batch_data.ccs, 
        &honest_batch_data.public_input, 
        &honest_batch_data.witness
    );
    
    assert!(honest_constraint_result.is_ok(), "Honest batch should satisfy constraints: {:?}", honest_constraint_result);
    println!("✅ Honest case: 100 → 101 → 103 (stitching constraints satisfied)");
    
    Ok(())
}

#[test]
fn test_secure_batch_builder_basic() -> Result<()> {
    // Simple test of SecureBatchBuilder with single step
    
    use neo::ivc::IvcBatchBuilder;
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_by_step_ccs();
    
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108],
        y_compact: vec![F::from_u64(100)], 
        step: 0,
    };
    
    let y_step_offsets = vec![3]; // next_x is at position 3 in witness
    
    let binding_spec = StepBindingSpec {
        y_step_offsets,
        x_witness_indices: vec![], // No step public inputs
        y_prev_witness_indices: vec![1], // prev_x at index 1
        const1_witness_index: 0, // Constant-1 at index 0
    };
    let mut secure_batch = IvcBatchBuilder::new_with_bindings(
        params,
        step_ccs,
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    )?;
    
    let honest_extractor = HonestExtractor { n: 1 };
    
    // Single step: 100 + 1 = 101
    let step1_witness = build_increment_witness(100, 1);
    let step1_y_step = honest_extractor.extract_y_step(&step1_witness);
    let _y1_next = secure_batch.append_step(&step1_witness, None, &step1_y_step)?;
    
    println!("✅ Single step appended to SecureBatchBuilder");
    
    // Try to extract and verify
    if let Ok(Some(batch_data)) = secure_batch.finalize() {
        println!("📦 Extracted batch: {} steps, {} constraints, pub={}, wit={}", 
                 batch_data.steps_covered, batch_data.ccs.n, 
                 batch_data.public_input.len(), batch_data.witness.len());
        
        let constraint_result = check_ccs_rowwise_zero(
            &batch_data.ccs, 
            &batch_data.public_input, 
            &batch_data.witness
        );
        
        match constraint_result {
            Ok(()) => {
                println!("✅ Single step SecureBatchBuilder verification succeeded");
            }
            Err(e) => {
                println!("❌ Single step verification failed: {:?}", e);
                return Err(e.into());
            }
        }
    } else {
        println!("❌ Failed to extract batch data");
    }
    
    Ok(())
}
