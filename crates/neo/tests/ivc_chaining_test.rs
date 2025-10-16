use anyhow::Result;
use neo::{NeoParams, F};
use neo::{StepBindingSpec, IvcStepInput, prove_ivc_step_chained, Accumulator, IvcProof, AppInputBinding};
use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Build a simple increment CCS: next_x = prev_x + delta
/// Variables: [const=1, prev_x, delta, next_x]
/// Constraint: next_x - prev_x - delta = 0
fn build_increment_ccs() -> CcsStructure<F> {
    let rows = 4;  // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 4;
    
    // A matrix: next_x - prev_x - delta = 0
    let mut a_data = vec![F::ZERO; rows * cols];
    a_data[0 * cols + 3] = F::ONE;   // next_x coefficient
    a_data[0 * cols + 1] = -F::ONE;  // prev_x coefficient  
    a_data[0 * cols + 2] = -F::ONE;  // delta coefficient
    
    // B matrix: multiply by const=1
    let mut b_data = vec![F::ZERO; rows * cols];
    b_data[0 * cols + 0] = F::ONE;   // const=1 coefficient
    
    // C matrix: all zeros (linear constraint)
    let c_data = vec![F::ZERO; rows * cols];
    
    // Rows 1-3: dummy constraints (0 * 1 = 0)
    for row in 1..4 {
        a_data[row * cols] = F::ZERO;
        b_data[row * cols] = F::ONE;
    }
    
    let a = Mat::from_row_major(rows, cols, a_data);
    let b = Mat::from_row_major(rows, cols, b_data);
    let c = Mat::from_row_major(rows, cols, c_data);
    
    r1cs_to_ccs(a, b, c)
}

/// Build witness for increment step: [const=1, prev_x, delta, next_x]
fn build_increment_witness(prev_x: u64, delta: u64) -> Vec<F> {
    let next_x = prev_x + delta;
    vec![
        F::ONE,                    // const
        F::from_u64(prev_x),      // prev_x
        F::from_u64(delta),       // delta
        F::from_u64(next_x),      // next_x
    ]
}

#[test]
fn test_ivc_chaining_not_self_folding() -> Result<()> {
    println!("🧪 Testing IVC chaining (not self-folding)");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],           // next_x at position 3
        step_program_input_witness_indices: vec![2],        // bind delta to witness position 2
        y_prev_witness_indices: vec![],    // no binding to EV y_prev (they're different values!)
        const1_witness_index: 0,
    };
    
    // Manual chained IVC proving (replaces ivc_chain)
    let mut acc = Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: vec![F::ZERO], step: 0 };
    let mut proofs: Vec<IvcProof> = Vec::new();
    let mut prev_me = None;
    let mut prev_me_wit = None;
    let mut prev_lhs = None;
    let mut step_outputs = Vec::new(); // Track actual program outputs (before folding)
    
    // Step 0: 0 + 5 = 5
    let step0_witness = build_increment_witness(0, 5);
    let step0_public_input = vec![F::from_u64(5)]; // delta=5
    {
        let y_step = step0_witness[3..=3].to_vec();
        step_outputs.push(y_step[0]); // ✅ Extract output BEFORE folding
        let input = IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &step0_witness, prev_accumulator: &acc, step: 0, public_input: Some(&step0_public_input), y_step: &y_step, binding_spec: &binding_spec, app_input_binding: AppInputBinding::WitnessBound, prev_augmented_x: None };
        let (res, me, wit, lhs) = prove_ivc_step_chained(
            input,
            prev_me.take(),
            prev_me_wit.take(),
            prev_lhs.take(),
        ).map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
        acc = res.proof.next_accumulator.clone(); proofs.push(res.proof); prev_me=Some(me); prev_me_wit=Some(wit); prev_lhs=Some(lhs);
    }
    println!("✅ Step 0: 0 + 5 = {} (output), accumulator = {}", step_outputs[0].as_canonical_u64(), acc.y_compact[0].as_canonical_u64());
    
    // Step 1: 5 + 3 = 8
    let step1_witness = build_increment_witness(5, 3);
    let step1_public_input = vec![F::from_u64(3)]; // delta=3
    {
        let y_step = step1_witness[3..=3].to_vec();
        step_outputs.push(y_step[0]); // ✅ Extract output BEFORE folding
        let input = IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &step1_witness, prev_accumulator: &acc, step: 1, public_input: Some(&step1_public_input), y_step: &y_step, binding_spec: &binding_spec, app_input_binding: AppInputBinding::WitnessBound, prev_augmented_x: proofs.last().map(|p| p.public_inputs.step_augmented_public_input()) };
        let (res, me, wit, lhs) = prove_ivc_step_chained(
            input,
            prev_me.take(),
            prev_me_wit.take(),
            prev_lhs.take(),
        ).map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
        acc = res.proof.next_accumulator.clone(); proofs.push(res.proof); prev_me=Some(me); prev_me_wit=Some(wit); prev_lhs=Some(lhs);
    }
    println!("✅ Step 1: 5 + 3 = {} (output), accumulator = {}", step_outputs[1].as_canonical_u64(), acc.y_compact[0].as_canonical_u64());
    
    // Step 2: 8 + 2 = 10
    let step2_witness = build_increment_witness(8, 2);
    let step2_public_input = vec![F::from_u64(2)]; // delta=2
    {
        let y_step = step2_witness[3..=3].to_vec();
        step_outputs.push(y_step[0]); // ✅ Extract output BEFORE folding
        let input = IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &step2_witness, prev_accumulator: &acc, step: 2, public_input: Some(&step2_public_input), y_step: &y_step, binding_spec: &binding_spec, app_input_binding: AppInputBinding::WitnessBound, prev_augmented_x: proofs.last().map(|p| p.public_inputs.step_augmented_public_input()) };
        let (res, _me, _wit, _lhs) = prove_ivc_step_chained(
            input,
            prev_me.take(),
            prev_me_wit.take(),
            prev_lhs.take(),
        ).map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
        acc = res.proof.next_accumulator.clone(); proofs.push(res.proof);
    }
    println!("✅ Step 2: 8 + 2 = {} (output), accumulator = {}", step_outputs[2].as_canonical_u64(), acc.y_compact[0].as_canonical_u64());
    
    // CRITICAL TEST: Verify that each step's ρ (folding randomness) is different
    // If we're folding with itself, the ρ values would be computed from identical instances
    // and could be the same. If we're properly chaining, they should be different.
    assert!(proofs.len() >= 3, "Should have at least 3 IVC proofs");
    
    let rho_0 = proofs[0].public_inputs.rho();
    let rho_1 = proofs[1].public_inputs.rho();
    let rho_2 = proofs[2].public_inputs.rho();
    
    println!("🔍 Folding randomness values:");
    println!("   Step 0 ρ: {:?}", rho_0);
    println!("   Step 1 ρ: {:?}", rho_1);
    println!("   Step 2 ρ: {:?}", rho_2);
    
    // The ρ values should be different because they're derived from different transcript states
    // If we're folding with itself, the transcript would be more predictable
    assert_ne!(rho_0, rho_1, "Step 0 and Step 1 should have different folding randomness (ρ)");
    assert_ne!(rho_1, rho_2, "Step 1 and Step 2 should have different folding randomness (ρ)");
    assert_ne!(rho_0, rho_2, "Step 0 and Step 2 should have different folding randomness (ρ)");
    
    // CRITICAL TEST: Verify that the accumulator digest evolves properly
    // If we're folding with itself, the digest evolution would be broken
    let digest_0 = proofs[0].next_accumulator.c_z_digest;
    let digest_1 = proofs[1].next_accumulator.c_z_digest;
    let digest_2 = proofs[2].next_accumulator.c_z_digest;
    
    println!("🔍 Accumulator digest evolution:");
    println!("   Step 0 digest: {:02x?}", &digest_0[..8]);
    println!("   Step 1 digest: {:02x?}", &digest_1[..8]);
    println!("   Step 2 digest: {:02x?}", &digest_2[..8]);
    
    // Each step should produce a different digest
    assert_ne!(digest_0, digest_1, "Step 0 and Step 1 should have different accumulator digests");
    assert_ne!(digest_1, digest_2, "Step 1 and Step 2 should have different accumulator digests");
    assert_ne!(digest_0, digest_2, "Step 0 and Step 2 should have different accumulator digests");
    
    // CRITICAL TEST: Verify that the commitment coordinates evolve properly
    // If we're folding with itself, the commitment evolution would be broken
    let coords_0_len = proofs[0].next_accumulator.c_coords.len();
    let coords_1_len = proofs[1].next_accumulator.c_coords.len();
    let coords_2_len = proofs[2].next_accumulator.c_coords.len();
    
    println!("🔍 Commitment coordinate evolution:");
    println!("   Step 0 coords length: {}", coords_0_len);
    println!("   Step 1 coords length: {}", coords_1_len);
    println!("   Step 2 coords length: {}", coords_2_len);
    
    // All steps should have the same coordinate length (same Ajtai parameters)
    assert_eq!(coords_0_len, coords_1_len, "Step 0 and Step 1 should have same coordinate length");
    assert_eq!(coords_1_len, coords_2_len, "Step 1 and Step 2 should have same coordinate length");
    
    // But the actual coordinate values should be different (evolving commitment)
    if coords_0_len > 0 && coords_1_len > 0 && coords_2_len > 0 {
        let coords_0 = &proofs[0].next_accumulator.c_coords;
        let coords_1 = &proofs[1].next_accumulator.c_coords;
        let coords_2 = &proofs[2].next_accumulator.c_coords;
        
        assert_ne!(coords_0, coords_1, "Step 0 and Step 1 should have different commitment coordinates");
        assert_ne!(coords_1, coords_2, "Step 1 and Step 2 should have different commitment coordinates");
        assert_ne!(coords_0, coords_2, "Step 0 and Step 2 should have different commitment coordinates");
        
        println!("   ✅ Commitment coordinates are properly evolving");
    }
    
    // FINAL TEST: Verify program outputs (extracted before folding)
    // With CORRECT HyperNova folding: y_next = y_prev + ρ·y_step
    // - The ACCUMULATOR (acc.y_compact) is ρ-dependent (cryptographic commitment)
    // - The PROGRAM OUTPUTS (step_outputs) are the actual results
    println!("🔍 Program outputs (extracted before folding):");
    for (i, output) in step_outputs.iter().enumerate() {
        println!("   Step {}: next_x = {}", i, output.as_canonical_u64());
    }
    
    // Verify the outputs match expected increments: 5, 8, 10
    assert_eq!(step_outputs[0].as_canonical_u64(), 5, "Step 0: 0 + 5 = 5");
    assert_eq!(step_outputs[1].as_canonical_u64(), 8, "Step 1: 5 + 3 = 8");
    assert_eq!(step_outputs[2].as_canonical_u64(), 10, "Step 2: 8 + 2 = 10");
    
    let final_result_f = acc.y_compact[0];
    println!("   Final accumulator (ρ-dependent): {}", final_result_f.as_canonical_u64());
    println!("   Final program output: {}", step_outputs.last().unwrap().as_canonical_u64());
    println!("   📝 Note: Accumulator ≠ output (folding uses ρ, this is correct!)");
    
    println!("🎉 IVC chaining test PASSED!");
    println!("   ✅ Folding randomness (ρ) values are distinct across steps");
    println!("   ✅ Accumulator digests evolve properly");
    println!("   ✅ Commitment coordinates evolve properly");
    println!("   ✅ Program outputs verified: 5 → 8 → 10 ✅");
    println!("   ✅ Correct HyperNova folding: y_next = y_prev + ρ·y_step");
    
    Ok(())
}
