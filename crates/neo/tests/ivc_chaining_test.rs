use anyhow::Result;
use neo::{NeoParams, F};
use neo::ivc::StepBindingSpec;
use neo::ivc_chain;
use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Build a simple increment CCS: next_x = prev_x + delta
/// Variables: [const=1, prev_x, delta, next_x]
/// Constraint: next_x - prev_x - delta = 0
fn build_increment_ccs() -> CcsStructure<F> {
    let rows = 1;
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
        x_witness_indices: vec![2],        // bind delta to witness position 2
        y_prev_witness_indices: vec![],    // no binding to EV y_prev (they're different values!)
        const1_witness_index: 0,
    };
    
    // Initialize IVC chain
    let initial_y = vec![F::ZERO]; // Start with 0
    let mut state = ivc_chain::State::new(params, step_ccs, initial_y, binding_spec)?;
    
    // Step 0: 0 + 5 = 5
    let step0_witness = build_increment_witness(0, 5);
    let step0_public_input = vec![F::from_u64(5)]; // delta=5
    state = ivc_chain::step(state, &step0_public_input, &step0_witness)?;
    println!("✅ Step 0: {} + 5 = {}", 0, state.accumulator.y_compact[0].as_canonical_u64());
    
    // Step 1: 5 + 3 = 8
    let step1_witness = build_increment_witness(5, 3);
    let step1_public_input = vec![F::from_u64(3)]; // delta=3
    state = ivc_chain::step(state, &step1_public_input, &step1_witness)?;
    println!("✅ Step 1: {} + 3 = {}", 5, state.accumulator.y_compact[0].as_canonical_u64());
    
    // Step 2: 8 + 2 = 10
    let step2_witness = build_increment_witness(8, 2);
    let step2_public_input = vec![F::from_u64(2)]; // delta=2
    state = ivc_chain::step(state, &step2_public_input, &step2_witness)?;
    println!("✅ Step 2: {} + 2 = {}", 8, state.accumulator.y_compact[0].as_canonical_u64());
    
    // CRITICAL TEST: Verify that each step's ρ (folding randomness) is different
    // If we're folding with itself, the ρ values would be computed from identical instances
    // and could be the same. If we're properly chaining, they should be different.
    assert!(state.ivc_proofs.len() >= 3, "Should have at least 3 IVC proofs");
    
    let rho_0 = state.ivc_proofs[0].step_rho;
    let rho_1 = state.ivc_proofs[1].step_rho;
    let rho_2 = state.ivc_proofs[2].step_rho;
    
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
    let digest_0 = state.ivc_proofs[0].next_accumulator.c_z_digest;
    let digest_1 = state.ivc_proofs[1].next_accumulator.c_z_digest;
    let digest_2 = state.ivc_proofs[2].next_accumulator.c_z_digest;
    
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
    let coords_0_len = state.ivc_proofs[0].next_accumulator.c_coords.len();
    let coords_1_len = state.ivc_proofs[1].next_accumulator.c_coords.len();
    let coords_2_len = state.ivc_proofs[2].next_accumulator.c_coords.len();
    
    println!("🔍 Commitment coordinate evolution:");
    println!("   Step 0 coords length: {}", coords_0_len);
    println!("   Step 1 coords length: {}", coords_1_len);
    println!("   Step 2 coords length: {}", coords_2_len);
    
    // All steps should have the same coordinate length (same Ajtai parameters)
    assert_eq!(coords_0_len, coords_1_len, "Step 0 and Step 1 should have same coordinate length");
    assert_eq!(coords_1_len, coords_2_len, "Step 1 and Step 2 should have same coordinate length");
    
    // But the actual coordinate values should be different (evolving commitment)
    if coords_0_len > 0 && coords_1_len > 0 && coords_2_len > 0 {
        let coords_0 = &state.ivc_proofs[0].next_accumulator.c_coords;
        let coords_1 = &state.ivc_proofs[1].next_accumulator.c_coords;
        let coords_2 = &state.ivc_proofs[2].next_accumulator.c_coords;
        
        assert_ne!(coords_0, coords_1, "Step 0 and Step 1 should have different commitment coordinates");
        assert_ne!(coords_1, coords_2, "Step 1 and Step 2 should have different commitment coordinates");
        assert_ne!(coords_0, coords_2, "Step 0 and Step 2 should have different commitment coordinates");
        
        println!("   ✅ Commitment coordinates are properly evolving");
    }
    
    // FINAL TEST: Verify the EV folding equation accumulates correctly over steps
    // Our EV picks y_step = next_x (application next state) via y_step_offsets = [3]
    // Therefore: y_final = Σ rho_i * next_x_i, where next_x_i evolves as prev_x + delta_i over F
    let mut expected_y = F::ZERO;
    let mut app_x = F::ZERO; // initial prev_x = 0
    for (i, proof) in state.ivc_proofs.iter().enumerate() {
        // App input (delta) is the last element of step_x
        let delta = proof.step_public_input.last().copied().expect("step_public_input not empty");
        app_x += delta;               // next_x = prev_x + delta
        let next_x = app_x;           // y_step for this EV configuration
        let rho = proof.step_rho;
        expected_y += rho * next_x;
        println!("   Accumulate step {}: expected_y += rho*next_x = {:?}", i, (rho * next_x).as_canonical_u64());
    }

    let final_result_f = state.accumulator.y_compact[0];
    assert_eq!(final_result_f, expected_y, 
               "Final IVC y should equal Σ rho_i*delta_i; got {:?} vs {:?}", 
               final_result_f.as_canonical_u64(), expected_y.as_canonical_u64());
    
    println!("🎉 IVC chaining test PASSED!");
    println!("   ✅ Folding randomness (ρ) values are distinct across steps");
    println!("   ✅ Accumulator digests evolve properly");
    println!("   ✅ Commitment coordinates evolve properly");
    println!("   ✅ Final folded y matches: {}", final_result_f.as_canonical_u64());
    
    Ok(())
}
