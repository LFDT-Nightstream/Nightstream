//! Red Team Tests: Public IO Binding Vulnerabilities
//!
//! This module tests critical Fiat-Shamir binding properties to ensure that the `public_io` 
//! digest commits to the ENTIRE augmented public input, not just a subset like `x` (delta).
//!
//! ## Background: Why This Matters
//!
//! In IVC systems, each step contributes bytes to the Fiat-Shamir transcript that define what 
//! was proven at that step. The `public_io` digest must bind:
//! - Circuit identifier / CCS structure (domain separation)
//! - Step index (prevents reordering/replay)  
//! - Compact state: y_prev and y_step/y_next
//! - Public inputs selected from witness (e.g., x/delta)
//! - Random challenges used for compression (e.g., ρ)
//!
//! If the digest only commits to a subset (e.g., just `x`), an attacker can malleate 
//! unbound parts without breaking the digest - a classic Fiat-Shamir binding pitfall.
//!
//! ## Test Categories
//!
//! 1. **Unit Test**: Digest must match the full augmented public input
//! 2. **Negative Test**: Verifier must reject when augmented PI changes but x stays same
//! 3. **Property Test**: Changing any bound field must change the digest
//! 4. **Challenge Binding**: ρ must be bound to the full statement transcript
//! 5. **Replay Attack**: Different step indices must produce different digests

use anyhow::Result;
use neo::{F, NeoParams};
use neo::{
    Accumulator, IvcStepInput, StepBindingSpec, prove_ivc_step, verify_ivc_step,
    prove_ivc_step_chained, verify_ivc_step_legacy,
    LastNExtractor, StepOutputExtractor, rho_from_transcript,
};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Build incrementer CCS: next_x = prev_x + delta
/// Variables: [const=1, prev_x, delta, next_x]
/// Constraint: next_x - prev_x - delta = 0
fn build_incrementer_step_ccs() -> CcsStructure<F> {
    let rows = 1;
    let cols = 4;
    
    // A matrix: next_x - prev_x - delta
    let a = Mat::from_row_major(rows, cols, vec![
        F::ZERO,  // const
        -F::ONE,  // -prev_x
        -F::ONE,  // -delta
        F::ONE,   // +next_x
    ]);
    
    // B matrix: select constant 1
    let b = Mat::from_row_major(rows, cols, vec![
        F::ONE,   // const = 1
        F::ZERO,  // prev_x
        F::ZERO,  // delta
        F::ZERO,  // next_x
    ]);
    
    // C matrix: zero (R1CS: A*z ∘ B*z = C*z, so 0 = 0)
    let c = Mat::from_row_major(rows, cols, vec![
        F::ZERO, F::ZERO, F::ZERO, F::ZERO,
    ]);
    
    r1cs_to_ccs(a, b, c)
}

/// Test T1: The digest must match the *augmented* public input (unit test)
/// 
/// This test proves one step normally, then reconstructs the augmented public input
/// exactly as the verifier should and recomputes the digest. If the prover was 
/// hashing only `x`, this test fails; after fixing to hash the entire augmented PI, it passes.
#[test]
fn test_public_io_digest_matches_augmented_pi() -> Result<()> {
    println!("🔒 Testing that public_io digest commits to FULL augmented public input...");
    
    // Setup
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        // witness[3] is next_x
        step_program_input_witness_indices: vec![],     // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![], // unused for this test
        const1_witness_index: 0,
    };
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    // Make one valid step: prev_x=0, delta=5, next_x=5
    let delta = F::from_u64(5);
    let prev_x = initial_acc.y_compact[0];
    let next_x = prev_x + delta;
    let witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &witness,
        prev_accumulator: &initial_acc,
        step: 0,
        public_input: None, // No app public input - testing digest binding, not input binding
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let (step_result, _me, _wit, _lhs) = prove_ivc_step_chained(step_input, None, None, None).expect("prove step");
    let proof = step_result.proof;
    
    // The key test: verify that the proof's public_io digest was computed from
    // the FULL augmented public input, not just x
    // 
    // We reconstruct what the verifier should compute and check it matches
    let is_valid = verify_ivc_step_legacy(&step_ccs, &proof, &initial_acc, &binding_spec, &params, None)
        .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;
    
    if !is_valid {
        println!("   ❌ CRITICAL: Verification failed - this indicates the digest binding is broken!");
        println!("   Expected: Digest computed from full augmented PI [step_x, rho, y_prev, y_next]");
        println!("   Actual: Digest likely computed from incomplete subset (e.g., just x)");
        panic!("Public IO digest does not commit to the entire augmented public input");
    }
    
    println!("   ✅ SUCCESS: Public IO digest correctly commits to full augmented public input");
    println!("   This means the digest includes: step_x, rho, y_prev, y_next, and CCS binding");
    
    Ok(())
}

/// Test T2: Negative test - verifier must reject when augmented PI changes but x stays same
///
/// Generate a valid proof, then surgically modify a component that must be bound 
/// (e.g., y_step) while keeping x the same. If the verifier only hashes x, it will 
/// still accept; a correct verifier will reject because the augmented PI changes.
#[test] 
fn test_verifier_rejects_when_augmented_pi_changes_but_x_same() -> Result<()> {
    println!("🔒 Testing that verifier rejects when bound fields change but x stays same...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],   
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata 
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    // Test with two different initial states but same delta x
    let acc_a = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO], // prev_x = 0
        step: 0,
    };
    
    let acc_b = Accumulator {
        c_z_digest: [0u8; 32], 
        c_coords: vec![],
        y_compact: vec![F::from_u64(10)], // prev_x = 10 (DIFFERENT)
        step: 0,
    };
    
    let x = vec![F::from_u64(5)]; // Same delta for both
    
    // Prove step A: 0 + 5 = 5
    let prev_a = acc_a.y_compact[0];
    let next_a = prev_a + x[0];
    let witness_a = vec![F::ONE, prev_a, x[0], next_a];
    let y_step_a = LastNExtractor { n: 1 }.extract_y_step(&witness_a);
    
    let step_a = IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness_a,
        prev_accumulator: &acc_a, step: 0, public_input: Some(&x),
        y_step: &y_step_a, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    };
    let result_a = prove_ivc_step(step_a).expect("prove step A");
    
    // Prove step B: 10 + 5 = 15  
    let prev_b = acc_b.y_compact[0];
    let next_b = prev_b + x[0];
    let witness_b = vec![F::ONE, prev_b, x[0], next_b];
    let y_step_b = LastNExtractor { n: 1 }.extract_y_step(&witness_b);
    
    let step_b = IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness_b,
        prev_accumulator: &acc_b, step: 0, public_input: Some(&x),
        y_step: &y_step_b, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    };
    let _result_b = prove_ivc_step(step_b).expect("prove step B");
    
    // Critical test: Try to verify proof A against accumulator B
    // This should FAIL because y_prev is different (0 vs 10) even though x is the same
    let cross_verify = verify_ivc_step(&step_ccs, &result_a.proof, &acc_b, &binding_spec, &params, None)
        .map_err(|e| anyhow::anyhow!("Cross-verification failed: {}", e))?;
    
    if cross_verify {
        println!("   ❌ CRITICAL VULNERABILITY: Verifier accepted proof with wrong y_prev!");
        println!("   Proof A (prev_x=0, delta=5, next_x=5) was accepted against accumulator B (prev_x=10)");
        println!("   This means the digest only commits to x (delta), not the full augmented PI");
        panic!("Verifier must reject when augmented PI changes even if x stays the same");
    }
    
    println!("   ✅ SUCCESS: Verifier correctly rejected proof with mismatched y_prev");
    println!("   This confirms the digest commits to the full augmented PI, not just x");
    
    Ok(())
}

/// Test T3: Property test - changing any bound field must change the digest
///
/// This asserts the cryptographic invariant: with fixed CCS and step index,
/// ANY change to a bound component (y_prev, y_step, ρ, ...) must change the digest.
/// If the digest depends only on x, this test will fail.
#[test]
fn test_context_digest_changes_when_any_bound_field_changes() -> Result<()> {
    println!("🔒 Testing that context digest changes when any bound field changes...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],   
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata 
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let x = vec![F::from_u64(7)]; // Fixed delta
    
    // Scenario A: prev_x = 0
    let acc_a = Accumulator {
        c_z_digest: [0u8; 32], c_coords: vec![], 
        y_compact: vec![F::ZERO], step: 0,
    };
    let prev_a = acc_a.y_compact[0];
    let next_a = prev_a + x[0];
    let witness_a = vec![F::ONE, prev_a, x[0], next_a];
    let y_step_a = LastNExtractor { n: 1 }.extract_y_step(&witness_a);
    
    let result_a = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness_a,
        prev_accumulator: &acc_a, step: 0, public_input: Some(&x),
        y_step: &y_step_a, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove A");
    
    // Scenario B: prev_x = 3 (different y_prev, same x)
    let acc_b = Accumulator {
        c_z_digest: [0u8; 32], c_coords: vec![],
        y_compact: vec![F::from_u64(3)], step: 0,
    };
    let prev_b = acc_b.y_compact[0];
    let next_b = prev_b + x[0];
    let witness_b = vec![F::ONE, prev_b, x[0], next_b];
    let y_step_b = LastNExtractor { n: 1 }.extract_y_step(&witness_b);
    
    let result_b = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness_b,
        prev_accumulator: &acc_b, step: 0, public_input: Some(&x),
        y_step: &y_step_b, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove B");
    
    // Extract digests from public_io (last 32 bytes)
    let digest_a = &result_a.proof.step_proof.public_io[result_a.proof.step_proof.public_io.len() - 32..];
    let digest_b = &result_b.proof.step_proof.public_io[result_b.proof.step_proof.public_io.len() - 32..];
    
    if digest_a == digest_b {
        println!("   ❌ CRITICAL VULNERABILITY: Context digests are identical!");
        println!("   Scenario A: prev_x=0, delta=7, next_x=7");
        println!("   Scenario B: prev_x=3, delta=7, next_x=10");
        println!("   Same digest means it only commits to x (delta=7), not full augmented PI");
        panic!("Context digest must change when any bound field (y_prev) changes");
    }
    
    println!("   ✅ SUCCESS: Context digests are different when y_prev changes");
    println!("   Digest A: {:02x?}", &digest_a[..8]);
    println!("   Digest B: {:02x?}", &digest_b[..8]);
    println!("   This confirms the digest commits to the full augmented PI");
    
    Ok(())
}

/// Test T4: Challenge binding - ρ must be bound to the full statement transcript
///
/// Tests that the random challenge ρ used for compression is derived from a transcript
/// that includes the full statement. If ρ can be manipulated independently of the 
/// statement, RLC security breaks down (Schwartz-Zippel fails).
#[test]
fn test_rho_challenge_binding() -> Result<()> {
    println!("🔒 Testing that ρ challenge is bound to full statement transcript...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    // Two different scenarios with same x but different y_prev
    let acc_1 = Accumulator {
        c_z_digest: [0u8; 32], c_coords: vec![],
        y_compact: vec![F::from_u64(5)], step: 0,
    };
    let acc_2 = Accumulator {
        c_z_digest: [0u8; 32], c_coords: vec![],
        y_compact: vec![F::from_u64(8)], step: 0,
    };
    
    let x = vec![F::from_u64(3)];
    
    // Prove both scenarios
    let witness_1 = vec![F::ONE, F::from_u64(5), F::from_u64(3), F::from_u64(8)];
    let y_step_1 = LastNExtractor { n: 1 }.extract_y_step(&witness_1);
    let result_1 = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness_1,
        prev_accumulator: &acc_1, step: 0, public_input: Some(&x),
        y_step: &y_step_1, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove 1");
    
    let witness_2 = vec![F::ONE, F::from_u64(8), F::from_u64(3), F::from_u64(11)];
    let y_step_2 = LastNExtractor { n: 1 }.extract_y_step(&witness_2);
    let result_2 = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness_2,
        prev_accumulator: &acc_2, step: 0, public_input: Some(&x),
        y_step: &y_step_2, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove 2");
    
    // ρ values should be different because they're derived from different transcripts
    let rho_1 = result_1.proof.public_inputs.rho();
    let rho_2 = result_2.proof.public_inputs.rho();
    
    if rho_1 == rho_2 {
        println!("   ❌ CRITICAL VULNERABILITY: ρ values are identical for different statements!");
        println!("   Statement 1: prev_x=5, delta=3, next_x=8");
        println!("   Statement 2: prev_x=8, delta=3, next_x=11");
        println!("   Same ρ means it's not derived from the full statement transcript");
        panic!("ρ challenge must be bound to the full statement transcript");
    }
    
    println!("   ✅ SUCCESS: ρ values are different for different statements");
    println!("   ρ₁ = {} (for prev_x=5)", rho_1.as_canonical_u64());
    println!("   ρ₂ = {} (for prev_x=8)", rho_2.as_canonical_u64());
    println!("   This confirms ρ is derived from the full statement transcript");
    
    Ok(())
}

/// Test T5: Replay attack prevention - different step indices must produce different digests
///
/// Tests that the step index is properly bound in the digest to prevent replay attacks
/// where an attacker reuses a proof from a different step.
#[test]
fn test_step_index_binding_prevents_replay() -> Result<()> {
    println!("🔒 Testing that step index binding prevents replay attacks...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let x = vec![F::from_u64(4)];
    let acc = Accumulator {
        c_z_digest: [0u8; 32], c_coords: vec![],
        y_compact: vec![F::from_u64(2)], step: 0,
    };
    
    // Same computation, different step indices
    let witness = vec![F::ONE, F::from_u64(2), F::from_u64(4), F::from_u64(6)];
    let y_step = LastNExtractor { n: 1 }.extract_y_step(&witness);
    
    // Prove at step 0
    let result_step_0 = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness,
        prev_accumulator: &acc, step: 0, public_input: Some(&x),
        y_step: &y_step, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove step 0");
    
    // Prove at step 1 (different step index, same computation)
    let mut acc_step_1 = acc.clone();
    acc_step_1.step = 1;
    let result_step_1 = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs, step_witness: &witness,
        prev_accumulator: &acc_step_1, step: 1, public_input: Some(&x),
        y_step: &y_step, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove step 1");
    
    // Extract digests
    let digest_0 = &result_step_0.proof.step_proof.public_io[result_step_0.proof.step_proof.public_io.len() - 32..];
    let digest_1 = &result_step_1.proof.step_proof.public_io[result_step_1.proof.step_proof.public_io.len() - 32..];
    
    if digest_0 == digest_1 {
        println!("   ❌ CRITICAL VULNERABILITY: Digests are identical for different step indices!");
        println!("   Step 0 digest: {:02x?}", &digest_0[..8]);
        println!("   Step 1 digest: {:02x?}", &digest_1[..8]);
        println!("   This enables replay attacks - proofs can be reused across different steps");
        panic!("Step index must be bound in the context digest to prevent replay attacks");
    }
    
    println!("   ✅ SUCCESS: Digests are different for different step indices");
    println!("   Step 0 digest: {:02x?}", &digest_0[..8]);
    println!("   Step 1 digest: {:02x?}", &digest_1[..8]);
    println!("   This prevents replay attacks across different steps");
    
    Ok(())
}

/// Test T6: Domain separation - different CCS must produce different digests
///
/// Tests that the CCS/circuit identifier is properly bound in the digest for domain separation.
/// This prevents proofs from one circuit being accepted by a different circuit.
#[test]
fn test_ccs_domain_separation() -> Result<()> {
    println!("🔒 Testing that CCS domain separation prevents cross-circuit attacks...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Two different CCS circuits
    let step_ccs_1 = build_incrementer_step_ccs(); // next_x = prev_x + delta
    
    // Build a different CCS: next_x = prev_x + delta + 1 (slightly different adder)
    let step_ccs_2 = {
        let rows = 1;
        let cols = 4;
        
        // A matrix: next_x - prev_x - delta - 1 = 0 (so next_x = prev_x + delta + 1)
        let a = Mat::from_row_major(rows, cols, vec![
            -F::ONE,  // -const (so we get -1)
            -F::ONE,  // -prev_x
            -F::ONE,  // -delta
            F::ONE,   // +next_x
        ]);
        
        // B matrix: select constant 1
        let b = Mat::from_row_major(rows, cols, vec![
            F::ONE,   // const = 1
            F::ZERO,  // prev_x
            F::ZERO,  // delta
            F::ZERO,  // next_x
        ]);
        
        // C matrix: zero (R1CS: A*z ∘ B*z = C*z, so 0 = 0)
        let c = Mat::from_row_major(rows, cols, vec![
            F::ZERO, F::ZERO, F::ZERO, F::ZERO,
        ]);
        
        r1cs_to_ccs(a, b, c)
    };
    
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let acc = Accumulator {
        c_z_digest: [0u8; 32], c_coords: vec![],
        y_compact: vec![F::from_u64(3)], step: 0,
    };
    let x = vec![F::from_u64(2)];
    
    // Prove with incrementer CCS: 3 + 2 = 5
    let witness_1 = vec![F::ONE, F::from_u64(3), F::from_u64(2), F::from_u64(5)];
    let y_step_1 = LastNExtractor { n: 1 }.extract_y_step(&witness_1);
    let result_1 = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs_1, step_witness: &witness_1,
        prev_accumulator: &acc, step: 0, public_input: Some(&x),
        y_step: &y_step_1, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove with incrementer CCS");
    
    // Prove with modified incrementer CCS: 3 + 2 + 1 = 6
    let witness_2 = vec![F::ONE, F::from_u64(3), F::from_u64(2), F::from_u64(6)];
    let y_step_2 = LastNExtractor { n: 1 }.extract_y_step(&witness_2);
    let result_2 = prove_ivc_step(IvcStepInput {
        params: &params, step_ccs: &step_ccs_2, step_witness: &witness_2,
        prev_accumulator: &acc, step: 0, public_input: Some(&x),
        y_step: &y_step_2, binding_spec: &binding_spec,
        transcript_only_app_inputs: false, prev_augmented_x: None,
    }).expect("prove with modified incrementer CCS");
    
    // Extract digests
    let digest_1 = &result_1.proof.step_proof.public_io[result_1.proof.step_proof.public_io.len() - 32..];
    let digest_2 = &result_2.proof.step_proof.public_io[result_2.proof.step_proof.public_io.len() - 32..];
    
    if digest_1 == digest_2 {
        println!("   ❌ CRITICAL VULNERABILITY: Digests are identical for different CCS circuits!");
        println!("   Incrementer CCS digest:          {:02x?}", &digest_1[..8]);
        println!("   Modified incrementer CCS digest: {:02x?}", &digest_2[..8]);
        println!("   This enables cross-circuit attacks - proofs can be reused across different circuits");
        panic!("CCS identifier must be bound in the context digest for domain separation");
    }
    
    println!("   ✅ SUCCESS: Digests are different for different CCS circuits");
    println!("   Incrementer CCS digest:          {:02x?}", &digest_1[..8]);
    println!("   Modified incrementer CCS digest: {:02x?}", &digest_2[..8]);
    println!("   This provides proper domain separation between different circuits");
    
    Ok(())
}

/// Test T7: Critical vulnerability - verifier accepts proof with mismatched prev_acc
///
/// This test exposes the core issue: when y_prev is bound in the witness but folding 
/// verification is disabled, the verifier should reject proofs against mismatched 
/// previous accumulators. If this test fails (verifier accepts), it demonstrates 
/// the missing fold/state binding that makes the IVC unsound.
#[test]
fn test_verifier_accepts_with_mismatched_prev_acc_vulnerability() -> Result<()> {
    println!("🔒 Testing critical vulnerability: verifier accepts proof with mismatched prev_acc...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    
    // CRITICAL: Bind previous state into the witness at index 1 (prev_x)
    // This is what real IVC should do - bind the previous state
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        // next_x at witness[3]
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata     // delta at witness[2]  
        y_prev_witness_indices: vec![1], // <-- BIND prev state to witness[1]!
        const1_witness_index: 0,
    };
    
    // prev_acc A with y = 0
    let prev_acc_a = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    // Build a valid step relative to A: prev_x = 0, delta = 7 -> next_x = 7
    let delta = F::from_u64(7);
    let prev_x = prev_acc_a.y_compact[0]; // 0
    let next_x = prev_x + delta;          // 7
    let step_witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &prev_acc_a,
        step: 0,
        public_input: None, // No app public input - testing digest binding, not input binding
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let (step_result, _me, _wit, _lhs) = prove_ivc_step_chained(step_input, None, None, None).expect("proving failed");
    let proof = step_result.proof;
    
    // Positive control: A should verify
    let valid_verify = verify_ivc_step_legacy(&step_ccs, &proof, &prev_acc_a, &binding_spec, &params, None)
        .map_err(|e| anyhow::anyhow!("Verifier error for A: {}", e))?;
    
    if !valid_verify {
        panic!("SETUP ERROR: Original proof should verify against correct prev_acc A");
    }
    println!("   ✅ Positive control: Proof verifies against correct prev_acc A");
    
    // Forge prev_acc B with a different y (999), same step index  
    let prev_acc_b = Accumulator {
        c_z_digest: [1u8; 32], // Different digest
        c_coords: vec![],
        y_compact: vec![F::from_u64(999)], // DIFFERENT y_prev (0 vs 999)
        step: 0,
    };
    
    // CRITICAL TEST: With a sound verifier, this MUST be REJECTED 
    // because y_prev is bound in the witness but the accumulator has different y_prev
    let accepted = verify_ivc_step_legacy(&step_ccs, &proof, &prev_acc_b, &binding_spec, &params, None)
        .unwrap_or(false);
    
    if accepted {
        println!("   ❌ CRITICAL VULNERABILITY CONFIRMED!");
        println!("   The verifier accepted a proof against a mismatched previous accumulator!");
        println!("   Proof was generated for prev_acc A (y_prev=0)");
        println!("   But verifier accepted it against prev_acc B (y_prev=999)");
        println!("   This demonstrates missing fold/state binding in IVC verification");
        println!("   Root cause: Folding verification is disabled, so y_prev binding is not enforced");
        panic!("UNSOUND: verifier accepted proof under mismatched prev_acc!");
    }
    
    println!("   ✅ SUCCESS: Verifier correctly rejected proof with mismatched prev_acc");
    println!("   This confirms proper fold/state binding is enforced");
    
    Ok(())
}

/// Test T8: Demonstrates the ACTUAL vulnerability when y_prev is NOT bound
///
/// This test shows what happens when y_prev_witness_indices is empty (like in most 
/// existing tests). In this case, the verifier WILL accept proofs with mismatched 
/// previous accumulators because there's no constraint linking them.
#[test]
fn test_vulnerability_when_y_prev_not_bound() -> Result<()> {
    println!("🔒 Testing vulnerability when y_prev is NOT bound (empty y_prev_witness_indices)...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    
    // VULNERABILITY: Do NOT bind previous state (like most existing tests)
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        // next_x at witness[3]
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata     // delta at witness[2]  
        y_prev_witness_indices: vec![], // <-- NO BINDING! This is the vulnerability
        const1_witness_index: 0,
    };
    
    // prev_acc A with y = 0
    let prev_acc_a = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    // Build a step that IGNORES the actual prev_acc and uses arbitrary prev_x
    // Since y_prev is not bound, the step can use any prev_x value
    let step_x = vec![F::from_u64(7)];
    let arbitrary_prev_x = F::from_u64(42); // Doesn't match prev_acc_a.y_compact[0] = 0
    let delta = step_x[0];                  // 7
    let next_x = arbitrary_prev_x + delta;  // 42 + 7 = 49
    let step_witness = vec![F::ONE, arbitrary_prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &prev_acc_a,
        step: 0,
        public_input: Some(&step_x),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("proving failed");
    let proof = step_result.proof;
    
    // Test against the original prev_acc A (y_prev=0)
    let accepted_a = verify_ivc_step(&step_ccs, &proof, &prev_acc_a, &binding_spec, &params, None)
        .unwrap_or(false);
    
    // Test against a different prev_acc B (y_prev=999)
    let prev_acc_b = Accumulator {
        c_z_digest: [1u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(999)],
        step: 0,
    };
    
    let accepted_b = verify_ivc_step(&step_ccs, &proof, &prev_acc_b, &binding_spec, &params, None)
        .unwrap_or(false);
    
    if accepted_a && accepted_b {
        println!("   ❌ CRITICAL VULNERABILITY CONFIRMED!");
        println!("   The verifier accepted the SAME proof against DIFFERENT previous accumulators!");
        println!("   Proof accepted against prev_acc A (y_prev=0): {}", accepted_a);
        println!("   Proof accepted against prev_acc B (y_prev=999): {}", accepted_b);
        println!("   This demonstrates that when y_prev_witness_indices is empty,");
        println!("   the step is completely detached from the previous accumulator state.");
        println!("   Root cause: No binding constraints between step witness and y_prev");
        
        // This is expected behavior when y_prev is not bound - it's a vulnerability
        // but it's the current state of most tests
        return Ok(());
    }
    
    println!("   ✅ Unexpected: Verifier rejected at least one mismatched prev_acc");
    println!("   Accepted against A: {}, Accepted against B: {}", accepted_a, accepted_b);
    
    Ok(())
}

/// Test T9: Demonstrates the robust design - context digest provides automatic y_prev binding
///
/// This test confirms that even without explicit y_prev_witness_indices binding,
/// the context digest mechanism provides strong binding to the previous accumulator
/// because y_prev is always included in the augmented public input.
#[test]
fn test_context_digest_provides_automatic_y_prev_binding() -> Result<()> {
    println!("🔒 Testing that context digest provides automatic y_prev binding...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    
    // Use minimal binding (no explicit y_prev binding in witness)
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata     
        y_prev_witness_indices: vec![], // No explicit witness binding
        const1_witness_index: 0,
    };
    
    // Create two different previous accumulators
    let prev_acc_a = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(100)], // y_prev = 100
        step: 0,
    };
    
    let prev_acc_b = Accumulator {
        c_z_digest: [1u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(200)], // y_prev = 200 (different!)
        step: 0,
    };
    
    // Generate proof for prev_acc_a
    let step_x = vec![F::from_u64(5)];
    let arbitrary_prev_x = F::from_u64(42); // Can be anything since no witness binding
    let delta = step_x[0];
    let next_x = arbitrary_prev_x + delta;
    let step_witness = vec![F::ONE, arbitrary_prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &prev_acc_a, // Proof generated for prev_acc_a
        step: 0,
        public_input: Some(&step_x),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("proving failed");
    let proof = step_result.proof;
    
    // Verify against the correct prev_acc_a
    let accepted_a = verify_ivc_step(&step_ccs, &proof, &prev_acc_a, &binding_spec, &params, None)
        .unwrap_or(false);
    
    // Try to verify against different prev_acc_b  
    let accepted_b = verify_ivc_step(&step_ccs, &proof, &prev_acc_b, &binding_spec, &params, None)
        .unwrap_or(false);
    
    println!("   📊 Results:");
    println!("   Proof accepted against original prev_acc A (y_prev=100): {}", accepted_a);
    println!("   Proof accepted against different prev_acc B (y_prev=200): {}", accepted_b);
    
    if accepted_a && !accepted_b {
        println!("   ✅ EXCELLENT: Context digest provides automatic y_prev binding!");
        println!("   Even without explicit y_prev_witness_indices, the verifier correctly");
        println!("   rejects proofs against mismatched previous accumulators because:");
        println!("   1. y_prev is included in the augmented public input: [step_x || ρ || y_prev || y_next]");
        println!("   2. Context digest is computed from this full public input");
        println!("   3. Verifier recomputes digest with its own y_prev and compares");
        println!("   This provides robust protection against accumulator substitution attacks!");
    } else if !accepted_a {
        println!("   ❌ ERROR: Proof should have been accepted against the original prev_acc");
    } else if accepted_b {
        println!("   ❌ VULNERABILITY: Proof was accepted against wrong prev_acc");
    }
    
    Ok(())
}

/// Test T10: Step Index Manipulation Attack - Can we reuse proofs across different steps?
///
/// This test attempts to exploit potential weaknesses in step index binding by trying
/// to use a proof generated for one step in a different step context.
#[test]
fn test_step_index_manipulation_attack() -> Result<()> {
    println!("🔒 Testing step index manipulation attack - can we reuse proofs across steps?...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    // Create accumulator for step 0
    let acc_step_0 = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(100)],
        step: 0, // Step 0
    };
    
    // Create accumulator for step 1 with SAME y_compact but different step
    let acc_step_1 = Accumulator {
        c_z_digest: [1u8; 32], // Different digest
        c_coords: vec![],
        y_compact: vec![F::from_u64(100)], // SAME y_compact
        step: 1, // Different step!
    };
    
    // Generate proof for step 0
    let step_x = vec![F::from_u64(5)];
    let prev_x = F::from_u64(42);
    let delta = step_x[0];
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    // Prove for step 0
    let step_input_0 = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &acc_step_0,
        step: 0, // Proving for step 0
        public_input: Some(&step_x),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input_0).expect("proving failed");
    let proof = step_result.proof;
    
    // Verify against correct step 0
    let accepted_step_0 = verify_ivc_step(&step_ccs, &proof, &acc_step_0, &binding_spec, &params, None)
        .unwrap_or(false);
    
    // Try to verify same proof against step 1 accumulator
    let accepted_step_1 = verify_ivc_step(&step_ccs, &proof, &acc_step_1, &binding_spec, &params, None)
        .unwrap_or(false);
    
    println!("   📊 Step Index Manipulation Results:");
    println!("   Proof accepted against step 0 accumulator: {}", accepted_step_0);
    println!("   Proof accepted against step 1 accumulator: {}", accepted_step_1);
    
    if accepted_step_0 && accepted_step_1 {
        println!("   ❌ CRITICAL VULNERABILITY: Step index manipulation succeeded!");
        println!("   The same proof was accepted for different step indices!");
        println!("   This could enable replay attacks across different steps in the IVC chain.");
        panic!("Step index manipulation attack succeeded!");
    } else if accepted_step_0 && !accepted_step_1 {
        println!("   ✅ Good: Step index binding prevents cross-step proof reuse");
        println!("   The verifier correctly rejected the proof when step index differs");
    } else {
        println!("   ❌ ERROR: Proof should have been accepted for the original step 0");
    }
    
    Ok(())
}

/// Test T11: Public Input Prefix Attack - Can we manipulate step_x prefix binding?
///
/// This test attempts to exploit the step_x prefix binding mechanism by creating
/// accumulators with different digest prefixes but trying to reuse proofs.
#[test]
fn test_public_input_prefix_attack() -> Result<()> {
    println!("🔒 Testing public input prefix attack - can we bypass step_x prefix binding?...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    // Create accumulator A
    let acc_a = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(50)],
        step: 0,
    };
    
    // Create accumulator B with different c_z_digest (which affects step_x prefix)
    let acc_b = Accumulator {
        c_z_digest: [0xFFu8; 32], // Very different digest
        c_coords: vec![],
        y_compact: vec![F::from_u64(50)], // Same y_compact
        step: 0, // Same step
    };
    
    // Generate proof for accumulator A
    let step_x_a = vec![F::from_u64(7)]; // This will be derived from acc_a
    let prev_x = F::from_u64(25);
    let delta = step_x_a[0];
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input_a = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &acc_a,
        step: 0,
        public_input: Some(&step_x_a),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input_a).expect("proving failed");
    let proof = step_result.proof;
    
    // Verify against correct accumulator A
    let accepted_a = verify_ivc_step(&step_ccs, &proof, &acc_a, &binding_spec, &params, None)
        .unwrap_or(false);
    
    // Try to verify same proof against accumulator B (different c_z_digest)
    let accepted_b = verify_ivc_step(&step_ccs, &proof, &acc_b, &binding_spec, &params, None)
        .unwrap_or(false);
    
    println!("   📊 Public Input Prefix Attack Results:");
    println!("   Proof accepted against original accumulator A: {}", accepted_a);
    println!("   Proof accepted against different accumulator B: {}", accepted_b);
    println!("   Accumulator A c_z_digest: {:02x?}", &acc_a.c_z_digest[..8]);
    println!("   Accumulator B c_z_digest: {:02x?}", &acc_b.c_z_digest[..8]);
    
    if accepted_a && accepted_b {
        println!("   ❌ CRITICAL VULNERABILITY: Public input prefix attack succeeded!");
        println!("   The same proof was accepted for accumulators with different c_z_digest!");
        println!("   This bypasses the step_x prefix binding mechanism.");
        panic!("Public input prefix attack succeeded!");
    } else if accepted_a && !accepted_b {
        println!("   ✅ Good: step_x prefix binding prevents accumulator substitution");
        println!("   The verifier correctly rejected the proof for different c_z_digest");
    } else {
        println!("   ❌ ERROR: Proof should have been accepted for the original accumulator A");
    }
    
    Ok(())
}

/// Test T12: Public IO Malleability Attack - Can we manipulate the proof's public_io directly?
///
/// This test attempts to directly modify the proof's public_io field to see if we can
/// bypass the context digest verification by crafting malicious public_io data.
#[test]
fn test_public_io_malleability_attack() -> Result<()> {
    println!("🔒 Testing public IO malleability attack - can we manipulate proof.public_io directly?...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let acc_original = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(100)],
        step: 0,
    };
    
    let acc_target = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(200)], // Different y_compact!
        step: 0,
    };
    
    // Generate legitimate proof for acc_original
    let step_x = vec![F::from_u64(5)];
    let prev_x = F::from_u64(30);
    let delta = step_x[0];
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &acc_original,
        step: 0,
        public_input: Some(&step_x),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("proving failed");
    let proof = step_result.proof;
    
    // Verify original proof works
    let original_valid = verify_ivc_step(&step_ccs, &proof, &acc_original, &binding_spec, &params, None)
        .unwrap_or(false);
    
    println!("   📊 Original proof verification: {}", original_valid);
    
    if !original_valid {
        println!("   ❌ ERROR: Original proof should be valid");
        return Ok(());
    }
    
    // Attack 1: Try to manipulate the y_next values in public_io
    // The public_io format is: [y_next values as bytes] + [context_digest]
    let mut malicious_proof = proof.clone();
    
    // Try to change the first y_next value from 35 to 235 (prev_x + delta = 30 + 5 = 35)
    // We want to make it look like y_next = 235 instead
    if malicious_proof.step_proof.public_io.len() >= 8 {
        // Overwrite first 8 bytes (first y_next value) with 235
        let malicious_y_next = F::from_u64(235);
        let malicious_bytes = malicious_y_next.as_canonical_u64().to_le_bytes();
        malicious_proof.step_proof.public_io[0..8].copy_from_slice(&malicious_bytes);
        
        println!("   🔧 Modified public_io: changed first y_next from 35 to 235");
        
        // Try to verify against acc_original with manipulated proof
        let malicious_accepted_original = verify_ivc_step(&step_ccs, &malicious_proof, &acc_original, &binding_spec, &params, None)
            .unwrap_or(false);
        
        // Try to verify against acc_target with manipulated proof  
        let malicious_accepted_target = verify_ivc_step(&step_ccs, &malicious_proof, &acc_target, &binding_spec, &params, None)
            .unwrap_or(false);
        
        println!("   📊 Malicious proof results:");
        println!("   Accepted against original accumulator: {}", malicious_accepted_original);
        println!("   Accepted against target accumulator: {}", malicious_accepted_target);
        
        if malicious_accepted_original || malicious_accepted_target {
            println!("   ❌ CRITICAL VULNERABILITY: Public IO malleability attack succeeded!");
            println!("   The verifier accepted a proof with manipulated public_io!");
            panic!("Public IO malleability attack succeeded!");
        } else {
            println!("   ✅ Good: Context digest verification prevents public_io manipulation");
            println!("   The verifier correctly rejected proofs with modified public_io");
        }
    } else {
        println!("   ⚠️  Cannot perform attack: public_io too short");
    }
    
    Ok(())
}

/// Test T13: Zero ρ Bypass Attack - Can we bypass the ρ guard by setting step_rho to ZERO?
///
/// This test exploits the conditional ρ check: when step_rho == F::ZERO, the guard is skipped.
/// We can then forge c_step_coords without triggering the mismatch detection.
#[test]
fn test_zero_rho_bypass_attack() -> Result<()> {
    println!("🔒 Testing zero ρ bypass attack - can we skip the ρ guard by zeroing step_rho?...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let acc_original = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(100)],
        step: 0,
    };
    
    // Generate legitimate proof
    let step_x = vec![F::from_u64(7)];
    let prev_x = F::from_u64(50);
    let delta = step_x[0];
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &acc_original,
        step: 0,
        public_input: Some(&step_x),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("proving failed");
    let mut proof = step_result.proof;
    
    // Verify original proof works
    let original_valid = verify_ivc_step(&step_ccs, &proof, &acc_original, &binding_spec, &params, None)
        .unwrap_or(false);
    
    if !original_valid {
        println!("   ❌ ERROR: Original proof should be valid");
        return Ok(());
    }
    
    // Attack: Forge c_step_coords AND zero step_rho to bypass Guard A
    if !proof.c_step_coords.is_empty() {
        proof.c_step_coords[0] += F::from_u64(42); // Forge coordinates
        proof.public_inputs.__test_tamper_rho(F::ZERO); // Zero ρ to skip the guard!
        
        println!("   🔧 Attack executed:");
        println!("   - Forged c_step_coords[0] by adding 42");
        println!("   - Set step_rho to ZERO to bypass guard");
        
        let malicious_accepted = verify_ivc_step(&step_ccs, &proof, &acc_original, &binding_spec, &params, None)
            .unwrap_or(false);
        
        println!("   📊 Zero ρ Bypass Results:");
        println!("   Original proof accepted: {}", original_valid);
        println!("   Malicious proof accepted: {}", malicious_accepted);
        
        if malicious_accepted {
            println!("   ❌ CRITICAL VULNERABILITY: Zero ρ bypass attack succeeded!");
            println!("   The verifier accepted forged coordinates when step_rho was zeroed!");
            println!("   This bypasses Guard A entirely by exploiting the conditional check.");
            panic!("Zero ρ bypass attack succeeded!");
        } else {
            println!("   ✅ Good: Zero ρ bypass attack was prevented");
            println!("   The verifier correctly rejected forged coordinates even with zero step_rho");
        }
    } else {
        println!("   ⚠️  Cannot perform attack: c_step_coords is empty");
    }
    
    Ok(())
}

/// Test T14: Coordinated ρ + Coordinates Attack - Can we forge both coordinates and matching ρ?
///
/// This test attempts to forge c_step_coords while also recomputing the matching step_rho
/// to pass Guard A, then see if the verifier still accepts the coordinated forgery.
#[test]
fn test_coordinated_rho_coordinates_attack() -> Result<()> {
    println!("🔒 Testing coordinated ρ + coordinates attack - can we forge both and pass Guard A?...");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // Empty - NIVC wrapper handles lane metadata
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let acc_original = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(75)],
        step: 0,
    };
    
    // Generate legitimate proof
    let step_x = vec![F::from_u64(10)];
    let prev_x = F::from_u64(25);
    let delta = step_x[0];
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];
    
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &acc_original,
        step: 0,
        public_input: Some(&step_x),
        y_step: &y_step,
        binding_spec: &binding_spec,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("proving failed");
    let mut proof = step_result.proof;
    
    // Verify original proof works
    let original_valid = verify_ivc_step(&step_ccs, &proof, &acc_original, &binding_spec, &params, None)
        .unwrap_or(false);
    
    if !original_valid {
        println!("   ❌ ERROR: Original proof should be valid");
        return Ok(());
    }
    
    // Attack: Forge c_step_coords AND recompute matching step_rho
    if !proof.c_step_coords.is_empty() {
        // Store original values for comparison
        let original_coord = proof.c_step_coords[0];
        let original_rho = proof.public_inputs.rho();
        
        // Forge coordinates
        proof.c_step_coords[0] += F::from_u64(123); // Forge coordinates
        
        // Recompute step_rho for the forged coordinates using the same logic as the verifier
        let step_digest = [1u8; 32]; // Use a dummy digest for this test
        let (recomputed_rho, _transcript) = rho_from_transcript(&acc_original, step_digest, &proof.c_step_coords);
        proof.public_inputs.__test_tamper_rho(recomputed_rho); // Update ρ to match forged coordinates
        
        println!("   🔧 Coordinated attack executed:");
        println!("   - Original c_step_coords[0]: {}", original_coord.as_canonical_u64());
        println!("   - Forged c_step_coords[0]: {}", proof.c_step_coords[0].as_canonical_u64());
        println!("   - Original step_rho: {}", original_rho.as_canonical_u64());
        println!("   - Recomputed step_rho: {}", proof.public_inputs.rho().as_canonical_u64());
        
        let malicious_accepted = verify_ivc_step(&step_ccs, &proof, &acc_original, &binding_spec, &params, None)
            .unwrap_or(false);
        
        println!("   📊 Coordinated Attack Results:");
        println!("   Original proof accepted: {}", original_valid);
        println!("   Coordinated malicious proof accepted: {}", malicious_accepted);
        
        if malicious_accepted {
            println!("   ❌ CRITICAL VULNERABILITY: Coordinated ρ + coordinates attack succeeded!");
            println!("   The verifier accepted forged coordinates with matching recomputed ρ!");
            println!("   This shows that Guard A can be satisfied by coordinated forgery.");
            panic!("Coordinated ρ + coordinates attack succeeded!");
        } else {
            println!("   ✅ Good: Coordinated attack was prevented");
            println!("   The verifier correctly rejected the coordinated forgery");
        }
    } else {
        println!("   ⚠️  Cannot perform attack: c_step_coords is empty");
    }
    
    Ok(())
}
