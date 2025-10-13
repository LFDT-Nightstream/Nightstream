/// Regression test for all-zero witness attack
/// This reproduces the exact attack pattern from the external integration logs
/// where all z_vars and z_digits are 0, making all AJTAI-CHECK rows 0=0

use neo::{F, NeoParams};
use neo::{Accumulator, StepBindingSpec, prove_ivc_step_with_extractor, LastNExtractor, verify_ivc_step_legacy};
use neo_ccs::{Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a simple increment CCS: next = prev + delta
fn build_increment_ccs() -> neo_ccs::CcsStructure<F> {
    let rows = 1;
    let cols = 4;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    // (next - prev - delta) * const1 = 0
    a[3] = F::ONE;   // +next
    a[1] = -F::ONE;  // -prev
    a[2] = -F::ONE;  // -delta
    b[0] = F::ONE;   // * const1
    
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

#[test]
fn test_all_zero_witness_attack_blocked() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![2],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };
    
    // ATTACK: All-zero witness (including const1 = 0!)
    // This is the exact attack from the external logs:
    //   z_vars[*] = 0, z_digits[*] = 0
    //   AJTAI-CHECK rows: lhs=0 rhs=0
    let all_zero_witness = vec![
        F::ZERO,  // const1 = 0 ❌ (ATTACK!)
        F::ZERO,  // prev = 0
        F::ZERO,  // delta = 0
        F::ZERO,  // next = 0
    ];
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![F::ZERO];
    
    println!("🚨 ATTACK: Attempting to prove with ALL-ZERO witness (const1=0)");
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &all_zero_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    // Post-fix: The augmented CCS has a constraint w_const1 * ρ = ρ
    // With const1=0, this becomes: 0 * ρ = ρ, which is FALSE (since ρ ≠ 0)
    // The verifier MUST reject this.
    
    match result {
        Ok(step_result) => {
            println!("⚠️  Prover generated proof (prover-side check may be disabled)");
            println!("   Testing VERIFIER with in-circuit const-1 enforcement...");
            
            let verify_result = verify_ivc_step_legacy(
                &step_ccs,
                &step_result.proof,
                &prev_acc,
                &binding,
                &params,
                None,
            );
            
            match verify_result {
                Ok(is_valid) => {
                    if is_valid {
                        println!("\n❌❌❌ CRITICAL SOUNDNESS BUG ❌❌❌");
                        println!("   All-zero witness VERIFIED successfully!");
                        println!("   const1=0, all other values=0");
                        println!("   This means w_const1 * ρ = ρ constraint is NOT working!");
                        println!("\n   Expected behavior:");
                        println!("   - Augmented CCS should have: w_const1 * ρ = ρ");
                        println!("   - With const1=0: 0 * ρ = ρ (FALSE since ρ ≠ 0)");
                        println!("   - Verifier should REJECT");
                        panic!("SOUNDNESS BUG: All-zero witness accepted!");
                    } else {
                        println!("✅ GOOD: Verifier REJECTED all-zero witness");
                        println!("   The const-1 enforcement constraint is working!");
                    }
                }
                Err(e) => {
                    println!("✅ GOOD: Verification failed with error: {}", e);
                    println!("   In-circuit constraints working");
                }
            }
        }
        Err(e) => {
            println!("✅ GOOD: Prover failed early: {}", e);
            println!("   (Prover-side check or constraint validation caught it)");
        }
    }
}

#[test]
fn test_all_zero_witness_with_nonzero_const1_still_invalid() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![2],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };
    
    // Even with const1=1, if the CCS constraints aren't satisfied, it should fail
    // Constraint: next - prev - delta = 0
    // With all zeros except const1: 0 - 0 - 0 = 0 ✓ (actually satisfies!)
    // But if we claim a non-zero public input, binding should fail
    let mostly_zero_witness = vec![
        F::ONE,   // const1 = 1 ✓
        F::ZERO,  // prev = 0
        F::ZERO,  // delta = 0 (but we'll claim delta=5 in public)
        F::ZERO,  // next = 0
    ];
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![F::from_u64(5)]; // Claim delta=5 (lie!)
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &mostly_zero_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    // The app-input binding should catch this: witness[2]=0 but public claims 5
    match result {
        Ok(step_result) => {
            let verify_result = verify_ivc_step_legacy(
                &step_ccs,
                &step_result.proof,
                &prev_acc,
                &binding,
                &params,
                None,
            );
            
            match verify_result {
                Ok(is_valid) => {
                    assert!(!is_valid, 
                        "Verifier should reject when witness[delta]=0 but public claims delta=5");
                }
                Err(_e) => {
                    // Verification failed (good - binding caught the mismatch)
                }
            }
        }
        Err(_e) => {
            // Prover failed (also acceptable)
        }
    }
}

#[test]
fn test_valid_zero_state_with_const1_one() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![2],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };
    
    // VALID: const1=1, and 0 + 0 = 0 is a valid computation
    let valid_zero_witness = vec![
        F::ONE,   // const1 = 1 ✓
        F::ZERO,  // prev = 0 ✓
        F::ZERO,  // delta = 0 ✓
        F::ZERO,  // next = 0 ✓ (0 + 0 = 0)
    ];
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![F::ZERO]; // Honestly claim delta=0
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &valid_zero_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    // This SHOULD succeed - it's a valid proof of 0+0=0 with const1=1
    let step_result = result.expect("Valid zero-state proof should succeed");
    
    let ok = verify_ivc_step_legacy(
        &step_ccs,
        &step_result.proof,
        &prev_acc,
        &binding,
        &params,
        None,
    ).expect("Verification should not error");
    
    assert!(ok, "Valid zero-state proof (with const1=1) should verify");
}

#[test]
fn test_all_zero_witness_attack_with_ivc_session() {
    use neo::session::{IvcSession, NeoStep, StepArtifacts, StepSpec};
    
    struct AllZeroAttackStepper {
        ccs: neo_ccs::CcsStructure<F>,
        spec: StepSpec,
    }
    
    impl AllZeroAttackStepper {
        fn new() -> Self {
            let ccs = build_increment_ccs();
            let spec = StepSpec {
                y_len: 1,
                const1_index: 0,
                y_step_indices: vec![3],
                y_prev_indices: Some(vec![1]),
                app_input_indices: None,  // No app inputs for this test
            };
            Self { ccs, spec }
        }
    }
    
    impl NeoStep for AllZeroAttackStepper {
        type ExternalInputs = ();
        
        fn state_len(&self) -> usize { 1 }
        fn step_spec(&self) -> StepSpec { self.spec.clone() }
        
        fn synthesize_step(
            &mut self,
            _step_idx: usize,
            _z_prev: &[F],
            _inputs: &Self::ExternalInputs,
        ) -> StepArtifacts {
            let all_zero_witness = vec![
                F::ZERO,  // const1 = 0 ❌
                F::ZERO,  // prev = 0
                F::ZERO,  // delta = 0
                F::ZERO,  // next = 0
            ];
            
            StepArtifacts {
                ccs: self.ccs.clone(),
                witness: all_zero_witness,
                public_app_inputs: vec![],
                spec: self.spec.clone(),
            }
        }
    }
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let mut session = IvcSession::new(&params, Some(vec![F::ZERO]), 0);
    let mut stepper = AllZeroAttackStepper::new();
    
    let step_result = session.prove_step(&mut stepper, &());
    
    match step_result {
        Ok(_) => panic!("IvcSession should reject all-zero witness (const1=0)"),
        Err(e) => {
            let error = e.to_string();
            assert!(
                error.contains("constant-1") || error.contains("const1") || error.contains("must be 1"),
                "Error should mention const-1: {}", error
            );
        }
    }
}

#[test]
fn test_valid_ivc_session_with_proper_const1() {
    use neo::session::{IvcSession, NeoStep, StepArtifacts, StepSpec};
    
    struct HonestStepper {
        ccs: neo_ccs::CcsStructure<F>,
        spec: StepSpec,
    }
    
    impl HonestStepper {
        fn new() -> Self {
            let ccs = build_increment_ccs();
            let spec = StepSpec {
                y_len: 1,
                const1_index: 0,
                y_step_indices: vec![3],
                y_prev_indices: Some(vec![1]),
                app_input_indices: None,
            };
            Self { ccs, spec }
        }
    }
    
    impl NeoStep for HonestStepper {
        type ExternalInputs = ();
        
        fn state_len(&self) -> usize { 1 }
        fn step_spec(&self) -> StepSpec { self.spec.clone() }
        
        fn synthesize_step(
            &mut self,
            _step_idx: usize,
            z_prev: &[F],
            _inputs: &Self::ExternalInputs,
        ) -> StepArtifacts {
            let prev = z_prev[0];
            let delta = F::from_u64(3);
            let next = prev + delta;
            
            let valid_witness = vec![
                F::ONE,   // const1 = 1 ✓
                prev,     // prev
                delta,    // delta
                next,     // next
            ];
            
            StepArtifacts {
                ccs: self.ccs.clone(),
                witness: valid_witness,
                public_app_inputs: vec![],
                spec: self.spec.clone(),
            }
        }
    }
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let mut session = IvcSession::new(&params, Some(vec![F::ZERO]), 0);
    let mut stepper = HonestStepper::new();
    
    let step_result = session.prove_step(&mut stepper, &());
    assert!(step_result.is_ok(), "Honest proof with const1=1 should succeed");
}

/// CRITICAL TEST: All-zero witness (everything is 0, including const1)
/// This tests if the system incorrectly accepts a witness that satisfies NO constraints.
/// 
/// Expected behavior:
/// - Prover should reject (const-1 check catches it)
/// - If prover is malicious and bypasses check, VERIFIER must reject
/// 
/// Bug scenario (if this test fails):
/// - System generates proof with all zeros
/// - Verifier accepts the proof
/// - This would be a CRITICAL soundness bug
#[test]
fn test_complete_zero_witness_critical() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![2],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };
    
    // CRITICAL: Complete all-zero witness
    // Constraint: next - prev - delta = 0
    // With all zeros: 0 - 0 - 0 = 0 ✓ (step CCS is satisfied!)
    // BUT const1=0 should make this invalid via in-circuit enforcement
    let complete_zero_witness = vec![
        F::ZERO,  // const1 = 0 ❌ (CRITICAL!)
        F::ZERO,  // prev = 0
        F::ZERO,  // delta = 0
        F::ZERO,  // next = 0
    ];
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![F::ZERO]; // Claim delta=0 (matches witness)
    
    println!("\n🔥 CRITICAL TEST: Complete all-zero witness");
    println!("================================================");
    println!("Witness: [0, 0, 0, 0] (const1=0!)");
    println!("Public input: [0]");
    println!("Step CCS constraint: 0 - 0 - 0 = 0 ✓ (satisfied by zeros)");
    println!("\n⚠️  If prover generates proof AND verifier accepts:");
    println!("   => SOUNDNESS BUG CONFIRMED");
    println!("\n✅ Expected: Prover rejects (const1 check) OR Verifier rejects (in-circuit)");
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &complete_zero_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    match result {
        Ok(step_result) => {
            println!("\n⚠️  PROVER GENERATED PROOF (prover-side check bypassed or disabled)");
            println!("   Testing VERIFIER...");
            
            let verify_result = verify_ivc_step_legacy(
                &step_ccs,
                &step_result.proof,
                &prev_acc,
                &binding,
                &params,
                None,
            );
            
            match verify_result {
                Ok(is_valid) => {
                    if is_valid {
                        println!("\n❌❌❌ CRITICAL SOUNDNESS BUG FOUND ❌❌❌");
                        println!("   Complete zero witness VERIFIED successfully!");
                        println!("   const1=0, all values=0");
                        println!("   This means:");
                        println!("   1. The w_const1 * ρ = ρ constraint is NOT working");
                        println!("   2. OR the constraint is missing from augmented CCS");
                        println!("   3. OR ρ is somehow zero (impossible!)");
                        panic!("CRITICAL: Complete zero witness accepted by verifier!");
                    } else {
                        println!("\n✅ GOOD: Verifier REJECTED complete zero witness");
                        println!("   In-circuit const-1 enforcement is working!");
                    }
                }
                Err(e) => {
                    println!("\n✅ GOOD: Verification failed with error: {}", e);
                    println!("   In-circuit constraints caught the zero witness");
                }
            }
        }
        Err(e) => {
            println!("\n✅ GOOD: Prover rejected complete zero witness");
            println!("   Error: {}", e);
            
            // Verify it's the const-1 check
            let error = e.to_string();
            if error.contains("constant-1") || error.contains("const1") || error.contains("must be 1") {
                println!("   Caught by const-1 security check ✓");
            } else {
                println!("   Caught by different check: {}", error);
            }
        }
    }
}
