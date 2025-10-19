//! Direct test of in-circuit constraint enforcement
//! 
//! This test bypasses the session API and directly tests prove_ivc_step to see
//! if the augmented CCS properly enforces step CCS constraints.

use neo::{F, NeoParams};
use neo::{Accumulator, StepBindingSpec, prove_ivc_step_with_extractor, verify_ivc_step, LastNExtractor};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::{PrimeCharacteristicRing};

/// Build a simple CCS that enforces: next = prev + delta
/// Variables: [const, prev, delta, next]
/// Constraint: next - prev - delta = 0
fn build_increment_ccs() -> CcsStructure<F> {
    let rows = 4;  // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 4;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];

    // Constraint: (next - prev - delta) * 1 = 0
    a[0 * cols + 3] = F::ONE;   // +next
    a[0 * cols + 1] = -F::ONE;  // -prev
    a[0 * cols + 2] = -F::ONE;  // -delta
    b[0 * cols + 0] = F::ONE;   // *const1

    // Rows 1-3: dummy constraints (0 * 1 = 0)
    for row in 1..4 {
        a[row * cols] = F::ZERO;
        b[row * cols] = F::ONE;
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn test_valid_witness_passes() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],  // prev state = 0
        step: 0,
    };
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],           // next is at witness[3]
        step_program_input_witness_indices: vec![2],  // delta is at witness[2], binds to public input
        const1_witness_index: 0,
    };
    
    // VALID witness: next = prev + delta = 0 + 3 = 3
    let delta = F::from_u64(3);
    let valid_witness = vec![
        F::ONE,              // const
        F::ZERO,             // prev = 0
        delta,               // delta = 3
        F::from_u64(3),      // next = 3 ✓
    ];
    
    let check = check_ccs_rowwise_zero(&step_ccs, &[], &valid_witness);
    assert!(check.is_ok(), "Valid witness should satisfy step CCS");
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![delta];
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &valid_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    let step_result = result.expect("Valid witness should be accepted");
    
    let ok = verify_ivc_step(
        &step_ccs,
        &step_result.proof,
        &prev_acc,
        &binding,
        &params,
        None,
    ).expect("verify should not error");
    
    assert!(ok, "Valid proof should verify");
}

#[test]
fn test_invalid_witness_in_circuit_enforcement() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],  // prev state = 0
        step: 0,
    };
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],           // next is at witness[3]
        step_program_input_witness_indices: vec![2],  // delta is at witness[2]
        const1_witness_index: 0,
    };
    
    // INVALID witness: next should be 3, but we set it to 10
    let delta = F::from_u64(3);
    let invalid_witness = vec![
        F::ONE,              // const
        F::ZERO,             // prev = 0
        delta,               // delta = 3
        F::from_u64(10),     // next = 10 ❌ (should be 3!)
    ];
    
    let check = check_ccs_rowwise_zero(&step_ccs, &[], &invalid_witness);
    assert!(check.is_err(), "Invalid witness should NOT satisfy step CCS");
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![delta];
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &invalid_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    let step_result = result.expect("Prover should generate proof (prover-side check disabled)");
    
    let verify_result = verify_ivc_step(
        &step_ccs,
        &step_result.proof,
        &prev_acc,
        &binding,
        &params,
        None,
    );
    
    match verify_result {
        Ok(is_valid) => {
            assert!(!is_valid, "SOUNDNESS BUG: Verifier accepted proof with invalid witness!");
        }
        Err(_e) => {
            // Verification failed (good - in-circuit constraints working)
        }
    }
}

/// REGRESSION TEST: Explicitly test the const-1 = 0 attack vector
/// This is the exact attack that was possible before the soundness fix.
/// A malicious prover sets witness[const1_index] = 0 to neutralize all
/// linear constraints that multiply by const-1.
#[test]
fn test_const1_zero_attack_is_blocked() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
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
        const1_witness_index: 0,  // This is what the attacker targets
    };
    
    // ATTACK: Forge a witness with const1=0 to try to zero out constraints
    // The step CCS constraint is: (next - prev - delta) * const1 = 0
    // If const1=0, this becomes: (next - prev - delta) * 0 = 0 (always true!)
    let delta = F::from_u64(3);
    let malicious_witness = vec![
        F::ZERO,             // const1 = 0 ❌ (ATTACK!)
        F::ZERO,             // prev = 0
        delta,               // delta = 3
        F::from_u64(999),    // next = 999 (completely wrong, should be 3)
    ];
    
    // Verify this witness would satisfy the step CCS if const1 could be 0
    // (it multiplies by 0, making the constraint trivial)
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![delta];
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &malicious_witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    // Post-fix: The augmented CCS has a constraint w_const1 * ρ = ρ
    // which forces const1 = 1. This should catch the attack.
    match result {
        Ok(step_result) => {
            // Prover generated proof, but verifier must reject
            let verify_result = verify_ivc_step(
                &step_ccs,
                &step_result.proof,
                &prev_acc,
                &binding,
                &params,
                None,
            );
            
            match verify_result {
                Ok(is_valid) => {
                    assert!(!is_valid, "CRITICAL: const1=0 attack was NOT blocked! The w_const1 * ρ = ρ constraint failed.");
                }
                Err(_e) => {
                    // Verification failed (good - const1 enforcement working)
                }
            }
        }
        Err(_e) => {
            // Prover failed (also acceptable - caught early)
        }
    }
}
