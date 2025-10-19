/// Test that app inputs are cryptographically bound to witness positions
/// to prevent public input malleability attacks.

use neo::{F, NeoParams};
use neo::{Accumulator, StepBindingSpec, prove_ivc_step_with_extractor, verify_ivc_step, LastNExtractor};
use neo_ccs::{Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a simple increment CCS: next = prev + delta
/// Padded to 4 rows to satisfy ℓ >= 2 requirement
fn build_increment_ccs() -> neo_ccs::CcsStructure<F> {
    // Variables: [const1, prev, delta, next]
    // Constraint: next - prev - delta = 0
    let rows = 4;  // Minimum for ℓ >= 2
    let cols = 4;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    // Row 0: (next - prev - delta) * const1 = 0
    a[0 * cols + 3] = F::ONE;   // +next
    a[0 * cols + 1] = -F::ONE;  // -prev
    a[0 * cols + 2] = -F::ONE;  // -delta
    b[0 * cols + 0] = F::ONE;   // * const1
    
    // Rows 1-3: Padding constraints (0 * const1 = 0)
    // These are always satisfied and don't affect the circuit logic
    for row in 1..rows {
        b[row * cols + 0] = F::ONE;  // * const1
        // a and c remain zero for these rows
    }
    
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

#[test]
fn test_app_input_binding_prevents_malleability() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    // CCS: next = prev + delta
    // Variables: [const1, prev, delta, next]
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],           // next is at witness[3]
        step_program_input_witness_indices: vec![2],  // delta is at witness[2] (public input)
        const1_witness_index: 0,
    };
    
    // ATTACK ATTEMPT: Prove with delta=5 in witness, but claim delta=10 in public
    let actual_delta = F::from_u64(5);
    let claimed_delta = F::from_u64(10);
    let prev = F::ZERO;
    let next = prev + actual_delta; // = 5
    
    let witness = vec![
        F::ONE,         // const1 = 1
        prev,           // prev = 0
        actual_delta,   // delta = 5 (what's actually in the witness)
        next,           // next = 5
    ];
    
    let extractor = LastNExtractor { n: 1 };
    let malicious_public_input = vec![claimed_delta]; // Claim delta=10 (lie!)
    
    // Try to prove with the malicious public input
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &witness,
        &prev_acc,
        0,
        Some(&malicious_public_input),
        &extractor,
        &binding,
    );
    
    // The augmented CCS should have binding rows that enforce:
    // step_x[i] = witness[step_program_input_witness_indices[i]]
    // In this case: public_input[0] (delta) = witness[2]
    // Since public_input[0] = 10 but witness[2] = 5, this should FAIL
    
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
                    assert!(!is_valid, 
                        "SECURITY BUG: Verifier accepted proof with malicious public input! \
                         Witness had delta=5 but public claimed delta=10.");
                }
                Err(_e) => {
                    // Verification failed (good - binding working)
                }
            }
        }
        Err(_e) => {
            // Prover failed (also acceptable - caught early by prover-side check or constraint building)
        }
    }
}

#[test]
fn test_app_input_binding_allows_honest_proof() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };
    
    // CCS: next = prev + delta
    // Variables: [const1, prev, delta, next]
    let binding = StepBindingSpec {
        y_step_offsets: vec![3],           // next is at witness[3]
        step_program_input_witness_indices: vec![2],  // delta is at witness[2] (public input)
        const1_witness_index: 0,
    };
    
    // HONEST PROOF: delta matches between witness and public
    let delta = F::from_u64(5);
    let prev = F::ZERO;
    let next = prev + delta; // = 5
    
    let witness = vec![
        F::ONE,   // const1 = 1
        prev,     // prev = 0
        delta,    // delta = 5
        next,     // next = 5
    ];
    
    let extractor = LastNExtractor { n: 1 };
    let public_input = vec![delta]; // Honest: matches witness[2]
    
    let result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &witness,
        &prev_acc,
        0,
        Some(&public_input),
        &extractor,
        &binding,
    );
    
    // Honest proof should succeed
    let step_result = result.expect("Honest proof should be generated");
    
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
            if !is_valid {
                println!("❌ Honest proof REJECTED by verifier");
                println!("   witness: {:?}", witness);
                println!("   public_input: {:?}", public_input);
                panic!("Honest proof should verify successfully");
            }
        }
        Err(e) => {
            println!("❌ Verification failed with error: {}", e);
            panic!("Verification should not error for honest proof");
        }
    }
}
