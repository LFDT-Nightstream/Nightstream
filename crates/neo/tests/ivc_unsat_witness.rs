//! IVC negative test: unsatisfiable step witness should be rejected by verify_ivc_step.
//! This test intentionally does NOT call `check_ccs_rowwise_zero` beforehand.
//!
//! Uses a 4-row CCS (ℓ=2, minimum required for security) to ensure the verifier 
//! can detect invalid witnesses via the initial_sum == 0 check at the base case.

use neo::{F, NeoParams};
use neo::{
    Accumulator, LastNExtractor, StepBindingSpec,
    prove_ivc_step_with_extractor, verify_ivc_step,
};
use neo_ccs::{Mat, SparsePoly, Term, CcsStructure};
use p3_field::PrimeCharacteristicRing;

/// Build a 4-row CCS to achieve ℓ=2 (minimum 4 rows required for security)
/// CCS polynomial: f(M0, M1, M2) = M0·M1 - M2
/// Witness format: [const1, z0, z1]
/// Constraints:
///   Row 0: z0 * z0 = z0  (z0 is boolean)
///   Row 1: z1 * z1 = z1  (z1 is boolean)
///   Row 2: z0 * z1 = z0  (if z0=1, then z1 must be 1)
///   Row 3: 0 * 1 = 0     (dummy constraint for security)
fn tiny_r1cs_to_ccs() -> neo_ccs::CcsStructure<F> {
    let m0 = Mat::from_row_major(4, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ZERO,  // Row 3: 0 (dummy)
    ]);
    let m1 = Mat::from_row_major(4, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ZERO, F::ONE,   // Row 2: z1
        F::ONE, F::ZERO, F::ZERO,   // Row 3: 1 (dummy)
    ]);
    let m2 = Mat::from_row_major(4, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ZERO,  // Row 3: 0 (dummy)
    ]);
    
    // CCS polynomial: f(X0, X1, X2) = X0·X1 - X2
    let terms = vec![
        Term { coeff: F::ONE, exps: vec![1, 1, 0] },   // X0 * X1
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // -X2
    ];
    let f = SparsePoly::new(3, terms);
    
    CcsStructure::new(vec![m0, m1, m2], f)
        .expect("valid CCS structure")
}

#[test]
fn ivc_unsat_step_witness_should_fail_verify() {
    // Step CCS and params
    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Base accumulator: no prior commitment, no y (y_len = 0)
    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    // Binding spec: no y, no app-input binding; const-1 witness at index 0
    // We ensure witness[0] == 1 below to satisfy the const-1 convention.
    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // INVALID witness: [const1=1, z0=1, z1=5]
    // Row 0: 1*1 - 1 = 0 ✓
    // Row 1: 5*5 - 5 = 20 ≠ 0 ❌
    // Row 2: 1*5 - 1 = 4 ≠ 0 ❌
    let step_witness = vec![F::ONE, F::ONE, F::from_u64(5)];

    // No app public inputs; y_len == 0, so extractor returns empty y_step
    let extractor = LastNExtractor { n: 0 };

    // Prove one IVC step (prover can always produce a transcript; soundness is in verify)
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &step_witness,
        &prev_acc,
        prev_acc.step,
        None,                 // no app public inputs
        &extractor,
        &binding,
    ).expect("IVC step proving should not error");

    // Verify should REJECT because the step witness violates the step CCS
    let ok = verify_ivc_step(
        &step_ccs,
        &step_res.proof,
        &prev_acc,
        &binding,
        &params,
        None, // prev_augmented_x
    ).expect("verify_ivc_step should not error");

    // Expect rejection; if this assertion fails, it demonstrates the bug the user reported.
    assert!(!ok, "IVC verification accepted an unsatisfiable step witness");
}

#[test]
fn ivc_proof_with_invalid_witness_from_generation() {
    // This test generates a proof using an INVALID witness (one that doesn't satisfy
    // the step CCS constraint). The prover will compute everything "honestly" from
    // this invalid witness, so the digit witnesses will be consistent with the ME instances.
    // However, the witness itself violates the CCS, so verification should reject it.
    //
    // For the 4-row CCS with constraints:
    //   Row 0: z0 * z0 = z0  (z0 is boolean)
    //   Row 1: z1 * z1 = z1  (z1 is boolean)
    //   Row 2: z0 * z1 = z0  (if z0=1, then z1 must be 1)
    // Valid witness examples: [1, 1, 1], [1, 0, 0], etc.
    // INVALID witness: [1, 1, 7] -> Row 1: 7*7 - 7 = 42 ≠ 0, Row 2: 1*7 - 1 = 6 ≠ 0
    
    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // INVALID witness: [const1=1, z0=1, z1=7]
    // Row 0: 1*1 - 1 = 0 ✓
    // Row 1: 7*7 - 7 = 42 ≠ 0 ❌
    // Row 2: 1*7 - 1 = 6 ≠ 0 ❌
    let invalid_witness = vec![F::ONE, F::ONE, F::from_u64(7)];
    let extractor = LastNExtractor { n: 0 };

    // Prove with the INVALID witness
    // The prover will compute digit witnesses, commitments, etc. consistently
    // from this invalid witness, but the witness itself violates the CCS
    let proof_res = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &invalid_witness,  // <-- INVALID WITNESS HERE
        &prev_acc,
        prev_acc.step,
        None,
        &extractor,
        &binding,
    ).expect("Proving should complete (soundness check is in verify)");

    // Verify should REJECT because the witness doesn't satisfy the CCS
    let ok = verify_ivc_step(
        &step_ccs,
        &proof_res.proof,
        &prev_acc,
        &binding,
        &params,
        None,
    ).expect("verify_ivc_step should not error");

    // Expect rejection: even though digit witnesses are consistent with ME instances,
    // the underlying witness violates the step CCS constraints
    assert!(!ok, "IVC verification accepted a proof generated with an invalid witness!");
}

#[test]
fn ivc_cross_link_vulnerability_pi_ccs_rhs_vs_parent_me() {
    // THE VULNERABILITY (from security review):
    // Π-CCS RHS ME (pi_ccs_outputs[1]) is not cross-linked to the parent ME
    // (which is bound to Z via check_me_consistency).
    //
    // A sophisticated malicious prover could:
    // 1. Generate Π-CCS proof with manipulated y_scalars (makes terminal check pass)
    // 2. Provide digit witnesses/ME that are self-consistent (passes tie check)  
    // 3. Exploit: NO check that pi_ccs_outputs[1].y_scalars == me_parent.y_scalars
    // 4. Verifier accepts even though witness doesn't satisfy the CCS
    
    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Start with a VALID witness to get a valid proof structure
    // Valid witness: [const1=1, z0=1, z1=1]
    let valid_witness = vec![F::ONE, F::ONE, F::ONE];
    let extractor = LastNExtractor { n: 0 };

    let valid_proof_res = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &valid_witness,
        &prev_acc,
        prev_acc.step,
        None,
        &extractor,
        &binding,
    ).expect("Valid proof generation should succeed");

    // Now construct a MALICIOUS proof that exploits the missing cross-link:
    // We'll tamper with the Π-CCS RHS ME's y_scalars to different values
    // while keeping the digit witnesses and digit ME instances unchanged.
    //
    // Without the cross-link check, the verifier will:
    // - Accept the Π-CCS proof (because we'll keep it internally consistent OR it fails earlier)
    // - Accept the tie check (because digit ME and digit witnesses match)
    // - But NOT check that pi_ccs_outputs[1].y_scalars matches the parent ME's y_scalars
    
    let malicious_proof = {
        let mut malicious_folding = valid_proof_res.proof.folding_proof.clone()
            .expect("Folding proof should exist");
        
        // Tamper with the RHS ME's y_scalars
        if malicious_folding.pi_ccs_outputs.len() >= 2 {
            let rhs_me = &mut malicious_folding.pi_ccs_outputs[1];
            
            // Change the y_scalars to wrong values
            // This breaks the tie relationship: these y_scalars won't match
            // what you'd compute from the actual digit witnesses
            if !rhs_me.y_scalars.is_empty() {
                rhs_me.y_scalars[0] += neo_math::K::from(F::from_u64(999));
            }
        }
        
        // Construct the malicious IvcProof with tampered pi_ccs_outputs
        let mut malicious_proof_inner = valid_proof_res.proof.clone();
        malicious_proof_inner.folding_proof = Some(malicious_folding); // TAMPERED folding proof
        malicious_proof_inner
    };

    // Attempt verification
    let ok = verify_ivc_step(
        &step_ccs,
        &malicious_proof,
        &prev_acc,
        &binding,
        &params,
        None,
    ).expect("verify_ivc_step should not error");

    if ok {
        // TEST FAILS - VULNERABILITY EXISTS!
        panic!(
            "The verifier ACCEPTED a malicious proof with tampered Π-CCS RHS y_scalars!"
        );
    }
    
    println!("✅ Test PASSED: Verifier correctly REJECTED the tampered proof.");
}
