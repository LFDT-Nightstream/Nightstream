//! IVC negative test: unsatisfiable step witness should be rejected by verify_ivc_step.
//! This test intentionally does NOT call `check_ccs_rowwise_zero` beforehand.

use neo::{F, NeoParams};
use neo::ivc::{
    Accumulator, LastNExtractor, StepBindingSpec,
    prove_ivc_step_with_extractor, verify_ivc_step,
};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Tiny R1CS-encoded CCS for constraint: (z0 - z1) * z0 = 0
fn tiny_r1cs_to_ccs() -> neo_ccs::CcsStructure<F> {
    // One row, two vars: A = [1, -1], B = [1, 0], C = [0, 0]
    // Row-wise, (A z)[i] * (B z)[i] - (C z)[i] = 0  =>  (z0 - z1) * z0 = 0
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
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
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // UNSAT step witness for (z0 - z1) * z0 = 0: choose z = [1, 5]
    // (1 - 5) * 1 = -4 != 0, so the step CCS is violated.
    // Note: witness[0] == 1 satisfies the const-1 convention used by the augmented CCS.
    let step_witness = vec![F::ONE, F::from_u64(5)];

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
    // For constraint: (z0 - z1) * z0 = 0
    // Valid witness examples: [1, 1], [0, 0], [0, 5], etc.
    // INVALID witness: [2, 5] -> (2 - 5) * 2 = -6 ≠ 0
    
    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // INVALID witness: [1, 7] doesn't satisfy (z0 - z1) * z0 = 0
    // because (1 - 7) * 1 = -6 ≠ 0
    // Note: First element is 1 to satisfy const-1 convention
    let invalid_witness = vec![F::ONE, F::from_u64(7)];
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
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Start with a VALID witness to get a valid proof structure
    let valid_witness = vec![F::ONE, F::ONE];
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
        neo::ivc::IvcProof {
            step_proof: valid_proof_res.proof.step_proof.clone(),
            next_accumulator: valid_proof_res.proof.next_accumulator.clone(),
            step: valid_proof_res.proof.step,
            metadata: valid_proof_res.proof.metadata.clone(),
            step_public_input: valid_proof_res.proof.step_public_input.clone(),
            step_augmented_public_input: valid_proof_res.proof.step_augmented_public_input.clone(),
            prev_step_augmented_public_input: valid_proof_res.proof.prev_step_augmented_public_input.clone(),
            step_rho: valid_proof_res.proof.step_rho,
            step_y_prev: valid_proof_res.proof.step_y_prev.clone(),
            step_y_next: valid_proof_res.proof.step_y_next.clone(),
            c_step_coords: valid_proof_res.proof.c_step_coords.clone(),
            me_instances: valid_proof_res.proof.me_instances.clone(), // Unchanged
            digit_witnesses: valid_proof_res.proof.digit_witnesses.clone(), // Unchanged
            folding_proof: Some(malicious_folding), // TAMPERED folding proof
        }
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
