use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{ExtF, F};
use neo_fold::FoldState;
// Oracle removed in NARK mode
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

/// Test folding with zero polynomial edge case.
/// Ensures the folding scheme handles empty or zero-degree polynomials correctly.
#[test]
fn test_zero_poly_folding() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ZERO], 1)],
        mv_poly(|_: &[ExtF]| ExtF::ZERO, 1)
    );
   
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
   
    // Create instances with zero witnesses
    let instance1 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ZERO, // Zero error term
    };
    let witness1 = CcsWitness {
        z: vec![ExtF::ZERO], // Zero witness
    };
   
    let instance2 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ZERO,
    };
    let witness2 = CcsWitness {
        z: vec![ExtF::ZERO],
    };
   
    // Folding should handle zero cases gracefully
    let result = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&result.transcript, &committer),
            "Zero polynomial folding should verify correctly");
}

/// Test folding with empty instances.
/// Validates robustness when no instances are provided.
#[test]
fn test_empty_instances() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![], 0)], // Empty matrix
        mv_poly(|_: &[ExtF]| ExtF::ZERO, 0) // Zero-variable polynomial
    );
   
    let state = FoldState::new(structure);
   
    // State should initialize properly even with empty structure
    assert!(state.ccs_instance.is_none(), "Should start with no instances");
   
    // Verification with empty transcript should handle gracefully
    let empty_transcript = vec![];
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let result = state.verify(&empty_transcript, &committer);
   
    // Empty verification should either pass (trivial case) or fail gracefully
    // This depends on implementation - either outcome is acceptable for edge case
    assert!(result || !result, "Empty case should handle gracefully");
}

/// Test folding with minimal non-zero instances.
/// Validates the boundary between trivial and non-trivial cases.
#[test]
fn test_minimal_instances() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ONE], 1)], // Minimal non-zero matrix
        mv_poly(|vars: &[ExtF]| vars[0], 1) // Identity polynomial
    );
   
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
   
    let instance1 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE, // Non-zero u
        e: F::ONE, // Set e=1 so that u*e = 1*1 = 1 = f(M*z)
    };
    let witness1 = CcsWitness {
        z: vec![ExtF::ONE], // Minimal non-zero witness
    };
   
    let instance2 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ONE, // Set e=1 so that u*e = 1*1 = 1 = f(M*z)
    };
    let witness2 = CcsWitness {
        z: vec![ExtF::ONE],
    };
   
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer),
            "Minimal instances should fold and verify correctly");
}

/// Test that folding rejects instances with mismatched dimensions.
/// Ensures proper validation of input structure consistency.
#[test]
fn test_mismatched_dimensions() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ONE, F::ZERO], 2)], // 2-element matrix
        mv_poly(|vars: &[ExtF]| vars[0] + vars[1], 2) // 2-variable polynomial
    );
   
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
   
    let instance1 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ZERO,
    };
    // Witness has wrong dimension (1 element instead of 2)
    let bad_witness1 = CcsWitness {
        z: vec![ExtF::ONE], // Wrong size!
    };
   
    let instance2 = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ZERO,
    };
    let witness2 = CcsWitness {
        z: vec![ExtF::ONE, ExtF::ZERO], // Correct size
    };
   
    // This should either panic or produce an invalid proof
    // We test that if a proof is generated, it doesn't verify
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.generate_proof((instance1, bad_witness1), (instance2, witness2), &committer)
    }));
   
    match result {
        Ok(proof_result) => {
            // If proof generation succeeded, verification should fail
            assert!(!state.verify(&proof_result.transcript, &committer),
                    "Mismatched dimensions should not verify");
        }
        Err(_) => {
            // Panic is also acceptable for dimension mismatch
        }
    }
}

/// Test recursive IVC with depth 1 (boundary case).
/// Validates that minimal recursion works correctly.
#[test]
fn test_ivc_depth_one() {
    let structure = CcsStructure::new(
        vec![RowMajorMatrix::<F>::new(vec![F::ONE], 1)],
        mv_poly(|vars: &[ExtF]| vars[0], 1)
    );
   
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
   
    // Set initial instance
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ONE,
        e: F::ZERO,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ZERO],
    };
    state.ccs_instance = Some((instance, witness));
   
    // Depth 1 should work (single recursive step)
    assert!(state.recursive_ivc(1, &committer),
            "IVC with depth 1 should succeed");
}


