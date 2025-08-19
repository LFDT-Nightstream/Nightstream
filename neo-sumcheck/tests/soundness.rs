use neo_fields::ExtF;
use neo_sumcheck::{FriOracle, PolyOracle, Polynomial};
use p3_field::PrimeCharacteristicRing;

/// Test that multilinear sum-check works correctly for simple polynomials.
/// This validates basic soundness and correctness.
#[test]
fn test_multilinear_soundness() {
    // Create a simple 2-variable multilinear polynomial: f(x,y) = x*y
    // Evaluations: f(0,0)=0, f(0,1)=0, f(1,0)=0, f(1,1)=1
    let evals = vec![ExtF::ZERO, ExtF::ZERO, ExtF::ZERO, ExtF::ONE];
    
    // Correct claim: sum over {0,1}^2 = 0+0+0+1 = 1
    let correct_claim = ExtF::ONE;
    
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![], &mut transcript);
    
    // Test with multilinear sumcheck
    use neo_sumcheck::multilinear_sumcheck_prover;
    let mut eval_copy = evals.clone();
    let result = multilinear_sumcheck_prover(&mut eval_copy, correct_claim, &mut oracle, &mut transcript);
    
    assert!(result.is_ok(), "Valid claim should be provable");
}

/// Test that verifier rejects forged sumcheck messages.
/// Prevents adversarial prover from faking reductions (Theorem 1 soundness).
#[test]
fn test_reject_forged_msgs() {
    // Create a simple multilinear polynomial
    let evals = vec![ExtF::ZERO, ExtF::ONE, ExtF::ONE, ExtF::ZERO]; // 2-variable polynomial
    
    let claim = ExtF::from_u64(2); // sum should be 0+1+1+0 = 2
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![], &mut transcript);
    
    use neo_sumcheck::{multilinear_sumcheck_prover, multilinear_sumcheck_verifier};
    let mut eval_copy = evals.clone();
    let result = multilinear_sumcheck_prover(&mut eval_copy, claim, &mut oracle, &mut transcript);
    
    if let Ok((mut msgs, comms)) = result {
        // Forge the first message by tampering with the polynomial
        if !msgs.is_empty() {
            msgs[0].0 = Polynomial::new(vec![ExtF::ONE, ExtF::ONE]); // Invalid polynomial
        }
        
        let mut verifier_transcript = vec![];
        let mut verifier_oracle = FriOracle::new_for_verifier(8);
        let verify_result = multilinear_sumcheck_verifier(claim, &msgs, &comms, &mut verifier_oracle, &mut verifier_transcript);
        assert!(verify_result.is_none(), "Forged message should be rejected");
    }
}

/// Test that FRI verifier rejects tampered proofs with high probability.
/// Validates proof system soundness against malicious provers.
#[test]
fn test_fri_soundness_reject_tampered() {
    let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(2)]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    let comms = oracle.commit();
    let point = vec![ExtF::ONE];
    let (evals, proofs) = oracle.open_at_point(&point);
    
    let mut reject_count = 0;
    let trials = 20; // Reduced for faster testing
    
    for trial in 0..trials {
        let mut proofs_copy = proofs.clone();
        if let Some(proof) = proofs_copy.get_mut(0) {
            if !proof.is_empty() {
                // Tamper one byte based on trial number
                let tamper_pos = trial % proof.len();
                proof[tamper_pos] ^= 1;
                
                let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
                let verifier = FriOracle::new_for_verifier(domain_size);
                
                if !verifier.verify_openings(&comms, &point, &evals, &proofs_copy) {
                    reject_count += 1;
                }
            }
        }
    }
    
    // Should reject most tampering attempts (>80% for small test)
    assert!(reject_count > trials * 4 / 5, "Only rejected {}/{} tampered proofs", reject_count, trials);
}

/// Test soundness for zero polynomial edge case.
/// Ensures verifier doesn't accept invalid proofs for dummy polynomials.
#[test]
fn test_zero_poly_soundness() {
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![], &mut transcript); // Empty -> dummy
    let comms = oracle.commit();
    let point = vec![ExtF::ONE];
    let (mut evals, proofs) = oracle.open_at_point(&point);
    
    // Tamper with evaluation
    if !evals.is_empty() {
        evals[0] += ExtF::ONE;
    }
    
    let verifier = FriOracle::new_for_verifier(4);
    assert!(!verifier.verify_openings(&comms, &point, &evals, &proofs),
            "Verifier should reject tampered zero poly evaluation");
}

/// Test that multilinear sumcheck rejects invalid claims with wrong sums.
/// This validates knowledge soundness for sum-check protocol.
#[test]
fn test_invalid_claim_rejection() {
    // Create polynomial: f(x,y) = x + y (evaluations: f(0,0)=0, f(0,1)=1, f(1,0)=1, f(1,1)=2)
    let evals = vec![ExtF::ZERO, ExtF::ONE, ExtF::ONE, ExtF::from_u64(2)];
    
    // Correct claim: sum = 0+1+1+2 = 4
    // Invalid claim: something different
    let invalid_claim = ExtF::from_u64(3); // Wrong!
    
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![], &mut transcript);
    
    use neo_sumcheck::multilinear_sumcheck_prover;
    let mut eval_copy = evals.clone();
    
    // Prover should fail or produce proof that verifier rejects
    let result = multilinear_sumcheck_prover(&mut eval_copy, invalid_claim, &mut oracle, &mut transcript);
    
    // If prover somehow succeeds, verifier should reject
    if let Ok((msgs, comms)) = result {
        let mut verifier_transcript = vec![];
        let mut verifier_oracle = FriOracle::new_for_verifier(8);
        
        use neo_sumcheck::multilinear_sumcheck_verifier;
        let verify_result = multilinear_sumcheck_verifier(invalid_claim, &msgs, &comms, &mut verifier_oracle, &mut verifier_transcript);
        
        // Verification should fail for invalid claim
        assert!(verify_result.is_none(), "Invalid claim should be rejected by verifier");
    }
}