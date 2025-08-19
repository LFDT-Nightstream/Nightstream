use neo_fields::ExtF;
use neo_poly::Polynomial;
use neo_sumcheck::oracle::FriOracle;
use neo_sumcheck::PolyOracle;
use p3_field::PrimeCharacteristicRing;
use quickcheck_macros::quickcheck;

/// Test that blinding hides polynomial evaluations.
/// Different transcripts should produce different commitments for the same polynomial.
/// This validates zero-knowledge property (paper §8 ZK hiding).
#[test]
fn test_blinding_hides_poly() {
    let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(2)]);
    
    // Same polynomial, different transcript seeds should give different commitments
    let mut transcript1 = b"seed_alpha".to_vec();
    let mut oracle1 = FriOracle::new(vec![poly.clone()], &mut transcript1);
    let commit1 = oracle1.commit();
    
    let mut transcript2 = b"seed_beta".to_vec();
    let mut oracle2 = FriOracle::new(vec![poly.clone()], &mut transcript2);
    let commit2 = oracle2.commit();
    
    assert_ne!(commit1, commit2, "Different transcripts should produce different commitments");
    assert_ne!(oracle1.blinds, oracle2.blinds, "Blinds should differ with different seeds");
}

/// Test that same transcript produces deterministic commitments.
/// This ensures the blinding is deterministic given the same randomness source.
#[test]
fn test_blinding_deterministic_same_seed() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    
    let mut transcript1 = b"fixed_seed".to_vec();
    let mut oracle1 = FriOracle::new(vec![poly.clone()], &mut transcript1);
    let commit1 = oracle1.commit();
    
    let mut transcript2 = b"fixed_seed".to_vec();
    let mut oracle2 = FriOracle::new(vec![poly.clone()], &mut transcript2);
    let commit2 = oracle2.commit();
    
    assert_eq!(commit1, commit2, "Same transcript should produce same commitment");
    assert_eq!(oracle1.blinds, oracle2.blinds, "Same transcript should produce same blinds");
}

/// Test that blinded openings verify correctly while hiding the original polynomial.
/// This validates that ZK doesn't break correctness.
#[test]
fn test_blinded_opening_correctness() {
    let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(3)]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    let point = vec![ExtF::from_u64(2)];
    let (evals, proofs) = oracle.open_at_point(&point);
    let commit = oracle.commit();
    
    // Opened evaluation should include blind
    let expected_raw = poly.eval(point[0]);
    let expected_blinded = expected_raw + oracle.blinds[0];
    assert_eq!(evals[0], expected_blinded, "Opened eval should be blinded");
    
    // Verification should pass with blinded evaluation
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(verifier.verify_openings(&commit, &point, &evals, &proofs),
            "Blinded opening should verify correctly");
    
    // Direct polynomial evaluation should be hidden (different from opened value)
    assert_ne!(evals[0], expected_raw, "ZK should hide the raw polynomial evaluation");
}

/// Property-based test: multiple polynomials with same coefficients but different blinds.
/// Tests that batch commitments maintain ZK properties.
#[quickcheck]
fn prop_batch_blinding_independence(coeff: u64) -> bool {
    let poly1 = Polynomial::new(vec![ExtF::from_u64(coeff), ExtF::ONE]);
    let poly2 = Polynomial::new(vec![ExtF::from_u64(coeff), ExtF::ONE]); // Same coefficients
    
    let mut transcript1 = b"batch_seed_1".to_vec();
    let mut oracle1 = FriOracle::new(vec![poly1, poly2.clone()], &mut transcript1);
    let commit1 = oracle1.commit();
    
    let mut transcript2 = b"batch_seed_2".to_vec();
    let mut oracle2 = FriOracle::new(vec![poly2], &mut transcript2);
    let commit2 = oracle2.commit();
    
    // Even with same polynomials, different batching/seeds should give different commits
    commit1 != commit2
}

/// Test that ZK holds for edge cases like zero polynomials.
/// Ensures blinding works even for trivial inputs.
#[test]
fn test_zero_poly_zk() {
    let zero_poly = Polynomial::new(vec![ExtF::ZERO]);
    
    let mut transcript1 = b"zero_test_1".to_vec();
    let mut oracle1 = FriOracle::new(vec![zero_poly.clone()], &mut transcript1);
    let commit1 = oracle1.commit();
    
    let mut transcript2 = b"zero_test_2".to_vec();
    let mut oracle2 = FriOracle::new(vec![zero_poly], &mut transcript2);
    let commit2 = oracle2.commit();
    
    assert_ne!(commit1, commit2, "Zero polynomial should still have ZK hiding");
    
    // Blinds should be non-zero to provide hiding
    assert_ne!(oracle1.blinds[0], ExtF::ZERO, "Blind should be non-zero for hiding");
    assert_ne!(oracle2.blinds[0], ExtF::ZERO, "Blind should be non-zero for hiding");
}

/// Test computational hiding: many samples from same polynomial should not reveal it.
/// Statistical test for hiding property across multiple commitments.
#[test]
fn test_computational_hiding() {
    let secret_poly = Polynomial::new(vec![ExtF::from_u64(42), ExtF::from_u64(17)]);
    let mut commitments = Vec::new();
    
    // Generate many commitments to the same polynomial
    for i in 0..20 {
        let mut transcript = format!("hiding_test_{}", i).into_bytes();
        let mut oracle = FriOracle::new(vec![secret_poly.clone()], &mut transcript);
        commitments.push(oracle.commit());
    }
    
    // All commitments should be different (hiding property)
    for i in 0..commitments.len() {
        for j in i+1..commitments.len() {
            assert_ne!(commitments[i], commitments[j], 
                      "Commitments {} and {} should differ", i, j);
        }
    }
}
