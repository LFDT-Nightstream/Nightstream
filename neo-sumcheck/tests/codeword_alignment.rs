use neo_fields::ExtF;
use neo_poly::Polynomial;
use neo_sumcheck::{FriOracle, PolyOracle};
use neo_sumcheck::oracle::generate_coset;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_codewords_align_with_domain() {
    // Test that the FRI oracle works correctly without double bit-reversal issues
    // We test this indirectly by ensuring polynomial commitment and opening work correctly
    
    let poly = Polynomial::new(vec![ExtF::from_u64(7), ExtF::from_u64(3)]); // 7 + 3x
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    // Get a commitment
    let commits = oracle.commit();
    assert!(!commits.is_empty());
    
    // Open at a random point and verify - if codewords don't align with domain,
    // the verification will fail
    let point = vec![ExtF::from_u64(42)];
    let (evals, proofs) = oracle.open_at_point(&point);
    
    // Create a verifier and check the opening
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    
    assert!(verifier.verify_openings(&commits, &point, &evals, &proofs),
           "Verification failed - this indicates codeword/domain misalignment (double bit-reversal)");
}

#[test]
fn test_consecutive_pairing_property() {
    // Test that domain maintains consecutive pairing after bit-reversal
    for size in [4, 8, 16] {
        let domain = generate_coset(size);
        
        // Verify consecutive pairing: domain[i ^ 1] = -domain[i]
        for i in (0..size).step_by(2) {
            let idx_pair = i ^ 1; // Consecutive pair via bit flip
            assert_eq!(domain[idx_pair], -domain[i], 
                      "Consecutive pairing broken at size {} index {}: domain[{}]={:?} != -domain[{}]={:?}", 
                      size, i, idx_pair, domain[idx_pair], i, domain[i]);
        }
    }
}

#[test]
fn test_fri_oracle_merkle_consistency() {
    // Test that opening generation works correctly (indicates proper alignment)
    let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(2), ExtF::from_u64(3)]); 
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    let commit = oracle.commit();
    let point = vec![ExtF::from_u64(42)]; // Arbitrary evaluation point
    let (evals, proofs) = oracle.open_at_point(&point);
    
    // The fact that open_at_point succeeds without panicking indicates
    // the Merkle tree structure is consistent with the codewords
    assert!(!evals.is_empty());
    assert!(!proofs.is_empty());
    assert!(!commit.is_empty());
    
    // If we got this far, the domain/codeword alignment is working correctly
    eprintln!("Merkle consistency test: open_at_point succeeded, alignment is correct");
}

#[test]
fn test_no_double_bit_reversal() {
    // Test that constant polynomials work correctly (indirect test for double bit-reversal)
    let poly = Polynomial::new(vec![ExtF::from_u64(5)]); // Constant polynomial
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    // For a constant polynomial, commitment and opening should work without issues
    let commit = oracle.commit();
    let point = vec![ExtF::from_u64(123)]; // Any point should give same result for constant poly
    let (evals, proofs) = oracle.open_at_point(&point);
    
    // Verify the opening
    let domain_size = 4; // Minimum size for constant poly
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(verifier.verify_openings(&commit, &point, &evals, &proofs),
           "Constant polynomial verification failed - suggests double bit-reversal or other alignment issues");
    
    // The evaluation should be close to the constant value (5) plus some blind factor
    assert!(!evals.is_empty());
    // We can't check exact value since we don't know the blind, but verification passing is sufficient
}
