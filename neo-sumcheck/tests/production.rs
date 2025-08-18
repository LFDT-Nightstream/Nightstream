use neo_fields::{ExtF, from_base, F};
use neo_poly::Polynomial;
use neo_sumcheck::oracle::{FriOracle, generate_coset};
use neo_sumcheck::PolyOracle;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_fri_prod_with_pow() {
    let poly = Polynomial::new(vec![ExtF::ONE]);
    let mut t = vec![];
    let mut oracle = FriOracle::new(vec![poly], &mut t);
    let commit = oracle.commit()[0].clone();
    let z = ExtF::ONE;
    let p_z = oracle.blinds[0] + ExtF::ONE;
    let proof = oracle.generate_fri_proof(0, z, p_z);
    assert!(oracle.verify_fri_proof(&commit, z, p_z, &proof));
    assert_ne!(proof.final_pow, 0, "PoW nonce should be non-zero");
}

#[test]
fn test_fri_prod_blinding() {
    let poly = Polynomial::new(vec![ExtF::ZERO]);
    let mut t1 = b"seed1".to_vec();
    let oracle1 = FriOracle::new(vec![poly.clone()], &mut t1);
    let mut t2 = b"seed2".to_vec();
    let oracle2 = FriOracle::new(vec![poly], &mut t2);
    assert_ne!(oracle1.blinds, oracle2.blinds, "Blinds should differ with different seeds");
}

#[test]
fn test_fri_prod_folding() {
    let size = 4;
    let evals = (0..size).map(|i| from_base(F::from_u64(i as u64))).collect::<Vec<_>>();
    let domain = generate_coset(size);
    let challenge = from_base(F::from_u64(5));
    let oracle = FriOracle::new_for_verifier(size);
    let (new_evals, _new_domain) = oracle.fold_evals(&evals, &domain, challenge);
    assert_eq!(new_evals.len(), 2);
    let two_inv = ExtF::ONE / from_base(F::from_u64(2));
    // Use first half vs second half pairing to match our implementation
    let half = size / 2;
    let expected0 = (evals[0] + evals[half]) * two_inv + challenge * (evals[0] - evals[half]) * two_inv / domain[0];
    assert_eq!(new_evals[0], expected0);
}

#[test]
fn test_fri_prod_eval_consistency() {
    // Test that blinded evaluations are consistent
    let poly = Polynomial::new(vec![ExtF::ONE, from_base(F::from_u64(2))]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    let point = vec![from_base(F::from_u64(3))];
    let (evals, proofs) = oracle.open_at_point(&point);
    let commit = oracle.commit();
    
    // The opened eval should be poly.eval(point) + blind
    let expected_unblinded = poly.eval(point[0]);
    let actual_unblinded = evals[0] - oracle.blinds[0];
    println!("Poly evaluation: {:?}", expected_unblinded);
    println!("Blinded eval: {:?}", evals[0]);
    println!("Blind factor: {:?}", oracle.blinds[0]);
    println!("Unblinded eval: {:?}", actual_unblinded);
    assert_eq!(actual_unblinded, expected_unblinded, "Unblinded eval should match polynomial evaluation");
    
    // Verification should pass with the blinded evaluation
    let domain_size = 8; // Size should match polynomial degree + blowup
    let verifier = FriOracle::new_for_verifier(domain_size);
    println!("Testing verification with blinded eval: {:?}", evals[0]);
    assert!(verifier.verify_openings(&commit, &point, &evals, &proofs));
}

#[test]  
fn test_fri_prod_domain_pairing() {
    // Test that domain maintains proper pairing relationships
    for size in [4, 8, 16] {
        let domain = generate_coset(size);
        for i in 0..size/2 {
            assert_eq!(domain[i + size/2], -domain[i], 
                      "Domain pairing broken at size {} index {}", size, i);
        }
    }
}
