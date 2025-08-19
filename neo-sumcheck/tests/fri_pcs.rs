use neo_sumcheck::{from_base, ExtF, F, FriOracle, PolyOracle, Polynomial};
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_fri_roundtrip() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = oracle.commit();
    assert_eq!(comms.len(), 1);
    let point = vec![from_base(F::from_u64(5))];
    let (evals, proofs) = oracle.open_at_point(&point);
    assert_eq!(evals.len(), 1);
    assert!(oracle.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_fri_tampered_proof_rejected() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = oracle.commit();
    let point = vec![from_base(F::from_u64(2))];
    let (evals, mut proofs) = oracle.open_at_point(&point);
    if let Some(p) = proofs.get_mut(0) {
        if !p.is_empty() {
            p[0] ^= 1;
        }
    }
    assert!(!oracle.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_fri_zk_hides_poly() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t1 = b"seed1".to_vec();
    let mut oracle1 = FriOracle::new(vec![poly.clone()], &mut t1);
    let comm1 = oracle1.commit();
    let mut t2 = b"seed2".to_vec();
    let mut oracle2 = FriOracle::new(vec![poly.clone()], &mut t2);
    let comm2 = oracle2.commit();
    assert_ne!(comm1, comm2);
}

#[test]
fn test_fri_batch_multi_poly() {
    let p1 = Polynomial::new(vec![ExtF::ONE]);
    let p2 = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut oracle = FriOracle::new(vec![p1, p2], &mut t);
    let comms = oracle.commit();
    assert_eq!(comms.len(), 2);
    let point = vec![from_base(F::from_u64(4))];
    let (evals, proofs) = oracle.open_at_point(&point);
    assert_eq!(evals.len(), 2);
    assert!(oracle.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_fri_rejects_high_degree() {
    let mut coeffs = vec![ExtF::ZERO; 5];
    coeffs.push(ExtF::ONE); // degree 5
    let high_deg_poly = Polynomial::new(coeffs);
    let mut t = vec![];
    let mut prover = FriOracle::new(vec![high_deg_poly], &mut t);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (evals, proofs) = prover.open_at_point(&point);
    // Verifier assumes smaller domain (degree 1)
    let domain_size = (1usize + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(!verifier.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_fri_blinding_deterministic() {
    let poly = Polynomial::new(vec![ExtF::ZERO]);
    let mut t1 = b"seed1".to_vec();
    let oracle1 = FriOracle::new(vec![poly.clone()], &mut t1);
    let mut t2 = b"seed1".to_vec();
    let oracle2 = FriOracle::new(vec![poly.clone()], &mut t2);
    assert_eq!(oracle1.blinds, oracle2.blinds);
    let mut t3 = b"seed2".to_vec();
    let oracle3 = FriOracle::new(vec![poly], &mut t3);
    assert_ne!(oracle1.blinds, oracle3.blinds);
}


