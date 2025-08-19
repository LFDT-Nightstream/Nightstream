use byteorder::{BigEndian, WriteBytesExt};
use std::collections::HashSet;

use neo_sumcheck::oracle::{
    deserialize_fri_proof, extf_pow, generate_coset, hash_extf, serialize_fri_proof,
    verify_merkle_opening, FriLayerQuery, FriOracle, FriProof, FriQuery, MerkleTree, NUM_QUERIES,
};
use neo_sumcheck::{from_base, ExtF, PolyOracle, Polynomial, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;


#[test]
fn test_domain_generation_roots_of_unity() {
    let size = 8;
    let roots = generate_coset(size);
    assert_eq!(roots.len(), size);
    let gen = roots[1] / roots[0];
    assert_eq!(extf_pow(gen, size as u64), ExtF::ONE);
    let mut seen = HashSet::new();
    for &r in &roots {
        assert!(seen.insert(r));
    }
}

#[test]
fn test_fri_coset_domain_secure() {
    let size = 8;
    let roots = generate_coset(size);
    for i in 0..size / 2 {
        assert_eq!(roots[i + size / 2], -roots[i]);
    }
}

#[test]
fn test_fri_verifier_no_secrets() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut prover = FriOracle::new(vec![poly.clone()], &mut transcript);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (evals, proofs) = prover.open_at_point(&point);
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(verifier.verify_openings(&comms, &point, &evals, &proofs));
    let mut reject = 0;
    for _ in 0..100 {
        let mut bad = deserialize_fri_proof(&proofs[0]).unwrap();
        bad.queries[0].f_val += ExtF::ONE;
        let bad_bytes = serialize_fri_proof(&bad);
        if !verifier.verify_openings(&comms, &point, &evals, &[bad_bytes]) {
            reject += 1;
        }
    }
    assert!(reject > 95);
}

#[test]
fn test_low_degree_rejects_non_constant() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut prover = FriOracle::new(vec![poly.clone()], &mut transcript);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (evals, proofs) = prover.open_at_point(&point);
    let mut bad_proof = deserialize_fri_proof(&proofs[0]).unwrap();
    bad_proof.final_eval += ExtF::ONE;
    let bad_bytes = serialize_fri_proof(&bad_proof);
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(!verifier.verify_openings(&comms, &point, &evals, &[bad_bytes]));
}

#[test]
fn test_fri_query_tamper_rejected() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut prover = FriOracle::new(vec![poly.clone()], &mut transcript);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (evals, mut proofs) = prover.open_at_point(&point);
    let mut bad = deserialize_fri_proof(&proofs[0]).unwrap();
    bad.queries[0].layers[0].sib_val += ExtF::ONE;
    proofs[0] = serialize_fri_proof(&bad);
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(!verifier.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_fold_evals() {
    let oracle = FriOracle::new_for_verifier(8);
    let evals = vec![ExtF::ZERO; 8];
    let domain = generate_coset(8);
    let chal = ExtF::ONE;
    let (new_evals, new_domain) = oracle.fold_evals(&evals, &domain, chal);
    assert_eq!(new_evals.len(), 4);
    assert_eq!(new_domain[0], domain[0] * domain[0]);
}

#[test]
fn test_fri_folding_correctness() {
    let oracle = FriOracle::new_for_verifier(4);
    let domain = generate_coset(4);
    let evals = vec![
        ExtF::from_u64(1),
        ExtF::from_u64(3),
        ExtF::from_u64(5),
        ExtF::from_u64(7),
    ];
    let chal = ExtF::from_u64(2);
    let (new_evals, _) = oracle.fold_evals(&evals, &domain, chal);
    let g = domain[0];
    let expected = (evals[0] + evals[2]) / ExtF::from_u64(2)
        + chal * (evals[0] - evals[2]) / (ExtF::from_u64(2) * g);
    assert_eq!(new_evals[0], expected);
}

#[test]
fn test_fri_folding_proximity() {
    let oracle = FriOracle::new_for_verifier(4);
    let domain = generate_coset(4);
    let evals = (0..4).map(|i| ExtF::from_u64(i as u64)).collect::<Vec<_>>();
    let chal = ExtF::from_u64(3);
    let (new_evals, _) = oracle.fold_evals(&evals, &domain, chal);
    let g = domain[0];
    let expected = (evals[0] + evals[2]) / ExtF::from_u64(2)
        + chal * (evals[0] - evals[2]) / (ExtF::from_u64(2) * g);
    assert_eq!(new_evals[0], expected);
}

#[test]
fn test_fri_folding_correct_low_degree() {
    let oracle = FriOracle::new_for_verifier(4);
    let domain = generate_coset(4);
    let evals = domain.iter().map(|&d| d).collect::<Vec<_>>();
    let chal = ExtF::ONE;
    let (new_evals, _) = oracle.fold_evals(&evals, &domain, chal);
    let g = domain[0];
    let expected = (evals[0] + evals[2]) * (ExtF::ONE / ExtF::from_u64(2))
        + chal * (evals[0] - evals[2]) / (ExtF::from_u64(2) * g);
    assert_eq!(new_evals[0], expected);
}

#[test]
fn test_fri_folding_non_vanishing() {
    let mut rng = rand::rng();
    for _ in 0..10 {
        let s0 = from_base(F::from_u64(rng.random()));
        let s1 = from_base(F::from_u64(rng.random()));
        if s0 == s1 {
            continue;
        }
        let chal = from_base(F::from_u64(rng.random()));
        let two_inv = ExtF::ONE / (ExtF::ONE + ExtF::ONE);
        let folded = (s0 + s1 + chal * (s0 - s1)) * two_inv;
        assert_ne!(folded, (s0 + s1) * two_inv);
    }
}

#[test]
fn test_fri_quotient_consistency() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    let comms = oracle.commit();
    let z = ExtF::ONE;
    let p_z = poly.eval(z) + oracle.blinds[0];
    let proof = oracle.generate_fri_proof(0, z, p_z);
    assert!(oracle.verify_fri_proof(&comms[0], z, p_z, &proof));
    let mut bad_proof = proof.clone();
    bad_proof.final_eval += ExtF::ONE;
    assert!(!oracle.verify_fri_proof(&comms[0], z, p_z, &bad_proof));
}

#[test]
fn test_fold_evals_correct() {
    let oracle = FriOracle::new_for_verifier(4);
    let domain = generate_coset(4);
    let evals = vec![
        ExtF::from_u64(1),
        ExtF::from_u64(2),
        ExtF::from_u64(3),
        ExtF::from_u64(4),
    ];
    let challenge = ExtF::from_u64(5);
    let (new_evals, _) = oracle.fold_evals(&evals, &domain, challenge);
    let g = domain[0];
    let expected = (evals[0] + evals[2]) * (ExtF::ONE / ExtF::from_u64(2))
        + challenge * (evals[0] - evals[2]) * (ExtF::ONE / ExtF::from_u64(2)) / g;
    assert_eq!(new_evals[0], expected);
}

#[test]
fn test_quotient_verification() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE, ExtF::from_u64(2)]);
    let mut t = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = oracle.commit();
    let z = ExtF::from_u64(3);
    let p_z = poly.eval(z) + oracle.blinds[0];
    let proof = oracle.generate_fri_proof(0, z, p_z);
    assert!(oracle.verify_fri_proof(&comms[0], z, p_z, &proof));
    let mut bad_proof = proof;
    bad_proof.queries[0].f_val += ExtF::ONE;
    assert!(!oracle.verify_fri_proof(&comms[0], z, p_z, &bad_proof));
}

#[test]
fn test_fri_quotient_full_check() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = oracle.commit();
    let z = ExtF::ONE;
    let p_z = poly.eval(z) + oracle.blinds[0];
    let mut proof = oracle.generate_fri_proof(0, z, p_z);
    proof.queries[0].layers[0].val += ExtF::ONE;
    assert!(!oracle.verify_fri_proof(&comms[0], z, p_z, &proof));
}

#[test]
fn test_blinding_discrete() {
    use std::collections::HashMap;
    let mut rng = ChaCha20Rng::from_seed([0; 32]);
    let mut hist = HashMap::new();
    let q = F::ORDER_U64;
    let half = q / 2;
    for _ in 0..1000 {
        let blind = FriOracle::sample_discrete_gaussian(&mut rng, 3.2);
        let val = blind.to_array()[0].as_canonical_u64();
        let signed = if val > half {
            val as i64 - q as i64
        } else {
            val as i64
        };
        *hist.entry(signed).or_insert(0) += 1;
    }
    let mean = hist.iter().map(|(&k, &v)| k as f64 * v as f64).sum::<f64>() / 1000.0;
    assert!(mean.abs() < 0.5);
}

#[test]
fn test_discrete_gaussian_unbiased() {
    use std::collections::HashMap;
    let mut rng = ChaCha20Rng::from_seed([0; 32]);
    let mut hist = HashMap::new();
    for _ in 0..10000 {
        let sample = FriOracle::sample_discrete_gaussian(&mut rng, 3.2);
        let val = sample.to_array()[0].as_canonical_u64();
        let q = F::ORDER_U64;
        let half = q / 2;
        let z = if val > half {
            val as i64 - q as i64
        } else {
            val as i64
        };
        *hist.entry(z).or_insert(0) += 1;
    }
    let mean: f64 = hist
        .iter()
        .map(|(&k, &v)| k as f64 * v as f64)
        .sum::<f64>()
        / 10000.0;
    assert!(mean.abs() < 0.5); // Unbiased mean ~0
}

#[test]
fn test_deep_binding_correct() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut prover = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (evals, proofs) = prover.open_at_point(&point);
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(verifier.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_deep_binding_rejects_forge() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut prover = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (mut evals, proofs) = prover.open_at_point(&point);
    evals[0] += ExtF::ONE;
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    let mut rejects = 0;
    for _ in 0..100 {
        if !verifier.verify_openings(&comms, &point, &evals, &proofs) {
            rejects += 1;
        }
    }
    assert!(rejects > 95);
}

#[test]
fn test_pairing_verification() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t = vec![];
    let mut prover = FriOracle::new(vec![poly.clone()], &mut t);
    let comms = prover.commit();
    let point = vec![ExtF::ONE];
    let (evals, mut proofs) = prover.open_at_point(&point);
    let mut bad_proof = deserialize_fri_proof(&proofs[0]).unwrap();
    bad_proof.queries[0].layers[0].sib_idx += 1;
    proofs[0] = serialize_fri_proof(&bad_proof);
    let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
    let verifier = FriOracle::new_for_verifier(domain_size);
    assert!(!verifier.verify_openings(&comms, &point, &evals, &proofs));
}

#[test]
fn test_hash_extf_deterministic() {
    let e = ExtF::new_complex(F::from_u64(1), F::from_u64(2));
    let h1 = hash_extf(e);
    let h2 = hash_extf(e);
    assert_eq!(h1, h2);
}

#[test]
fn test_generate_coset_power_of_two() {
    let size = 4;
    let coset = generate_coset(size);
    assert_eq!(coset.len(), size);
    assert_eq!(coset[0] * coset[2], -ExtF::ONE * coset[0] * coset[0]);
}

#[test]
fn test_merkle_roundtrip() {
    let values = vec![
        ExtF::ZERO,
        ExtF::ONE,
        from_base(F::from_u64(2)),
        from_base(F::from_u64(3)),
    ];
    let tree = MerkleTree::new(&values);
    let root = tree.root();
    let index = 1;
    let proof = tree.open(index);
    assert!(verify_merkle_opening(
        &root,
        values[1],
        index,
        &proof,
        tree.leaves
    ));
}

#[test]
fn test_merkle_rejects_tamper() {
    let values = vec![ExtF::ZERO, ExtF::ONE];
    let tree = MerkleTree::new(&values);
    let root = tree.root();
    let index = 0;
    let mut proof = tree.open(index);
    if let Some(p) = proof.get_mut(0) {
        p[0] ^= 1;
    }
    assert!(!verify_merkle_opening(
        &root,
        values[0],
        index,
        &proof,
        tree.leaves
    ));
}

#[test]
fn test_generate_fri_proof_structure() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let oracle = FriOracle::new(vec![poly], &mut transcript);
    let proof = oracle.generate_fri_proof(0, ExtF::ONE, ExtF::ONE);
    assert!(!proof.layer_roots.is_empty());
    assert_eq!(proof.queries.len(), NUM_QUERIES);
}

#[test]
fn test_verify_fri_proof_correct() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    let p_z = poly.eval(ExtF::ONE) + oracle.blinds[0];
    let proof = oracle.generate_fri_proof(0, ExtF::ONE, p_z);
    let commit = oracle.commit();
    let root = &commit[0];
    assert!(oracle.verify_fri_proof(root, ExtF::ONE, p_z, &proof));
}

#[test]
fn test_verify_fri_proof_rejects_invalid() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    let p_z = poly.eval(ExtF::ONE) + oracle.blinds[0];
    let mut proof = oracle.generate_fri_proof(0, ExtF::ONE, p_z);
    proof.final_eval += ExtF::ONE;
    let commit = oracle.commit();
    let root = &commit[0];
    assert!(!oracle.verify_fri_proof(root, ExtF::ONE, p_z, &proof));
}

#[test]
fn test_fri_serialization_roundtrip() {
    let proof = FriProof {
        layer_roots: vec![[1u8; 32], [2u8; 32]],
        queries: vec![FriQuery {
            idx: 0,
            f_val: ExtF::ZERO,
            f_path: vec![[3u8; 32]; 2],
            layers: vec![],
        }],
        final_eval: ExtF::ONE,
        final_pow: 42,
    };
    let bytes = serialize_fri_proof(&proof);
    let de = deserialize_fri_proof(&bytes).unwrap();
    assert_eq!(proof.layer_roots.len(), de.layer_roots.len());
    assert_eq!(proof.final_pow, de.final_pow);
}

#[test]
fn test_fri_serialization_variable_lengths() {
    let proof = FriProof {
        layer_roots: vec![[1u8; 32]],
        queries: vec![FriQuery {
            idx: 0,
            f_val: ExtF::ZERO,
            f_path: vec![[3u8; 32]; 3],
            layers: vec![FriLayerQuery {
                idx: 0,
                sib_idx: 1,
                val: ExtF::ONE,
                sib_val: from_base(F::from_u64(2)),
                path: vec![[4u8; 32]; 2],
                sib_path: vec![[5u8; 32]; 1],
            }],
        }],
        final_eval: ExtF::ONE,
        final_pow: 42,
    };
    let bytes = serialize_fri_proof(&proof);
    let de = deserialize_fri_proof(&bytes).unwrap();
    assert_eq!(proof.queries[0].f_path.len(), de.queries[0].f_path.len());
    assert_eq!(
        proof.queries[0].layers[0].path.len(),
        de.queries[0].layers[0].path.len()
    );
    assert_eq!(
        proof.queries[0].layers[0].sib_path.len(),
        de.queries[0].layers[0].sib_path.len()
    );
}

#[test]
fn test_deserialize_rejects_large() {
    let mut bytes = vec![];
    bytes.write_u32::<BigEndian>(1001).unwrap();
    assert!(deserialize_fri_proof(&bytes).is_err());
}

#[test]
fn test_deserialize_rejects_malformed() {
    let bytes = vec![255u8; 100];
    assert!(deserialize_fri_proof(&bytes).is_err());
}

#[test]
fn test_fri_oracle_trait_impl() {
    let poly = Polynomial::new(vec![ExtF::ZERO]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly], &mut transcript);
    let comms = oracle.commit();
    let point = vec![ExtF::ZERO];
    let (evals, proofs) = oracle.open_at_point(&point);
    assert!(oracle.verify_openings(&comms, &point, &evals, &proofs));
}



#[test]
fn test_blinding_seed_full_entropy() {
    let poly = Polynomial::new(vec![ExtF::ZERO]);
    let mut t1 = b"seed1".to_vec();
    let o1 = FriOracle::new(vec![poly.clone()], &mut t1);
    let mut t2 = b"seed1".to_vec();
    let o2 = FriOracle::new(vec![poly.clone()], &mut t2);
    assert_eq!(o1.blinds, o2.blinds);
    let mut t3 = b"seed2".to_vec();
    let o3 = FriOracle::new(vec![poly], &mut t3);
    assert_ne!(o1.blinds, o3.blinds);
}

#[test]
fn test_fri_proof_serde_roundtrip_complex() {
    let proof = FriProof {
        layer_roots: vec![[0u8; 32], [255u8; 32]],
        queries: vec![FriQuery {
            idx: 42,
            f_val: ExtF::new_complex(F::from_u64(1), F::from_u64(2)),
            f_path: vec![[3u8; 32], [4u8; 32]],
            layers: vec![FriLayerQuery {
                idx: 5,
                sib_idx: 6,
                val: ExtF::ONE,
                sib_val: ExtF::from_u64(2),
                path: vec![[7u8; 32]],
                sib_path: vec![[8u8; 32]],
            }],
        }],
        final_eval: ExtF::new_complex(F::from_u64(3), F::from_u64(4)),
        final_pow: 42,
    };
    let bytes = serialize_fri_proof(&proof);
    let de = deserialize_fri_proof(&bytes).unwrap();
    assert_eq!(proof, de);
}


