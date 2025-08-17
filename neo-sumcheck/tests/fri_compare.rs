use neo_sumcheck::{ExtF, FriOracle, PolyOracle, Polynomial, from_base};
use neo_fields::{random_extf, random_extf_with_flag};
use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_field::extension::BinomialExtensionField;
use core::mem::transmute;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use neo_sumcheck::oracle::{deserialize_fri_proof, serialize_fri_proof};
use rand::{rng, Rng};

type Val = Goldilocks;
type Challenge = Val;
type Perm = Poseidon2Goldilocks<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type FriPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Challenge2 = BinomialExtensionField<Val, 2>;
type ChallengeMmcs2 = ExtensionMmcs<Val, Challenge2, ValMmcs>;
type FriPcs2 = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs2>;

fn p3_pcs() -> (FriPcs, Challenger) {
    let mut rng = rng();
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 40,
        proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = FriPcs::new(Dft::default(), val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    (pcs, challenger)
}

fn p3_pcs_ext() -> (FriPcs2, Challenger) {
    let mut rng = rng();
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs2::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 40,
        proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = FriPcs2::new(Dft::default(), val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    (pcs, challenger)
}

fn to_p3(e: ExtF) -> Val {
    let [r, _] = e.to_array();
    r
}

fn challenge2_from_array(arr: [Val; 2]) -> Challenge2 {
    unsafe { transmute::<[Val; 2], Challenge2>(arr) }
}

fn challenge2_to_array(c: Challenge2) -> [Val; 2] {
    unsafe { transmute::<Challenge2, [Val; 2]>(c) }
}

fn to_p3_ext(e: ExtF) -> Challenge2 {
    challenge2_from_array(e.to_array())
}


#[test]
fn test_fri_equivalence() {
    let mut rng = rng();
    let degree = 7;
    // Use real_only=true to match p3-fri base field
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf_with_flag(&mut rng, true)).collect();
    let poly = Polynomial::new(coeffs.clone());

    // Use blowup=2 to match p3-fri's log_blowup=1 (2^1=2)
    let mut our_oracle = FriOracle::new_with_blinds_and_blowup(vec![poly.clone()], vec![ExtF::ZERO], 2);
    let _our_comms = our_oracle.commit();
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    let (our_evals, _our_proofs) = our_oracle.open_at_point(&point);

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Val> = domain.clone().into_iter().map(|x| {
        let x_ext = from_base(x);
        to_p3(poly.eval(x_ext))
    }).collect();
    let matrix = RowMajorMatrix::new(p3_vals.clone(), 1);
    let (p3_comm, prover_data) = <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&prover_data, vec![vec![p3_point]])], &mut p_challenger);
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![(p3_comm, vec![(domain, vec![(p3_point, opened[0][0][0].clone())])])];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_challenger).is_ok());

    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3(our_evals[0]), p3_eval, "Eval mismatch");
}

#[test]
fn test_fri_equivalence_constant() {
    use neo_fields::F;
    
    let mut rng = rng();

    let constant_val = from_base(F::from_u64(5));
    let poly = Polynomial::new(vec![constant_val]);

    // Neo FRI
    let mut t = vec![];
    // Use blowup=2 to match p3-fri's log_blowup=1 (2^1=2)
    let mut neo_oracle = FriOracle::new_with_blowup(vec![poly.clone()], &mut t, 2);
    let neo_comms = neo_oracle.commit();
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));
    assert_eq!(neo_evals[0], constant_val + neo_oracle.blinds[0]);

    // p3-fri
    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1);
    let p3_vals: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly.eval(from_base(x))))
        .collect();
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![p3_point]])],
        &mut p_challenger,
    );
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![(domain, vec![(p3_point, opened[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3(neo_evals[0] - neo_oracle.blinds[0]), p3_eval);
}

#[test]
#[ignore]
fn test_fri_equivalence_linear() {
    let mut rng = rng();
    let coeffs: Vec<ExtF> = (0..=1).map(|_| random_extf(&mut rng)).collect();
    let poly = Polynomial::new(coeffs.clone());

    let mut t = vec![];
    let mut neo_oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let neo_comms = neo_oracle.commit();
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 2);
    let p3_vals: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly.eval(from_base(x))))
        .collect();
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![p3_point]])],
        &mut p_challenger,
    );
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![(domain, vec![(Val::ZERO, opened[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3(neo_evals[0] - neo_oracle.blinds[0]), p3_eval);
}

#[test]
#[ignore]
fn test_fri_equivalence_large() {
    let mut rng = rng();
    let degree = 1024;
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf(&mut rng)).collect();
    let poly = Polynomial::new(coeffs.clone());

    let mut t = vec![];
    let mut neo_oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let neo_comms = neo_oracle.commit();
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));

    let (pcs, challenger0) = p3_pcs();
    let domain =
        <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly.eval(from_base(x))))
        .collect();
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![p3_point]])],
        &mut p_challenger,
    );
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![(domain, vec![(Val::ZERO, opened[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3(neo_evals[0] - neo_oracle.blinds[0]), p3_eval);
}

#[test]
#[ignore]
fn test_fri_equivalence_custom_config() {
    let mut rng = rng();
    let degree = 16;
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf(&mut rng)).collect();
    let poly = Polynomial::new(coeffs.clone());

    let mut t = vec![];
    let mut neo_oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let neo_comms = neo_oracle.commit();
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));

    // p3-fri with custom parameters
    let mut rng2 = rand::rng();
    let perm = Perm::new_from_rng_128(&mut rng2);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 50,
        proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = FriPcs::new(Dft::default(), val_mmcs, fri_params);
    let domain =
        <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly.eval(from_base(x))))
        .collect();
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = Challenger::new(perm.clone());
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![p3_point]])],
        &mut p_challenger,
    );
    let mut v_challenger = Challenger::new(perm);
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![(domain, vec![(p3_point, opened[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3(neo_evals[0] - neo_oracle.blinds[0]), p3_eval);
}

#[test]
#[ignore]
fn test_fri_equivalence_tampered_proof() {
    let mut rng = rng();
    let degree = 8;
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf(&mut rng)).collect();
    let poly = Polynomial::new(coeffs.clone());

    // Neo FRI tampering
    let mut t = vec![];
    let mut neo_oracle = FriOracle::new(vec![poly.clone()], &mut t);
    let neo_comms = neo_oracle.commit();
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    let (mut neo_evals, mut neo_proofs) = neo_oracle.open_at_point(&point);
    let original_evals = neo_evals.clone();
    let original_proofs = neo_proofs.clone();
    // corrupt eval
    neo_evals[0] += ExtF::ONE;
    assert!(!neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));
    neo_evals = original_evals.clone();
    // corrupt proof
    neo_proofs[0][0] ^= 1;
    assert!(!neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));
    neo_proofs = original_proofs;
    // wrong commitment
    let mut bad_comm = neo_comms.clone();
    bad_comm[0][0] ^= 1;
    assert!(!neo_oracle.verify_openings(&bad_comm, &point, &neo_evals, &neo_proofs));
    // mismatched point
    let bad_point = vec![point[0] + ExtF::ONE];
    assert!(!neo_oracle.verify_openings(&neo_comms, &bad_point, &neo_evals, &neo_proofs));

    // p3-fri tampering
    let (pcs, challenger0) = p3_pcs();
    let domain =
        <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly.eval(from_base(x))))
        .collect();
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened_orig, proof_orig) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![p3_point]])],
        &mut p_challenger,
    );
    // corrupt eval
    let mut opened = opened_orig.clone();
    let mut proof = proof_orig.clone();
    opened[0][0][0][0] += Val::ONE;
    let mut v_challenger = challenger0.clone();
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm.clone(),
            vec![(domain.clone(), vec![(p3_point, opened[0][0][0].clone())])],
        ),
    ];
    assert!(
        <FriPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_challenger)
            .is_err()
    );
    // corrupt proof
    opened = opened_orig.clone();
    proof = proof_orig.clone();
    proof.final_poly[0] += Val::ONE;
    let mut v_challenger2 = challenger0.clone();
    v_challenger2.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm.clone(),
            vec![(domain.clone(), vec![(p3_point, opened[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger2,
    )
    .is_err());
    // wrong commitment
    let mut v_challenger3 = challenger0.clone();
    use p3_symmetric::Hash;
    let mut arr: [Val; 8] = p3_comm.into();
    arr[0] += Val::ONE;
    let bad_comm: Hash<Val, Val, 8> = arr.into();
    v_challenger3.observe(bad_comm.clone());
    let claims = vec![
        (
            bad_comm.clone(),
            vec![(domain.clone(), vec![(p3_point, opened_orig[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof_orig,
        &mut v_challenger3,
    )
    .is_err());
    // mismatched point
    let mut v_challenger4 = challenger0;
    v_challenger4.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![
                (
                    domain,
                    vec![(p3_point + Val::ONE, opened_orig[0][0][0].clone())],
                ),
            ],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof_orig,
        &mut v_challenger4,
    )
    .is_err());
}

#[test]
#[ignore]
fn test_fri_equivalence_multi_poly() {
    let mut rng = rng();
    let degree = 4;
    // Use real_only=true to match p3-fri base field
    let poly1 = Polynomial::new((0..=degree).map(|_| random_extf_with_flag(&mut rng, true)).collect());
    let poly2 = Polynomial::new((0..=degree).map(|_| random_extf_with_flag(&mut rng, true)).collect());

    // Use blowup=2 to match p3-fri's log_blowup=1 (2^1=2) and zero blinds like working tests
    let mut neo_oracle = FriOracle::new_with_blinds_and_blowup(vec![poly1.clone(), poly2.clone()], vec![ExtF::ZERO, ExtF::ZERO], 2);
    println!("Poly1 degree: {}, Poly2 degree: {}", poly1.degree(), poly2.degree());
    println!("Blinds: {:?}", neo_oracle.blinds);
    
    let neo_comms = neo_oracle.commit();
    println!("Commitments: {} bytes each", neo_comms[0].len());
    
    let p3_point = rng.random::<Val>();
    let point = vec![from_base(p3_point)];
    println!("Opening at point: {:?}", point[0]);
    
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    println!("Neo evals: {:?}", neo_evals);
    println!("Proof sizes: {:?}", neo_proofs.iter().map(|p| p.len()).collect::<Vec<_>>());
    
    let verification_result = neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs);
    println!("Verification result: {}", verification_result);
    assert!(verification_result);

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals1: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly1.eval(from_base(x))))
        .collect();
    let p3_vals2: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly2.eval(from_base(x))))
        .collect();
    let matrix1 = RowMajorMatrix::new(p3_vals1, 1);
    let matrix2 = RowMajorMatrix::new(p3_vals2, 1);
    let inputs = vec![(domain.clone(), matrix1), (domain.clone(), matrix2)];
    let (p3_comm, prover_data) = <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, inputs);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let z = p3_point;
    let open_points = vec![vec![z], vec![z]];
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, open_points)],
        &mut p_challenger,
    );
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![
                (domain.clone(), vec![(p3_point, opened[0][0][0].clone())]),
                (domain.clone(), vec![(p3_point, opened[0][1][0].clone())]),
            ],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
    let p3_eval1 = opened[0][0][0][0];
    let p3_eval2 = opened[0][1][0][0];
    assert_eq!(to_p3(neo_evals[0] - neo_oracle.blinds[0]), p3_eval1);
    assert_eq!(to_p3(neo_evals[1] - neo_oracle.blinds[1]), p3_eval2);
}

#[test]
fn test_fri_equivalence_zero_poly() {
    let poly = Polynomial::new(vec![ExtF::ZERO]);

    let mut t = vec![];
    // Use blowup=2 to match p3-fri's log_blowup=1 (2^1=2)
    let mut neo_oracle = FriOracle::new_with_blowup(vec![poly.clone()], &mut t, 2);
    let neo_comms = neo_oracle.commit();
    let point = vec![ExtF::ZERO];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));
    assert_eq!(neo_evals[0], neo_oracle.blinds[0]);

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1);
    let p3_vals: Vec<Val> = domain
        .clone()
        .into_iter()
        .map(|x| to_p3(poly.eval(from_base(x))))
        .collect();
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![Val::ZERO]])],
        &mut p_challenger,
    );
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![
        (
            p3_comm,
            vec![(domain, vec![(Val::ZERO, opened[0][0][0].clone())])],
        ),
    ];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3(neo_evals[0] - neo_oracle.blinds[0]), p3_eval);
}

#[test]
fn test_fri_equivalence_high_norm_reject() {
    use neo_fields::{F, MAX_BLIND_NORM};
    let high = ExtF::new_complex(F::from_u64(MAX_BLIND_NORM + 1), F::ZERO);
    let poly = Polynomial::new(vec![high]);
    let mut neo_oracle = FriOracle::new_with_blinds(vec![poly.clone()], vec![ExtF::ZERO]);
    let neo_comms = neo_oracle.commit();
    let point = vec![ExtF::ONE];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(!neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1);
    let p3_high = to_p3(high);
    let p3_vals: Vec<Val> = vec![p3_high; domain.size()];
    let matrix = RowMajorMatrix::new(p3_vals, 1);
    let (p3_comm, prover_data) =
        <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let (opened, proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, vec![vec![Val::ONE]])],
        &mut p_challenger,
    );
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![(p3_comm, vec![(domain, vec![(Val::ONE, opened[0][0][0].clone())])])];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        claims,
        &proof,
        &mut v_challenger,
    )
    .is_ok());
}

#[test]
fn test_fri_blinding_variability() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut t1 = b"seed1".to_vec();
    let mut oracle1 = FriOracle::new(vec![poly.clone()], &mut t1);
    let mut t2 = b"seed2".to_vec();
    let mut oracle2 = FriOracle::new(vec![poly.clone()], &mut t2);
    let comm1 = oracle1.commit();
    let comm2 = oracle2.commit();
    assert_ne!(comm1, comm2);
    assert_ne!(oracle1.blinds, oracle2.blinds);

    let point = vec![ExtF::ONE];
    let (evals1, proofs1) = oracle1.open_at_point(&point);
    let (evals2, proofs2) = oracle2.open_at_point(&point);
    assert!(oracle1.verify_openings(&comm1, &point, &evals1, &proofs1));
    assert!(oracle2.verify_openings(&comm2, &point, &evals2, &proofs2));
    assert_eq!(evals1[0] - oracle1.blinds[0], evals2[0] - oracle2.blinds[0]);
}

#[test]
fn test_fri_equivalence_zk_blinding() {
    use std::collections::HashSet;
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE, ExtF::from_u64(2)]);
    let point = vec![ExtF::ONE];
    let num_runs = 50;
    let mut commits = vec![];
    let mut unblinded_evals = vec![];
    for i in 0u64..num_runs {
        let mut t = i.to_be_bytes().to_vec();
        let mut oracle = FriOracle::new(vec![poly.clone()], &mut t);
        let comm = oracle.commit();
        commits.push(comm[0].clone());
        let (evals, proofs) = oracle.open_at_point(&point);
        assert!(oracle.verify_openings(&comm, &point, &evals, &proofs));
        unblinded_evals.push(evals[0] - oracle.blinds[0]);
    }
    let unique = commits.iter().collect::<HashSet<_>>().len();
    assert!(unique as f64 / num_runs as f64 > 0.9, "Poor hiding: only {} unique commits", unique);
    let first = unblinded_evals[0];
    assert!(unblinded_evals.iter().all(|&e| e == first), "Unblinded mismatch");
}

#[test]
fn test_fri_equivalence_serialization_tamper() {
    use bincode;
    let mut rng = rng();
    let degree = 7;
    // Use real_only=true to match p3-fri base field
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf_with_flag(&mut rng, true)).collect();
    let poly = Polynomial::new(coeffs.clone());
    let blind = random_extf_with_flag(&mut rng, true);
    // Use blowup=2 to match p3-fri's log_blowup=1 (2^1=2)
    let mut neo_oracle = FriOracle::new_with_blinds_and_blowup(vec![poly.clone()], vec![blind], 2);
    let _neo_comm = neo_oracle.commit();
    let z = random_extf_with_flag(&mut rng, true);
    let p_z = poly.eval(z) + blind;
    let proof = neo_oracle.generate_fri_proof(0, z, p_z);
    let bytes = serialize_fri_proof(&proof);
    let de_proof = deserialize_fri_proof(&bytes).unwrap();
    assert_eq!(proof, de_proof, "Neo serialization mismatch");
    let mut bad_bytes = bytes.clone();
    if !bad_bytes.is_empty() { bad_bytes[0] ^= 1; }
    assert!(deserialize_fri_proof(&bad_bytes).is_err(), "Neo accepts tampered serialization");

    let (pcs, challenger0) = p3_pcs();
    let domain = <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Val> = domain.clone().into_iter().map(|x| {
        let x_ext = from_base(x);
        to_p3(poly.eval(x_ext))
    }).collect();
    let matrix = RowMajorMatrix::new(p3_vals.clone(), 1);
    let (p3_comm, prover_data) = <FriPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let p3_point = rng.random::<Val>();
    let (opened, p3_proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&prover_data, vec![vec![p3_point]])], &mut p_challenger);
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![(p3_comm, vec![(domain, vec![(p3_point, opened[0][0][0].clone())])])];
    assert!(<FriPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &p3_proof, &mut v_challenger).is_ok());
    let p3_proof_bytes = bincode::serialize(&p3_proof).unwrap();
    let _: <FriPcs as Pcs<Challenge, Challenger>>::Proof = bincode::deserialize(&p3_proof_bytes).unwrap();
    let mut bad_p3_bytes = p3_proof_bytes.clone();
    if !bad_p3_bytes.is_empty() { bad_p3_bytes[0] ^= 1; }
    assert!(bincode::deserialize::< <FriPcs as Pcs<Challenge, Challenger>>::Proof >(&bad_p3_bytes).is_err(),
        "p3 accepts tampered");
}

#[test]
#[ignore]
fn test_fri_equivalence_extension_field() {
    let mut rng = rng();
    let degree = 8;
    let coeffs: Vec<ExtF> = (0..=degree).map(|_| random_extf(&mut rng)).collect();
    let poly = Polynomial::new(coeffs.clone());
    let blind = random_extf(&mut rng);
    let mut neo_oracle = FriOracle::new_with_blinds(vec![poly.clone()], vec![blind]);
    let neo_comms = neo_oracle.commit();
    let point = vec![random_extf(&mut rng)];
    let (neo_evals, neo_proofs) = neo_oracle.open_at_point(&point);
    assert!(neo_oracle.verify_openings(&neo_comms, &point, &neo_evals, &neo_proofs));

    let (pcs, challenger0) = p3_pcs_ext();
    let domain = <FriPcs2 as Pcs<Challenge2, Challenger>>::natural_domain_for_degree(&pcs, degree + 1);
    let p3_vals: Vec<Challenge2> = domain.clone().into_iter().map(|x| {
        let x_ext = from_base(x);
        to_p3_ext(poly.eval(x_ext))
    }).collect();
    let width = p3_vals.len();
    let flattened: Vec<Val> = p3_vals
        .iter()
        .flat_map(|&c| challenge2_to_array(c))
        .collect();
    let matrix = RowMajorMatrix::new(flattened, width);
    let (p3_comm, prover_data) = <FriPcs2 as Pcs<Challenge2, Challenger>>::commit(&pcs, [(domain.clone(), matrix)]);
    let mut p_challenger = challenger0.clone();
    p_challenger.observe(p3_comm.clone());
    let p3_point = to_p3_ext(point[0]);
    let (opened, proof) = <FriPcs2 as Pcs<Challenge2, Challenger>>::open(&pcs, vec![(&prover_data, vec![vec![p3_point]])], &mut p_challenger);
    let mut v_challenger = challenger0;
    v_challenger.observe(p3_comm.clone());
    let claims = vec![(p3_comm, vec![(domain, vec![(p3_point, opened[0][0][0].clone())])])];
    assert!(<FriPcs2 as Pcs<Challenge2, Challenger>>::verify(&pcs, claims, &proof, &mut v_challenger).is_ok());
    let p3_eval = opened[0][0][0][0];
    assert_eq!(to_p3_ext(neo_evals[0] - blind), p3_eval);
}
