use neo_sumcheck::{
    batched_sumcheck_prover, batched_sumcheck_verifier, multilinear_sumcheck_prover,
    batched_multilinear_sumcheck_prover, fiat_shamir_challenge,
    ExtF, FnOracle, FriOracle, UnivPoly, Commitment, OpeningProof, PolyOracle, from_base, F
};
use p3_field::PrimeCharacteristicRing;
use neo_poly::Polynomial;
use quickcheck_macros::quickcheck;
use rand::{rng, Rng};
use neo_fields::MAX_BLIND_NORM;

struct TestPoly;
impl UnivPoly for TestPoly {
    fn evaluate(&self, _point: &[ExtF]) -> ExtF { ExtF::ZERO }
    fn degree(&self) -> usize { 3 }
    fn max_individual_degree(&self) -> usize { 1 }
}

#[test]
fn test_batched_sumcheck_no_clone_panic() {
    let poly: Box<dyn UnivPoly> = Box::new(TestPoly);
    let mut oracle = FnOracle::new(|_| vec![]);
    let mut transcript = vec![];
    let result = batched_sumcheck_prover(&[ExtF::ZERO], &[&*poly], &mut oracle, &mut transcript);
    assert!(result.is_ok());
}

#[test]
fn test_fiat_shamir_challenge_ext_field() {
    // Determinism: same input yields same output; domain separation yields different.
    let ch1 = fiat_shamir_challenge(b"12345678");
    let ch2 = fiat_shamir_challenge(b"12345678");
    assert_eq!(ch1, ch2);
    let mut t = b"12345678".to_vec();
    t.extend_from_slice(b"|sep");
    let ch3 = fiat_shamir_challenge(&t);
    assert_ne!(ch1, ch3);
}

#[test]
fn test_domain_sep_multilinear() {
    let mut evals1 = vec![ExtF::ONE; 4];
    let claim = from_base(F::from_u64(4));
    let mut oracle1 = FnOracle::new(|_| vec![]);
    let mut t1 = vec![];
    multilinear_sumcheck_prover(&mut evals1, claim, &mut oracle1, &mut t1).unwrap();

    let mut evals2 = vec![ExtF::ONE; 4];
    let mut oracle2 = FnOracle::new(|_| vec![]);
    let mut t2 = b"extra".to_vec();
    multilinear_sumcheck_prover(&mut evals2, claim, &mut oracle2, &mut t2).unwrap();

    assert_ne!(t1, t2);
}

#[test]
fn test_multilinear_zk_blinding() {
    if std::env::var("RUN_LONG_TESTS").is_err() {
        return;
    }
    let claim = from_base(F::from_u64(4));
    let mut rng = rng();
    let mut evals1 = vec![ExtF::ONE; 4];
    let mut oracle = FnOracle::new(|_| vec![]);
    let mut t1 = vec![];
    let prefix1 = rng.random::<u64>().to_be_bytes().to_vec();
    t1.extend(prefix1);
    let (msgs1, _) =
        multilinear_sumcheck_prover(&mut evals1, claim, &mut oracle, &mut t1).unwrap();
    let mut evals2 = vec![ExtF::ONE; 4];
    let mut t2 = vec![];
    let prefix2 = rng.random::<u64>().to_be_bytes().to_vec();
    t2.extend(prefix2);
    let (msgs2, _) =
        multilinear_sumcheck_prover(&mut evals2, claim, &mut oracle, &mut t2).unwrap();
    assert_ne!(msgs1[0].0, msgs2[0].0);
}

#[test]
fn test_sumcheck_unit_correctness() {
    struct SquarePoly;
    impl UnivPoly for SquarePoly {
        fn evaluate(&self, point: &[ExtF]) -> ExtF {
            if point.len() != 1 {
                ExtF::ZERO
            } else {
                point[0] * point[0]
            }
        }
        fn degree(&self) -> usize {
            1
        }
        fn max_individual_degree(&self) -> usize {
            2
        }
    }

    let poly: Box<dyn UnivPoly> = Box::new(SquarePoly);
    let claim = from_base(F::from_u64(1));
    let dense = Polynomial::new(vec![ExtF::ZERO, ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![dense], &mut transcript);
    let (msgs, _comms) =
        batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle, &mut transcript).unwrap();
    let (first_uni, _) = &msgs[0];
    assert_eq!(
        first_uni.eval(ExtF::ZERO) + first_uni.eval(ExtF::ONE),
        claim
    );
}

#[test]
fn test_batched_multilinear_domain_sep() {
    let claims = vec![ExtF::ONE];
    let mut evals1 = vec![vec![ExtF::ONE; 4]];
    let mut oracle1 = FnOracle::new(|_| vec![]);
    let mut t1 = vec![];
    batched_multilinear_sumcheck_prover(&claims, &mut evals1, &mut oracle1, &mut t1).unwrap();

    let mut evals2 = vec![vec![ExtF::ONE; 4]];
    let mut oracle2 = FnOracle::new(|_| vec![]);
    let mut t2 = b"extra".to_vec();
    batched_multilinear_sumcheck_prover(&claims, &mut evals2, &mut oracle2, &mut t2).unwrap();

    assert_ne!(t1, t2);
}

#[test]
fn test_batched_sumcheck_domain_sep() {
    struct ConstPoly;
    impl UnivPoly for ConstPoly {
        fn evaluate(&self, _: &[ExtF]) -> ExtF {
            ExtF::ONE
        }
        fn degree(&self) -> usize {
            1
        }
        fn max_individual_degree(&self) -> usize {
            1
        }
    }
    let poly: Box<dyn UnivPoly> = Box::new(ConstPoly);
    let claim = ExtF::ONE + ExtF::ONE; // sum over {0,1}
    let mut oracle1 = FnOracle::new(|_| vec![]);
    let mut t1 = vec![];
    batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle1, &mut t1).unwrap();

    let mut oracle2 = FnOracle::new(|_| vec![]);
    let mut t2 = b"extra".to_vec();
    batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle2, &mut t2).unwrap();

    assert_ne!(t1, t2);
}

#[test]
fn test_verifier_rejects_high_norm_eval() {
    struct ConstPoly;
    impl UnivPoly for ConstPoly {
        fn evaluate(&self, _: &[ExtF]) -> ExtF {
            ExtF::ZERO
        }
        fn degree(&self) -> usize {
            1
        }
        fn max_individual_degree(&self) -> usize {
            1
        }
    }
    let poly: Box<dyn UnivPoly> = Box::new(ConstPoly);
    let claim = ExtF::ZERO;
    let mut prover_oracle = FnOracle::new(|_| vec![ExtF::ZERO]);
    let mut transcript = vec![];
    let (msgs, comms) =
        batched_sumcheck_prover(&[claim], &[&*poly], &mut prover_oracle, &mut transcript)
            .unwrap();

    struct HighNormOracle;
    impl PolyOracle for HighNormOracle {
        fn commit(&mut self) -> Vec<Commitment> {
            vec![]
        }
        fn open_at_point(&mut self, _point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>) {
            let val = F::from_u64(MAX_BLIND_NORM / 2 + 1);
            let high_norm = ExtF::new_complex(val, val);
            (vec![high_norm], vec![])
        }
        fn verify_openings(
            &self,
            _comms: &[Commitment],
            _point: &[ExtF],
            _evals: &[ExtF],
            _proofs: &[OpeningProof],
        ) -> bool {
            true
        }
    }
    let mut high_oracle = HighNormOracle;
    let mut vt = vec![];
    let result = batched_sumcheck_verifier(&[claim], &msgs, &comms, &mut high_oracle, &mut vt, &[]);
    assert!(result.is_none());
}

#[quickcheck]
fn prop_blind_vanishes(coeffs: Vec<i64>) -> bool {
    if coeffs.is_empty() {
        return true;
    }
    let coeffs_ext: Vec<ExtF> = coeffs.iter().map(|&c| from_base(F::from_i64(c))).collect();
    let blind_poly = Polynomial::new(coeffs_ext);
    let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
    let blind_factor = x_poly * xm1_poly * blind_poly;
    blind_factor.eval(ExtF::ZERO) == ExtF::ZERO && blind_factor.eval(ExtF::ONE) == ExtF::ZERO
}
