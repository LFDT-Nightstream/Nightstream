use neo_math::{Fq, K};
use neo_reductions::sumcheck::{poly_eval_k, run_sumcheck_prover, verify_sumcheck_rounds, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

struct DummyOracle {
    coeffs: Vec<K>,
    rounds: usize,
    degree: usize,
}

impl RoundOracle for DummyOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        points
            .iter()
            .map(|&x| poly_eval_k(&self.coeffs, x))
            .collect()
    }

    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        self.degree
    }
    fn fold(&mut self, _r: K) {}
}

#[test]
fn run_sumcheck_prover_round_trip() {
    // Polynomial p(x) = 3 + 2x + x^2
    let coeffs = vec![K::from(Fq::from_u64(3)), K::from(Fq::from_u64(2)), K::ONE];
    let mut oracle = DummyOracle {
        coeffs: coeffs.clone(),
        rounds: 1,
        degree: 2,
    };
    let initial_sum = poly_eval_k(&coeffs, K::ZERO) + poly_eval_k(&coeffs, K::ONE);

    let mut tr = Poseidon2Transcript::new(b"sumcheck/prover/test");
    let (round_polys, challenges) =
        run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).expect("prover should succeed");

    assert_eq!(round_polys.len(), 1);
    assert_eq!(challenges.len(), 1);

    let mut tr_v = Poseidon2Transcript::new(b"sumcheck/prover/test");
    let (verifier_chals, final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, initial_sum, &round_polys);
    assert!(ok);
    assert_eq!(verifier_chals.len(), 1);
    assert_eq!(verifier_chals[0], challenges[0]);

    let expected_final = poly_eval_k(&coeffs, challenges[0]);
    assert_eq!(final_sum, expected_final);
}

#[test]
fn run_sumcheck_linear_round_trip() {
    // p(x) = 5 + 7x
    let coeffs = vec![K::from(Fq::from_u64(5)), K::from(Fq::from_u64(7))];
    let mut oracle = DummyOracle {
        coeffs: coeffs.clone(),
        rounds: 1,
        degree: 1,
    };
    let initial_sum = poly_eval_k(&coeffs, K::ZERO) + poly_eval_k(&coeffs, K::ONE);

    let mut tr = Poseidon2Transcript::new(b"sumcheck/prover/linear");
    let (round_polys, chals) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).expect("prover should succeed");
    assert_eq!(round_polys.len(), 1);

    let mut tr_v = Poseidon2Transcript::new(b"sumcheck/prover/linear");
    let (_c, final_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 1, initial_sum, &round_polys);
    assert!(ok);
    assert_eq!(final_sum, poly_eval_k(&coeffs, chals[0]));
}
