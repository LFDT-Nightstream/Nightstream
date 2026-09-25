use neo_math::{Fq, K};
use neo_reductions::sumcheck::{
    interpolate_from_evals, poly_eval_k, poly_eval_k_base, run_batched_sumcheck_prover, run_sumcheck_prover,
    verify_batched_sumcheck_rounds, verify_sumcheck_rounds, BatchedClaim, RoundOracle, SumcheckError,
};
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

#[derive(Clone)]
struct ConsistentLinearOracle {
    rounds: usize,
    round_idx: usize,
    target_sum: K,
    anchor_seed: u64,
}

impl ConsistentLinearOracle {
    fn new(rounds: usize, target_sum: K, anchor_seed: u64) -> Self {
        Self {
            rounds,
            round_idx: 0,
            target_sum,
            anchor_seed,
        }
    }

    #[inline]
    fn anchor(&self) -> K {
        let v = self
            .anchor_seed
            .wrapping_add((self.round_idx as u64).wrapping_mul(17))
            .wrapping_add(1);
        K::from(Fq::from_u64(v))
    }

    #[inline]
    fn eval_linear(&self, x: K) -> K {
        let p0 = self.anchor();
        let p1 = self.target_sum - p0;
        p0 + (p1 - p0) * x
    }
}

impl RoundOracle for ConsistentLinearOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        points.iter().map(|&x| self.eval_linear(x)).collect()
    }

    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn fold(&mut self, r: K) {
        self.target_sum = self.eval_linear(r);
        self.round_idx += 1;
    }
}

fn run_batched_case(
    num_claims: usize,
    rounds: usize,
) -> (Vec<K>, Vec<Vec<Vec<K>>>, Vec<K>, Vec<K>, Vec<&'static [u8]>) {
    const LABEL: &[u8] = b"batched/test_claim";
    let mut tr = Poseidon2Transcript::new(b"sumcheck/batched/test");

    let mut oracles: Vec<ConsistentLinearOracle> = (0..num_claims)
        .map(|i| {
            let claimed = K::from(Fq::from_u64((i as u64 + 1) * 13));
            ConsistentLinearOracle::new(rounds, claimed, 1000 + i as u64)
        })
        .collect();

    let claimed_sums: Vec<K> = oracles.iter().map(|o| o.target_sum).collect();
    let labels: Vec<&'static [u8]> = vec![LABEL; num_claims];
    let mut claims: Vec<BatchedClaim<'_>> = oracles
        .iter_mut()
        .map(|oracle| {
            let claimed_sum = oracle.target_sum;
            BatchedClaim {
                oracle,
                claimed_sum,
                label: LABEL,
            }
        })
        .collect();

    let (shared_challenges, per_claim_results) =
        run_batched_sumcheck_prover(&mut tr, claims.as_mut_slice()).expect("batched sumcheck should succeed");

    let round_polys: Vec<Vec<Vec<K>>> = per_claim_results
        .iter()
        .map(|r| r.round_polys.clone())
        .collect();
    let final_values: Vec<K> = per_claim_results.iter().map(|r| r.final_value).collect();
    (shared_challenges, round_polys, final_values, claimed_sums, labels)
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
fn polynomial_helpers_match_base_field_and_interpolation() {
    let coeffs = [
        K::from(Fq::from_u64(3)),
        K::from(Fq::from_u64(2)),
        K::from(Fq::from_u64(5)),
    ];
    let x = Fq::from_u64(7);
    assert_eq!(poly_eval_k(&coeffs, K::from(x)), K::from(Fq::from_u64(262)));
    assert_eq!(poly_eval_k_base(&coeffs, x), poly_eval_k(&coeffs, K::from(x)));
    assert_eq!(poly_eval_k(&[], K::from(x)), K::ZERO);

    let xs = [K::ZERO, K::ONE, K::from(Fq::from_u64(2))];
    let ys = [
        K::from(Fq::from_u64(1)),
        K::from(Fq::from_u64(6)),
        K::from(Fq::from_u64(17)),
    ];
    let interpolated = interpolate_from_evals(&xs, &ys);
    for (&point, &expected) in xs.iter().zip(&ys) {
        assert_eq!(poly_eval_k(&interpolated, point), expected);
    }
}

#[test]
fn verifier_rejects_tampered_round_polynomial() {
    let coeffs = vec![K::from(Fq::from_u64(5))];
    let initial_sum = K::from(Fq::from_u64(10));
    let mut oracle = DummyOracle {
        coeffs,
        rounds: 1,
        degree: 1,
    };
    let mut prover_transcript = Poseidon2Transcript::new(b"sumcheck/prover/tamper");
    let (mut round_polys, _) =
        run_sumcheck_prover(&mut prover_transcript, &mut oracle, initial_sum).expect("honest round must be valid");
    round_polys[0][0] += K::ONE;

    let mut verifier_transcript = Poseidon2Transcript::new(b"sumcheck/prover/tamper");
    let (_, _, valid) = verify_sumcheck_rounds(&mut verifier_transcript, 1, initial_sum, &round_polys);
    assert!(!valid, "the verifier must reject a changed round polynomial");
}

#[test]
fn verifier_rejects_round_above_degree_bound() {
    let round_poly = vec![K::ONE, K::from(Fq::from_u64(2)), K::from(Fq::from_u64(3))];
    let initial_sum = poly_eval_k(&round_poly, K::ZERO) + poly_eval_k(&round_poly, K::ONE);
    let mut transcript = Poseidon2Transcript::new(b"sumcheck/prover/degree");
    let (_, _, valid) = verify_sumcheck_rounds(&mut transcript, 1, initial_sum, &[round_poly]);
    assert!(!valid, "the verifier must reject a polynomial above the degree bound");
}

#[test]
fn prover_rejects_wrong_initial_sum() {
    let mut oracle = DummyOracle {
        coeffs: vec![K::from(Fq::from_u64(5))],
        rounds: 1,
        degree: 1,
    };
    let mut transcript = Poseidon2Transcript::new(b"sumcheck/prover/wrong-initial-sum");
    let error = run_sumcheck_prover(&mut transcript, &mut oracle, K::from(Fq::from_u64(999)))
        .expect_err("the prover must reject an inconsistent initial sum");
    assert!(matches!(error, SumcheckError::Invariant { round: 0, .. }));
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

#[test]
fn run_batched_sumcheck_round_trip_many_claims() {
    let num_claims = 24usize;
    let rounds = 5usize;
    let (shared_chals, round_polys, final_values, claimed_sums, labels) = run_batched_case(num_claims, rounds);

    let mut tr_v = Poseidon2Transcript::new(b"sumcheck/batched/test");
    let degree_bounds = vec![1usize; num_claims];
    let (v_chals, v_finals, ok) = verify_batched_sumcheck_rounds(
        &mut tr_v,
        &round_polys,
        &claimed_sums,
        labels.as_slice(),
        degree_bounds.as_slice(),
    );
    assert!(ok, "batched verifier should accept prover output");
    assert_eq!(v_chals, shared_chals, "shared challenges must match");
    assert_eq!(v_finals, final_values, "final values must match");
}

#[test]
fn run_batched_sumcheck_is_deterministic_across_thread_pool_configs() {
    let serial_like = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread pool");
    let out_single = serial_like.install(|| run_batched_case(24, 5));
    let out_default = run_batched_case(24, 5);
    assert_eq!(out_single, out_default, "batched sumcheck output must be deterministic");
}
