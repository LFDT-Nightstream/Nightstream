use neo_fold::sumcheck::{RoundOracle, run_sumcheck, run_sumcheck_skip_eval_at_one, verify_sumcheck_rounds};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

struct ZeroOracle { ell: usize, d: usize }

impl RoundOracle for ZeroOracle {
    fn num_rounds(&self) -> usize { self.ell }
    fn degree_bound(&self) -> usize { self.d }
    fn evals_at(&mut self, xs: &[K]) -> Vec<K> { vec![K::ZERO; xs.len()] }
    fn fold(&mut self, _r_i: K) {}
}

#[test]
fn engine_accepts_zero_polynomial() {
    let mut tr_p = Poseidon2Transcript::new(b"sumcheck");
    let mut oracle = ZeroOracle { ell: 3, d: 2 };
    // sample points: 0, 1, 2 over K
    let xs: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();

    let out = run_sumcheck(&mut tr_p, &mut oracle, K::ZERO, &xs)
        .expect("sumcheck should accept zero polynomial");
    assert_eq!(out.rounds.len(), 3);

    // verifier recomputes the same r-vector and accepts
    let mut tr_v = Poseidon2Transcript::new(b"sumcheck");
    let (r, _sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, K::ZERO, &out.rounds);
    assert!(ok, "verifier should accept rounds for zero polynomial");
    assert_eq!(r.len(), 3);
}

// ---- Additional engine tests ----

struct CountingOracle {
    ell: usize,
    d: usize,
    folds: usize,
    // maintain the current running sum S_i that the round must satisfy: p(0)+p(1) = S_i
    cur_sum: K,
    // remember last round polynomial to update cur_sum on fold
    last_a: K,
    last_b: K,
}

impl RoundOracle for CountingOracle {
    fn num_rounds(&self) -> usize { self.ell }
    fn degree_bound(&self) -> usize { self.d }
    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        // Choose p(X) = a + b X with a=0, b=cur_sum so that p(0)+p(1)=cur_sum
        self.last_a = K::ZERO;
        self.last_b = self.cur_sum;
        xs.iter().map(|x| self.last_a + self.last_b * *x).collect()
    }
    fn fold(&mut self, r_i: K) {
        self.folds += 1;
        // Update running sum for next round: S_{i+1} = p(r_i)
        self.cur_sum = self.last_a + self.last_b * r_i;
    }
}

#[test]
fn zero_rounds_final_sum_equals_initial() {
    let mut tr_p = Poseidon2Transcript::new(b"sumcheck");
    let initial = K::from_u64(123);
    let mut oracle = CountingOracle { ell: 0, d: 1, folds: 0, cur_sum: initial, last_a: K::ZERO, last_b: K::ZERO };
    let xs: Vec<K> = (0..=1u64).map(|u| K::from(F::from_u64(u))).collect();
    let out = run_sumcheck(&mut tr_p, &mut oracle, initial, &xs).unwrap();
    assert_eq!(out.rounds.len(), 0);
    assert_eq!(out.final_sum, initial);
    assert_eq!(oracle.folds, 0);
    let mut tr_v = Poseidon2Transcript::new(b"sumcheck");
    let (_r, sum, ok) = verify_sumcheck_rounds(&mut tr_v, 1, initial, &out.rounds);
    assert!(ok);
    assert_eq!(sum, initial);
}

#[test]
fn determinism_and_fold_count() {
    let ell = 3usize;
    let xs: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();

    // run #1
    let mut tr1 = Poseidon2Transcript::new(b"sumcheck");
    let mut o1 = CountingOracle { ell, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let out1 = run_sumcheck(&mut tr1, &mut o1, K::ZERO, &xs).unwrap();
    assert_eq!(o1.folds, ell);

    // run #2: same inputs ⇒ identical rounds & challenges
    let mut tr2 = Poseidon2Transcript::new(b"sumcheck");
    let mut o2 = CountingOracle { ell, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let out2 = run_sumcheck(&mut tr2, &mut o2, K::ZERO, &xs).unwrap();

    assert_eq!(out1.rounds, out2.rounds);
    assert_eq!(out1.challenges, out2.challenges);

    // verifier accepts both
    let mut tr_v1 = Poseidon2Transcript::new(b"sumcheck");
    let (_r1, _s1, ok1) = verify_sumcheck_rounds(&mut tr_v1, 2, K::ZERO, &out1.rounds);
    let mut tr_v2 = Poseidon2Transcript::new(b"sumcheck");
    let (_r2, _s2, ok2) = verify_sumcheck_rounds(&mut tr_v2, 2, K::ZERO, &out2.rounds);
    assert!(ok1 && ok2);
}

#[test]
fn rejects_wrong_eval_count() {
    struct BadCount { ell: usize, d: usize }
    impl RoundOracle for BadCount {
        fn num_rounds(&self) -> usize { self.ell }
        fn degree_bound(&self) -> usize { self.d }
        fn evals_at(&mut self, xs: &[K]) -> Vec<K> { vec![K::ZERO; xs.len() - 1] }
        fn fold(&mut self, _r_i: K) {}
    }
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = BadCount { ell: 1, d: 1 };
    let xs: Vec<K> = (0..=1u64).map(|u| K::from(F::from_u64(u))).collect();
    assert!(run_sumcheck(&mut tr, &mut o, K::ZERO, &xs).is_err());
}

#[test]
fn rejects_p0_plus_p1_mismatch() {
    struct BadPoly { ell: usize, d: usize }
    impl RoundOracle for BadPoly {
        fn num_rounds(&self) -> usize { self.ell }
        fn degree_bound(&self) -> usize { self.d }
        fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
            xs.iter().map(|_| K::ONE).collect()
        }
        fn fold(&mut self, _r_i: K) {}
    }
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = BadPoly { ell: 1, d: 0 };
    let xs = vec![K::ZERO]; // degree 0 ⇒ one point
    assert!(run_sumcheck(&mut tr, &mut o, K::ZERO, &xs).is_err());
}

#[test]
fn verifier_rejects_tampered_coeffs() {
    // honest run
    let mut tr_p = Poseidon2Transcript::new(b"sumcheck");
    let mut oracle = CountingOracle { ell: 2, d: 1, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let xs: Vec<K> = (0..=1u64).map(|u| K::from(F::from_u64(u))).collect();
    let out = run_sumcheck(&mut tr_p, &mut oracle, K::ZERO, &xs).unwrap();

    // tamper: flip one coefficient
    let mut tampered = out.rounds.clone();
    tampered[0][0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new(b"sumcheck");
    let (_r, _sum, ok) = verify_sumcheck_rounds(&mut tr_v, 1, K::ZERO, &tampered);
    assert!(!ok);
}

#[test]
fn verifier_rejects_degree_overflow() {
    // fabricate a round polynomial with degree 3 while d_sc == 2
    let rounds = vec![ vec![K::ZERO, K::ONE, K::ONE, K::ONE] ]; // len=4 ⇒ deg=3
    let mut tr_v = Poseidon2Transcript::new(b"sumcheck");
    let (_r, _sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, K::ZERO, &rounds);
    assert!(!ok);
}

// ---- Skip-eval-at-1 optimization tests ----

#[test]
fn skip_eval_at_one_matches_baseline_quadratic_points() {
    let xs_full: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();

    let mut tr1 = Poseidon2Transcript::new(b"sumcheck");
    let mut tr2 = Poseidon2Transcript::new(b"sumcheck");

    let mut o1 = CountingOracle { ell: 3, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let mut o2 = CountingOracle { ell: 3, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };

    let out1 = run_sumcheck(&mut tr1, &mut o1, K::ZERO, &xs_full).unwrap();
    let out2 = run_sumcheck_skip_eval_at_one(&mut tr2, &mut o2, K::ZERO, &xs_full).unwrap();

    assert_eq!(out1.rounds, out2.rounds, "coefficients must match");
    assert_eq!(out1.challenges, out2.challenges, "r-vector must match");
    assert_eq!(o1.folds, o2.folds, "same number of folds");

    let mut tv1 = Poseidon2Transcript::new(b"sumcheck");
    let mut tv2 = Poseidon2Transcript::new(b"sumcheck");
    let (_r1, _s1, ok1) = verify_sumcheck_rounds(&mut tv1, 2, K::ZERO, &out1.rounds);
    let (_r2, _s2, ok2) = verify_sumcheck_rounds(&mut tv2, 2, K::ZERO, &out2.rounds);
    assert!(ok1 && ok2);
}

#[test]
fn skip_eval_at_one_rejects_if_missing_zero_or_one() {
    let xs_missing_one = vec![K::from(F::from_u64(0)), K::from(F::from_u64(2))];
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = CountingOracle { ell: 1, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    assert!(run_sumcheck_skip_eval_at_one(&mut tr, &mut o, K::ZERO, &xs_missing_one).is_err());

    let xs_missing_zero = vec![K::from(F::from_u64(1)), K::from(F::from_u64(2))];
    let mut tr2 = Poseidon2Transcript::new(b"sumcheck");
    let mut o2 = CountingOracle { ell: 1, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    assert!(run_sumcheck_skip_eval_at_one(&mut tr2, &mut o2, K::ZERO, &xs_missing_zero).is_err());
}

#[test]
fn skip_eval_at_one_rejects_insufficient_points_for_degree_bound() {
    // d_sc = 2 but only two points {0,1}
    let xs_full = vec![K::ZERO, K::ONE];
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = CountingOracle { ell: 1, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    assert!(run_sumcheck_skip_eval_at_one(&mut tr, &mut o, K::ZERO, &xs_full).is_err());
}

// Red-team: degree overflow under skip path
struct QuadOracleSkip { ell: usize, d_bound: usize }
impl RoundOracle for QuadOracleSkip {
    fn num_rounds(&self) -> usize { self.ell }
    fn degree_bound(&self) -> usize { self.d_bound }
    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        // degree 2 polynomial: p(X)=X^2 - X; p(0)+p(1)=0
        xs.iter().map(|x| *x * *x - *x).collect()
    }
    fn fold(&mut self, _r: K) {}
}

#[test]
fn skip_eval_at_one_rejects_degree_overflow() {
    // 3 points allow interpolation to detect true degree 2 > d_bound=1
    let xs_full: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = QuadOracleSkip { ell: 1, d_bound: 1 };
    assert!(run_sumcheck_skip_eval_at_one(&mut tr, &mut o, K::ZERO, &xs_full).is_err());
}

#[test]
fn duplicated_sample_points_are_rejected() {
    // Using ZeroOracle; engine should error out due to duplicate sample points
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = ZeroOracle { ell: 1, d: 2 };
    let xs = vec![K::from(F::from_u64(0)), K::from(F::from_u64(0)), K::from(F::from_u64(1))];
    let res = run_sumcheck(&mut tr, &mut o, K::ZERO, &xs);
    assert!(res.is_err());
}

// ---- Red-team: prover tries to sneak in higher degree when nodes are sufficient ----
struct QuadOracle { ell: usize, d_bound: usize }
impl RoundOracle for QuadOracle {
    fn num_rounds(&self) -> usize { self.ell }
    fn degree_bound(&self) -> usize { self.d_bound } // claims ≤ 1
    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        // actually degree 2: p(X) = X^2 ; p(0)+p(1)=1 so this also violates the invariant for initial_sum=0,
        // but we provide a compensating constant so that p(0)+p(1)=0 while deg=2:
        // take p(X)=X^2 - X. Then p(0)+p(1)=0, deg=2 (should still be rejected by degree bound when nodes≥3)
        xs.iter().map(|x| *x * *x - *x).collect()
    }
    fn fold(&mut self, _r: K) {}
}

#[test]
fn prover_degree_overflow_rejected_when_nodes_sufficient() {
    let mut tr = Poseidon2Transcript::new(b"sumcheck");
    let mut o = QuadOracle { ell: 1, d_bound: 1 };
    // 3 nodes ⇒ interpolation reveals true degree 2
    let xs: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();
    let err = run_sumcheck(&mut tr, &mut o, K::ZERO, &xs);
    assert!(err.is_err(), "engine should reject round poly of degree > d_sc");
}

#[test]
fn verifier_final_sum_matches_prover_final_sum() {
    // Use a simple oracle: p(X)=a + bX with a=0, b=0 ⇒ p≡0 across rounds.
    struct ZeroLin { ell: usize, d: usize }
    impl RoundOracle for ZeroLin {
        fn num_rounds(&self) -> usize { self.ell }
        fn degree_bound(&self) -> usize { self.d }
        fn evals_at(&mut self, xs: &[K]) -> Vec<K> { xs.iter().map(|_| K::ZERO).collect() }
        fn fold(&mut self, _r: K) {}
    }
    let xs: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();
    let mut tr_p = Poseidon2Transcript::new(b"sumcheck");
    let mut o = ZeroLin { ell: 4, d: 2 };
    let out = run_sumcheck(&mut tr_p, &mut o, K::ZERO, &xs).unwrap();

    let mut tr_v = Poseidon2Transcript::new(b"sumcheck");
    let (_r, v_sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, K::ZERO, &out.rounds);
    assert!(ok);
    assert_eq!(v_sum, out.final_sum, "verifier and prover must agree on final reduced claim");
}

#[test]
fn verifier_rejects_reordered_rounds() {
    // honest run with an oracle that maintains the invariant across rounds
    let xs: Vec<K> = (0..=1u64).map(|u| K::from(F::from_u64(u))).collect();
    let mut tr_p = Poseidon2Transcript::new(b"sumcheck");
    let mut o = CountingOracle { ell: 2, d: 1, folds: 0, cur_sum: K::ONE, last_a: K::ZERO, last_b: K::ZERO };
    let out = run_sumcheck(&mut tr_p, &mut o, K::ONE, &xs).unwrap();

    // tamper: swap round order
    let mut tampered = out.rounds.clone();
    tampered.reverse();

    let mut tr_v = Poseidon2Transcript::new(b"sumcheck");
    let (_r, _s, ok) = verify_sumcheck_rounds(&mut tr_v, 1, K::ONE, &tampered);
    assert!(!ok, "reordering rounds must fail due to transcript/challenges mismatch");
}
