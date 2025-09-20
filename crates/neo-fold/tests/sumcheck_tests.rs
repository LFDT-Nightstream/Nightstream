use neo_fold::sumcheck::{RoundOracle, run_sumcheck, verify_sumcheck_rounds};
use neo_fold::transcript::FoldTranscript;
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
    let mut tr_p = FoldTranscript::default();
    let mut oracle = ZeroOracle { ell: 3, d: 2 };
    // sample points: 0, 1, 2 over K
    let xs: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();

    let out = run_sumcheck(&mut tr_p, &mut oracle, K::ZERO, &xs)
        .expect("sumcheck should accept zero polynomial");
    assert_eq!(out.rounds.len(), 3);

    // verifier recomputes the same r-vector and accepts
    let mut tr_v = FoldTranscript::default();
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
    let mut tr_p = FoldTranscript::default();
    let initial = K::from_u64(123);
    let mut oracle = CountingOracle { ell: 0, d: 1, folds: 0, cur_sum: initial, last_a: K::ZERO, last_b: K::ZERO };
    let xs: Vec<K> = (0..=1u64).map(|u| K::from(F::from_u64(u))).collect();
    let out = run_sumcheck(&mut tr_p, &mut oracle, initial, &xs).unwrap();
    assert_eq!(out.rounds.len(), 0);
    assert_eq!(out.final_sum, initial);
    assert_eq!(oracle.folds, 0);
    let mut tr_v = FoldTranscript::default();
    let (_r, sum, ok) = verify_sumcheck_rounds(&mut tr_v, 1, initial, &out.rounds);
    assert!(ok);
    assert_eq!(sum, initial);
}

#[test]
fn determinism_and_fold_count() {
    let ell = 3usize;
    let xs: Vec<K> = (0..=2u64).map(|u| K::from(F::from_u64(u))).collect();

    // run #1
    let mut tr1 = FoldTranscript::default();
    let mut o1 = CountingOracle { ell, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let out1 = run_sumcheck(&mut tr1, &mut o1, K::ZERO, &xs).unwrap();
    assert_eq!(o1.folds, ell);

    // run #2: same inputs ⇒ identical rounds & challenges
    let mut tr2 = FoldTranscript::default();
    let mut o2 = CountingOracle { ell, d: 2, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let out2 = run_sumcheck(&mut tr2, &mut o2, K::ZERO, &xs).unwrap();

    assert_eq!(out1.rounds, out2.rounds);
    assert_eq!(out1.challenges, out2.challenges);

    // verifier accepts both
    let mut tr_v1 = FoldTranscript::default();
    let (_r1, _s1, ok1) = verify_sumcheck_rounds(&mut tr_v1, 2, K::ZERO, &out1.rounds);
    let mut tr_v2 = FoldTranscript::default();
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
    let mut tr = FoldTranscript::default();
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
    let mut tr = FoldTranscript::default();
    let mut o = BadPoly { ell: 1, d: 0 };
    let xs = vec![K::ZERO]; // degree 0 ⇒ one point
    assert!(run_sumcheck(&mut tr, &mut o, K::ZERO, &xs).is_err());
}

#[test]
fn verifier_rejects_tampered_coeffs() {
    // honest run
    let mut tr_p = FoldTranscript::default();
    let mut oracle = CountingOracle { ell: 2, d: 1, folds: 0, cur_sum: K::ZERO, last_a: K::ZERO, last_b: K::ZERO };
    let xs: Vec<K> = (0..=1u64).map(|u| K::from(F::from_u64(u))).collect();
    let out = run_sumcheck(&mut tr_p, &mut oracle, K::ZERO, &xs).unwrap();

    // tamper: flip one coefficient
    let mut tampered = out.rounds.clone();
    tampered[0][0] += K::ONE;

    let mut tr_v = FoldTranscript::default();
    let (_r, _sum, ok) = verify_sumcheck_rounds(&mut tr_v, 1, K::ZERO, &tampered);
    assert!(!ok);
}

#[test]
fn verifier_rejects_degree_overflow() {
    // fabricate a round polynomial with degree 3 while d_sc == 2
    let rounds = vec![ vec![K::ZERO, K::ONE, K::ONE, K::ONE] ]; // len=4 ⇒ deg=3
    let mut tr_v = FoldTranscript::default();
    let (_r, _sum, ok) = verify_sumcheck_rounds(&mut tr_v, 2, K::ZERO, &rounds);
    assert!(!ok);
}

#[test]
fn duplicated_sample_points_are_rejected() {
    // Using ZeroOracle; engine should error out due to duplicate sample points
    let mut tr = FoldTranscript::default();
    let mut o = ZeroOracle { ell: 1, d: 2 };
    let xs = vec![K::from(F::from_u64(0)), K::from(F::from_u64(0)), K::from(F::from_u64(1))];
    let res = run_sumcheck(&mut tr, &mut o, K::ZERO, &xs);
    assert!(res.is_err());
}
