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

