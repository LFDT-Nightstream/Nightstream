use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;

/// RoundOracle wrapper that truncates the reported number of rounds.
///
/// Used for Route A batching: some oracles have extra rounds (e.g. Shout lookup has
/// time+address rounds) but we batch only the first `ell_n` time rounds.
pub struct RoundOraclePrefix<'a> {
    inner: &'a mut dyn RoundOracle,
    num_rounds: usize,
}

impl<'a> RoundOraclePrefix<'a> {
    pub fn new(inner: &'a mut dyn RoundOracle, num_rounds: usize) -> Self {
        Self { inner, num_rounds }
    }
}

impl<'a> RoundOracle for RoundOraclePrefix<'a> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.inner.evals_at(points)
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.inner.degree_bound()
    }

    fn fold(&mut self, r: K) {
        self.inner.fold(r)
    }
}
