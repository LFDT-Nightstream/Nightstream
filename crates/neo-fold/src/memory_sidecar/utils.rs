use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

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

// ============================================================================
// Aggregated bitness weights
// ============================================================================

#[inline]
fn bitness_weight_base(r_cycle: &[K], domain_sep: u64) -> K {
    // Deterministic weight base derived from transcript-known `r_cycle` and a small domain separator.
    //
    // Security note: Weights are needed to prevent cross-column cancellation in the aggregated
    // booleanity check (Σ_i w_i * b_i*(b_i-1) == 0).
    let mut base = if r_cycle.is_empty() {
        K::from(neo_math::F::from_u64(domain_sep))
    } else {
        r_cycle[0] + K::from(neo_math::F::from_u64(domain_sep))
    };
    if base == K::ZERO || base == K::ONE || base == -K::ONE {
        base += K::from(neo_math::F::ONE);
    }
    if base == K::ZERO {
        base = K::ONE;
    }
    base
}

#[inline]
pub fn bitness_weights(r_cycle: &[K], n: usize, domain_sep: u64) -> Vec<K> {
    let base = bitness_weight_base(r_cycle, domain_sep);
    let mut out = Vec::with_capacity(n);
    let mut cur = K::ONE;
    for _ in 0..n {
        out.push(cur);
        cur *= base;
    }
    out
}
