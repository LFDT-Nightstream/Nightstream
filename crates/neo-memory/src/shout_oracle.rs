//! Shout-specific sumcheck oracle (lookup correctness).

use crate::twist_oracle::{broadcast_cycle, build_eq_table, ProductRoundOracle};
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;

/// Oracle for Shout lookup correctness:
/// rv(r_cycle) = Σ_{k,j} eq(r_cycle, j) · ra(k, j) · Table(k)
pub struct ShoutReadCheckOracle {
    core: ProductRoundOracle,
    pub ell_addr: usize,
    pub ell_cycle: usize,
}

impl ShoutReadCheckOracle {
    pub fn new(ra_table: Vec<K>, table_vals: Vec<K>, has_lookup_table: Vec<K>, r_cycle: &[K]) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        assert_eq!(
            ra_table.len() % pow2_cycle,
            0,
            "ra_table length must be multiple of cycle domain size"
        );
        let pow2_addr = ra_table.len() / pow2_cycle;
        assert_eq!(pow2_addr, table_vals.len(), "table size mismatch");
        assert!(pow2_addr.is_power_of_two(), "address domain must be power-of-two");

        let eq_cycle = build_eq_table(r_cycle);
        let eq_cycle_table = broadcast_cycle(&eq_cycle, pow2_addr);
        let has_lookup_broadcast = broadcast_cycle(&has_lookup_table, pow2_addr);

        // Broadcast table_vals across cycles
        let mut table_table = Vec::with_capacity(pow2_addr * pow2_cycle);
        for _ in 0..pow2_cycle {
            table_table.extend_from_slice(&table_vals);
        }

        let core = ProductRoundOracle::new(vec![eq_cycle_table, has_lookup_broadcast, ra_table, table_table], 4);
        Self {
            core,
            ell_addr: pow2_addr.trailing_zeros() as usize,
            ell_cycle: r_cycle.len(),
        }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }
}

impl RoundOracle for ShoutReadCheckOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        self.core.evals_at(points)
    }
    fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }
    fn degree_bound(&self) -> usize {
        self.core.degree_bound()
    }
    fn fold(&mut self, r: K) {
        self.core.fold(r)
    }
}
