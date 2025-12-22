//! Twist (and Shout) sumcheck oracles built from multilinear factor tables.
//!
//! This module provides oracles for the **index-bit addressing** architecture:
//! instead of materializing huge one-hot tables, we compute eq(bits, r_addr)
//! dynamically from committed bit columns.
//!
//! ## Key Oracles
//!
//! - `ProductRoundOracle`: Generic multilinear product sumcheck
//! - `TwistReadCheckOracle`: Proves rv(t) = Val(ra_t, t) via bit-factors
//! - `TwistWriteCheckOracle`: Proves Inc(k, t) = wa(k, t) * (wv - Val(k, t))
//! - `TwistValEvalOracle`: Proves Val(r_addr, r_cycle) = Σ Inc * LT
//! - `IndexAdapterOracle`: Proves eq(bits_t, r_addr) consistency (IDX→OH bridge)
//! - `BitnessOracle`: Proves bit columns are binary

use crate::bit_ops::{eq_bit_affine, eq_bits_prod_table};
use crate::mle::{eq_points, lt_eval};
#[cfg(feature = "debug-logs")]
use neo_math::KExtensions;
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::{Field, PrimeCharacteristicRing};

macro_rules! impl_round_oracle_via_core {
    ($ty:ty) => {
        impl RoundOracle for $ty {
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
    };
}

// ============================================================================
// Core ProductRoundOracle
// ============================================================================

/// Helper that runs sumcheck for a product of multilinear factors.
///
/// Each factor table is a length-2^ℓ vector enumerating the factor on the
/// Boolean hypercube, using little-endian bit order.
pub struct ProductRoundOracle {
    factors: Vec<Vec<K>>,
    rounds_remaining: usize,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl ProductRoundOracle {
    pub fn new(factors: Vec<Vec<K>>, degree_bound: usize) -> Self {
        let len = factors.first().map(|f| f.len()).unwrap_or(1);
        assert!(len.is_power_of_two(), "factor length must be a power of two");
        for f in factors.iter() {
            assert_eq!(f.len(), len, "all factors must have the same length");
        }
        let total_rounds = log2_pow2(len);
        Self {
            factors,
            rounds_remaining: total_rounds,
            degree_bound,
            challenges: Vec::with_capacity(total_rounds),
        }
    }

    pub fn value(&self) -> Option<K> {
        if self.factors.iter().all(|f| f.len() == 1) {
            let mut acc = K::ONE;
            for f in self.factors.iter() {
                acc *= f[0];
            }
            Some(acc)
        } else {
            None
        }
    }

    pub fn challenges(&self) -> &[K] {
        &self.challenges
    }

    pub fn sum_over_hypercube(&self) -> K {
        let n = self.factors.first().map(|f| f.len()).unwrap_or(0);
        let mut sum = K::ZERO;
        for t in 0..n {
            let mut prod = K::ONE;
            for f in &self.factors {
                prod *= f[t];
            }
            sum += prod;
        }
        sum
    }
}

impl RoundOracle for ProductRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.rounds_remaining == 0 {
            // Return the single value for all points
            let val = self.value().unwrap_or(K::ZERO);
            return vec![val; points.len()];
        }
        let half = 1usize << (self.rounds_remaining - 1);
        let mut ys = vec![K::ZERO; points.len()];

        for (idx_point, &x) in points.iter().enumerate() {
            let mut acc = K::ZERO;
            for pair in 0..half {
                let mut prod = K::ONE;
                for factor in self.factors.iter() {
                    let f0 = factor[2 * pair];
                    let f1 = factor[2 * pair + 1];
                    prod *= f0 + (f1 - f0) * x;
                }
                acc += prod;
            }
            ys[idx_point] = acc;
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.rounds_remaining == 0 {
            return;
        }
        let half = 1usize << (self.rounds_remaining - 1);
        for factor in self.factors.iter_mut() {
            for i in 0..half {
                let f0 = factor[2 * i];
                let f1 = factor[2 * i + 1];
                factor[i] = f0 + (f1 - f0) * r;
            }
            factor.truncate(half);
        }
        self.rounds_remaining -= 1;
        self.challenges.push(r);
    }
}

// ============================================================================
// Bit-Factor Helpers (for index-bit addressing)
// ============================================================================

/// Build eq(bit_column, r_bit) factor from a single bit column.
///
/// For each step t: eq(b_t, r) = b_t * r + (1 - b_t) * (1 - r)
///                             = b_t * (2r - 1) + (1 - r)
fn build_single_bit_factor(bit_col: &[K], r_bit: K) -> Vec<K> {
    bit_col.iter().map(|&b| eq_bit_affine(b, r_bit)).collect()
}

/// Build all eq factors for address bits against r_addr.
///
/// Returns a vector of factors, one per bit column.
/// The product of all these factors gives eq(addr_t, r_addr) for each t.
pub fn build_bit_eq_factors(bit_cols: &[Vec<K>], r_addr: &[K]) -> Vec<Vec<K>> {
    assert_eq!(
        bit_cols.len(),
        r_addr.len(),
        "bit columns count must match r_addr length"
    );
    bit_cols
        .iter()
        .zip(r_addr.iter())
        .map(|(col, &r)| build_single_bit_factor(col, r))
        .collect()
}

/// Compute eq(bits_t, r_addr) for each step t.
/// This is the product of all bit-eq factors.
pub fn compute_eq_from_bits(bit_cols: &[Vec<K>], r_addr: &[K]) -> Vec<K> {
    eq_bits_prod_table(bit_cols, r_addr).expect("compute_eq_from_bits: invalid input")
}

fn build_cycle_gate_diff_bits_oracle(
    r_cycle: &[K],
    gate: Vec<K>,
    diff: Vec<K>,
    bit_cols: &[Vec<K>],
    r_addr: &[K],
) -> ProductRoundOracle {
    let eq_cycle = build_eq_table(r_cycle);
    assert_eq!(eq_cycle.len(), gate.len(), "eq_cycle length must match gate");
    assert_eq!(diff.len(), gate.len(), "diff length must match gate");

    let bit_eq_factors = build_bit_eq_factors(bit_cols, r_addr);
    let mut factors = Vec::with_capacity(3 + bit_eq_factors.len());
    factors.push(eq_cycle);
    factors.push(gate);
    factors.push(diff);
    factors.extend(bit_eq_factors);

    let degree_bound = factors.len();
    ProductRoundOracle::new(factors, degree_bound)
}

fn build_cycle_gate_bits_oracle(r_cycle: &[K], gate: Vec<K>, bit_cols: &[Vec<K>], r_addr: &[K]) -> ProductRoundOracle {
    let eq_cycle = build_eq_table(r_cycle);
    assert_eq!(eq_cycle.len(), gate.len(), "eq_cycle length must match gate");

    let bit_eq_factors = build_bit_eq_factors(bit_cols, r_addr);
    let mut factors = Vec::with_capacity(2 + bit_eq_factors.len());
    factors.push(eq_cycle);
    factors.push(gate);
    factors.extend(bit_eq_factors);

    let degree_bound = factors.len();
    ProductRoundOracle::new(factors, degree_bound)
}

fn build_cycle_bits_oracle(r_cycle: &[K], bit_cols: &[Vec<K>], r_addr: &[K]) -> ProductRoundOracle {
    let eq_cycle = build_eq_table(r_cycle);
    let bit_eq_factors = build_bit_eq_factors(bit_cols, r_addr);

    let mut factors = Vec::with_capacity(1 + bit_eq_factors.len());
    factors.push(eq_cycle);
    factors.extend(bit_eq_factors);

    let degree_bound = factors.len();
    ProductRoundOracle::new(factors, degree_bound)
}

fn build_bitness_oracle(col: Vec<K>) -> ProductRoundOracle {
    let col_minus_one: Vec<K> = col.iter().map(|&c| c - K::ONE).collect();
    ProductRoundOracle::new(vec![col, col_minus_one], 2)
}

fn build_cycle_bitness_oracle(r_cycle: &[K], col: Vec<K>) -> ProductRoundOracle {
    let eq_cycle = build_eq_table(r_cycle);
    assert_eq!(eq_cycle.len(), col.len(), "eq_cycle length must match domain");
    let col_minus_one: Vec<K> = col.iter().map(|&c| c - K::ONE).collect();
    ProductRoundOracle::new(vec![eq_cycle, col, col_minus_one], 3)
}

// ============================================================================
// Val-Evaluation Oracle (unchanged from original)
// ============================================================================

/// Val-evaluation oracle:
/// Proves V(r_addr, r_cycle) = Σ_{j'} Inc(r_addr, j') · LT(j', r_cycle)
pub struct TwistValEvalOracle {
    core: ProductRoundOracle,
}

impl TwistValEvalOracle {
    pub fn new<F: Field + Copy + Into<K>>(inc: &[F], k: usize, steps: usize, r_addr: &[K], r_cycle: &[K]) -> Self {
        let pow2_addr = 1usize << r_addr.len();
        let pow2_cycle = 1usize << r_cycle.len();
        let (inc_table, lt_table) = val_tables_from_inc(inc, k, steps, pow2_addr, pow2_cycle, r_addr, r_cycle);
        let core = ProductRoundOracle::new(vec![inc_table, lt_table], 2);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(TwistValEvalOracle);

// ============================================================================
// Val-Evaluation Oracle (SPARSE version)
// ============================================================================

/// Val-evaluation oracle using SPARSE Inc representation.
///
/// Proves: Val(r_addr, r_cycle) = init(r_addr) + Σ_t Inc(r_addr, t) · LT(t, r_cycle)
///
/// where Inc(r_addr, t) is computed from the sparse representation:
/// ```text
/// Inc(r_addr, t) = has_write(t) * eq(wa_bits(t), r_addr) * inc_at_write_addr(t)
/// ```
///
/// This oracle computes the Inc contribution (without the init term).
/// The verifier reconstructs Val = init_at_r_addr + inc_contribution.
///
/// ## Advantages over TwistValEvalOracle:
/// - Does not require the full k × steps Inc matrix
/// - Uses only committed sparse columns
/// - All data at consistent column offsets
/// - No X pollution or width mismatch issues
pub struct TwistValEvalOracleSparse {
    core: ProductRoundOracle,
}

impl TwistValEvalOracleSparse {
    /// Create a new sparse val-eval oracle.
    ///
    /// # Arguments
    /// - `wa_bits`: Write address bit columns (d*ell total, each length pow2_cycle)
    /// - `has_write`: Has-write flag column
    /// - `inc_at_write_addr`: Increment at write address column
    /// - `r_addr`: Random point for address dimension (d*ell bits)
    /// - `r_cycle`: Random point for cycle dimension
    ///
    /// # Returns
    /// - The oracle for sum-check
    /// - The Inc contribution (= Σ_t Inc(r_addr, t) · LT(t, r_cycle))
    pub fn new(
        wa_bits: &[Vec<K>],
        has_write: Vec<K>,
        inc_at_write_addr: Vec<K>,
        r_addr: &[K],
        r_cycle: &[K],
    ) -> (Self, K) {
        let pow2_cycle = 1usize << r_cycle.len();

        assert_eq!(wa_bits.len(), r_addr.len(), "wa_bits count must match r_addr length");
        assert_eq!(has_write.len(), pow2_cycle, "has_write length must match cycle domain");
        assert_eq!(
            inc_at_write_addr.len(),
            pow2_cycle,
            "inc_at_write_addr length must match cycle domain"
        );

        // Build LT(t, r_cycle) table: 1 if t < r_cycle, 0 otherwise (lexicographic)
        let lt_table = build_lt_table(r_cycle);

        // Build bit-eq factors for eq(wa_bits(t), r_addr).
        // Each factor is multilinear in the time variables, so the product is a
        // valid (higher-degree) sum-check target whose terminal evaluation is
        // verifier-computable from openings at r_val.
        let bit_eq_factors = build_bit_eq_factors(wa_bits, r_addr);

        // Sum-check oracle over:
        //   has_write(t) * inc_at_write_addr(t) * eq(wa_bits(t), r_addr) * LT(t, r_cycle)
        //
        // Degree bound equals the number of multilinear factors in the product.
        let mut factors = Vec::with_capacity(3 + bit_eq_factors.len());
        factors.push(has_write);
        factors.push(inc_at_write_addr);
        factors.push(lt_table);
        factors.extend(bit_eq_factors);

        let degree_bound = factors.len();
        let core = ProductRoundOracle::new(factors, degree_bound);
        let inc_contribution = core.sum_over_hypercube();

        (Self { core }, inc_contribution)
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(TwistValEvalOracleSparse);

// ============================================================================
// Total-Increment Oracle (SPARSE version)
// ============================================================================

/// Total-increment oracle using SPARSE Inc representation.
///
/// Proves:
/// ```text
/// Σ_t Inc(r_addr, t)
/// ```
///
/// where:
/// ```text
/// Inc(r_addr, t) = has_write(t) * eq(wa_bits(t), r_addr) * inc_at_write_addr(t)
/// ```
///
/// This is used to cryptographically link consecutive chunks by showing that the
/// end-of-chunk value at a random address point equals the next chunk's init at
/// that same point.
pub struct TwistTotalIncOracleSparse {
    core: ProductRoundOracle,
}

impl TwistTotalIncOracleSparse {
    /// Create a new sparse total-inc oracle.
    ///
    /// # Returns
    /// - The oracle for sum-check over the time variables
    /// - The claimed total increment (= Σ_t Inc(r_addr, t))
    pub fn new(wa_bits: &[Vec<K>], has_write: Vec<K>, inc_at_write_addr: Vec<K>, r_addr: &[K]) -> (Self, K) {
        let pow2_cycle = has_write.len();

        assert_eq!(wa_bits.len(), r_addr.len(), "wa_bits count must match r_addr length");
        assert_eq!(
            inc_at_write_addr.len(),
            pow2_cycle,
            "inc_at_write_addr length must match cycle domain"
        );

        let bit_eq_factors = build_bit_eq_factors(wa_bits, r_addr);

        // Sum-check oracle over:
        //   has_write(t) * inc_at_write_addr(t) * eq(wa_bits(t), r_addr)
        let mut factors = Vec::with_capacity(2 + bit_eq_factors.len());
        factors.push(has_write);
        factors.push(inc_at_write_addr);
        factors.extend(bit_eq_factors);

        let degree_bound = factors.len();
        let core = ProductRoundOracle::new(factors, degree_bound);
        let claimed_sum = core.sum_over_hypercube();

        (Self { core }, claimed_sum)
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(TwistTotalIncOracleSparse);

// ============================================================================
// Bit-Aware Read Check Oracle
// ============================================================================

/// Read-check oracle using bit-decomposed addresses:
///
/// Proves: Σ_t eq(r_cycle, t) * has_read(t) * eq(ra_bits_t, r_addr) * (Val(ra_t, t) - rv(t)) = 0
///
/// **Important**: The val_at_read_addr parameter contains Val(ra_t, t) - the memory value
/// at the **actual** read address ra_t for each step, NOT Val(r_addr, t) at the random point.
/// This is sound because:
/// - For steps where ra_t ≠ r_addr: eq(ra_bits_t, r_addr) = 0, so the term vanishes
/// - For steps where ra_t = r_addr: val_at_read_addr[t] = Val(ra_t, t) = Val(r_addr, t)
///
/// This design avoids needing to compute Val(r_addr, t) for all steps, which would
/// require expensive MLE evaluations.
pub struct TwistReadCheckOracle {
    core: ProductRoundOracle,
}

impl TwistReadCheckOracle {
    /// Create a read-check oracle with bit-decomposed addresses.
    ///
    /// # Arguments
    /// - `ra_bits`: d*ell bit columns for read addresses (flattened)
    /// - `val_at_read_addr`: Val(ra_t, t) - memory value at actual read address for each step
    /// - `rv`: Read values (what the VM observed)
    /// - `has_read`: Read flags
    /// - `r_cycle`: Random point for cycle dimension
    /// - `r_addr`: Random point for address dimension (ell_total = d*ell bits)
    pub fn new(
        ra_bits: &[Vec<K>],
        val_at_read_addr: Vec<K>,
        rv: Vec<K>,
        has_read: Vec<K>,
        r_cycle: &[K],
        r_addr: &[K],
    ) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        let n = val_at_read_addr.len();
        assert_eq!(n, pow2_cycle, "val_at_read_addr length must match cycle domain");
        assert_eq!(rv.len(), pow2_cycle, "rv length must match cycle domain");
        assert_eq!(has_read.len(), pow2_cycle, "has_read length must match cycle domain");
        assert_eq!(ra_bits.len(), r_addr.len(), "ra_bits count must match r_addr length");

        // (Val(ra_t, t) - rv(t)) - value at actual read address minus observed read value.
        // Note: val_at_read_addr contains Val(ra_t, t), the memory value at the actual read
        // address, NOT Val(r_addr, t) at the random challenge point.
        let diff: Vec<K> = val_at_read_addr
            .iter()
            .zip(rv.iter())
            .map(|(v, r)| *v - *r)
            .collect();

        #[cfg(feature = "debug-logs")]
        {
            use p3_field::PrimeField64;
            let format_k = |k: &K| -> String {
                let coeffs = k.as_coeffs();
                format!("K[{}, {}]", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
            };
            eprintln!("TwistReadCheckOracle::new()");
            eprintln!(
                "  pow2_cycle={}, r_cycle.len()={}, r_addr.len()={}",
                pow2_cycle,
                r_cycle.len(),
                r_addr.len()
            );
            eprintln!("  ra_bits.len()={}", ra_bits.len());

            let eq_cycle = build_eq_table(r_cycle);
            // Show first few values of each input
            eprintln!(
                "  val_at_read_addr[0..4]: [{}]",
                val_at_read_addr
                    .iter()
                    .take(4)
                    .map(|v| format_k(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "  rv[0..4]: [{}]",
                rv.iter()
                    .take(4)
                    .map(|v| format_k(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "  has_read[0..4]: [{}]",
                has_read
                    .iter()
                    .take(4)
                    .map(|v| format_k(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "  diff[0..4]: [{}]",
                diff.iter()
                    .take(4)
                    .map(|v| format_k(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "  eq_cycle[0..4]: [{}]",
                eq_cycle
                    .iter()
                    .take(4)
                    .map(|v| format_k(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            // Compute expected sum manually
            let mut manual_sum = K::ZERO;
            let eq_from_bits = compute_eq_from_bits(ra_bits, r_addr);
            for t in 0..pow2_cycle {
                let term = eq_cycle[t] * has_read[t] * diff[t] * eq_from_bits[t];
                manual_sum += term;
            }
            eprintln!("  Manual sum (should be 0): {}", format_k(&manual_sum));

            // Debug: show non-zero contributions
            let mut nonzero_count = 0;
            for t in 0..pow2_cycle.min(8) {
                let eq_from_bits_t = eq_from_bits[t];
                let term = eq_cycle[t] * has_read[t] * diff[t] * eq_from_bits_t;
                if term != K::ZERO || has_read[t] != K::ZERO {
                    eprintln!(
                        "    t={}: eq_cycle={}, has_read={}, diff={}, eq_bits={}, term={}",
                        t,
                        format_k(&eq_cycle[t]),
                        format_k(&has_read[t]),
                        format_k(&diff[t]),
                        format_k(&eq_from_bits_t),
                        format_k(&term)
                    );
                    nonzero_count += 1;
                }
            }
            if nonzero_count == 0 {
                eprintln!("    (all terms zero in first 8 steps)");
            }
        }

        let core = build_cycle_gate_diff_bits_oracle(r_cycle, has_read, diff, ra_bits, r_addr);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }
}

impl_round_oracle_via_core!(TwistReadCheckOracle);

// ============================================================================
// Bit-Aware Write Check Oracle
// ============================================================================

/// Write-check oracle using bit-decomposed addresses:
///
/// Proves: Σ_t eq(r_cycle, t) * has_write(t) * eq(wa_bits_t, r_addr) * (wv(t) - Val(wa_t, t) - Inc(wa_t, t)) = 0
///
/// **Important**: The val_at_write_addr and inc_at_write_addr parameters contain the memory
/// value and increment at the **actual** write address wa_t for each step, NOT at the random
/// point r_addr. This is sound because:
/// - For steps where wa_t ≠ r_addr: eq(wa_bits_t, r_addr) = 0, so the term vanishes
/// - For steps where wa_t = r_addr: the values match what would be computed at r_addr
///
/// The check verifies: wv(t) - Val(wa_t, t) = Inc(wa_t, t), i.e., the increment equals
/// the difference between the write value and the pre-write memory value.
pub struct TwistWriteCheckOracle {
    core: ProductRoundOracle,
}

impl TwistWriteCheckOracle {
    /// Create a write-check oracle with bit-decomposed addresses.
    pub fn new(
        wa_bits: &[Vec<K>],
        wv: Vec<K>,
        val_at_r_addr: Vec<K>,
        inc_at_r_addr: Vec<K>,
        has_write: Vec<K>,
        r_cycle: &[K],
        r_addr: &[K],
    ) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        let n = val_at_r_addr.len();
        assert_eq!(n, pow2_cycle, "val_at_r_addr length must match cycle domain");
        assert_eq!(wv.len(), pow2_cycle, "wv length must match cycle domain");
        assert_eq!(has_write.len(), pow2_cycle, "has_write length must match cycle domain");
        assert_eq!(wa_bits.len(), r_addr.len(), "wa_bits count must match r_addr length");
        assert_eq!(
            inc_at_r_addr.len(),
            pow2_cycle,
            "inc_at_r_addr length must match cycle domain"
        );

        // delta(t) = (wv - Val) - Inc
        let delta: Vec<K> = wv
            .iter()
            .zip(val_at_r_addr.iter())
            .zip(inc_at_r_addr.iter())
            .map(|((w, v), inc)| *w - *v - *inc)
            .collect();
        let core = build_cycle_gate_diff_bits_oracle(r_cycle, has_write, delta, wa_bits, r_addr);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }
}

impl_round_oracle_via_core!(TwistWriteCheckOracle);

// ============================================================================
// Address-Lane Oracles (addr rounds first, time summed)
// ============================================================================
//
// These are used by the "Phase 2" Route A integration to avoid materializing a
// time×addr table when Twist must share `r_time` with CCS:
//   1) Run an address-lane sum-check first (ell_addr rounds) to bind `r_addr`
//      and produce the *time-lane claimed sums* (one per check).
//   2) Run a time-lane sum-check (ell_n rounds) at the fixed `r_addr`.
//
// Concretely, these oracles implement the address-lane prefix of the same
// read/write check polynomials as the 2D oracles, but with the time variables
// summed internally. This keeps address rounds efficient and avoids allocating
// `pow2_time * pow2_addr`.
//
// Degree in each address variable is ≤ 2:
// - `Val_pre(addr, t)` is multilinear in `addr` (degree 1 per bit),
// - `eq(addr, bits(t))` is multilinear in `addr`,
// so their product has degree ≤ 2 per address bit.
//
// Variable order is little-endian address bits (bit 0 first), matching the
// rest of the module.

fn addrs_from_bits(bit_cols: &[Vec<K>]) -> Vec<usize> {
    let n = bit_cols.first().map(|c| c.len()).unwrap_or(0);
    let mut out = vec![0usize; n];
    for (b, col) in bit_cols.iter().enumerate() {
        for (t, &v) in col.iter().enumerate() {
            if v == K::ONE {
                out[t] |= 1usize << b;
            }
        }
    }
    out
}

fn fold_table_in_place(table: &mut Vec<K>, r: K) {
    let len = table.len();
    debug_assert!(len.is_power_of_two());
    if len <= 1 {
        return;
    }
    let half = len / 2;
    for i in 0..half {
        let f0 = table[2 * i];
        let f1 = table[2 * i + 1];
        table[i] = f0 + (f1 - f0) * r;
    }
    table.truncate(half);
}

fn update_prefix_weights_in_place(weights: &mut [K], addrs: &[usize], bit_idx: usize, r: K) {
    let r0 = K::ONE - r;
    for (w, &a) in weights.iter_mut().zip(addrs.iter()) {
        if ((a >> bit_idx) & 1) == 1 {
            *w *= r;
        } else {
            *w *= r0;
        }
    }
}

/// Address-lane prefix oracle for the Twist read-check.
///
/// Sums over time internally, and runs sum-check over address bits:
///   H(addr) = Σ_t χ_{r_cycle}(t)·has_read(t)·eq(addr, ra_bits(t))·(Val_pre(addr,t) - rv(t)).
///
/// The sum-check binds `addr = r_addr` and returns the time-lane claimed sum:
///   Σ_t χ_{r_cycle}(t)·has_read(t)·eq(r_addr, ra_bits(t))·(Val_pre(r_addr,t) - rv(t)).
pub struct TwistReadCheckAddrOracle {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    eq_cycle: Vec<K>,
    has_read: Vec<K>,
    rv: Vec<K>,

    has_write: Vec<K>,
    inc_at_write_addr: Vec<K>,

    ra_addrs: Vec<usize>,
    wa_addrs: Vec<usize>,

    init_fold: Vec<K>,
    ra_prefix_w: Vec<K>,
    wa_prefix_w: Vec<K>,
}

impl TwistReadCheckAddrOracle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init: Vec<K>,
        r_cycle: &[K],
        has_read: Vec<K>,
        rv: Vec<K>,
        ra_bits: &[Vec<K>],
        has_write: Vec<K>,
        wa_bits: &[Vec<K>],
        inc_at_write_addr: Vec<K>,
    ) -> Self {
        let pow2_time = 1usize << r_cycle.len();
        assert_eq!(has_read.len(), pow2_time, "has_read length must match time domain");
        assert_eq!(rv.len(), pow2_time, "rv length must match time domain");
        assert_eq!(has_write.len(), pow2_time, "has_write length must match time domain");
        assert_eq!(
            inc_at_write_addr.len(),
            pow2_time,
            "inc_at_write_addr length must match time domain"
        );

        let ell_addr = ra_bits.len();
        assert_eq!(wa_bits.len(), ell_addr, "wa_bits/ra_bits length mismatch");
        let pow2_addr = 1usize << ell_addr;
        assert_eq!(init.len(), pow2_addr, "init length must match address domain");

        for col in ra_bits {
            assert_eq!(col.len(), pow2_time, "ra_bits column length mismatch");
        }
        for col in wa_bits {
            assert_eq!(col.len(), pow2_time, "wa_bits column length mismatch");
        }

        let eq_cycle = build_eq_table(r_cycle);
        let ra_addrs = addrs_from_bits(ra_bits);
        let wa_addrs = addrs_from_bits(wa_bits);

        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            eq_cycle,
            has_read,
            rv,
            has_write,
            inc_at_write_addr,
            ra_addrs,
            wa_addrs,
            init_fold: init,
            ra_prefix_w: vec![K::ONE; pow2_time],
            wa_prefix_w: vec![K::ONE; pow2_time],
        }
    }
}

impl RoundOracle for TwistReadCheckAddrOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            // Constant polynomial; return value for all points.
            let mut ys = vec![K::ZERO; points.len()];
            let mut mem = self.init_fold.clone();
            for t in 0..self.eq_cycle.len() {
                let val = mem[0];
                let diff = val - self.rv[t];
                ys[0] += self.eq_cycle[t] * self.has_read[t] * self.ra_prefix_w[t] * diff;

                let has_w = self.has_write[t];
                if has_w != K::ZERO {
                    mem[0] += self.inc_at_write_addr[t] * has_w * self.wa_prefix_w[t];
                }
            }
            let val = ys[0];
            return vec![val; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];
        let mut mem = self.init_fold.clone();

        for t in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[t];
            let gate = self.has_read[t];
            if gate != K::ZERO {
                let ra = self.ra_addrs[t];
                let base = ra >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem[idx0];
                let v1 = mem[idx1];
                let dv = v1 - v0;
                let rv_t = self.rv[t];
                let prefix = self.ra_prefix_w[t];
                let bit = (ra >> bit_idx) & 1;

                for (i, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    let term = eq_t * gate * prefix * addr_factor * (val_x - rv_t);
                    ys[i] += term;
                }
            }

            let has_w = self.has_write[t];
            if has_w != K::ZERO {
                let wa = self.wa_addrs[t];
                let idx = wa >> bit_idx;
                mem[idx] += self.inc_at_write_addr[t] * has_w * self.wa_prefix_w[t];
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr.saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        fold_table_in_place(&mut self.init_fold, r);
        update_prefix_weights_in_place(&mut self.ra_prefix_w, &self.ra_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

/// Address-lane prefix oracle for the Twist write-check.
///
/// Sums over time internally, and runs sum-check over address bits:
///   H(addr) = Σ_t χ_{r_cycle}(t)·has_write(t)·eq(addr, wa_bits(t))·(wv(t) - Val_pre(addr,t) - inc(t)).
///
/// The sum-check binds `addr = r_addr` and returns the time-lane claimed sum:
///   Σ_t χ_{r_cycle}(t)·has_write(t)·eq(r_addr, wa_bits(t))·(wv(t) - Val_pre(r_addr,t) - inc(t)).
pub struct TwistWriteCheckAddrOracle {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    eq_cycle: Vec<K>,
    has_write: Vec<K>,
    wv: Vec<K>,
    inc_at_write_addr: Vec<K>,

    wa_addrs: Vec<usize>,

    init_fold: Vec<K>,
    wa_prefix_w: Vec<K>,
}

impl TwistWriteCheckAddrOracle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init: Vec<K>,
        r_cycle: &[K],
        has_write: Vec<K>,
        wv: Vec<K>,
        wa_bits: &[Vec<K>],
        inc_at_write_addr: Vec<K>,
    ) -> Self {
        let pow2_time = 1usize << r_cycle.len();
        assert_eq!(has_write.len(), pow2_time, "has_write length must match time domain");
        assert_eq!(wv.len(), pow2_time, "wv length must match time domain");
        assert_eq!(
            inc_at_write_addr.len(),
            pow2_time,
            "inc_at_write_addr length must match time domain"
        );

        let ell_addr = wa_bits.len();
        let pow2_addr = 1usize << ell_addr;
        assert_eq!(init.len(), pow2_addr, "init length must match address domain");
        for col in wa_bits {
            assert_eq!(col.len(), pow2_time, "wa_bits column length mismatch");
        }

        let eq_cycle = build_eq_table(r_cycle);
        let wa_addrs = addrs_from_bits(wa_bits);

        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            eq_cycle,
            has_write,
            wv,
            inc_at_write_addr,
            wa_addrs,
            init_fold: init,
            wa_prefix_w: vec![K::ONE; pow2_time],
        }
    }
}

impl RoundOracle for TwistWriteCheckAddrOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = self.init_fold.clone();
            let mut sum = K::ZERO;
            for t in 0..self.eq_cycle.len() {
                let val = mem[0];
                let delta = self.wv[t] - val - self.inc_at_write_addr[t];
                sum += self.eq_cycle[t] * self.has_write[t] * self.wa_prefix_w[t] * delta;

                let has_w = self.has_write[t];
                if has_w != K::ZERO {
                    mem[0] += self.inc_at_write_addr[t] * has_w * self.wa_prefix_w[t];
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];
        let mut mem = self.init_fold.clone();

        for t in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[t];
            let gate = self.has_write[t];
            if gate != K::ZERO {
                let wa = self.wa_addrs[t];
                let base = wa >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem[idx0];
                let v1 = mem[idx1];
                let dv = v1 - v0;
                let wv_t = self.wv[t];
                let inc_t = self.inc_at_write_addr[t];
                let prefix = self.wa_prefix_w[t];
                let bit = (wa >> bit_idx) & 1;

                for (i, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    let term = eq_t * gate * prefix * addr_factor * (wv_t - val_x - inc_t);
                    ys[i] += term;
                }
            }

            let has_w = self.has_write[t];
            if has_w != K::ZERO {
                let wa = self.wa_addrs[t];
                let idx = wa >> bit_idx;
                mem[idx] += self.inc_at_write_addr[t] * has_w * self.wa_prefix_w[t];
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr.saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        fold_table_in_place(&mut self.init_fold, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Index Adapter Oracle (IDX→OH Bridge)
// ============================================================================

/// Index adapter oracle: proves consistency between committed bit columns
/// and the conceptual one-hot MLE evaluations.
///
/// Without `has_lookup`: Proves Σ_t eq(r_cycle, t) * eq(bits_t, r_addr) = claimed_value
/// With `has_lookup`: Proves Σ_t eq(r_cycle, t) * has_lookup(t) * eq(bits_t, r_addr) = claimed_value
///
/// The `has_lookup` gated version is REQUIRED for Shout to be sound: the adapter must
/// prove the same weight polynomial that the lookup sum-check uses, otherwise non-lookup
/// steps with addr bits = 0 would contribute spurious weight at address 0.
pub struct IndexAdapterOracle {
    core: ProductRoundOracle,
}

impl IndexAdapterOracle {
    /// Create an ungated adapter oracle (for Twist use).
    /// Proves: Σ_t eq(r_cycle, t) * eq(bits_t, r_addr) = claimed_value
    pub fn new(bits: &[Vec<K>], r_cycle: &[K], r_addr: &[K]) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        assert_eq!(bits.len(), r_addr.len(), "bits count must match r_addr length");
        for col in bits {
            assert_eq!(col.len(), pow2_cycle, "bit column length must match cycle domain");
        }
        let core = build_cycle_bits_oracle(r_cycle, bits, r_addr);
        Self { core }
    }

    /// Create a gated adapter oracle (for Shout use).
    /// Proves: Σ_t eq(r_cycle, t) * has_lookup(t) * eq(bits_t, r_addr) = claimed_value
    ///
    /// This matches the weight polynomial used by AddressLookupOracle, ensuring
    /// the adapter proves the same quantity that the lookup check verifies against.
    pub fn new_with_gate(bits: &[Vec<K>], has_lookup: &[K], r_cycle: &[K], r_addr: &[K]) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        assert_eq!(bits.len(), r_addr.len(), "bits count must match r_addr length");
        assert_eq!(
            has_lookup.len(),
            pow2_cycle,
            "has_lookup length must match cycle domain"
        );
        for col in bits {
            assert_eq!(col.len(), pow2_cycle, "bit column length must match cycle domain");
        }
        let core = build_cycle_gate_bits_oracle(r_cycle, has_lookup.to_vec(), bits, r_addr);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(IndexAdapterOracle);

// ============================================================================
// Bitness Oracle (proves bit columns are binary)
// ============================================================================

/// Bitness oracle: proves that a column contains only 0/1 values.
///
/// Proves: Σ_t eq(r_cycle, t) * b(t) * (b(t) - 1) = 0
pub struct BitnessOracle {
    core: ProductRoundOracle,
}

impl BitnessOracle {
    pub fn new(bit_col: Vec<K>, r_cycle: &[K]) -> Self {
        let core = build_cycle_bitness_oracle(r_cycle, bit_col);
        Self { core }
    }
}

impl_round_oracle_via_core!(BitnessOracle);

// ============================================================================
// Address-Domain Lookup Oracle (New Architecture)
// ============================================================================

/// Address-domain lookup oracle for Shout.
///
/// Uses the identity:
///   val̃(r_cycle) = Σ_{a∈{0,1}^{ℓ_A}} Table(a) · Ã(r_cycle, a)
///
/// where Ã(r_cycle, a) is the MLE of the one-hot adapter matrix A(t,a) = 1[a = addr(t)].
///
/// The sum-check is over address variables (ℓ_A rounds), not time variables.
/// At the end, the verifier checks: S_final = Tablẽ(r_addr) · Ã(r_cycle, r_addr)
///
/// ## Advantages over time-domain approach:
/// - No need to commit `table_at_addr[t]` column
/// - Direct verification using table MLE (public) and adapter evaluation
/// - Sum-check domain is address space (often smaller than time)
pub struct AddressLookupOracle {
    /// Product oracle for Table(a) · weight(a) over address space
    core: ProductRoundOracle,
}

impl AddressLookupOracle {
    /// Create a new address-domain lookup oracle.
    ///
    /// # Parameters
    /// - `addr_bits`: Address bit columns (ell_addr total, each length pow2_cycle)
    /// - `has_lookup`: Indicator column (1 when a lookup occurs at that step)
    /// - `table`: The full lookup table (length = table_size)
    /// - `r_cycle`: Random challenge point for the cycle dimension
    /// - `ell_addr`: Number of address bits (determines sum-check domain size)
    ///
    /// # Returns
    /// - The oracle
    /// - The claimed sum (= val̃(r_cycle), to be checked against ME opening)
    ///
    /// # How it works
    /// 1. Build weight table: weight[addr] = Σ_{t: addr(t)=addr AND has_lookup(t)=1} eq(r_cycle, t)
    /// 2. The polynomial P(a) = Table(a) · weight(a) is degree 2 per variable
    /// 3. Sum-check proves: Σ_a P(a) = claimed_sum
    /// 4. Final check: S_final = Tablẽ(r_addr) · weight̃(r_addr)
    ///    where weight̃(r_addr) = Ã(r_cycle, r_addr) (the adapter evaluation)
    pub fn new(addr_bits: &[Vec<K>], has_lookup: &[K], table: &[K], r_cycle: &[K], ell_addr: usize) -> (Self, K) {
        let pow2_cycle = 1usize << r_cycle.len();
        let pow2_addr = 1usize << ell_addr;

        assert_eq!(addr_bits.len(), ell_addr, "addr_bits count must match ell_addr");
        for col in addr_bits {
            assert_eq!(col.len(), pow2_cycle, "bit column length must match cycle domain");
        }
        assert_eq!(
            has_lookup.len(),
            pow2_cycle,
            "has_lookup length must match cycle domain"
        );

        // Build eq(r_cycle, ·) table for time domain
        let eq_cycle_table = build_eq_table(r_cycle);

        // Build weight table: weight[addr] = Σ_{t: addr(t)=addr} has_lookup(t) · eq(r_cycle, t)
        // This represents Ã(r_cycle, a) on the Boolean address hypercube
        //
        // SECURITY FIX: Use scalar multiplication `has_lookup[t] * weight_t` instead of
        // conditional branching `if has_lookup[t] != K::ONE`. This allows has_lookup to be
        // any field element (the bitness sum-check proves it's binary), and correctly
        // handles the scalar case for extension field evaluations during sum-check.
        let mut weight_table = vec![K::ZERO; pow2_addr];
        let mut claimed_sum = K::ZERO; // Will also compute Σ_a Table(a) · weight(a)

        for t in 0..pow2_cycle {
            // Gate by has_lookup as a scalar (not a boolean check)
            // Bitness is proven separately, so has_lookup ∈ {0, 1} on hypercube
            let gate = has_lookup[t];
            if gate == K::ZERO {
                continue; // Skip zeros for efficiency (but scalar multiplication would also work)
            }

            // Decode address from bit columns at time t
            let mut addr_t: usize = 0;
            for b in 0..ell_addr {
                if addr_bits[b][t] == K::ONE {
                    addr_t |= 1 << b;
                }
            }

            // Add gated eq(r_cycle, t) weight to this address bucket
            let weight_t = eq_cycle_table[t] * gate;
            if addr_t < pow2_addr {
                weight_table[addr_t] += weight_t;
            }
        }

        // Compute claimed sum = Σ_a Table(a) · weight(a)
        // This should equal val̃(r_cycle) for correct lookups
        for addr in 0..pow2_addr.min(table.len()) {
            claimed_sum += table[addr] * weight_table[addr];
        }

        // Convert table to K and pad to pow2_addr
        let mut table_k: Vec<K> = table.iter().map(|&t| t).collect();
        table_k.resize(pow2_addr, K::ZERO);

        // P(a) = Table(a) · weight(a) - product of two multilinear functions
        // Degree = 2 per variable (1 from each factor)
        let core = ProductRoundOracle::new(vec![table_k, weight_table], 2);

        (Self { core }, claimed_sum)
    }

    /// Get the final value after all rounds (should equal Tablẽ(r_addr) · Ã(r_cycle, r_addr))
    pub fn final_value(&self) -> Option<K> {
        self.core.value()
    }

    /// Get the challenges accumulated during sum-check (= r_addr)
    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(AddressLookupOracle);

/// Compute the MLE of a table at a random point.
///
/// Tablẽ(r) = Σ_{a∈{0,1}^ℓ} Table[a] · eq(r, a)
pub fn table_mle_eval(table: &[K], r_addr: &[K]) -> K {
    let ell = r_addr.len();
    let pow2 = 1usize << ell;

    let mut result = K::ZERO;
    for (idx, &val) in table.iter().enumerate().take(pow2) {
        // eq(r, idx) = χ_r[idx]
        let weight = crate::mle::chi_at_index(r_addr, idx);
        result += val * weight;
    }
    result
}

/// Compute adapter evaluation Ã(r_cycle, r_addr) from committed bit columns.
///
/// Ã(r_cycle, r_addr) = Σ_t eq(r_cycle, t) · Π_j eq(r_addr_j, bit_j(t))
///
/// For correct bit columns (has_lookup masking), this equals the sum-check
/// final value of the adapter oracle.
pub fn adapter_eval_from_bits(addr_bits: &[Vec<K>], has_lookup: &[K], r_cycle: &[K], r_addr: &[K]) -> K {
    let pow2_cycle = 1usize << r_cycle.len();
    let ell_addr = addr_bits.len();

    assert_eq!(ell_addr, r_addr.len(), "addr_bits count must match r_addr length");

    let eq_cycle_table = build_eq_table(r_cycle);

    let mut result = K::ZERO;
    for t in 0..pow2_cycle {
        // Weight from cycle eq polynomial
        let mut weight = eq_cycle_table[t] * has_lookup[t];

        // Multiply by eq(r_addr_j, bit_j(t)) for each address bit
        for j in 0..ell_addr {
            let bit_jt = addr_bits[j][t];
            let r_j = r_addr[j];
            weight *= eq_bit_affine(bit_jt, r_j);
        }

        result += weight;
    }
    result
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build Val tables used by the Val-evaluation oracle.
///
/// **Optimization**: Uses sparse chi computation O(k * ℓ) instead of O(2^ℓ).
pub fn val_tables_from_inc<F: Field + Copy + Into<K>>(
    inc: &[F],
    k: usize,
    steps: usize,
    pow2_addr: usize,
    pow2_cycle: usize,
    r_addr: &[K],
    r_cycle: &[K],
) -> (Vec<K>, Vec<K>) {
    let ell_k = log2_pow2(pow2_addr);
    let ell_t = log2_pow2(pow2_cycle);
    assert_eq!(ell_k, r_addr.len(), "r_addr length mismatch");
    assert_eq!(ell_t, r_cycle.len(), "r_cycle length mismatch");

    // Sparse computation: only compute chi for cells 0..k, not full 2^ℓ
    let mut inc_at_addr = vec![K::ZERO; pow2_cycle];
    for cell in 0..k.min(pow2_addr) {
        // O(ℓ) per cell instead of O(2^ℓ) total
        let weight = crate::mle::chi_at_index(r_addr, cell);
        if weight == K::ZERO {
            continue;
        }
        // Note: j is used to compute idx = cell * steps + j, so index loop is clearer
        #[allow(clippy::needless_range_loop)]
        for j in 0..steps.min(pow2_cycle) {
            let idx = cell * steps + j;
            if idx < inc.len() {
                let inc_k: K = inc[idx].into();
                inc_at_addr[j] += inc_k * weight;
            }
        }
    }

    let lt_table = build_lt_table(r_cycle);
    (inc_at_addr, lt_table)
}

/// Build Inc(r_addr, t) table from flattened inc matrix.
/// Returns a vector of length pow2_cycle with Inc evaluated at r_addr for each timestep.
///
/// **Optimization**: Uses sparse chi computation O(k * ℓ) instead of O(2^ℓ) where ℓ = |r_addr|.
/// This is critical for scalability with large address spaces.
pub fn build_inc_at_r_addr<F: Field + Copy + Into<K>>(
    inc_flat: &[F],
    k: usize,
    steps: usize,
    pow2_cycle: usize,
    r_addr: &[K],
) -> Vec<K> {
    let mut result = vec![K::ZERO; pow2_cycle];

    // Sparse computation: only compute chi for cells 0..k, not the full 2^ℓ address space
    for cell in 0..k {
        // O(ℓ) per cell instead of O(2^ℓ) total
        let weight = crate::mle::chi_at_index(r_addr, cell);
        if weight == K::ZERO {
            continue;
        }
        // Note: j is used to compute idx = cell * steps + j, so index loop is clearer
        #[allow(clippy::needless_range_loop)]
        for j in 0..steps.min(pow2_cycle) {
            let idx = cell * steps + j;
            if idx < inc_flat.len() {
                result[j] += F::into(inc_flat[idx]) * weight;
            }
        }
    }
    result
}

/// Build Inc(r_addr, t) table using SPARSE representation.
///
/// This is a soundness-critical function that computes Inc(r_addr, t) from:
/// - `wa_bits[b][t]`: write address bit b at step t
/// - `has_write[t]`: whether step t has a write
/// - `inc_at_write_addr[t]`: the increment value at step t (if writing)
///
/// The formula is:
/// ```text
/// Inc(r_addr, t) = has_write(t) * eq(wa_bits(t), r_addr) * inc_at_write_addr(t)
/// ```
///
/// where `eq(wa_bits(t), r_addr) = Π_b eq(wa_bits[b][t], r_addr[b])`.
///
/// This sparse representation is equivalent to the full Inc matrix but:
/// - Avoids committing a k × steps matrix (which caused X pollution and width issues)
/// - Uses only O(steps) data instead of O(k × steps)
/// - Correctly aligns all data at the same column offset
pub fn build_inc_at_r_addr_sparse(
    wa_bits: &[Vec<K>],
    has_write: &[K],
    inc_at_write_addr: &[K],
    r_addr: &[K],
    pow2_cycle: usize,
) -> Vec<K> {
    assert_eq!(wa_bits.len(), r_addr.len(), "wa_bits count must match r_addr length");
    assert_eq!(has_write.len(), pow2_cycle, "has_write length must match pow2_cycle");
    assert_eq!(
        inc_at_write_addr.len(),
        pow2_cycle,
        "inc_at_write_addr length must match pow2_cycle"
    );
    for (b, wa_bit) in wa_bits.iter().enumerate() {
        assert_eq!(wa_bit.len(), pow2_cycle, "wa_bits[{b}] length must match pow2_cycle");
    }

    let mut result = vec![K::ZERO; pow2_cycle];

    for t in 0..pow2_cycle {
        // Skip if no write at this step
        if has_write[t] == K::ZERO {
            continue;
        }

        // Compute eq(wa_bits(t), r_addr) = Π_b eq(wa_bits[b][t], r_addr[b])
        // Using the formula: eq(b, u) = b*(2u-1) + (1-u) for binary b
        // Since we've proven bitness, wa_bits[b][t] ∈ {0, 1}
        let mut eq_addr = K::ONE;
        for (b, wa_bit) in wa_bits.iter().enumerate() {
            let bit = wa_bit[t];
            let u = r_addr[b];
            eq_addr *= eq_bit_affine(bit, u);
        }

        // Inc(r_addr, t) = has_write(t) * eq(wa_bits(t), r_addr) * inc_at_write_addr(t)
        result[t] = has_write[t] * eq_addr * inc_at_write_addr[t];
    }

    result
}

/// Build Val(r_addr, t) table: prefix sums of Inc(r_addr, t) starting from init_vals.
///
/// **Optimization**: Uses sparse chi computation O(k * ℓ) instead of O(2^ℓ).
pub fn build_val_at_r_addr<F: Field + Copy + Into<K>>(
    inc_flat: &[F],
    init_vals: &[F],
    k: usize,
    steps: usize,
    pow2_cycle: usize,
    r_addr: &[K],
) -> Vec<K> {
    // Compute initial value at r_addr using sparse chi computation
    let mut init_at_r_addr = K::ZERO;
    for (cell, &init_val) in init_vals.iter().enumerate().take(k) {
        // O(ℓ) per cell instead of building full chi table
        let weight = crate::mle::chi_at_index(r_addr, cell);
        init_at_r_addr += F::into(init_val) * weight;
    }

    // Build Inc(r_addr, t) - already uses sparse computation
    let inc_at_r_addr = build_inc_at_r_addr(inc_flat, k, steps, pow2_cycle, r_addr);

    // Val(r_addr, t) = init + Σ_{t' < t} Inc(r_addr, t')
    let mut result = vec![K::ZERO; pow2_cycle];
    let mut acc = init_at_r_addr;
    for j in 0..pow2_cycle {
        result[j] = acc;
        if j < steps {
            acc += inc_at_r_addr[j];
        }
    }
    result
}

pub fn build_eq_table(target: &[K]) -> Vec<K> {
    let ell = target.len();
    let mut out = Vec::with_capacity(1usize << ell);
    for mask in 0..(1usize << ell) {
        let bits = mask_to_bits(mask, ell);
        out.push(eq_points(&bits, target));
    }
    out
}

pub(crate) fn build_lt_table(r_cycle: &[K]) -> Vec<K> {
    let ell = r_cycle.len();
    let mut out = Vec::with_capacity(1usize << ell);
    for mask in 0..(1usize << ell) {
        let bits = mask_to_bits(mask, ell);
        out.push(lt_eval(&bits, r_cycle));
    }
    out
}

fn mask_to_bits(mask: usize, ell: usize) -> Vec<K> {
    (0..ell)
        .map(|i| if (mask >> i) & 1 == 1 { K::ONE } else { K::ZERO })
        .collect()
}

fn log2_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    debug_assert!(n.is_power_of_two(), "expected power of two, got {n}");
    n.trailing_zeros() as usize
}

// ============================================================================
// Route A helpers
// ============================================================================

/// Lazy bitness oracle for Route A batched sum-check.
///
/// Proves: `Σ_t χ_{r_cycle}(t) * col(t) * (col(t) - 1) = 0` (i.e., col is binary)
/// via a random linear combination over `t`.
pub struct LazyBitnessOracle {
    core: ProductRoundOracle,
}

impl LazyBitnessOracle {
    pub fn new(col: Vec<K>) -> Self {
        let core = build_bitness_oracle(col);
        Self { core }
    }

    pub fn new_with_cycle(r_cycle: &[K], col: Vec<K>) -> Self {
        let core = build_cycle_bitness_oracle(r_cycle, col);
        Self { core }
    }

    pub fn compute_claim(&self) -> K {
        self.core.sum_over_hypercube()
    }

    /// Return the current folded value after all rounds have been folded.
    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }
}

impl_round_oracle_via_core!(LazyBitnessOracle);
