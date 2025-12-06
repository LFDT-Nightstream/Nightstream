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

use crate::mle::{build_chi_table, eq_points, lt_eval};
#[cfg(feature = "debug-logs")]
use neo_math::KExtensions;
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::{Field, PrimeCharacteristicRing};

// ============================================================================
// Core ProductRoundOracle
// ============================================================================

/// Helper that runs sumcheck for a product of multilinear factors.
///
/// Each factor table is a length-2^ℓ vector enumerating the factor on the
/// Boolean hypercube, using little-endian bit order.
pub struct ProductRoundOracle {
    factors: Vec<Vec<K>>,
    total_rounds: usize,
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
            total_rounds,
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
        self.total_rounds
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
    let c1 = r_bit + r_bit - K::ONE; // 2r - 1
    let c0 = K::ONE - r_bit; // 1 - r
    bit_col.iter().map(|&b| b * c1 + c0).collect()
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
    let n = bit_cols.first().map(|c| c.len()).unwrap_or(0);
    let mut result = vec![K::ONE; n];

    for (col, &r) in bit_cols.iter().zip(r_addr.iter()) {
        let c1 = r + r - K::ONE;
        let c0 = K::ONE - r;
        for (i, &b) in col.iter().enumerate() {
            result[i] *= b * c1 + c0;
        }
    }
    result
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

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl RoundOracle for TwistValEvalOracle {
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

// ============================================================================
// Bit-Aware Read Check Oracle
// ============================================================================

/// Read-check oracle using bit-decomposed addresses:
/// Proves: Σ_t eq(r_cycle, t) * has_read(t) * eq(ra_bits_t, r_addr) * (Val(r_addr, t) - rv(t)) = 0
///
/// This integrates the IDX→OH adapter directly into the read check.
pub struct TwistReadCheckOracle {
    core: ProductRoundOracle,
}

impl TwistReadCheckOracle {
    /// Create a read-check oracle with bit-decomposed addresses.
    ///
    /// # Arguments
    /// - `ra_bits`: d*ell bit columns for read addresses (flattened)
    /// - `val_at_r_addr`: Pre-computed Val(r_addr, t) for each step
    /// - `rv`: Read values
    /// - `has_read`: Read flags
    /// - `r_cycle`: Random point for cycle dimension
    /// - `r_addr`: Random point for address dimension (ell_total = d*ell bits)
    pub fn new(
        ra_bits: &[Vec<K>],
        val_at_r_addr: Vec<K>,
        rv: Vec<K>,
        has_read: Vec<K>,
        r_cycle: &[K],
        r_addr: &[K],
    ) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        let n = val_at_r_addr.len();
        assert_eq!(n, pow2_cycle, "val_at_r_addr length must match cycle domain");
        assert_eq!(rv.len(), pow2_cycle, "rv length must match cycle domain");
        assert_eq!(has_read.len(), pow2_cycle, "has_read length must match cycle domain");
        assert_eq!(ra_bits.len(), r_addr.len(), "ra_bits count must match r_addr length");

        // Build factors:
        // 1. eq(r_cycle, t)
        let eq_cycle = build_eq_table(r_cycle);

        // 2. has_read(t)
        // 3. (Val(r_addr, t) - rv(t))
        let diff: Vec<K> = val_at_r_addr
            .iter()
            .zip(rv.iter())
            .map(|(v, r)| *v - *r)
            .collect();

        // 4. eq(ra_bits_t, r_addr) = product of bit-eq factors
        let bit_eq_factors = build_bit_eq_factors(ra_bits, r_addr);

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

            // Show first few values of each input
            eprintln!(
                "  val_at_r_addr[0..4]: [{}]",
                val_at_r_addr
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

        // Combine all factors
        let mut factors = vec![eq_cycle, has_read, diff];
        factors.extend(bit_eq_factors);

        // Degree = number of factors
        let degree = factors.len();
        let core = ProductRoundOracle::new(factors, degree);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }
}

impl RoundOracle for TwistReadCheckOracle {
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

// ============================================================================
// Bit-Aware Write Check Oracle
// ============================================================================

/// Write-check oracle using bit-decomposed addresses:
/// Proves: Σ_t eq(r_cycle, t) * has_write(t) * eq(wa_bits_t, r_addr) * (wv(t) - Val(r_addr, t)) = Inc(r_addr, t)
///
/// More precisely, both sides are summed to the same expected value.
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

        // Build factors:
        // 1. eq(r_cycle, t)
        let eq_cycle = build_eq_table(r_cycle);

        // 2. has_write(t)
        // 3. delta(t) = (wv - Val) - Inc
        let delta: Vec<K> = wv
            .iter()
            .zip(val_at_r_addr.iter())
            .zip(inc_at_r_addr.iter())
            .map(|((w, v), inc)| *w - *v - *inc)
            .collect();

        // 4. eq(wa_bits_t, r_addr)
        let bit_eq_factors = build_bit_eq_factors(wa_bits, r_addr);

        let mut factors = vec![eq_cycle, has_write, delta];
        factors.extend(bit_eq_factors);

        let degree = factors.len();
        let core = ProductRoundOracle::new(factors, degree);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }
}

impl RoundOracle for TwistWriteCheckOracle {
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

// ============================================================================
// Index Adapter Oracle (IDX→OH Bridge)
// ============================================================================

/// Index adapter oracle: proves consistency between committed bit columns
/// and the conceptual one-hot MLE evaluations.
///
/// Proves: Σ_t eq(r_cycle, t) * eq(bits_t, r_addr) = claimed_value
///
/// This is used when Twist/Shout need an MLE opening of the conceptual
/// one-hot matrix at (r_cycle, r_addr).
pub struct IndexAdapterOracle {
    core: ProductRoundOracle,
}

impl IndexAdapterOracle {
    pub fn new(bits: &[Vec<K>], r_cycle: &[K], r_addr: &[K]) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        assert_eq!(bits.len(), r_addr.len(), "bits count must match r_addr length");
        for col in bits {
            assert_eq!(col.len(), pow2_cycle, "bit column length must match cycle domain");
        }

        let eq_cycle = build_eq_table(r_cycle);
        let bit_eq_factors = build_bit_eq_factors(bits, r_addr);

        let mut factors = vec![eq_cycle];
        factors.extend(bit_eq_factors);

        let degree = factors.len();
        let core = ProductRoundOracle::new(factors, degree);
        Self { core }
    }

    pub fn current_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl RoundOracle for IndexAdapterOracle {
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
        let eq_cycle = build_eq_table(r_cycle);
        let minus_one: Vec<K> = bit_col.iter().map(|&b| b - K::ONE).collect();

        let core = ProductRoundOracle::new(vec![eq_cycle, bit_col, minus_one], 3);
        Self { core }
    }
}

impl RoundOracle for BitnessOracle {
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

// ============================================================================
// Shout Lookup Oracle
// ============================================================================

/// Shout lookup check oracle:
/// Proves: Σ_t eq(r_cycle, t) * has_lookup(t) * eq(ra_bits_t, r_addr) * Table(r_addr) = Σ_t eq(r_cycle, t) * has_lookup(t) * val(t)
///
/// Simplified: proves that for each lookup, the returned value matches the table.
pub struct ShoutLookupOracle {
    core: ProductRoundOracle,
}

impl ShoutLookupOracle {
    pub fn new(
        ra_bits: &[Vec<K>],
        has_lookup: Vec<K>,
        val: Vec<K>,
        table_at_r_addr: K,
        r_cycle: &[K],
        r_addr: &[K],
    ) -> Self {
        let pow2_cycle = 1usize << r_cycle.len();
        assert_eq!(ra_bits.len(), r_addr.len(), "ra_bits count must match r_addr length");
        assert_eq!(val.len(), pow2_cycle, "val length must match cycle domain");

        let eq_cycle = build_eq_table(r_cycle);
        let bit_eq_factors = build_bit_eq_factors(ra_bits, r_addr);

        // delta(t) = val(t) - Table(r_addr)
        let delta: Vec<K> = val.iter().map(|v| *v - table_at_r_addr).collect();

        let mut factors = vec![eq_cycle, has_lookup, delta];
        factors.extend(bit_eq_factors);

        let degree = factors.len();
        let core = ProductRoundOracle::new(factors, degree);
        Self { core }
    }
}

impl RoundOracle for ShoutLookupOracle {
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Build Val tables used by the Val-evaluation oracle.
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

    let chi_addr = build_chi_table(r_addr);
    let mut inc_at_addr = vec![K::ZERO; pow2_cycle];
    for cell in 0..pow2_addr {
        let weight = chi_addr[cell];
        if weight == K::ZERO {
            continue;
        }
        for j in 0..steps.min(pow2_cycle) {
            if cell < k {
                let idx = cell * steps + j;
                if idx < inc.len() {
                    let inc_k: K = inc[idx].into();
                    inc_at_addr[j] += inc_k * weight;
                }
            }
        }
    }

    let lt_table = build_lt_table(r_cycle);
    (inc_at_addr, lt_table)
}

/// Build Inc(r_addr, t) table from flattened inc matrix.
/// Returns a vector of length pow2_cycle with Inc evaluated at r_addr for each timestep.
pub fn build_inc_at_r_addr<F: Field + Copy + Into<K>>(
    inc_flat: &[F],
    k: usize,
    steps: usize,
    pow2_cycle: usize,
    r_addr: &[K],
) -> Vec<K> {
    let pow2_addr = 1usize << r_addr.len();
    let chi_addr = build_chi_table(r_addr);

    let mut result = vec![K::ZERO; pow2_cycle];
    for cell in 0..k.min(pow2_addr) {
        let weight = chi_addr[cell];
        if weight == K::ZERO {
            continue;
        }
        for j in 0..steps.min(pow2_cycle) {
            let idx = cell * steps + j;
            if idx < inc_flat.len() {
                result[j] += F::into(inc_flat[idx]) * weight;
            }
        }
    }
    result
}

/// Build Val(r_addr, t) table: prefix sums of Inc(r_addr, t) starting from init_vals.
pub fn build_val_at_r_addr<F: Field + Copy + Into<K>>(
    inc_flat: &[F],
    init_vals: &[F],
    k: usize,
    steps: usize,
    pow2_cycle: usize,
    r_addr: &[K],
) -> Vec<K> {
    let pow2_addr = 1usize << r_addr.len();
    let chi_addr = build_chi_table(r_addr);

    // Compute initial value at r_addr
    let mut init_at_r_addr = K::ZERO;
    for cell in 0..k.min(pow2_addr).min(init_vals.len()) {
        init_at_r_addr += F::into(init_vals[cell]) * chi_addr[cell];
    }

    // Build Inc(r_addr, t)
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
    debug_assert!(n.is_power_of_two(), "expected power of two, got {}", n);
    n.trailing_zeros() as usize
}

// ============================================================================
// Legacy oracles (deprecated, kept for compatibility)
// ============================================================================

/// One-hot booleanity oracle (DEPRECATED - use BitnessOracle instead)
pub struct OneHotBooleanOracle {
    core: ProductRoundOracle,
}

impl OneHotBooleanOracle {
    pub fn new(v_table: Vec<K>, r_cycle: &[K], pow2_addr: usize) -> Self {
        let eq_cycle = build_eq_table(r_cycle);
        let eq_cycle_table = broadcast_cycle(&eq_cycle, pow2_addr);
        let minus_one: Vec<K> = v_table.iter().map(|x| *x - K::ONE).collect();
        let core = ProductRoundOracle::new(vec![eq_cycle_table, v_table, minus_one], 3);
        Self { core }
    }
}

impl RoundOracle for OneHotBooleanOracle {
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

/// One-hot Hamming oracle (DEPRECATED)
pub struct OneHotHammingOracle {
    core: ProductRoundOracle,
}

impl OneHotHammingOracle {
    pub fn new(v_table: Vec<K>, r_cycle: &[K], pow2_addr: usize) -> Self {
        let eq_cycle = build_eq_table(r_cycle);
        let eq_cycle_table = broadcast_cycle(&eq_cycle, pow2_addr);
        let core = ProductRoundOracle::new(vec![eq_cycle_table, v_table], 2);
        Self { core }
    }
}

impl RoundOracle for OneHotHammingOracle {
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

pub(crate) fn broadcast_cycle(base: &[K], pow2_addr: usize) -> Vec<K> {
    let mut out = Vec::with_capacity(base.len() * pow2_addr);
    for &v in base.iter() {
        for _ in 0..pow2_addr {
            out.push(v);
        }
    }
    out
}

#[allow(dead_code)]
pub(crate) fn broadcast_addr(base: &[K], pow2_cycle: usize) -> Vec<K> {
    let mut out = Vec::with_capacity(base.len() * pow2_cycle);
    for _ in 0..pow2_cycle {
        out.extend_from_slice(base);
    }
    out
}
