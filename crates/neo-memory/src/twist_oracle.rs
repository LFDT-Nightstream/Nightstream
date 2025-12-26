//! Twist (and Shout) sumcheck oracles built from multilinear factor tables.
//!
//! This module provides oracles for the **index-bit addressing** architecture:
//! instead of materializing huge one-hot tables, we compute eq(bits, r_addr)
//! dynamically from committed bit columns.
//!
//! ## Key Oracles
//!
//! - `ProductRoundOracle`: Generic multilinear product sumcheck (used for address-domain checks)
//! - `AddressLookupOracle`: Shout address-domain lookup sumcheck
//! - `IndexAdapterOracleSparseTime`: Sparse-in-time IDX→OH adapter (time-domain)
//! - `LazyBitnessOracleSparseTime`: Sparse-in-time bitness checks (time-domain)
//! - `TwistReadCheckOracleSparseTime` / `TwistWriteCheckOracleSparseTime`: Twist time-lane checks (time-domain)
//! - `TwistValEvalOracleSparseTime` / `TwistTotalIncOracleSparseTime`: Twist val reconstruction (time-domain)

use crate::bit_ops::eq_bit_affine;
use crate::mle::lt_eval;
use crate::sparse_time::SparseIdxVec;
#[cfg(feature = "debug-logs")]
use neo_math::KExtensions;
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

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
// Sparse-in-time helpers and oracles (Track A)
// ============================================================================

#[inline]
fn eq_single(a: K, b: K) -> K {
    (K::ONE - a) * (K::ONE - b) + a * b
}

#[inline]
fn chi_at_bool_index(r: &[K], idx: usize) -> K {
    crate::mle::chi_at_index(r, idx)
}

/// Compute χ_{r_cycle}(t) children for the current time sumcheck round.
///
/// Variable order is little-endian (bit 0 first), matching `ProductRoundOracle`.
#[inline]
fn chi_cycle_children(r_cycle: &[K], bit_idx: usize, prefix_eq: K, pair_idx: usize) -> (K, K) {
    debug_assert!(bit_idx < r_cycle.len());

    // Higher bits (bit_idx+1..ell) come from pair_idx, little-endian.
    let mut suffix = K::ONE;
    let mut shift = 0usize;
    for b in (bit_idx + 1)..r_cycle.len() {
        let bit = (pair_idx >> shift) & 1;
        suffix *= if bit == 1 { r_cycle[b] } else { K::ONE - r_cycle[b] };
        shift += 1;
    }

    let r = r_cycle[bit_idx];
    let child0 = prefix_eq * (K::ONE - r) * suffix;
    let child1 = prefix_eq * r * suffix;
    (child0, child1)
}

fn gather_pairs_from_sparse(entries: &[(usize, K)]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(entries.len());
    let mut prev: Option<usize> = None;
    for &(idx, _v) in entries {
        let p = idx >> 1;
        if prev != Some(p) {
            out.push(p);
            prev = Some(p);
        }
    }
    out
}

/// Sparse Route A oracle for Shout value:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · val(t)
pub struct ShoutValueOracleSparse {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl ShoutValueOracleSparse {
    pub fn new(r_cycle: &[K], has_lookup: SparseIdxVec<K>, val: SparseIdxVec<K>) -> (Self, K) {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            let v = val.get(t);
            if v == K::ZERO {
                continue;
            }
            claim += chi_at_bool_index(r_cycle, t) * gate * v;
        }

        (
            Self {
                bit_idx: 0,
                r_cycle: r_cycle.to_vec(),
                prefix_eq: K::ONE,
                has_lookup,
                val,
                degree_bound: 3,
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }
}

impl RoundOracle for ShoutValueOracleSparse {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let v = self.prefix_eq * self.has_lookup.singleton_value() * self.val.singleton_value();
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let (chi0_base, chi1_base) = if self.bit_idx < self.r_cycle.len() {
            // Per-pair child weights depend on higher-bit assignment (pair index).
            (K::ZERO, K::ZERO)
        } else {
            (K::ZERO, K::ZERO)
        };
        let _ = (chi0_base, chi1_base);

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = chi0 + (chi1 - chi0) * x;
                let gate_x = gate0 + (gate1 - gate0) * x;
                let val_x = val0 + (val1 - val0) * x;
                ys[i] += chi_x * gate_x * val_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

#[inline]
fn interp(f0: K, f1: K, x: K) -> K {
    f0 + (f1 - f0) * x
}

fn fill_time_point(out: &mut [K], prefix: &[K], bit_idx: usize, bit_value: K, pair_idx: usize) {
    debug_assert!(bit_idx < out.len(), "bit_idx out of range");
    debug_assert_eq!(prefix.len(), bit_idx, "prefix length mismatch");

    out[..bit_idx].copy_from_slice(prefix);
    out[bit_idx] = bit_value;

    let mut shift = 0usize;
    for b in (bit_idx + 1)..out.len() {
        let bit = (pair_idx >> shift) & 1;
        out[b] = if bit == 1 { K::ONE } else { K::ZERO };
        shift += 1;
    }
}

fn lt_eval_at_bool_index(idx: usize, point: &[K]) -> K {
    let mut suffix = K::ONE;
    let mut acc = K::ZERO;
    for i in (0..point.len()).rev() {
        let bit = (idx >> i) & 1;
        if bit == 0 {
            acc += point[i] * suffix;
            suffix *= K::ONE - point[i];
        } else {
            suffix *= point[i];
        }
    }
    acc
}

fn val_pre_from_inc_terms(init_at_r_addr: K, inc_terms: &[(usize, K)], t_point: &[K]) -> K {
    let mut acc = init_at_r_addr;
    for &(u, inc_u) in inc_terms.iter() {
        acc += inc_u * lt_eval_at_bool_index(u, t_point);
    }
    acc
}

fn build_inc_terms_at_r_addr(
    wa_bits: &[SparseIdxVec<K>],
    has_write: &SparseIdxVec<K>,
    inc_at_write_addr: &SparseIdxVec<K>,
    r_addr: &[K],
) -> Vec<(usize, K)> {
    let mut out: Vec<(usize, K)> = Vec::new();
    for &(t, has_w) in has_write.entries() {
        let inc_t = inc_at_write_addr.get(t);
        if has_w == K::ZERO || inc_t == K::ZERO {
            continue;
        }

        let mut eq_addr = K::ONE;
        for (b, col) in wa_bits.iter().enumerate() {
            let bit = col.get(t);
            eq_addr *= eq_bit_affine(bit, r_addr[b]);
        }

        let inc_at_r_addr = has_w * inc_t * eq_addr;
        if inc_at_r_addr != K::ZERO {
            out.push((t, inc_at_r_addr));
        }
    }
    out
}

// ============================================================================
// Sparse-in-time Route A oracles (Track A, shared CPU bus)
// ============================================================================

/// Sparse Route A oracle for Shout adapter:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · Π_b eq(addr_bits_b(t), r_addr_b)
pub struct IndexAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    addr_bits: Vec<SparseIdxVec<K>>,
    r_addr: Vec<K>,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl IndexAdapterOracleSparseTime {
    pub fn new_with_gate(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        addr_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
    ) -> (Self, K) {
        let ell_n = r_cycle.len();
        let ell_addr = addr_bits.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(r_addr.len(), ell_addr);
        for (b, col) in addr_bits.iter().enumerate() {
            debug_assert_eq!(
                col.len(),
                1usize << ell_n,
                "addr_bits[{b}] length must match time domain"
            );
        }

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            let mut eq_addr = K::ONE;
            for (b, col) in addr_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }
            claim += chi_at_bool_index(r_cycle, t) * gate * eq_addr;
        }

        (
            Self {
                bit_idx: 0,
                r_cycle: r_cycle.to_vec(),
                prefix_eq: K::ONE,
                has_lookup,
                addr_bits,
                r_addr: r_addr.to_vec(),
                degree_bound: 2 + ell_addr,
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }
}

impl RoundOracle for IndexAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.addr_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let v = self.prefix_eq * self.has_lookup.singleton_value() * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            // Per-bit eq factors at the children (after any time folding).
            let mut eq0s: Vec<K> = Vec::with_capacity(self.addr_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.addr_bits.len());
            for (b, col) in self.addr_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let gate_x = interp(gate0, gate1, x);
                let mut prod = chi_x * gate_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        for col in self.addr_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for χ_{r_cycle}-weighted bitness:
///   Σ_t χ_{r_cycle}(t) · col(t) · (col(t) - 1)
pub struct LazyBitnessOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    col: SparseIdxVec<K>,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl LazyBitnessOracleSparseTime {
    pub fn new_with_cycle(r_cycle: &[K], col: SparseIdxVec<K>) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(col.len(), 1usize << ell_n);
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            col,
            degree_bound: 3,
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for LazyBitnessOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.col.len() == 1 {
            let b = self.col.singleton_value();
            let v = self.prefix_eq * b * (b - K::ONE);
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.col.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let b0 = self.col.get(child0);
            let b1 = self.col.get(child1);
            if b0 == K::ZERO && b1 == K::ZERO {
                continue;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let b_x = interp(b0, b1, x);
                ys[i] += chi_x * b_x * (b_x - K::ONE);
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.col.fold_round_in_place(r);
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for Twist read-check (time rounds only).
pub struct TwistReadCheckOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    degree_bound: usize,

    r_addr: Vec<K>,
    ra_bits: Vec<SparseIdxVec<K>>,
    has_read: SparseIdxVec<K>,
    rv: SparseIdxVec<K>,

    init_at_r_addr: K,
    inc_terms_at_r_addr: Vec<(usize, K)>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    challenges: Vec<K>,
}

impl TwistReadCheckOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: Vec<SparseIdxVec<K>>,
        // Write stream (for Val_pre).
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_read.len(), 1usize << ell_n);
        debug_assert_eq!(rv.len(), 1usize << ell_n);
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(ra_bits.len(), ell_addr);
        debug_assert_eq!(wa_bits.len(), ell_addr);

        let inc_terms_at_r_addr = build_inc_terms_at_r_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 3 + ell_addr,
            r_addr: r_addr.to_vec(),
            ra_bits,
            has_read,
            rv,
            init_at_r_addr,
            inc_terms_at_r_addr,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for TwistReadCheckOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_read.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.ra_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let t_point = self.challenges.as_slice();
            let val_pre = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, t_point);
            let diff = val_pre - self.rv.singleton_value();
            let v = self.prefix_eq * self.has_read.singleton_value() * diff * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_read.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_read.get(child0);
            let gate1 = self.has_read.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
            fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
            let val_pre0 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child0);
            let val_pre1 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child1);

            let diff0 = val_pre0 - self.rv.get(child0);
            let diff1 = val_pre1 - self.rv.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.ra_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.ra_bits.len());
            for (b, col) in self.ra_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let gate_x = interp(gate0, gate1, x);
                let diff_x = interp(diff0, diff1, x);
                let mut prod = chi_x * gate_x * diff_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_read.fold_round_in_place(r);
        self.rv.fold_round_in_place(r);
        for col in self.ra_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for Twist write-check (time rounds only).
pub struct TwistWriteCheckOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    degree_bound: usize,

    r_addr: Vec<K>,
    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    wv: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,

    init_at_r_addr: K,
    inc_terms_at_r_addr: Vec<(usize, K)>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    challenges: Vec<K>,
}

impl TwistWriteCheckOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(wv.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);

        let inc_terms_at_r_addr = build_inc_terms_at_r_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 3 + ell_addr,
            r_addr: r_addr.to_vec(),
            wa_bits,
            has_write,
            wv,
            inc_at_write_addr,
            init_at_r_addr,
            inc_terms_at_r_addr,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for TwistWriteCheckOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let t_point = self.challenges.as_slice();
            let val_pre = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, t_point);
            let delta = self.wv.singleton_value() - val_pre - self.inc_at_write_addr.singleton_value();
            let v = self.prefix_eq * self.has_write.singleton_value() * delta * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_write.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get(child0);
            let gate1 = self.has_write.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
            fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
            let val_pre0 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child0);
            let val_pre1 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child1);

            let delta0 = self.wv.get(child0) - val_pre0 - self.inc_at_write_addr.get(child0);
            let delta1 = self.wv.get(child1) - val_pre1 - self.inc_at_write_addr.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            for (b, col) in self.wa_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let gate_x = interp(gate0, gate1, x);
                let delta_x = interp(delta0, delta1, x);
                let mut prod = chi_x * gate_x * delta_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_write.fold_round_in_place(r);
        self.wv.fold_round_in_place(r);
        self.inc_at_write_addr.fold_round_in_place(r);
        for col in self.wa_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse time-domain val-evaluation oracle:
///   Σ_t has_write(t) · inc(t) · LT(t, r_time) · Π_b eq(wa_bit_b(t), r_addr_b)
pub struct TwistValEvalOracleSparseTime {
    bit_idx: usize,
    degree_bound: usize,

    r_time: Vec<K>,
    r_addr: Vec<K>,

    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    challenges: Vec<K>,
}

impl TwistValEvalOracleSparseTime {
    pub fn new(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
        r_time: &[K],
    ) -> (Self, K) {
        let ell_n = r_time.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);
        for (b, col) in wa_bits.iter().enumerate() {
            debug_assert_eq!(
                col.len(),
                1usize << ell_n,
                "wa_bits[{b}] length must match time domain"
            );
        }

        let mut claim = K::ZERO;
        for &(t, gate) in has_write.entries() {
            let inc_t = inc_at_write_addr.get(t);
            if gate == K::ZERO || inc_t == K::ZERO {
                continue;
            }
            let lt_t = lt_eval_at_bool_index(t, r_time);
            let mut eq_addr = K::ONE;
            for (b, col) in wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }
            claim += gate * inc_t * lt_t * eq_addr;
        }

        (
            Self {
                bit_idx: 0,
                degree_bound: 3 + ell_addr,
                r_time: r_time.to_vec(),
                r_addr: r_addr.to_vec(),
                wa_bits,
                has_write,
                inc_at_write_addr,
                t_child0: vec![K::ZERO; ell_n],
                t_child1: vec![K::ZERO; ell_n],
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }
}

impl RoundOracle for TwistValEvalOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let lt = lt_eval(&self.challenges, &self.r_time);
            let v = self.has_write.singleton_value() * self.inc_at_write_addr.singleton_value() * eq_addr * lt;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_write.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get(child0);
            let gate1 = self.has_write.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }
            let inc0 = self.inc_at_write_addr.get(child0);
            let inc1 = self.inc_at_write_addr.get(child1);

            fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
            fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
            let lt0 = lt_eval(&self.t_child0, &self.r_time);
            let lt1 = lt_eval(&self.t_child1, &self.r_time);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            for (b, col) in self.wa_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let gate_x = interp(gate0, gate1, x);
                let inc_x = interp(inc0, inc1, x);
                let lt_x = interp(lt0, lt1, x);
                let mut prod = gate_x * inc_x * lt_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_time.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.has_write.fold_round_in_place(r);
        self.inc_at_write_addr.fold_round_in_place(r);
        for col in self.wa_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse time-domain total-increment oracle:
///   Σ_t has_write(t) · inc(t) · Π_b eq(wa_bit_b(t), r_addr_b)
pub struct TwistTotalIncOracleSparseTime {
    degree_bound: usize,

    r_addr: Vec<K>,
    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,
}

impl TwistTotalIncOracleSparseTime {
    pub fn new(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
    ) -> (Self, K) {
        let ell_n = log2_pow2(has_write.len());
        let ell_addr = r_addr.len();
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);
        for (b, col) in wa_bits.iter().enumerate() {
            debug_assert_eq!(
                col.len(),
                1usize << ell_n,
                "wa_bits[{b}] length must match time domain"
            );
        }

        let mut claim = K::ZERO;
        for &(t, gate) in has_write.entries() {
            let inc_t = inc_at_write_addr.get(t);
            if gate == K::ZERO || inc_t == K::ZERO {
                continue;
            }
            let mut eq_addr = K::ONE;
            for (b, col) in wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }
            claim += gate * inc_t * eq_addr;
        }

        (
            Self {
                degree_bound: 2 + ell_addr,
                r_addr: r_addr.to_vec(),
                wa_bits,
                has_write,
                inc_at_write_addr,
            },
            claim,
        )
    }
}

impl RoundOracle for TwistTotalIncOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let v = self.has_write.singleton_value() * self.inc_at_write_addr.singleton_value() * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_write.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get(child0);
            let gate1 = self.has_write.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }
            let inc0 = self.inc_at_write_addr.get(child0);
            let inc1 = self.inc_at_write_addr.get(child1);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            for (b, col) in self.wa_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let gate_x = interp(gate0, gate1, x);
                let inc_x = interp(inc0, inc1, x);
                let mut prod = gate_x * inc_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        log2_pow2(self.has_write.len())
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.has_write.len() == 1 {
            return;
        }
        self.has_write.fold_round_in_place(r);
        self.inc_at_write_addr.fold_round_in_place(r);
        for col in self.wa_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
    }
}

// ============================================================================
// Val-Evaluation Oracle (SPARSE version)
// ============================================================================

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

// ============================================================================
// Address-lane oracles (Track A sparse-in-time variants)
// ============================================================================

fn addr_from_sparse_bits_at_time(bit_cols: &[SparseIdxVec<K>], t: usize) -> usize {
    let mut out = 0usize;
    for (b, col) in bit_cols.iter().enumerate() {
        if col.get(t) == K::ONE {
            out |= 1usize << b;
        }
    }
    out
}

fn merge_sparse_time_indices(a: &[(usize, K)], b: &[(usize, K)]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() || j < b.len() {
        let next = match (a.get(i), b.get(j)) {
            (Some(&(ai, _)), Some(&(bj, _))) => {
                if ai <= bj {
                    i += 1;
                    ai
                } else {
                    j += 1;
                    bj
                }
            }
            (Some(&(ai, _)), None) => {
                i += 1;
                ai
            }
            (None, Some(&(bj, _))) => {
                j += 1;
                bj
            }
            (None, None) => break,
        };
        if out.last().copied() != Some(next) {
            out.push(next);
        }
    }
    out
}

/// Sparse-in-time address-lane prefix oracle for the Twist read-check.
///
/// Same proof semantics as `TwistReadCheckAddrOracle`, but internal time iteration scales with
/// activity (`nnz(has_read) + nnz(has_write)`), not `T = 2^ell_n`.
pub struct TwistReadCheckAddrOracleSparseTime {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    mem_scratch: std::collections::HashMap<usize, K>,

    // Per-event (sparse time list) arrays, sorted by time index.
    eq_cycle: Vec<K>,
    has_read: Vec<K>,
    rv: Vec<K>,
    has_write: Vec<K>,
    inc_at_write_addr: Vec<K>,
    ra_addrs: Vec<usize>,
    wa_addrs: Vec<usize>,
    ra_prefix_w: Vec<K>,
    wa_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,
}

impl TwistReadCheckAddrOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init_sparse: Vec<(usize, K)>,
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: &[SparseIdxVec<K>],
        has_write: SparseIdxVec<K>,
        wa_bits: &[SparseIdxVec<K>],
        inc_at_write_addr: SparseIdxVec<K>,
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
        for (addr, _) in init_sparse.iter() {
            assert!(*addr < pow2_addr, "init address out of range");
        }
        for (b, col) in ra_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_time, "ra_bits[{b}] length mismatch");
        }
        for (b, col) in wa_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_time, "wa_bits[{b}] length mismatch");
        }

        let times = merge_sparse_time_indices(has_read.entries(), has_write.entries());
        let mut eq_cycle_out = Vec::with_capacity(times.len());
        let mut has_read_out = Vec::with_capacity(times.len());
        let mut rv_out = Vec::with_capacity(times.len());
        let mut has_write_out = Vec::with_capacity(times.len());
        let mut inc_out = Vec::with_capacity(times.len());
        let mut ra_addrs = Vec::with_capacity(times.len());
        let mut wa_addrs = Vec::with_capacity(times.len());

        for &t in times.iter() {
            eq_cycle_out.push(chi_at_bool_index(r_cycle, t));
            let hr = has_read.get(t);
            let hw = has_write.get(t);
            has_read_out.push(hr);
            rv_out.push(rv.get(t));
            has_write_out.push(hw);
            inc_out.push(inc_at_write_addr.get(t));

            ra_addrs.push(if hr != K::ZERO { addr_from_sparse_bits_at_time(ra_bits, t) } else { 0 });
            wa_addrs.push(if hw != K::ZERO { addr_from_sparse_bits_at_time(wa_bits, t) } else { 0 });
        }

        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            mem_scratch: std::collections::HashMap::with_capacity(init_addrs.len()),
            eq_cycle: eq_cycle_out,
            has_read: has_read_out,
            rv: rv_out,
            has_write: has_write_out,
            inc_at_write_addr: inc_out,
            ra_addrs,
            wa_addrs,
            ra_prefix_w: vec![K::ONE; times.len()],
            wa_prefix_w: vec![K::ONE; times.len()],
            init_prefix_w: vec![K::ONE; init_addrs.len()],
            init_addrs,
            init_vals,
        }
    }
}

impl RoundOracle for TwistReadCheckAddrOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            for i in 0..self.eq_cycle.len() {
                let eq_t = self.eq_cycle[i];
                let gate_r = self.has_read[i];
                if gate_r != K::ZERO {
                    sum += eq_t * gate_r * self.ra_prefix_w[i] * (mem - self.rv[i]);
                }

                let gate_w = self.has_write[i];
                if gate_w != K::ZERO {
                    mem += self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];

        self.mem_scratch.clear();
        let mem = &mut self.mem_scratch;
        for ((&addr, &val), &w) in self
            .init_addrs
            .iter()
            .zip(self.init_vals.iter())
            .zip(self.init_prefix_w.iter())
        {
            let idx = addr >> bit_idx;
            let contrib = val * w;
            if contrib != K::ZERO {
                *mem.entry(idx).or_insert(K::ZERO) += contrib;
            }
        }

        for i in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[i];

            let gate_r = self.has_read[i];
            if gate_r != K::ZERO {
                let ra = self.ra_addrs[i];
                let base = ra >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                let dv = v1 - v0;
                let rv_t = self.rv[i];
                let prefix = self.ra_prefix_w[i];
                let bit = (ra >> bit_idx) & 1;

                for (j, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    ys[j] += eq_t * gate_r * prefix * addr_factor * (val_x - rv_t);
                }
            }

            let gate_w = self.has_write[i];
            if gate_w != K::ZERO {
                let wa = self.wa_addrs[i];
                let idx = wa >> bit_idx;
                let delta = self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                if delta != K::ZERO {
                    *mem.entry(idx).or_insert(K::ZERO) += delta;
                }
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
        update_prefix_weights_in_place(&mut self.init_prefix_w, &self.init_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.ra_prefix_w, &self.ra_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

/// Sparse-in-time address-lane prefix oracle for the Twist write-check.
pub struct TwistWriteCheckAddrOracleSparseTime {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    mem_scratch: std::collections::HashMap<usize, K>,

    eq_cycle: Vec<K>,
    has_write: Vec<K>,
    wv: Vec<K>,
    inc_at_write_addr: Vec<K>,
    wa_addrs: Vec<usize>,
    wa_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,
}

impl TwistWriteCheckAddrOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init_sparse: Vec<(usize, K)>,
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        wa_bits: &[SparseIdxVec<K>],
        inc_at_write_addr: SparseIdxVec<K>,
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
        for (addr, _) in init_sparse.iter() {
            assert!(*addr < pow2_addr, "init address out of range");
        }
        for (b, col) in wa_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_time, "wa_bits[{b}] length mismatch");
        }

        let times: Vec<usize> = has_write.entries().iter().map(|(t, _)| *t).collect();
        let mut eq_cycle_out = Vec::with_capacity(times.len());
        let mut has_write_out = Vec::with_capacity(times.len());
        let mut wv_out = Vec::with_capacity(times.len());
        let mut inc_out = Vec::with_capacity(times.len());
        let mut wa_addrs = Vec::with_capacity(times.len());

        for &t in times.iter() {
            eq_cycle_out.push(chi_at_bool_index(r_cycle, t));
            let hw = has_write.get(t);
            has_write_out.push(hw);
            wv_out.push(wv.get(t));
            inc_out.push(inc_at_write_addr.get(t));
            wa_addrs.push(addr_from_sparse_bits_at_time(wa_bits, t));
        }

        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            mem_scratch: std::collections::HashMap::with_capacity(init_addrs.len()),
            eq_cycle: eq_cycle_out,
            has_write: has_write_out,
            wv: wv_out,
            inc_at_write_addr: inc_out,
            wa_addrs,
            init_prefix_w: vec![K::ONE; init_addrs.len()],
            init_addrs,
            init_vals,
            wa_prefix_w: vec![K::ONE; times.len()],
        }
    }
}

impl RoundOracle for TwistWriteCheckAddrOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            for i in 0..self.eq_cycle.len() {
                let delta = self.wv[i] - mem - self.inc_at_write_addr[i];
                sum += self.eq_cycle[i] * self.has_write[i] * self.wa_prefix_w[i] * delta;

                let gate_w = self.has_write[i];
                if gate_w != K::ZERO {
                    mem += self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];

        self.mem_scratch.clear();
        let mem = &mut self.mem_scratch;
        for ((&addr, &val), &w) in self
            .init_addrs
            .iter()
            .zip(self.init_vals.iter())
            .zip(self.init_prefix_w.iter())
        {
            let idx = addr >> bit_idx;
            let contrib = val * w;
            if contrib != K::ZERO {
                *mem.entry(idx).or_insert(K::ZERO) += contrib;
            }
        }

        for i in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[i];
            let gate = self.has_write[i];
            if gate != K::ZERO {
                let wa = self.wa_addrs[i];
                let base = wa >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                let dv = v1 - v0;
                let wv_t = self.wv[i];
                let inc_t = self.inc_at_write_addr[i];
                let prefix = self.wa_prefix_w[i];
                let bit = (wa >> bit_idx) & 1;

                for (j, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    ys[j] += eq_t * gate * prefix * addr_factor * (wv_t - val_x - inc_t);
                }
            }

            let gate_w = self.has_write[i];
            if gate_w != K::ZERO {
                let wa = self.wa_addrs[i];
                let idx = wa >> bit_idx;
                let delta = self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                if delta != K::ZERO {
                    *mem.entry(idx).or_insert(K::ZERO) += delta;
                }
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
        update_prefix_weights_in_place(&mut self.init_prefix_w, &self.init_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

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
    /// Create a new address-domain lookup oracle from sparse-in-time columns.
    ///
    /// Track A: proof statement/verifier semantics are unchanged; only time iteration is sparse.
    pub fn new(
        addr_bits: &[SparseIdxVec<K>],
        has_lookup: &SparseIdxVec<K>,
        table: &[K],
        r_cycle: &[K],
        ell_addr: usize,
    ) -> (Self, K) {
        let pow2_cycle = 1usize << r_cycle.len();
        let pow2_addr = 1usize << ell_addr;

        assert_eq!(addr_bits.len(), ell_addr, "addr_bits count must match ell_addr");
        for (b, col) in addr_bits.iter().enumerate() {
            assert_eq!(
                col.len(),
                pow2_cycle,
                "addr_bits[{b}] length must match cycle domain"
            );
        }
        assert_eq!(
            has_lookup.len(),
            pow2_cycle,
            "has_lookup length must match cycle domain"
        );

        let mut claimed_sum = K::ZERO;
        let mut weight_table = vec![K::ZERO; pow2_addr];

        for &(t, gate) in has_lookup.entries() {
            if gate == K::ZERO {
                continue;
            }
            let weight_t = chi_at_bool_index(r_cycle, t) * gate;
            if weight_t == K::ZERO {
                continue;
            }

            let mut addr_t = 0usize;
            for (b, col) in addr_bits.iter().enumerate() {
                if col.get(t) == K::ONE {
                    addr_t |= 1usize << b;
                }
            }
            if addr_t < pow2_addr {
                weight_table[addr_t] += weight_t;
            }
        }

        for addr in 0..pow2_addr.min(table.len()) {
            claimed_sum += table[addr] * weight_table[addr];
        }

        let mut table_k: Vec<K> = table.iter().copied().collect();
        table_k.resize(pow2_addr, K::ZERO);
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

fn log2_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    debug_assert!(n.is_power_of_two(), "expected power of two, got {n}");
    n.trailing_zeros() as usize
}
