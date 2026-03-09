use crate::bit_ops::eq_bit_affine;
use crate::mle::eq_single;
use crate::sparse_time::SparseIdxVec;
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

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

pub struct ProductRoundOracle {
    factors: Vec<Vec<K>>,
    rounds_remaining: usize,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl ProductRoundOracle {
    pub fn new(factors: Vec<Vec<K>>, degree_bound: usize) -> Self {
        let len = factors.first().map(|f| f.len()).unwrap_or(1);
        debug_assert!(len.is_power_of_two(), "factor length must be a power of two");
        for f in factors.iter() {
            debug_assert_eq!(f.len(), len, "all factors must have the same length");
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
        let n = self.factors.first().map(|f| f.len()).unwrap_or(1);
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
            let val = self
                .value()
                .expect("ProductRoundOracle invariant broken: rounds_remaining==0 but value() is None");
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

fn chi_at_bool_index(r: &[K], idx: usize) -> K {
    crate::mle::chi_at_index(r, idx)
}

fn chi_cycle_children(r_cycle: &[K], bit_idx: usize, prefix_eq: K, pair_idx: usize) -> (K, K) {
    debug_assert!(bit_idx < r_cycle.len());

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

macro_rules! for_each_sparse_parent_pair {
    ($entries:expr, $pair:ident, $body:block) => {{
        let mut prev_pair = usize::MAX;
        for &(idx, _) in $entries {
            let $pair = idx >> 1;
            #[cfg(debug_assertions)]
            if prev_pair != usize::MAX {
                debug_assert!(
                    $pair >= prev_pair,
                    "for_each_sparse_parent_pair requires nondecreasing parent pair order"
                );
            }
            if $pair == prev_pair {
                continue;
            }
            prev_pair = $pair;
            $body
        }
    }};
}

fn pow2_weights_32() -> &'static [K] {
    static W: OnceLock<[K; 32]> = OnceLock::new();
    W.get_or_init(|| std::array::from_fn(|i| K::from_u64(1u64 << i)))
}

fn pow2_weights_64() -> &'static [K] {
    static W: OnceLock<[K; 64]> = OnceLock::new();
    W.get_or_init(|| std::array::from_fn(|i| K::from_u64(1u64 << i)))
}

fn pow2_weights_5() -> &'static [K] {
    static W: OnceLock<[K; 5]> = OnceLock::new();
    W.get_or_init(|| std::array::from_fn(|i| K::from_u64(1u64 << i)))
}

fn expr_id1(cols: &[K; 1]) -> K {
    cols[0]
}

pub struct ShoutValueOracleSparse {
    core: SparseTimeExprOracle<1>,
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
                core: SparseTimeExprOracle::new(r_cycle, has_lookup, [val], 3, expr_id1),
            },
            claim,
        )
    }
}
impl_round_oracle_via_core!(ShoutValueOracleSparse);

type SparseExprFn<const N: usize> = fn(&[K; N]) -> K;

struct SparseTimeExprOracle<const N: usize> {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    cols: [SparseIdxVec<K>; N],
    degree_bound: usize,
    expr_fn: SparseExprFn<N>,
}

impl<const N: usize> SparseTimeExprOracle<N> {
    fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        cols: [SparseIdxVec<K>; N],
        degree_bound: usize,
        expr_fn: SparseExprFn<N>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        assert_cols_match_time(&cols, 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            cols,
            degree_bound,
            expr_fn,
        }
    }
}

impl<const N: usize> RoundOracle for SparseTimeExprOracle<N> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let cols = std::array::from_fn(|i| self.cols[i].singleton_value());
            let expr = (self.expr_fn)(&cols);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        for_each_sparse_parent_pair!(self.has_lookup.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let cols0: [K; N] = std::array::from_fn(|i| self.cols[i].get(child0));
            let cols1: [K; N] = std::array::from_fn(|i| self.cols[i].get(child1));

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let cols_x: [K; N] = std::array::from_fn(|j| interp(cols0[j], cols1[j], x));
                let expr_x = (self.expr_fn)(&cols_x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        });
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
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

fn expr_rv32_packed_add(cols: &[K; 4]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let carry = cols[2];
    let val = cols[3];
    lhs + rhs - val - carry * K::from_u64(1u64 << 32)
}

fn expr_rv32_packed_sub(cols: &[K; 4]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let borrow = cols[2];
    let val = cols[3];
    lhs - rhs - val + borrow * K::from_u64(1u64 << 32)
}

fn expr_rv32_packed_mulhsu_adapter(cols: &[K; 6]) -> K {
    let rhs = cols[1];
    let lhs_sign = cols[2];
    let hi = cols[3];
    let borrow = cols[4];
    let val = cols[5];
    hi - lhs_sign * rhs - val + borrow * K::from_u64(1u64 << 32)
}

fn expr_rv32_packed_slt(cols: &[K; 6]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let lhs_sign = cols[2];
    let rhs_sign = cols[3];
    let diff = cols[4];
    let out = cols[5];
    let two = K::from_u64(2);
    let two31 = K::from_u64(1u64 << 31);
    let two32 = K::from_u64(1u64 << 32);
    let lhs_b = lhs + (K::ONE - two * lhs_sign) * two31;
    let rhs_b = rhs + (K::ONE - two * rhs_sign) * two31;
    lhs_b - rhs_b - diff + out * two32
}

fn expr_rv32_packed_sltu(cols: &[K; 4]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let diff = cols[2];
    let out = cols[3];
    lhs - rhs - diff + out * K::from_u64(1u64 << 32)
}

fn expr_rv32_packed_divu(cols: &[K; 5]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let rem = cols[2];
    let z = cols[3];
    let quot = cols[4];
    let all_ones = K::from_u64(u32::MAX as u64);
    z * (quot - all_ones) + (K::ONE - z) * (lhs - rhs * quot - rem)
}

fn expr_rv64_packed_divu(cols: &[K; 5]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let rem = cols[2];
    let z = cols[3];
    let quot = cols[4];
    let all_ones = K::from_u64(u64::MAX);
    z * (quot - all_ones) + (K::ONE - z) * (lhs - rhs * quot - rem)
}

fn expr_rv32_packed_remu(cols: &[K; 5]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let quot = cols[2];
    let z = cols[3];
    let rem = cols[4];
    z * (rem - lhs) + (K::ONE - z) * (lhs - rhs * quot - rem)
}

fn expr_rv64_packed_remu(cols: &[K; 5]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let quot = cols[2];
    let z = cols[3];
    let rem = cols[4];
    z * (rem - lhs) + (K::ONE - z) * (lhs - rhs * quot - rem)
}

fn expr_rv32_packed_div(cols: &[K; 6]) -> K {
    let lhs_sign = cols[0];
    let rhs_sign = cols[1];
    let z = cols[2];
    let q_abs = cols[3];
    let q_is_zero = cols[4];
    let val = cols[5];
    let two = K::from_u64(2);
    let two32 = K::from_u64(1u64 << 32);
    let all_ones = K::from_u64(u32::MAX as u64);
    let div_sign = lhs_sign + rhs_sign - two * lhs_sign * rhs_sign;
    let neg_q = (K::ONE - q_is_zero) * (two32 - q_abs);
    let q_signed = (K::ONE - div_sign) * q_abs + div_sign * neg_q;
    z * (val - all_ones) + (K::ONE - z) * (val - q_signed)
}

fn expr_rv64_packed_div(cols: &[K; 6]) -> K {
    let lhs_sign = cols[0];
    let rhs_sign = cols[1];
    let z = cols[2];
    let q_abs = cols[3];
    let q_is_zero = cols[4];
    let val = cols[5];
    let two = K::from_u64(2);
    let two64_mod = K::from_u64(u32::MAX as u64);
    let all_ones = K::from_u64(u64::MAX);
    let div_sign = lhs_sign + rhs_sign - two * lhs_sign * rhs_sign;
    let neg_q = (K::ONE - q_is_zero) * (two64_mod - q_abs);
    let q_signed = (K::ONE - div_sign) * q_abs + div_sign * neg_q;
    z * (val - all_ones) + (K::ONE - z) * (val - q_signed)
}

fn expr_rv32_packed_rem(cols: &[K; 6]) -> K {
    let lhs = cols[0];
    let lhs_sign = cols[1];
    let z = cols[2];
    let r_abs = cols[3];
    let r_is_zero = cols[4];
    let val = cols[5];
    let two32 = K::from_u64(1u64 << 32);
    let neg_r = (K::ONE - r_is_zero) * (two32 - r_abs);
    let r_signed = (K::ONE - lhs_sign) * r_abs + lhs_sign * neg_r;
    z * (val - lhs) + (K::ONE - z) * (val - r_signed)
}

fn expr_rv64_packed_rem(cols: &[K; 6]) -> K {
    let lhs = cols[0];
    let lhs_sign = cols[1];
    let z = cols[2];
    let r_abs = cols[3];
    let r_is_zero = cols[4];
    let val = cols[5];
    let two64_mod = K::from_u64(u32::MAX as u64);
    let neg_r = (K::ONE - r_is_zero) * (two64_mod - r_abs);
    let r_signed = (K::ONE - lhs_sign) * r_abs + lhs_sign * neg_r;
    z * (val - lhs) + (K::ONE - z) * (val - r_signed)
}

fn expr_rv32_packed_mul(cols: &[K; 3], limb_sum: K) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let val = cols[2];
    lhs * rhs - val - limb_sum * K::from_u64(1u64 << 32)
}

fn expr_rv32_packed_mulhu(cols: &[K; 3], limb_sum: K) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let val = cols[2];
    lhs * rhs - limb_sum - val * K::from_u64(1u64 << 32)
}

fn expr_rv64_packed_mul(cols: &[K; 3], limb_sum: K) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let val = cols[2];
    let two64_mod = K::from_u64(u32::MAX as u64);
    lhs * rhs - val - limb_sum * two64_mod
}

fn expr_rv64_packed_mulhu(cols: &[K; 3], limb_sum: K) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let val = cols[2];
    let two64_mod = K::from_u64(u32::MAX as u64);
    lhs * rhs - limb_sum - val * two64_mod
}

struct SparseShiftRemBoundOracle {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    rem_bits: Vec<SparseIdxVec<K>>,
}

fn shamt_values_singleton(shamt_bits: &[SparseIdxVec<K>]) -> [K; 5] {
    let mut shamt = [K::ZERO; 5];
    for (i, b) in shamt_bits.iter().enumerate() {
        shamt[i] = b.singleton_value();
    }
    shamt
}

fn shamt_values_pair(shamt_bits: &[SparseIdxVec<K>], child0: usize, child1: usize) -> ([K; 5], [K; 5]) {
    let mut b0s = [K::ZERO; 5];
    let mut b1s = [K::ZERO; 5];
    for (i, b) in shamt_bits.iter().enumerate() {
        b0s[i] = b.get(child0);
        b1s[i] = b.get(child1);
    }
    (b0s, b1s)
}

fn shamt_values_interp(b0s: &[K; 5], b1s: &[K; 5], x: K) -> [K; 5] {
    let mut shamt = [K::ZERO; 5];
    for j in 0..5 {
        shamt[j] = interp(b0s[j], b1s[j], x);
    }
    shamt
}

fn pow2_from_shamt(shamt: &[K; 5]) -> K {
    let pow2_const = [2u64, 4, 16, 256, 65536];
    let mut pow2 = K::ONE;
    for (b, c) in shamt.iter().zip(pow2_const.iter()) {
        let c = K::from_u64(*c);
        pow2 *= K::ONE + *b * (c - K::ONE);
    }
    pow2
}

fn shift_rem_bound_expr<F>(shamt: [K; 5], rem_bits_len: usize, mut rem_at: F) -> K
where
    F: FnMut(usize) -> K,
{
    let mut tail_sum: [K; 32] = [K::ZERO; 32];
    let mut tail = K::ZERO;
    for j in (0..rem_bits_len).rev() {
        tail += rem_at(j) * K::from_u64(1u64 << j);
        tail_sum[j] = tail;
    }

    let mut expr = K::ZERO;
    for s in 0..32usize {
        let mut prod = K::ONE;
        for j in 0..5usize {
            let b = shamt[j];
            prod *= if ((s >> j) & 1) == 1 { b } else { K::ONE - b };
        }
        expr += prod * tail_sum[s];
    }
    expr
}

impl SparseShiftRemBoundOracle {
    fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        rem_bits: Vec<SparseIdxVec<K>>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        assert_cols_match_time(&shamt_bits, 1usize << ell_n);
        debug_assert!(rem_bits.len() <= 32);
        assert_cols_match_time(&rem_bits, 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            shamt_bits,
            rem_bits,
        }
    }
}

impl RoundOracle for SparseShiftRemBoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let shamt = shamt_values_singleton(&self.shamt_bits);
            let expr = shift_rem_bound_expr(shamt, self.rem_bits.len(), |j| self.rem_bits[j].singleton_value());
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        for_each_sparse_parent_pair!(self.has_lookup.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (b0s, b1s) = shamt_values_pair(&self.shamt_bits, child0, child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let shamt = shamt_values_interp(&b0s, &b1s, x);
                let expr_x = shift_rem_bound_expr(shamt, self.rem_bits.len(), |j| {
                    interp(self.rem_bits[j].get(child0), self.rem_bits[j].get(child1), x)
                });

                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        });

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        8
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        for b in self.rem_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

type SparseShiftExprFn = fn(lhs: K, val: K, pow2: K, limb_sum: K, sign: K) -> K;

struct SparseShiftExprOracle {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    bits: Vec<SparseIdxVec<K>>,
    sign: Option<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
    expr_fn: SparseShiftExprFn,
}

impl SparseShiftExprOracle {
    fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        bits: Vec<SparseIdxVec<K>>,
        sign: Option<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
        degree_bound: usize,
        expr_fn: SparseShiftExprFn,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        assert_cols_match_time(&shamt_bits, 1usize << ell_n);
        assert_cols_match_time(&bits, 1usize << ell_n);
        if let Some(s) = sign.as_ref() {
            debug_assert_eq!(s.len(), 1usize << ell_n);
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            shamt_bits,
            bits,
            sign,
            val,
            degree_bound,
            expr_fn,
        }
    }
}

impl RoundOracle for SparseShiftExprOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let val = self.val.singleton_value();
            let shamt = shamt_values_singleton(&self.shamt_bits);
            let pow2 = pow2_from_shamt(&shamt);

            let mut limb_sum = K::ZERO;
            for (i, b) in self.bits.iter().enumerate() {
                limb_sum += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let sign = self
                .sign
                .as_ref()
                .map(|s| s.singleton_value())
                .unwrap_or(K::ZERO);
            let expr = (self.expr_fn)(lhs, val, pow2, limb_sum, sign);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        for_each_sparse_parent_pair!(self.has_lookup.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);
            let sign0 = self.sign.as_ref().map(|s| s.get(child0)).unwrap_or(K::ZERO);
            let sign1 = self.sign.as_ref().map(|s| s.get(child1)).unwrap_or(K::ZERO);

            let (b0s, b1s) = shamt_values_pair(&self.shamt_bits, child0, child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let val_x = interp(val0, val1, x);
                let sign_x = interp(sign0, sign1, x);

                let shamt_x = shamt_values_interp(&b0s, &b1s, x);
                let pow2_x = pow2_from_shamt(&shamt_x);

                let mut limb_sum_x = K::ZERO;
                for (j, b) in self.bits.iter().enumerate() {
                    let bit_x = interp(b.get(child0), b.get(child1), x);
                    limb_sum_x += bit_x * K::from_u64(1u64 << j);
                }

                let expr_x = (self.expr_fn)(lhs_x, val_x, pow2_x, limb_sum_x, sign_x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        });
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
        self.lhs.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        for b in self.bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        if let Some(s) = self.sign.as_mut() {
            s.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

fn expr_rv32_packed_sll(lhs: K, val: K, pow2: K, carry: K, _sign: K) -> K {
    lhs * pow2 - val - carry * K::from_u64(1u64 << 32)
}

fn expr_rv32_packed_srl(lhs: K, val: K, pow2: K, rem: K, _sign: K) -> K {
    lhs - val * pow2 - rem
}

fn expr_rv32_packed_sra(lhs: K, val: K, pow2: K, rem: K, sign: K) -> K {
    lhs - val * pow2 - rem - sign * K::from_u64(1u64 << 32) * (K::ONE - pow2)
}

type SparseColsBitsExprFn<const N: usize, const W: usize> = fn(&[K; N], &[K], K, &[K; W]) -> K;

struct SparseColsBitsExprOracle<const N: usize, const W: usize> {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    cols: [SparseIdxVec<K>; N],
    bits: Vec<SparseIdxVec<K>>,
    bit_weights: Cow<'static, [K]>,
    bits_x_scratch: Vec<K>,
    expr_weights: [K; W],
    degree_bound: usize,
    expr_fn: SparseColsBitsExprFn<N, W>,
}

impl<const N: usize, const W: usize> SparseColsBitsExprOracle<N, W> {
    fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        cols: [SparseIdxVec<K>; N],
        bits: Vec<SparseIdxVec<K>>,
        bit_weights: Cow<'static, [K]>,
        expr_weights: [K; W],
        degree_bound: usize,
        expr_fn: SparseColsBitsExprFn<N, W>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        assert_cols_match_time(&cols, 1usize << ell_n);
        assert_cols_match_time(&bits, 1usize << ell_n);
        debug_assert!(bit_weights.is_empty() || bit_weights.len() == bits.len());

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            cols,
            bits_x_scratch: vec![K::ZERO; bits.len()],
            bits,
            bit_weights,
            expr_weights,
            degree_bound,
            expr_fn,
        }
    }
}

impl<const N: usize, const W: usize> RoundOracle for SparseColsBitsExprOracle<N, W> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let cols: [K; N] = std::array::from_fn(|i| self.cols[i].singleton_value());
            let bits: Vec<K> = self
                .bits
                .iter()
                .map(SparseIdxVec::singleton_value)
                .collect();
            let bit_sum = if self.bit_weights.is_empty() {
                K::ZERO
            } else {
                bits.iter()
                    .zip(self.bit_weights.iter())
                    .fold(K::ZERO, |acc, (b, w)| acc + *b * *w)
            };
            let expr = (self.expr_fn)(&cols, &bits, bit_sum, &self.expr_weights);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut bits_x = std::mem::take(&mut self.bits_x_scratch);
        if bits_x.len() != self.bits.len() {
            bits_x.resize(self.bits.len(), K::ZERO);
        }
        for_each_sparse_parent_pair!(self.has_lookup.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let cols0: [K; N] = std::array::from_fn(|i| self.cols[i].get(child0));
            let cols1: [K; N] = std::array::from_fn(|i| self.cols[i].get(child1));

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let cols_x: [K; N] = std::array::from_fn(|j| interp(cols0[j], cols1[j], x));
                for (j, b) in self.bits.iter().enumerate() {
                    bits_x[j] = interp(b.get(child0), b.get(child1), x);
                }
                let bit_sum_x = if self.bit_weights.is_empty() {
                    K::ZERO
                } else {
                    bits_x
                        .iter()
                        .zip(self.bit_weights.iter())
                        .fold(K::ZERO, |acc, (b, w)| acc + *b * *w)
                };
                let expr_x = (self.expr_fn)(&cols_x, &bits_x, bit_sum_x, &self.expr_weights);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        });
        self.bits_x_scratch = bits_x;
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
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        for b in self.bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

fn expr_rv32_packed_mulh_adapter(cols: &[K; 7], _bits: &[K], _bit_sum: K, w: &[K; 2]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let lhs_sign = cols[2];
    let rhs_sign = cols[3];
    let hi = cols[4];
    let k = cols[5];
    let val = cols[6];
    let eq_expr = hi - lhs_sign * rhs - rhs_sign * lhs + k * K::from_u64(1u64 << 32) - val;
    let range = k * (k - K::ONE) * (k - K::from_u64(2));
    w[0] * eq_expr + w[1] * range
}

fn expr_rv64_packed_mulh_adapter(cols: &[K; 7], _bits: &[K], _bit_sum: K, w: &[K; 2]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let lhs_sign = cols[2];
    let rhs_sign = cols[3];
    let hi = cols[4];
    let k = cols[5];
    let val = cols[6];
    let two64_mod = K::from_u64(u32::MAX as u64);
    let eq_expr = hi - lhs_sign * rhs - rhs_sign * lhs + k * two64_mod - val;
    let range = k * (k - K::ONE) * (k - K::from_u64(2));
    w[0] * eq_expr + w[1] * range
}

fn expr_rv32_packed_divremu_adapter(cols: &[K; 4], _bits: &[K], bit_sum: K, w: &[K; 4]) -> K {
    let rhs = cols[0];
    let z = cols[1];
    let rem = cols[2];
    let diff = cols[3];
    let c0 = z * (K::ONE - z);
    let c1 = z * rhs;
    let c2 = (K::ONE - z) * (rem - rhs - diff + K::from_u64(1u64 << 32));
    let c3 = diff - bit_sum;
    w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3
}

fn expr_rv64_packed_divremu_adapter(cols: &[K; 4], _bits: &[K], bit_sum: K, w: &[K; 4]) -> K {
    let rhs = cols[0];
    let z = cols[1];
    let rem = cols[2];
    let diff = cols[3];
    let two64_mod = K::from_u64(u32::MAX as u64);
    let c0 = z * (K::ONE - z);
    let c1 = z * rhs;
    let c2 = (K::ONE - z) * (rem - rhs - diff + two64_mod);
    let c3 = diff - bit_sum;
    w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3
}

fn expr_rv32_packed_divrem_adapter(cols: &[K; 10], _bits: &[K], bit_sum: K, w: &[K; 7]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let z = cols[2];
    let lhs_sign = cols[3];
    let rhs_sign = cols[4];
    let q_abs = cols[5];
    let r_abs = cols[6];
    let mag = cols[7];
    let mag_z = cols[8];
    let diff = cols[9];
    let two = K::from_u64(2);
    let two32 = K::from_u64(1u64 << 32);
    let lhs_abs = lhs + lhs_sign * (two32 - two * lhs);
    let rhs_abs = rhs + rhs_sign * (two32 - two * rhs);
    let c0 = z * (K::ONE - z);
    let c1 = z * rhs;
    let c2 = mag_z * (K::ONE - mag_z);
    let c3 = mag_z * mag;
    let c4 = (K::ONE - z) * (lhs_abs - rhs_abs * q_abs - r_abs);
    let c5 = (K::ONE - z) * (r_abs - rhs_abs - diff + two32);
    let c6 = diff - bit_sum;
    w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3 + w[4] * c4 + w[5] * c5 + w[6] * c6
}

fn expr_rv64_packed_divrem_adapter(cols: &[K; 10], _bits: &[K], bit_sum: K, w: &[K; 7]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let z = cols[2];
    let lhs_sign = cols[3];
    let rhs_sign = cols[4];
    let q_abs = cols[5];
    let r_abs = cols[6];
    let mag = cols[7];
    let mag_z = cols[8];
    let diff = cols[9];
    let two = K::from_u64(2);
    let two64_mod = K::from_u64(u32::MAX as u64);
    let lhs_abs = lhs + lhs_sign * (two64_mod - two * lhs);
    let rhs_abs = rhs + rhs_sign * (two64_mod - two * rhs);
    let c0 = z * (K::ONE - z);
    let c1 = z * rhs;
    let c2 = mag_z * (K::ONE - mag_z);
    let c3 = mag_z * mag;
    let c4 = (K::ONE - z) * (lhs_abs - rhs_abs * q_abs - r_abs);
    let c5 = (K::ONE - z) * (r_abs - rhs_abs - diff + two64_mod);
    let c6 = diff - bit_sum;
    w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3 + w[4] * c4 + w[5] * c5 + w[6] * c6
}

fn expr_rv32_packed_eq_adapter(cols: &[K; 3], diff: K) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    let borrow = cols[2];
    lhs - rhs - diff + borrow * K::from_u64(1u64 << 32)
}

fn expr_rv64_packed_mulhsu_adapter(cols: &[K; 6]) -> K {
    let rhs = cols[1];
    let lhs_sign = cols[2];
    let hi = cols[3];
    let borrow = cols[4];
    let val = cols[5];
    let two64_mod = K::from_u64(u32::MAX as u64);
    hi - lhs_sign * rhs - val + borrow * two64_mod
}

fn expr_u_decomp(cols: &[K; 1], sum: K) -> K {
    cols[0] - sum
}

fn expr_rv32_packed_mul_bw0(cols: &[K; 3], _bits: &[K], bit_sum: K, _w: &[K; 0]) -> K {
    expr_rv32_packed_mul(cols, bit_sum)
}

fn expr_rv32_packed_mulhu_bw0(cols: &[K; 3], _bits: &[K], bit_sum: K, _w: &[K; 0]) -> K {
    expr_rv32_packed_mulhu(cols, bit_sum)
}

fn expr_rv64_packed_mul_bw0(cols: &[K; 3], _bits: &[K], bit_sum: K, _w: &[K; 0]) -> K {
    expr_rv64_packed_mul(cols, bit_sum)
}

fn expr_rv64_packed_mulhu_bw0(cols: &[K; 3], _bits: &[K], bit_sum: K, _w: &[K; 0]) -> K {
    expr_rv64_packed_mulhu(cols, bit_sum)
}

fn expr_rv32_packed_eq_adapter_bw0(cols: &[K; 3], _bits: &[K], bit_sum: K, _w: &[K; 0]) -> K {
    expr_rv32_packed_eq_adapter(cols, bit_sum)
}

fn expr_u_decomp_bw0(cols: &[K; 1], _bits: &[K], bit_sum: K, _w: &[K; 0]) -> K {
    expr_u_decomp(cols, bit_sum)
}

fn expr_eq_from_prod(val: K, prod: K) -> K {
    val - prod
}

fn expr_neq_from_prod(val: K, prod: K) -> K {
    val + prod - K::ONE
}

fn expr_virtual_write_domain(cols: &[K; 2]) -> K {
    cols[0] * (K::ONE - cols[1])
}

fn expr_nonvirtual_arch_domain(cols: &[K; 2]) -> K {
    (K::ONE - cols[0]) * cols[1]
}

fn expr_eq_from_prod_bits(cols: &[K; 1], bits: &[K], _bit_sum: K, _w: &[K; 0]) -> K {
    let mut prod = K::ONE;
    for &b in bits {
        prod *= K::ONE - b;
    }
    expr_eq_from_prod(cols[0], prod)
}

fn expr_neq_from_prod_bits(cols: &[K; 1], bits: &[K], _bit_sum: K, _w: &[K; 0]) -> K {
    let mut prod = K::ONE;
    for &b in bits {
        prod *= K::ONE - b;
    }
    expr_neq_from_prod(cols[0], prod)
}

macro_rules! define_sparse_time_expr_oracle {
    ($name:ident, $n:expr, $degree:expr, $expr:expr, [$($col:ident),+ $(,)?]) => {
        pub struct $name {
            core: SparseTimeExprOracle<$n>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                $($col: SparseIdxVec<K>,)+
            ) -> Self {
                Self {
                    core: SparseTimeExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [$($col),+],
                        $degree,
                        $expr,
                    ),
                }
            }
        }

        impl_round_oracle_via_core!($name);
    };
}

define_sparse_time_expr_oracle!(
    Rv32PackedAddOracleSparseTime,
    4,
    3,
    expr_rv32_packed_add,
    [lhs, rhs, carry, val]
);

define_sparse_time_expr_oracle!(
    Rv32PackedSubOracleSparseTime,
    4,
    3,
    expr_rv32_packed_sub,
    [lhs, rhs, borrow, val]
);

define_sparse_time_expr_oracle!(
    Rv32VirtualWriteDomainOracleSparseTime,
    2,
    4,
    expr_virtual_write_domain,
    [has_write, wa_bit5]
);

define_sparse_time_expr_oracle!(
    Rv32NonVirtualArchDomainOracleSparseTime,
    2,
    4,
    expr_nonvirtual_arch_domain,
    [is_virtual, addr_bit5]
);

macro_rules! define_weighted_bits32_oracle3 {
    ($name:ident, [$c0:ident, $c1:ident, $c2:ident], $expr:expr, $degree:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<3, 0>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                $c0: SparseIdxVec<K>,
                $c1: SparseIdxVec<K>,
                $c2: SparseIdxVec<K>,
                bits: Vec<SparseIdxVec<K>>,
            ) -> Self {
                debug_assert_eq!(bits.len(), 32);
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [$c0, $c1, $c2],
                        bits,
                        Cow::Borrowed(pow2_weights_32()),
                        [],
                        $degree,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

macro_rules! define_weighted_bits32_oracle3_bits_before_last {
    ($name:ident, [$c0:ident, $c1:ident], $bits:ident, $c2:ident, $expr:expr, $degree:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<3, 0>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                $c0: SparseIdxVec<K>,
                $c1: SparseIdxVec<K>,
                $bits: Vec<SparseIdxVec<K>>,
                $c2: SparseIdxVec<K>,
            ) -> Self {
                debug_assert_eq!($bits.len(), 32);
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [$c0, $c1, $c2],
                        $bits,
                        Cow::Borrowed(pow2_weights_32()),
                        [],
                        $degree,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_weighted_bits32_oracle3_bits_before_last!(
    Rv32PackedMulOracleSparseTime,
    [lhs, rhs],
    carry_bits,
    val,
    expr_rv32_packed_mul_bw0,
    4
);

define_weighted_bits32_oracle3_bits_before_last!(
    Rv32PackedMulhuOracleSparseTime,
    [lhs, rhs],
    lo_bits,
    val,
    expr_rv32_packed_mulhu_bw0,
    4
);

define_weighted_bits32_oracle3_bits_before_last!(
    Rv32PackedMulHiOracleSparseTime,
    [lhs, rhs],
    lo_bits,
    hi,
    expr_rv32_packed_mulhu_bw0,
    4
);

pub struct Rv64PackedMulOracleSparseTime {
    core: SparseColsBitsExprOracle<3, 0>,
}

impl Rv64PackedMulOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        carry_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        debug_assert_eq!(carry_bits.len(), 64);
        Self {
            core: SparseColsBitsExprOracle::new(
                r_cycle,
                has_lookup,
                [lhs, rhs, val],
                carry_bits,
                Cow::Borrowed(pow2_weights_64()),
                [],
                4,
                expr_rv64_packed_mul_bw0,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv64PackedMulOracleSparseTime);

pub struct Rv64PackedMulhuOracleSparseTime {
    core: SparseColsBitsExprOracle<3, 0>,
}

impl Rv64PackedMulhuOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lo_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        debug_assert_eq!(lo_bits.len(), 64);
        Self {
            core: SparseColsBitsExprOracle::new(
                r_cycle,
                has_lookup,
                [lhs, rhs, val],
                lo_bits,
                Cow::Borrowed(pow2_weights_64()),
                [],
                4,
                expr_rv64_packed_mulhu_bw0,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv64PackedMulhuOracleSparseTime);

pub struct Rv64PackedMulHiOracleSparseTime {
    core: SparseColsBitsExprOracle<3, 0>,
}

impl Rv64PackedMulHiOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lo_bits: Vec<SparseIdxVec<K>>,
        hi: SparseIdxVec<K>,
    ) -> Self {
        debug_assert_eq!(lo_bits.len(), 64);
        Self {
            core: SparseColsBitsExprOracle::new(
                r_cycle,
                has_lookup,
                [lhs, rhs, hi],
                lo_bits,
                Cow::Borrowed(pow2_weights_64()),
                [],
                4,
                expr_rv64_packed_mulhu_bw0,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv64PackedMulHiOracleSparseTime);

pub struct Rv32PackedMulhAdapterOracleSparseTime {
    core: SparseColsBitsExprOracle<7, 2>,
}

impl Rv32PackedMulhAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_sign: SparseIdxVec<K>,
        hi: SparseIdxVec<K>,
        k: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        weights: [K; 2],
    ) -> Self {
        Self {
            core: SparseColsBitsExprOracle::new(
                r_cycle,
                has_lookup,
                [lhs, rhs, lhs_sign, rhs_sign, hi, k, val],
                Vec::new(),
                Cow::Borrowed(&[]),
                weights,
                5,
                expr_rv32_packed_mulh_adapter,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv32PackedMulhAdapterOracleSparseTime);

pub struct Rv64PackedMulhAdapterOracleSparseTime {
    core: SparseColsBitsExprOracle<7, 2>,
}

impl Rv64PackedMulhAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_sign: SparseIdxVec<K>,
        hi: SparseIdxVec<K>,
        k: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        weights: [K; 2],
    ) -> Self {
        Self {
            core: SparseColsBitsExprOracle::new(
                r_cycle,
                has_lookup,
                [lhs, rhs, lhs_sign, rhs_sign, hi, k, val],
                Vec::new(),
                Cow::Borrowed(&[]),
                weights,
                5,
                expr_rv64_packed_mulh_adapter,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv64PackedMulhAdapterOracleSparseTime);

define_sparse_time_expr_oracle!(
    Rv32PackedMulhsuAdapterOracleSparseTime,
    6,
    4,
    expr_rv32_packed_mulhsu_adapter,
    [lhs, rhs, lhs_sign, hi, borrow, val]
);

define_sparse_time_expr_oracle!(
    Rv64PackedMulhsuAdapterOracleSparseTime,
    6,
    4,
    expr_rv64_packed_mulhsu_adapter,
    [lhs, rhs, lhs_sign, hi, borrow, val]
);

macro_rules! define_val_prod_bits_oracle {
    ($name:ident, $expr:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<1, 0>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                diff_bits: Vec<SparseIdxVec<K>>,
                val: SparseIdxVec<K>,
            ) -> Self {
                debug_assert_eq!(diff_bits.len(), 32);
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [val],
                        diff_bits,
                        Cow::Borrowed(&[]),
                        [],
                        34,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_val_prod_bits_oracle!(Rv32PackedEqOracleSparseTime, expr_eq_from_prod_bits);

define_weighted_bits32_oracle3!(
    Rv32PackedEqAdapterOracleSparseTime,
    [lhs, rhs, borrow],
    expr_rv32_packed_eq_adapter_bw0,
    3
);

define_val_prod_bits_oracle!(Rv32PackedNeqOracleSparseTime, expr_neq_from_prod_bits);

define_weighted_bits32_oracle3!(
    Rv32PackedNeqAdapterOracleSparseTime,
    [lhs, rhs, borrow],
    expr_rv32_packed_eq_adapter_bw0,
    3
);

define_sparse_time_expr_oracle!(
    Rv32PackedSltOracleSparseTime,
    6,
    3,
    expr_rv32_packed_slt,
    [lhs, rhs, lhs_sign, rhs_sign, diff, out]
);

define_sparse_time_expr_oracle!(
    Rv32PackedSltuOracleSparseTime,
    4,
    3,
    expr_rv32_packed_sltu,
    [lhs, rhs, diff, out]
);

macro_rules! define_shift_oracle_no_sign {
    ($name:ident, $bits_name:ident, $bits_len:expr, $expr:expr) => {
        pub struct $name {
            core: SparseShiftExprOracle,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                lhs: SparseIdxVec<K>,
                shamt_bits: Vec<SparseIdxVec<K>>,
                $bits_name: Vec<SparseIdxVec<K>>,
                val: SparseIdxVec<K>,
            ) -> Self {
                debug_assert_eq!($bits_name.len(), $bits_len);
                Self {
                    core: SparseShiftExprOracle::new(
                        r_cycle, has_lookup, lhs, shamt_bits, $bits_name, None, val, 8, $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

macro_rules! define_shift_oracle_with_sign {
    ($name:ident, $bits_len:expr, $expr:expr) => {
        pub struct $name {
            core: SparseShiftExprOracle,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                lhs: SparseIdxVec<K>,
                shamt_bits: Vec<SparseIdxVec<K>>,
                sign: SparseIdxVec<K>,
                rem_bits: Vec<SparseIdxVec<K>>,
                val: SparseIdxVec<K>,
            ) -> Self {
                debug_assert_eq!(rem_bits.len(), $bits_len);
                Self {
                    core: SparseShiftExprOracle::new(
                        r_cycle,
                        has_lookup,
                        lhs,
                        shamt_bits,
                        rem_bits,
                        Some(sign),
                        val,
                        8,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

macro_rules! define_shift_rem_bound_oracle {
    ($name:ident, $bits_len:expr) => {
        pub struct $name {
            core: SparseShiftRemBoundOracle,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                shamt_bits: Vec<SparseIdxVec<K>>,
                rem_bits: Vec<SparseIdxVec<K>>,
            ) -> Self {
                debug_assert_eq!(rem_bits.len(), $bits_len);
                Self {
                    core: SparseShiftRemBoundOracle::new(r_cycle, has_lookup, shamt_bits, rem_bits),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_shift_oracle_no_sign!(Rv32PackedSllOracleSparseTime, carry_bits, 32, expr_rv32_packed_sll);

define_shift_oracle_no_sign!(Rv32PackedSrlOracleSparseTime, rem_bits, 32, expr_rv32_packed_srl);

define_shift_rem_bound_oracle!(Rv32PackedSrlAdapterOracleSparseTime, 32);

define_shift_oracle_with_sign!(Rv32PackedSraOracleSparseTime, 31, expr_rv32_packed_sra);

define_shift_rem_bound_oracle!(Rv32PackedSraAdapterOracleSparseTime, 31);

define_sparse_time_expr_oracle!(
    Rv32PackedDivuOracleSparseTime,
    5,
    5,
    expr_rv32_packed_divu,
    [lhs, rhs, rem, rhs_is_zero, quot]
);

define_sparse_time_expr_oracle!(
    Rv64PackedDivuOracleSparseTime,
    5,
    5,
    expr_rv64_packed_divu,
    [lhs, rhs, rem, rhs_is_zero, quot]
);

define_sparse_time_expr_oracle!(
    Rv32PackedRemuOracleSparseTime,
    5,
    5,
    expr_rv32_packed_remu,
    [lhs, rhs, quot, rhs_is_zero, rem]
);

define_sparse_time_expr_oracle!(
    Rv64PackedRemuOracleSparseTime,
    5,
    5,
    expr_rv64_packed_remu,
    [lhs, rhs, quot, rhs_is_zero, rem]
);

macro_rules! define_bits_and_weights32_oracle {
    ($name:ident, $n:expr, $m:expr, [$($col:ident),+ $(,)?], $degree:expr, $expr:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<$n, $m>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                $($col: SparseIdxVec<K>,)+
                diff_bits: Vec<SparseIdxVec<K>>,
                weights: [K; $m],
            ) -> Self {
                debug_assert_eq!(diff_bits.len(), 32);
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [$($col),+],
                        diff_bits,
                        Cow::Borrowed(pow2_weights_32()),
                        weights,
                        $degree,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_bits_and_weights32_oracle!(
    Rv32PackedDivRemuAdapterOracleSparseTime,
    4,
    4,
    [rhs, rhs_is_zero, rem, diff],
    4,
    expr_rv32_packed_divremu_adapter
);

macro_rules! define_bits_and_weights64_oracle {
    ($name:ident, $n:expr, $m:expr, [$($col:ident),+ $(,)?], $degree:expr, $expr:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<$n, $m>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                $($col: SparseIdxVec<K>,)+
                diff_bits: Vec<SparseIdxVec<K>>,
                weights: [K; $m],
            ) -> Self {
                debug_assert_eq!(diff_bits.len(), 64);
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [$($col),+],
                        diff_bits,
                        Cow::Borrowed(pow2_weights_64()),
                        weights,
                        $degree,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_bits_and_weights64_oracle!(
    Rv64PackedDivRemuAdapterOracleSparseTime,
    4,
    4,
    [rhs, rhs_is_zero, rem, diff],
    4,
    expr_rv64_packed_divremu_adapter
);

define_sparse_time_expr_oracle!(
    Rv32PackedDivOracleSparseTime,
    6,
    7,
    expr_rv32_packed_div,
    [lhs_sign, rhs_sign, rhs_is_zero, q_abs, q_is_zero, val]
);

define_sparse_time_expr_oracle!(
    Rv64PackedDivOracleSparseTime,
    6,
    7,
    expr_rv64_packed_div,
    [lhs_sign, rhs_sign, rhs_is_zero, q_abs, q_is_zero, val]
);

define_sparse_time_expr_oracle!(
    Rv32PackedRemOracleSparseTime,
    6,
    7,
    expr_rv32_packed_rem,
    [lhs, lhs_sign, rhs_is_zero, r_abs, r_is_zero, val]
);

define_sparse_time_expr_oracle!(
    Rv64PackedRemOracleSparseTime,
    6,
    7,
    expr_rv64_packed_rem,
    [lhs, lhs_sign, rhs_is_zero, r_abs, r_is_zero, val]
);

define_bits_and_weights32_oracle!(
    Rv32PackedDivRemAdapterOracleSparseTime,
    10,
    7,
    [
        lhs,
        rhs,
        rhs_is_zero,
        lhs_sign,
        rhs_sign,
        q_abs,
        r_abs,
        mag,
        mag_is_zero,
        diff
    ],
    6,
    expr_rv32_packed_divrem_adapter
);

define_bits_and_weights64_oracle!(
    Rv64PackedDivRemAdapterOracleSparseTime,
    10,
    7,
    [
        lhs,
        rhs,
        rhs_is_zero,
        lhs_sign,
        rhs_sign,
        q_abs,
        r_abs,
        mag,
        mag_is_zero,
        diff
    ],
    6,
    expr_rv64_packed_divrem_adapter
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rv32PackedBitwiseOp2 {
    And,
    Andn,
    Or,
    Xor,
}

fn rv32_two_bit_digit_bits(inv2: K, inv6: K, x: K) -> (K, K) {
    let xm1 = x - K::ONE;
    let xm2 = x - K::from_u64(2);
    let xm3 = x - K::from_u64(3);

    let x_xm1 = x * xm1;
    let l1 = (x * xm2 * xm3) * inv2;
    let l3 = (x_xm1 * xm2) * inv6;
    let l2 = -(x_xm1 * xm3) * inv2;

    let bit0 = l1 + l3;
    let bit1 = l2 + l3;
    (bit0, bit1)
}

fn rv32_two_bit_digit_op(inv2: K, inv6: K, op: Rv32PackedBitwiseOp2, a: K, b: K) -> K {
    let (a0, a1) = rv32_two_bit_digit_bits(inv2, inv6, a);
    let (b0, b1) = rv32_two_bit_digit_bits(inv2, inv6, b);

    let two = K::from_u64(2);
    match op {
        Rv32PackedBitwiseOp2::And => {
            let r0 = a0 * b0;
            let r1 = a1 * b1;
            r0 + two * r1
        }
        Rv32PackedBitwiseOp2::Andn => {
            let r0 = a0 * (K::ONE - b0);
            let r1 = a1 * (K::ONE - b1);
            r0 + two * r1
        }
        Rv32PackedBitwiseOp2::Or => {
            let r0 = a0 + b0 - a0 * b0;
            let r1 = a1 + b1 - a1 * b1;
            r0 + two * r1
        }
        Rv32PackedBitwiseOp2::Xor => {
            let r0 = a0 + b0 - two * a0 * b0;
            let r1 = a1 + b1 - two * a1 * b1;
            r0 + two * r1
        }
    }
}

fn rv32_digit4_range_poly(x: K) -> K {
    x * (x - K::ONE) * (x - K::from_u64(2)) * (x - K::from_u64(3))
}

fn expr_rv32_packed_bitwise_adapter(cols: &[K; 2], bits: &[K], _bit_sum: K, w: &[K; 34]) -> K {
    let lhs = cols[0];
    let rhs = cols[1];
    debug_assert_eq!(bits.len(), 32);

    let mut lhs_recon = K::ZERO;
    let mut rhs_recon = K::ZERO;
    let mut range_sum = K::ZERO;
    for i in 0..16usize {
        let pow = K::from_u64(1u64 << (2 * i));
        let a = bits[i];
        let b = bits[16 + i];
        lhs_recon += a * pow;
        rhs_recon += b * pow;
        range_sum += w[2 + i] * rv32_digit4_range_poly(a);
        range_sum += w[2 + 16 + i] * rv32_digit4_range_poly(b);
    }

    w[0] * (lhs - lhs_recon) + w[1] * (rhs - rhs_recon) + range_sum
}

pub struct Rv32PackedBitwiseAdapterOracleSparseTime {
    core: SparseColsBitsExprOracle<2, 34>,
}

impl Rv32PackedBitwiseAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        weights: Vec<K>,
    ) -> Self {
        debug_assert_eq!(lhs_digits.len(), 16);
        debug_assert_eq!(rhs_digits.len(), 16);
        debug_assert_eq!(weights.len(), 34);
        let expr_weights: [K; 34] = weights
            .try_into()
            .unwrap_or_else(|v: Vec<K>| panic!("bitwise adapter weights length must be 34, got {}", v.len()));
        let mut digits = lhs_digits;
        digits.extend(rhs_digits);
        Self {
            core: SparseColsBitsExprOracle::new(
                r_cycle,
                has_lookup,
                [lhs, rhs],
                digits,
                Cow::Borrowed(&[]),
                expr_weights,
                6,
                expr_rv32_packed_bitwise_adapter,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv32PackedBitwiseAdapterOracleSparseTime);

fn expr_rv32_packed_bitwise(cols: &[K; 1], bits: &[K], _bit_sum: K, w: &[K; 2], op: Rv32PackedBitwiseOp2) -> K {
    let val = cols[0];
    debug_assert_eq!(bits.len(), 32);
    let mut out = K::ZERO;
    for i in 0..16usize {
        let a = bits[i];
        let b = bits[16 + i];
        let digit = rv32_two_bit_digit_op(w[0], w[1], op, a, b);
        out += digit * K::from_u64(1u64 << (2 * i));
    }
    out - val
}

fn expr_rv32_packed_and(cols: &[K; 1], bits: &[K], bit_sum: K, w: &[K; 2]) -> K {
    expr_rv32_packed_bitwise(cols, bits, bit_sum, w, Rv32PackedBitwiseOp2::And)
}

fn expr_rv32_packed_andn(cols: &[K; 1], bits: &[K], bit_sum: K, w: &[K; 2]) -> K {
    expr_rv32_packed_bitwise(cols, bits, bit_sum, w, Rv32PackedBitwiseOp2::Andn)
}

fn expr_rv32_packed_or(cols: &[K; 1], bits: &[K], bit_sum: K, w: &[K; 2]) -> K {
    expr_rv32_packed_bitwise(cols, bits, bit_sum, w, Rv32PackedBitwiseOp2::Or)
}

fn expr_rv32_packed_xor(cols: &[K; 1], bits: &[K], bit_sum: K, w: &[K; 2]) -> K {
    expr_rv32_packed_bitwise(cols, bits, bit_sum, w, Rv32PackedBitwiseOp2::Xor)
}

macro_rules! define_rv32_packed_bitwise_oracle {
    ($name:ident, $expr:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<1, 2>,
        }
        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                lhs_digits: Vec<SparseIdxVec<K>>,
                rhs_digits: Vec<SparseIdxVec<K>>,
                val: SparseIdxVec<K>,
            ) -> Self {
                debug_assert_eq!(lhs_digits.len(), 16);
                debug_assert_eq!(rhs_digits.len(), 16);
                let mut digits = lhs_digits;
                digits.extend(rhs_digits);
                let inv2 = K::from_u64(2).inverse();
                let inv6 = K::from_u64(6).inverse();
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [val],
                        digits,
                        Cow::Borrowed(&[]),
                        [inv2, inv6],
                        8,
                        $expr,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_rv32_packed_bitwise_oracle!(Rv32PackedAndOracleSparseTime, expr_rv32_packed_and);
define_rv32_packed_bitwise_oracle!(Rv32PackedAndnOracleSparseTime, expr_rv32_packed_andn);
define_rv32_packed_bitwise_oracle!(Rv32PackedOrOracleSparseTime, expr_rv32_packed_or);
define_rv32_packed_bitwise_oracle!(Rv32PackedXorOracleSparseTime, expr_rv32_packed_xor);

macro_rules! define_u_decomp_oracle {
    ($name:ident, $num_bits:expr) => {
        pub struct $name {
            core: SparseColsBitsExprOracle<1, 0>,
        }

        impl $name {
            pub fn new(
                r_cycle: &[K],
                has_lookup: SparseIdxVec<K>,
                x: SparseIdxVec<K>,
                bits: Vec<SparseIdxVec<K>>,
            ) -> Self {
                debug_assert_eq!(bits.len(), $num_bits);
                let weights: Cow<'static, [K]> = match $num_bits {
                    32 => Cow::Borrowed(pow2_weights_32()),
                    5 => Cow::Borrowed(pow2_weights_5()),
                    _ => Cow::Owned((0..$num_bits).map(|i| K::from_u64(1u64 << i)).collect()),
                };
                Self {
                    core: SparseColsBitsExprOracle::new(
                        r_cycle,
                        has_lookup,
                        [x],
                        bits,
                        weights,
                        [],
                        3,
                        expr_u_decomp_bw0,
                    ),
                }
            }
        }
        impl_round_oracle_via_core!($name);
    };
}

define_u_decomp_oracle!(U32DecompOracleSparseTime, 32);

define_u_decomp_oracle!(U5DecompOracleSparseTime, 5);

pub struct ZeroOracleSparseTime {
    rounds_remaining: usize,
    degree_bound: usize,
}

impl ZeroOracleSparseTime {
    pub fn new(num_rounds: usize, degree_bound: usize) -> Self {
        Self {
            rounds_remaining: num_rounds,
            degree_bound,
        }
    }
}

impl RoundOracle for ZeroOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        vec![K::ZERO; points.len()]
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, _r: K) {
        if self.rounds_remaining > 0 {
            self.rounds_remaining -= 1;
        }
    }
}

fn interp(f0: K, f1: K, x: K) -> K {
    f0 + (f1 - f0) * x
}

#[inline]
fn fold_dense_table_in_place(table: &mut Vec<K>, r: K) {
    if table.len() <= 1 {
        return;
    }
    let half = table.len() >> 1;
    for i in 0..half {
        let lo = table[2 * i];
        let hi = table[2 * i + 1];
        table[i] = interp(lo, hi, r);
    }
    table.truncate(half);
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

fn build_lt_prefix_table(inc_terms: &[(usize, K)], ell_n: usize) -> Vec<K> {
    let n = 1usize << ell_n;
    let mut deltas = vec![K::ZERO; n];
    for &(u, inc_u) in inc_terms.iter() {
        debug_assert!(u < n, "LT prefix table index out of range: u={u}, n={n}");
        let next = u.saturating_add(1);
        if next < n {
            deltas[next] += inc_u;
        }
    }
    let mut prefix = vec![K::ZERO; n];
    let mut run = K::ZERO;
    for i in 0..n {
        run += deltas[i];
        prefix[i] = run;
    }
    prefix
}

#[inline]
fn fill_one_minus_point(out: &mut Vec<K>, point: &[K]) {
    if out.len() != point.len() {
        out.resize(point.len(), K::ZERO);
    }
    for (dst, &src) in out.iter_mut().zip(point.iter()) {
        *dst = K::ONE - src;
    }
}

#[inline]
fn lt_eval_at_bool_index_with_one_minus(idx: usize, point: &[K], one_minus_point: &[K]) -> K {
    debug_assert_eq!(point.len(), one_minus_point.len());
    let mut suffix = K::ONE;
    let mut acc = K::ZERO;
    for i in (0..point.len()).rev() {
        let bit = (idx >> i) & 1;
        if bit == 0 {
            acc += point[i] * suffix;
            suffix *= one_minus_point[i];
        } else {
            suffix *= point[i];
        }
    }
    acc
}

fn val_pre_from_inc_terms(init_at_r_addr: K, inc_terms: &[(usize, K)], t_point: &[K], t_one_minus: &[K]) -> K {
    let mut acc = init_at_r_addr;
    for &(u, inc_u) in inc_terms.iter() {
        acc += inc_u * lt_eval_at_bool_index_with_one_minus(u, t_point, t_one_minus);
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

        let eq_addr = eq_addr_at_time(wa_bits, r_addr, t);
        let inc_at_r_addr = has_w * inc_t * eq_addr;
        if inc_at_r_addr != K::ZERO {
            out.push((t, inc_at_r_addr));
        }
    }
    out
}

fn eq_addr_at_time(bit_cols: &[SparseIdxVec<K>], r_addr: &[K], t: usize) -> K {
    debug_assert_eq!(bit_cols.len(), r_addr.len());
    let mut eq_addr = K::ONE;
    for (b, col) in bit_cols.iter().enumerate() {
        eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
    }
    eq_addr
}

fn eq_addr_singleton(bit_cols: &[SparseIdxVec<K>], r_addr: &[K]) -> K {
    debug_assert_eq!(bit_cols.len(), r_addr.len());
    let mut eq_addr = K::ONE;
    for (b, col) in bit_cols.iter().enumerate() {
        eq_addr *= eq_bit_affine(col.singleton_value(), r_addr[b]);
    }
    eq_addr
}

fn accumulate_pair_with_eq_addr_over_points<F>(
    ys: &mut [K],
    points: &[K],
    bit_cols: &[SparseIdxVec<K>],
    bit_hints: &mut [usize],
    r_addr: &[K],
    child0: usize,
    child1: usize,
    eq0s_scratch: &mut Vec<K>,
    d_eqs_scratch: &mut Vec<K>,
    mut coeff_at: F,
) where
    F: FnMut(K) -> K,
{
    debug_assert_eq!(bit_cols.len(), r_addr.len());
    debug_assert_eq!(bit_hints.len(), bit_cols.len());
    eq0s_scratch.clear();
    d_eqs_scratch.clear();
    eq0s_scratch.reserve(bit_cols.len());
    d_eqs_scratch.reserve(bit_cols.len());
    let mut const_prod = K::ONE;
    for (b, col) in bit_cols.iter().enumerate() {
        let e0 = eq_bit_affine(col.get_with_hint(child0, &mut bit_hints[b]), r_addr[b]);
        let de = eq_bit_affine(col.get_with_hint(child1, &mut bit_hints[b]), r_addr[b]) - e0;
        if de == K::ZERO {
            const_prod *= e0;
        } else {
            eq0s_scratch.push(e0);
            d_eqs_scratch.push(de);
        }
    }
    if const_prod == K::ZERO {
        return;
    }
    for (i, &x) in points.iter().enumerate() {
        let mut eq_addr = const_prod;
        for (e0, de) in eq0s_scratch.iter().zip(d_eqs_scratch.iter()) {
            eq_addr *= *e0 + *de * x;
        }
        if eq_addr == K::ZERO {
            continue;
        }
        ys[i] += coeff_at(x) * eq_addr;
    }
}

#[inline]
fn accumulate_pair_with_eq_addr_over_points_interp3(
    ys: &mut [K],
    points: &[K],
    bit_cols: &[SparseIdxVec<K>],
    bit_hints: &mut [usize],
    r_addr: &[K],
    child0: usize,
    child1: usize,
    eq0s_scratch: &mut Vec<K>,
    d_eqs_scratch: &mut Vec<K>,
    a0: K,
    a1: K,
    b0: K,
    b1: K,
    c0: K,
    c1: K,
) {
    debug_assert_eq!(bit_cols.len(), r_addr.len());
    debug_assert_eq!(bit_hints.len(), bit_cols.len());
    eq0s_scratch.clear();
    d_eqs_scratch.clear();
    eq0s_scratch.reserve(bit_cols.len());
    d_eqs_scratch.reserve(bit_cols.len());

    let mut const_prod = K::ONE;
    for (b, col) in bit_cols.iter().enumerate() {
        let e0 = eq_bit_affine(col.get_with_hint(child0, &mut bit_hints[b]), r_addr[b]);
        let de = eq_bit_affine(col.get_with_hint(child1, &mut bit_hints[b]), r_addr[b]) - e0;
        if de == K::ZERO {
            const_prod *= e0;
        } else {
            eq0s_scratch.push(e0);
            d_eqs_scratch.push(de);
        }
    }
    if const_prod == K::ZERO {
        return;
    }

    let da = a1 - a0;
    let db = b1 - b0;
    let dc = c1 - c0;

    for (i, &x) in points.iter().enumerate() {
        let mut eq_addr = const_prod;
        for (e0, de) in eq0s_scratch.iter().zip(d_eqs_scratch.iter()) {
            eq_addr *= *e0 + *de * x;
        }
        if eq_addr == K::ZERO {
            continue;
        }

        let ax = a0 + da * x;
        let bx = b0 + db * x;
        let cx = c0 + dc * x;
        let coeff = ax * bx * cx;
        if coeff == K::ZERO {
            continue;
        }
        ys[i] += coeff * eq_addr;
    }
}

#[derive(Clone, Debug)]
pub struct TwistLaneSparseCols {
    pub ra_bits: Vec<SparseIdxVec<K>>,
    pub wa_bits: Vec<SparseIdxVec<K>>,
    pub has_read: SparseIdxVec<K>,
    pub has_write: SparseIdxVec<K>,
    pub wv: SparseIdxVec<K>,
    pub rv: SparseIdxVec<K>,
    pub inc_at_write_addr: SparseIdxVec<K>,
}

pub struct IndexAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    addr_bits: Vec<SparseIdxVec<K>>,
    r_addr: Vec<K>,
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
        assert_cols_match_time(&addr_bits, 1usize << ell_n);

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            let eq_addr = eq_addr_at_time(&addr_bits, r_addr, t);
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
            },
            claim,
        )
    }
}

impl RoundOracle for IndexAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let eq_addr = eq_addr_singleton(&self.addr_bits, &self.r_addr);
            let v = self.prefix_eq * self.has_lookup.singleton_value() * eq_addr;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut eq0s_scratch = Vec::with_capacity(self.addr_bits.len());
        let mut d_eqs_scratch = Vec::with_capacity(self.addr_bits.len());
        let mut addr_hints = vec![0usize; self.addr_bits.len()];
        let mut has_lookup_hint = 0usize;
        for_each_sparse_parent_pair!(self.has_lookup.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get_with_hint(child0, &mut has_lookup_hint);
            let gate1 = self.has_lookup.get_with_hint(child1, &mut has_lookup_hint);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            accumulate_pair_with_eq_addr_over_points(
                &mut ys,
                points,
                &self.addr_bits,
                addr_hints.as_mut_slice(),
                &self.r_addr,
                child0,
                child1,
                &mut eq0s_scratch,
                &mut d_eqs_scratch,
                |x| interp(chi0, chi1, x) * interp(gate0, gate1, x),
            );
        });
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        2 + self.r_addr.len()
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
        self.bit_idx += 1;
    }
}

pub struct LazyWeightedBitnessOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    cols: Vec<SparseIdxVec<K>>,
    support: SparseIdxVec<K>,
    weights: Vec<K>,
    degree_bound: usize,
    pair_marks: Vec<u32>,
    pair_epoch: u32,
    pair_scratch: Vec<usize>,
    col_child0: Vec<K>,
    col_child1: Vec<K>,
}

impl LazyWeightedBitnessOracleSparseTime {
    #[inline]
    fn sparse_union_support(cols: &[SparseIdxVec<K>]) -> SparseIdxVec<K> {
        if cols.is_empty() {
            return SparseIdxVec::new(1);
        }
        let len = cols[0].len();
        let mut seen = vec![false; len];
        for col in cols {
            debug_assert_eq!(col.len(), len);
            for &(idx, _v) in col.entries() {
                seen[idx] = true;
            }
        }
        let mut entries = Vec::new();
        for (idx, hit) in seen.into_iter().enumerate() {
            if hit {
                entries.push((idx, K::ONE));
            }
        }
        SparseIdxVec::from_entries(len, entries)
    }

    pub fn new_with_cycle(r_cycle: &[K], cols: Vec<SparseIdxVec<K>>, weights: Vec<K>) -> Self {
        let ell_n = r_cycle.len();
        let col_count = cols.len();
        debug_assert_eq!(cols.len(), weights.len());
        for col in &cols {
            debug_assert_eq!(col.len(), 1usize << ell_n);
        }
        let support = Self::sparse_union_support(&cols);
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            cols,
            support,
            weights,
            degree_bound: 3,
            pair_marks: Vec::new(),
            pair_epoch: 1,
            pair_scratch: Vec::new(),
            col_child0: vec![K::ZERO; col_count],
            col_child1: vec![K::ZERO; col_count],
        }
    }
}

impl RoundOracle for LazyWeightedBitnessOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }

        if self.cols[0].len() == 1 {
            let mut acc = K::ZERO;
            for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                let b = col.singleton_value();
                acc += *w * b * (b - K::ONE);
            }
            let v = self.prefix_eq * acc;
            return vec![v; points.len()];
        }

        let pair_domain = self.support.len() >> 1;
        if self.pair_marks.len() < pair_domain {
            self.pair_marks.resize(pair_domain, 0);
        }
        self.pair_epoch = self.pair_epoch.wrapping_add(1);
        if self.pair_epoch == 0 {
            self.pair_marks.fill(0);
            self.pair_epoch = 1;
        }
        self.pair_scratch.clear();
        let epoch = self.pair_epoch;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if self.pair_marks[pair] != epoch {
                self.pair_marks[pair] = epoch;
                self.pair_scratch.push(pair);
            }
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut col_hints = vec![0usize; self.cols.len()];
        for &pair in self.pair_scratch.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            for (j, col) in self.cols.iter().enumerate() {
                self.col_child0[j] = col.get_with_hint(child0, &mut col_hints[j]);
                self.col_child1[j] = col.get_with_hint(child1, &mut col_hints[j]);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }

                let mut bit_sum = K::ZERO;
                for ((&b0, &b1), w) in self
                    .col_child0
                    .iter()
                    .zip(self.col_child1.iter())
                    .zip(self.weights.iter())
                {
                    if b0 == K::ZERO && b1 == K::ZERO {
                        continue;
                    }
                    let b_x = interp(b0, b1, x);
                    bit_sum += *w * b_x * (b_x - K::ONE);
                }

                ys[i] += chi_x * bit_sum;
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
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwistTimeCheckKind {
    Read,
    Write,
}

struct TwistTimeCheckOracleSparseTimeCore {
    kind: TwistTimeCheckKind,
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,

    r_addr: Vec<K>,
    addr_bits: Vec<SparseIdxVec<K>>,
    gate: SparseIdxVec<K>,
    value: SparseIdxVec<K>,
    inc_at_write_addr: Option<SparseIdxVec<K>>,

    init_at_r_addr: K,
    inc_terms_at_r_addr: Arc<Vec<(usize, K)>>,
    use_dense_lt_eval: bool,
    lt_prefix_folded: Vec<K>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    t_one_minus: Vec<K>,
    challenges: Vec<K>,
}

impl TwistTimeCheckOracleSparseTimeCore {
    const MAX_DENSE_LT_LOG_N: usize = 20;
    const FORCE_DENSE_LT_LOG_N: usize = 18;

    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: TwistTimeCheckKind,
        r_cycle: &[K],
        gate: SparseIdxVec<K>,
        value: SparseIdxVec<K>,
        addr_bits: Vec<SparseIdxVec<K>>,
        inc_at_write_addr: Option<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Arc<Vec<(usize, K)>>,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(gate.len(), 1usize << ell_n);
        debug_assert_eq!(value.len(), 1usize << ell_n);
        if let Some(inc) = inc_at_write_addr.as_ref() {
            debug_assert_eq!(inc.len(), 1usize << ell_n);
        }
        debug_assert_eq!(addr_bits.len(), ell_addr);
        let dense_cost = 1usize << ell_n;
        let sparse_cost = inc_terms_at_r_addr.len().saturating_mul(ell_n.max(1));
        let use_dense_lt_eval = if ell_n <= Self::FORCE_DENSE_LT_LOG_N {
            true
        } else {
            ell_n <= Self::MAX_DENSE_LT_LOG_N && sparse_cost > dense_cost
        };
        let lt_prefix_folded = if use_dense_lt_eval {
            build_lt_prefix_table(&inc_terms_at_r_addr, ell_n)
        } else {
            Vec::new()
        };

        Self {
            kind,
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            r_addr: r_addr.to_vec(),
            addr_bits,
            gate,
            value,
            inc_at_write_addr,
            init_at_r_addr,
            inc_terms_at_r_addr,
            use_dense_lt_eval,
            lt_prefix_folded,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            t_one_minus: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for TwistTimeCheckOracleSparseTimeCore {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let is_write = matches!(self.kind, TwistTimeCheckKind::Write);
        if self.gate.len() == 1 {
            let eq_addr = eq_addr_singleton(&self.addr_bits, &self.r_addr);
            let t_point = self.challenges.as_slice();
            let val_pre = if self.use_dense_lt_eval {
                self.init_at_r_addr + self.lt_prefix_folded.first().copied().unwrap_or(K::ZERO)
            } else {
                fill_one_minus_point(&mut self.t_one_minus, t_point);
                val_pre_from_inc_terms(
                    self.init_at_r_addr,
                    self.inc_terms_at_r_addr.as_ref(),
                    t_point,
                    self.t_one_minus.as_slice(),
                )
            };
            let inc = self
                .inc_at_write_addr
                .as_ref()
                .map(|c| c.singleton_value())
                .unwrap_or(K::ZERO);
            let term = if is_write {
                self.value.singleton_value() - val_pre - inc
            } else {
                val_pre - self.value.singleton_value()
            };
            let v = self.prefix_eq * self.gate.singleton_value() * term * eq_addr;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut eq0s_scratch = Vec::with_capacity(self.addr_bits.len());
        let mut d_eqs_scratch = Vec::with_capacity(self.addr_bits.len());
        let mut addr_hints = vec![0usize; self.addr_bits.len()];
        let mut gate_hint = 0usize;
        let mut value_hint = 0usize;
        let mut inc_hint = 0usize;
        for_each_sparse_parent_pair!(self.gate.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.gate.get_with_hint(child0, &mut gate_hint);
            let gate1 = self.gate.get_with_hint(child1, &mut gate_hint);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (val_pre0, val_pre1) = if self.use_dense_lt_eval {
                debug_assert!(child1 < self.lt_prefix_folded.len());
                (
                    self.init_at_r_addr + self.lt_prefix_folded[child0],
                    self.init_at_r_addr + self.lt_prefix_folded[child1],
                )
            } else {
                fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
                fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
                fill_one_minus_point(&mut self.t_one_minus, &self.t_child0);
                let pre0 = val_pre_from_inc_terms(
                    self.init_at_r_addr,
                    self.inc_terms_at_r_addr.as_ref(),
                    &self.t_child0,
                    self.t_one_minus.as_slice(),
                );
                fill_one_minus_point(&mut self.t_one_minus, &self.t_child1);
                let pre1 = val_pre_from_inc_terms(
                    self.init_at_r_addr,
                    self.inc_terms_at_r_addr.as_ref(),
                    &self.t_child1,
                    self.t_one_minus.as_slice(),
                );
                (pre0, pre1)
            };

            let value0 = self.value.get_with_hint(child0, &mut value_hint);
            let value1 = self.value.get_with_hint(child1, &mut value_hint);
            let (inc0, inc1) = if let Some(inc_col) = self.inc_at_write_addr.as_ref() {
                (
                    inc_col.get_with_hint(child0, &mut inc_hint),
                    inc_col.get_with_hint(child1, &mut inc_hint),
                )
            } else {
                (K::ZERO, K::ZERO)
            };
            let term0 = if is_write {
                value0 - val_pre0 - inc0
            } else {
                val_pre0 - value0
            };
            let term1 = if is_write {
                value1 - val_pre1 - inc1
            } else {
                val_pre1 - value1
            };

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            accumulate_pair_with_eq_addr_over_points_interp3(
                &mut ys,
                points,
                &self.addr_bits,
                addr_hints.as_mut_slice(),
                &self.r_addr,
                child0,
                child1,
                &mut eq0s_scratch,
                &mut d_eqs_scratch,
                chi0,
                chi1,
                gate0,
                gate1,
                term0,
                term1,
            );
        });
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        3 + self.r_addr.len()
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.gate.fold_round_in_place(r);
        self.value.fold_round_in_place(r);
        if let Some(inc) = self.inc_at_write_addr.as_mut() {
            inc.fold_round_in_place(r);
        }
        for col in self.addr_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        if self.use_dense_lt_eval {
            fold_dense_table_in_place(&mut self.lt_prefix_folded, r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

pub struct TwistReadCheckOracleSparseTime {
    core: TwistTimeCheckOracleSparseTimeCore,
}

impl TwistReadCheckOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
    ) -> Self {
        let ell_addr = r_addr.len();
        debug_assert_eq!(ra_bits.len(), ell_addr);
        debug_assert_eq!(wa_bits.len(), ell_addr);

        let inc_terms_at_r_addr = build_inc_terms_at_r_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr);

        Self::new_with_inc_terms_shared(
            r_cycle,
            has_read,
            rv,
            ra_bits,
            r_addr,
            init_at_r_addr,
            Arc::new(inc_terms_at_r_addr),
        )
    }

    pub fn new_with_inc_terms(
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Vec<(usize, K)>,
    ) -> Self {
        Self::new_with_inc_terms_shared(
            r_cycle,
            has_read,
            rv,
            ra_bits,
            r_addr,
            init_at_r_addr,
            Arc::new(inc_terms_at_r_addr),
        )
    }

    pub fn new_with_inc_terms_shared(
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Arc<Vec<(usize, K)>>,
    ) -> Self {
        Self {
            core: TwistTimeCheckOracleSparseTimeCore::new(
                TwistTimeCheckKind::Read,
                r_cycle,
                has_read,
                rv,
                ra_bits,
                None,
                r_addr,
                init_at_r_addr,
                inc_terms_at_r_addr,
            ),
        }
    }
}
impl_round_oracle_via_core!(TwistReadCheckOracleSparseTime);

pub struct TwistWriteCheckOracleSparseTime {
    core: TwistTimeCheckOracleSparseTimeCore,
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
        let ell_addr = r_addr.len();
        debug_assert_eq!(wa_bits.len(), ell_addr);

        let inc_terms_at_r_addr = build_inc_terms_at_r_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr);

        Self::new_with_inc_terms_shared(
            r_cycle,
            has_write,
            wv,
            inc_at_write_addr,
            wa_bits,
            r_addr,
            init_at_r_addr,
            Arc::new(inc_terms_at_r_addr),
        )
    }

    pub fn new_with_inc_terms(
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Vec<(usize, K)>,
    ) -> Self {
        Self::new_with_inc_terms_shared(
            r_cycle,
            has_write,
            wv,
            inc_at_write_addr,
            wa_bits,
            r_addr,
            init_at_r_addr,
            Arc::new(inc_terms_at_r_addr),
        )
    }

    pub fn new_with_inc_terms_shared(
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Arc<Vec<(usize, K)>>,
    ) -> Self {
        Self {
            core: TwistTimeCheckOracleSparseTimeCore::new(
                TwistTimeCheckKind::Write,
                r_cycle,
                has_write,
                wv,
                wa_bits,
                Some(inc_at_write_addr),
                r_addr,
                init_at_r_addr,
                inc_terms_at_r_addr,
            ),
        }
    }
}
impl_round_oracle_via_core!(TwistWriteCheckOracleSparseTime);

enum TwistWriteEqAddrMode {
    TotalInc,
    ValEval { lt_folded: Vec<K> },
}

struct TwistWriteEqAddrOracleSparseTimeCore {
    r_addr: Vec<K>,
    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,
    mode: TwistWriteEqAddrMode,
    challenges: Vec<K>,
}

fn claim_write_eq_addr<F>(
    wa_bits: &[SparseIdxVec<K>],
    has_write: &SparseIdxVec<K>,
    inc_at_write_addr: &SparseIdxVec<K>,
    r_addr: &[K],
    mut time_weight: F,
) -> K
where
    F: FnMut(usize) -> K,
{
    let mut claim = K::ZERO;
    for &(t, gate) in has_write.entries() {
        let inc_t = inc_at_write_addr.get(t);
        if gate == K::ZERO || inc_t == K::ZERO {
            continue;
        }
        let eq_addr = eq_addr_at_time(wa_bits, r_addr, t);
        claim += gate * inc_t * eq_addr * time_weight(t);
    }
    claim
}

impl TwistWriteEqAddrOracleSparseTimeCore {
    fn new_with_mode<F>(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
        mode: TwistWriteEqAddrMode,
        mut time_weight: F,
    ) -> (Self, K)
    where
        F: FnMut(usize) -> K,
    {
        let ell_n = log2_pow2(has_write.len());
        let ell_addr = r_addr.len();
        let pow2_n = 1usize << ell_n;
        debug_assert_eq!(inc_at_write_addr.len(), pow2_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);
        assert_cols_match_time(&wa_bits, pow2_n);

        let claim = claim_write_eq_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr, |t| time_weight(t));
        (
            Self {
                r_addr: r_addr.to_vec(),
                wa_bits,
                has_write,
                inc_at_write_addr,
                mode,
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }

    fn new_val(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
        r_time: &[K],
    ) -> (Self, K) {
        let ell_n = log2_pow2(has_write.len());
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        let lt_folded = (0..(1usize << ell_n))
            .map(|idx| lt_eval_at_bool_index(idx, r_time))
            .collect();
        Self::new_with_mode(
            wa_bits,
            has_write,
            inc_at_write_addr,
            r_addr,
            TwistWriteEqAddrMode::ValEval { lt_folded },
            |t| lt_eval_at_bool_index(t, r_time),
        )
    }

    fn new_total(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
    ) -> (Self, K) {
        Self::new_with_mode(
            wa_bits,
            has_write,
            inc_at_write_addr,
            r_addr,
            TwistWriteEqAddrMode::TotalInc,
            |_t| K::ONE,
        )
    }
}

impl RoundOracle for TwistWriteEqAddrOracleSparseTimeCore {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let eq_addr = eq_addr_singleton(&self.wa_bits, &self.r_addr);
            let lt = match &self.mode {
                TwistWriteEqAddrMode::TotalInc => K::ONE,
                TwistWriteEqAddrMode::ValEval { lt_folded } => lt_folded.first().copied().unwrap_or(K::ZERO),
            };
            let v = self.has_write.singleton_value() * self.inc_at_write_addr.singleton_value() * eq_addr * lt;
            return vec![v; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut eq0s_scratch = Vec::with_capacity(self.wa_bits.len());
        let mut d_eqs_scratch = Vec::with_capacity(self.wa_bits.len());
        let mut wa_hints = vec![0usize; self.wa_bits.len()];
        let mut has_write_hint = 0usize;
        let mut inc_hint = 0usize;
        for_each_sparse_parent_pair!(self.has_write.entries(), pair, {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get_with_hint(child0, &mut has_write_hint);
            let gate1 = self.has_write.get_with_hint(child1, &mut has_write_hint);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }
            let inc0 = self.inc_at_write_addr.get_with_hint(child0, &mut inc_hint);
            let inc1 = self.inc_at_write_addr.get_with_hint(child1, &mut inc_hint);

            let (lt0, lt1) = match &mut self.mode {
                TwistWriteEqAddrMode::TotalInc => (K::ONE, K::ONE),
                TwistWriteEqAddrMode::ValEval { lt_folded } => {
                    debug_assert!(child1 < lt_folded.len());
                    (lt_folded[child0], lt_folded[child1])
                }
            };

            accumulate_pair_with_eq_addr_over_points_interp3(
                &mut ys,
                points,
                &self.wa_bits,
                wa_hints.as_mut_slice(),
                &self.r_addr,
                child0,
                child1,
                &mut eq0s_scratch,
                &mut d_eqs_scratch,
                gate0,
                gate1,
                inc0,
                inc1,
                lt0,
                lt1,
            );
        });
        ys
    }

    fn num_rounds(&self) -> usize {
        log2_pow2(self.has_write.len())
    }

    fn degree_bound(&self) -> usize {
        match self.mode {
            TwistWriteEqAddrMode::TotalInc => 2 + self.r_addr.len(),
            TwistWriteEqAddrMode::ValEval { .. } => 3 + self.r_addr.len(),
        }
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
        if let TwistWriteEqAddrMode::ValEval { lt_folded } = &mut self.mode {
            fold_dense_table_in_place(lt_folded, r);
        }
        self.challenges.push(r);
    }
}

pub struct TwistValEvalOracleSparseTime {
    core: TwistWriteEqAddrOracleSparseTimeCore,
}

impl TwistValEvalOracleSparseTime {
    pub fn new(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
        r_time: &[K],
    ) -> (Self, K) {
        let (core, claim) =
            TwistWriteEqAddrOracleSparseTimeCore::new_val(wa_bits, has_write, inc_at_write_addr, r_addr, r_time);
        (Self { core }, claim)
    }
}
impl_round_oracle_via_core!(TwistValEvalOracleSparseTime);

pub struct TwistTotalIncOracleSparseTime {
    core: TwistWriteEqAddrOracleSparseTimeCore,
}

impl TwistTotalIncOracleSparseTime {
    pub fn new(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
    ) -> (Self, K) {
        let (core, claim) =
            TwistWriteEqAddrOracleSparseTimeCore::new_total(wa_bits, has_write, inc_at_write_addr, r_addr);
        (Self { core }, claim)
    }
}
impl_round_oracle_via_core!(TwistTotalIncOracleSparseTime);

fn update_prefix_weights_in_place<I>(weights: &mut [K], addrs: I, bit_idx: usize, r: K)
where
    I: IntoIterator<Item = usize>,
{
    let r0 = K::ONE - r;
    for (w, a) in weights.iter_mut().zip(addrs) {
        if ((a >> bit_idx) & 1) == 1 {
            *w *= r;
        } else {
            *w *= r0;
        }
    }
}

fn addr_from_sparse_bits_at_time(bit_cols: &[SparseIdxVec<K>], t: usize) -> usize {
    let mut out = 0usize;
    for (b, col) in bit_cols.iter().enumerate() {
        if col.get(t) == K::ONE {
            out |= 1usize << b;
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AddrEventKind {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug)]
struct AddrEvent {
    t: usize,
    kind: AddrEventKind,
    chi_t: K,
    gate: K,
    addr: usize,
    val: K,
    inc: K,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AddrCheckMode {
    Read,
    Write,
}

impl AddrCheckMode {
    fn wants_check(self, kind: AddrEventKind) -> bool {
        matches!(
            (self, kind),
            (AddrCheckMode::Read, AddrEventKind::Read) | (AddrCheckMode::Write, AddrEventKind::Write)
        )
    }

    fn check_expr(self, mem_x: K, val: K, inc: K) -> K {
        match self {
            AddrCheckMode::Read => mem_x - val,
            AddrCheckMode::Write => val - mem_x - inc,
        }
    }
}

struct AddrLaneCheckCore {
    ell_addr: usize,
    bit_idx: usize,
    mode: AddrCheckMode,

    events: Vec<AddrEvent>,
    event_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,

    mem_scratch: std::collections::HashMap<usize, K>,
}

impl AddrLaneCheckCore {
    fn new(init_sparse: Vec<(usize, K)>, ell_addr: usize, mode: AddrCheckMode, mut events: Vec<AddrEvent>) -> Self {
        let pow2_addr = 1usize << ell_addr;
        for (addr, _) in init_sparse.iter() {
            debug_assert!(*addr < pow2_addr, "init address out of range");
        }
        for event in events.iter() {
            debug_assert!(event.addr < pow2_addr, "event address out of range");
        }

        events.sort_unstable_by_key(|e| (e.t, e.kind));

        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        let n_events = events.len();
        let init_len = init_addrs.len();

        Self {
            ell_addr,
            bit_idx: 0,
            mode,
            events,
            event_prefix_w: vec![K::ONE; n_events],
            init_prefix_w: vec![K::ONE; init_len],
            init_addrs,
            init_vals,
            mem_scratch: std::collections::HashMap::with_capacity(init_len),
        }
    }
}

impl RoundOracle for AddrLaneCheckCore {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            let mut i = 0usize;
            while i < self.events.len() {
                let t = self.events[i].t;
                let start = i;
                while i < self.events.len() && self.events[i].t == t {
                    i += 1;
                }
                let end = i;

                for k in start..end {
                    let event = self.events[k];
                    if !self.mode.wants_check(event.kind) || event.gate == K::ZERO {
                        continue;
                    }
                    sum += event.chi_t
                        * event.gate
                        * self.event_prefix_w[k]
                        * self.mode.check_expr(mem, event.val, event.inc);
                }

                for k in start..end {
                    let event = self.events[k];
                    if event.kind != AddrEventKind::Write || event.gate == K::ZERO {
                        continue;
                    }
                    mem += event.inc * event.gate * self.event_prefix_w[k];
                }
            }

            return vec![sum; points.len()];
        }

        self.mem_scratch.clear();
        let bit_idx = self.bit_idx;
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

        let mut ys = vec![K::ZERO; points.len()];
        let mut i = 0usize;
        while i < self.events.len() {
            let t = self.events[i].t;
            let start = i;
            while i < self.events.len() && self.events[i].t == t {
                i += 1;
            }
            let end = i;

            for k in start..end {
                let event = self.events[k];
                if !self.mode.wants_check(event.kind) || event.gate == K::ZERO {
                    continue;
                }

                let base = event.addr >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                let dv = v1 - v0;
                let bit = (event.addr >> bit_idx) & 1;
                let pref = self.event_prefix_w[k];
                let coef = event.chi_t * event.gate * pref;

                for (j, &x) in points.iter().enumerate() {
                    let mem_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    ys[j] += coef * addr_factor * self.mode.check_expr(mem_x, event.val, event.inc);
                }
            }

            for k in start..end {
                let event = self.events[k];
                if event.kind != AddrEventKind::Write || event.gate == K::ZERO {
                    continue;
                }
                let idx = event.addr >> bit_idx;
                let delta = event.inc * event.gate * self.event_prefix_w[k];
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
        2
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        update_prefix_weights_in_place(
            &mut self.init_prefix_w,
            self.init_addrs.iter().copied(),
            self.bit_idx,
            r,
        );
        update_prefix_weights_in_place(
            &mut self.event_prefix_w,
            self.events.iter().map(|e| e.addr),
            self.bit_idx,
            r,
        );
        self.bit_idx += 1;
    }
}

fn push_read_events(
    out: &mut Vec<AddrEvent>,
    r_cycle: &[K],
    has_read: &SparseIdxVec<K>,
    rv: &SparseIdxVec<K>,
    ra_bits: &[SparseIdxVec<K>],
) {
    for &(t, gate) in has_read.entries() {
        if gate == K::ZERO {
            continue;
        }
        out.push(AddrEvent {
            t,
            kind: AddrEventKind::Read,
            chi_t: chi_at_bool_index(r_cycle, t),
            gate,
            addr: addr_from_sparse_bits_at_time(ra_bits, t),
            val: rv.get(t),
            inc: K::ZERO,
        });
    }
}

fn push_write_events(
    out: &mut Vec<AddrEvent>,
    r_cycle: &[K],
    has_write: &SparseIdxVec<K>,
    wa_bits: &[SparseIdxVec<K>],
    inc_at_write_addr: &SparseIdxVec<K>,
    wv: Option<&SparseIdxVec<K>>,
) {
    for &(t, gate) in has_write.entries() {
        if gate == K::ZERO {
            continue;
        }
        out.push(AddrEvent {
            t,
            kind: AddrEventKind::Write,
            chi_t: chi_at_bool_index(r_cycle, t),
            gate,
            addr: addr_from_sparse_bits_at_time(wa_bits, t),
            val: wv.map(|col| col.get(t)).unwrap_or(K::ZERO),
            inc: inc_at_write_addr.get(t),
        });
    }
}

fn assert_init_sparse_in_range(init_sparse: &[(usize, K)], ell_addr: usize) {
    let pow2_addr = 1usize << ell_addr;
    for (addr, _) in init_sparse.iter() {
        debug_assert!(*addr < pow2_addr);
    }
}

fn assert_cols_match_time(cols: &[SparseIdxVec<K>], pow2_time: usize) {
    for col in cols {
        debug_assert_eq!(col.len(), pow2_time);
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_singlelane_read_addr_events(
    r_cycle: &[K],
    has_read: &SparseIdxVec<K>,
    rv: &SparseIdxVec<K>,
    ra_bits: &[SparseIdxVec<K>],
    has_write: &SparseIdxVec<K>,
    wa_bits: &[SparseIdxVec<K>],
    inc_at_write_addr: &SparseIdxVec<K>,
) -> (usize, Vec<AddrEvent>) {
    let pow2_time = 1usize << r_cycle.len();
    let ell_addr = ra_bits.len();

    debug_assert_eq!(has_read.len(), pow2_time);
    debug_assert_eq!(rv.len(), pow2_time);
    debug_assert_eq!(has_write.len(), pow2_time);
    debug_assert_eq!(inc_at_write_addr.len(), pow2_time);
    debug_assert_eq!(wa_bits.len(), ell_addr);
    assert_cols_match_time(ra_bits, pow2_time);
    assert_cols_match_time(wa_bits, pow2_time);

    let mut events = Vec::new();
    push_read_events(&mut events, r_cycle, has_read, rv, ra_bits);
    push_write_events(&mut events, r_cycle, has_write, wa_bits, inc_at_write_addr, None);
    (ell_addr, events)
}

fn collect_singlelane_write_addr_events(
    r_cycle: &[K],
    has_write: &SparseIdxVec<K>,
    wv: &SparseIdxVec<K>,
    wa_bits: &[SparseIdxVec<K>],
    inc_at_write_addr: &SparseIdxVec<K>,
) -> (usize, Vec<AddrEvent>) {
    let pow2_time = 1usize << r_cycle.len();
    let ell_addr = wa_bits.len();

    debug_assert_eq!(has_write.len(), pow2_time);
    debug_assert_eq!(wv.len(), pow2_time);
    debug_assert_eq!(inc_at_write_addr.len(), pow2_time);
    assert_cols_match_time(wa_bits, pow2_time);

    let mut events = Vec::new();
    push_write_events(&mut events, r_cycle, has_write, wa_bits, inc_at_write_addr, Some(wv));
    (ell_addr, events)
}

fn collect_multilane_addr_events(
    r_cycle: &[K],
    lanes: &[TwistLaneSparseCols],
    mode: AddrCheckMode,
) -> (usize, Vec<AddrEvent>) {
    debug_assert!(!lanes.is_empty());
    let pow2_time = 1usize << r_cycle.len();
    let ell_addr = match mode {
        AddrCheckMode::Read => lanes[0].ra_bits.len(),
        AddrCheckMode::Write => lanes[0].wa_bits.len(),
    };
    let mut events = Vec::new();

    for lane in lanes {
        debug_assert_eq!(lane.has_write.len(), pow2_time);
        debug_assert_eq!(lane.inc_at_write_addr.len(), pow2_time);
        debug_assert_eq!(lane.wa_bits.len(), ell_addr);
        assert_cols_match_time(&lane.wa_bits, pow2_time);

        if matches!(mode, AddrCheckMode::Read) {
            debug_assert_eq!(lane.has_read.len(), pow2_time);
            debug_assert_eq!(lane.rv.len(), pow2_time);
            debug_assert_eq!(lane.ra_bits.len(), ell_addr);
            assert_cols_match_time(&lane.ra_bits, pow2_time);
            push_read_events(&mut events, r_cycle, &lane.has_read, &lane.rv, &lane.ra_bits);
            push_write_events(
                &mut events,
                r_cycle,
                &lane.has_write,
                &lane.wa_bits,
                &lane.inc_at_write_addr,
                None,
            );
        } else {
            debug_assert_eq!(lane.wv.len(), pow2_time);
            push_write_events(
                &mut events,
                r_cycle,
                &lane.has_write,
                &lane.wa_bits,
                &lane.inc_at_write_addr,
                Some(&lane.wv),
            );
        }
    }
    (ell_addr, events)
}

pub struct TwistReadCheckAddrOracleSparseTime {
    core: AddrLaneCheckCore,
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
        let (ell_addr, events) = collect_singlelane_read_addr_events(
            r_cycle,
            &has_read,
            &rv,
            ra_bits,
            &has_write,
            wa_bits,
            &inc_at_write_addr,
        );
        assert_init_sparse_in_range(&init_sparse, ell_addr);

        Self {
            core: AddrLaneCheckCore::new(init_sparse, ell_addr, AddrCheckMode::Read, events),
        }
    }
}
impl_round_oracle_via_core!(TwistReadCheckAddrOracleSparseTime);

pub struct TwistWriteCheckAddrOracleSparseTime {
    core: AddrLaneCheckCore,
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
        let (ell_addr, events) =
            collect_singlelane_write_addr_events(r_cycle, &has_write, &wv, wa_bits, &inc_at_write_addr);
        assert_init_sparse_in_range(&init_sparse, ell_addr);

        Self {
            core: AddrLaneCheckCore::new(init_sparse, ell_addr, AddrCheckMode::Write, events),
        }
    }
}
impl_round_oracle_via_core!(TwistWriteCheckAddrOracleSparseTime);

pub struct TwistReadCheckAddrOracleSparseTimeMultiLane {
    core: AddrLaneCheckCore,
}

impl TwistReadCheckAddrOracleSparseTimeMultiLane {
    pub fn new(init_sparse: Vec<(usize, K)>, r_cycle: &[K], lanes: &[TwistLaneSparseCols]) -> Self {
        let (ell_addr, events) = collect_multilane_addr_events(r_cycle, lanes, AddrCheckMode::Read);
        assert_init_sparse_in_range(&init_sparse, ell_addr);

        Self {
            core: AddrLaneCheckCore::new(init_sparse, ell_addr, AddrCheckMode::Read, events),
        }
    }
}
impl_round_oracle_via_core!(TwistReadCheckAddrOracleSparseTimeMultiLane);

pub struct TwistWriteCheckAddrOracleSparseTimeMultiLane {
    core: AddrLaneCheckCore,
}

impl TwistWriteCheckAddrOracleSparseTimeMultiLane {
    pub fn new(init_sparse: Vec<(usize, K)>, r_cycle: &[K], lanes: &[TwistLaneSparseCols]) -> Self {
        let (ell_addr, events) = collect_multilane_addr_events(r_cycle, lanes, AddrCheckMode::Write);
        assert_init_sparse_in_range(&init_sparse, ell_addr);

        Self {
            core: AddrLaneCheckCore::new(init_sparse, ell_addr, AddrCheckMode::Write, events),
        }
    }
}
impl_round_oracle_via_core!(TwistWriteCheckAddrOracleSparseTimeMultiLane);

pub struct AddressLookupOracle {
    core: ProductRoundOracle,
}

impl AddressLookupOracle {
    pub fn new(
        addr_bits: &[SparseIdxVec<K>],
        has_lookup: &SparseIdxVec<K>,
        table: &[K],
        r_cycle: &[K],
        ell_addr: usize,
    ) -> (Self, K) {
        let pow2_cycle = 1usize << r_cycle.len();
        let pow2_addr = 1usize << ell_addr;

        debug_assert_eq!(addr_bits.len(), ell_addr);
        for col in addr_bits.iter() {
            debug_assert_eq!(col.len(), pow2_cycle);
        }
        debug_assert_eq!(has_lookup.len(), pow2_cycle);

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

            let addr_t = addr_from_sparse_bits_at_time(addr_bits, t);
            weight_table[addr_t] += weight_t;
        }

        for addr in 0..pow2_addr.min(table.len()) {
            claimed_sum += table[addr] * weight_table[addr];
        }

        let mut table_k: Vec<K> = table.iter().copied().collect();
        table_k.resize(pow2_addr, K::ZERO);
        let core = ProductRoundOracle::new(vec![table_k, weight_table], 2);

        (Self { core }, claimed_sum)
    }

    pub fn final_value(&self) -> Option<K> {
        self.core.value()
    }

    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(AddressLookupOracle);

pub fn table_mle_eval(table: &[K], r_addr: &[K]) -> K {
    let ell = r_addr.len();
    let pow2 = 1usize << ell;

    let mut result = K::ZERO;
    for (idx, &val) in table.iter().enumerate().take(pow2) {
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
