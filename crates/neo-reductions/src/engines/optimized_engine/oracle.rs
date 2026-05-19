//! Optimized RoundOracle for Q(X) evaluation in Π_CCS.
//!
//! This oracle uses factored algebra, precomputed terms, and cached sparse formats
//! to efficiently evaluate the Q polynomial during sumcheck rounds. Mathematically
//! equivalent to paper-exact but ~10x faster.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{Fq, KExtensions, D, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::sync::Arc;

use crate::sumcheck::RoundOracle;

use super::common::Challenges;
pub use super::sparse::SparseCache;
use crate::superneo_eval::{SuperneoEvalCache, SuperneoZBlocks};

/// NC-only oracle for the split-NC Π_CCS variant.
///
/// Variable order (rounds): first the `ell_m` column bits, then the `ell_d` Ajtai bits.
///
/// This oracle evaluates the NC polynomial:
///   Q_nc(s, α) = eq(s, β_m) * eq(α, β_a) * Σ_i γ^{i+1} · N_i(Ẑ_i(α, s))
/// where `N_i(·)` is the digit-range (norm-check) range polynomial.
pub struct NcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    pub mcs_witnesses: &'a [CcsWitness<F>],
    pub me_witnesses: &'a [Mat<F>],
    pub ch: Challenges,

    pub ell_d: usize,
    pub ell_m: usize,
    pub d_sc: usize,

    pub round_idx: usize,
    pub col_chals: Vec<K>,
    pub ajtai_chals: Vec<K>,

    // Streaming tables over the remaining column bits.
    cur_len: usize,
    eq_beta_m_tbl: Vec<K>,
    // digits_tables[i][col_mask][rho] = balanced base-b digit lane for live logical columns.
    // Zero padding to the power-of-two sumcheck domain is implicit.
    digits_tables: Vec<Vec<[K; D]>>,
    // Bitmask of live digit lanes for each row in `digits_tables`; dense rows remain authority.
    digit_lane_masks: Vec<Vec<u64>>,
    // weights[i][rho] = γ^{i+1} * χ_{β_a}(rho)
    weights: Vec<[K; D]>,
    // Cached t^2 values for the symmetric range polynomial.
    range_t_sq: Vec<K>,
    // True while every entry in every `digits_tables[i]` has imag() == 0
    // (initially true: witnesses are base-field). Flipped to false the
    // first time `fold` runs with a challenge `r` having nonzero imag.
    digit_tables_all_real: bool,
}

impl<'a, F> NcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    pub fn new(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [CcsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_m: usize,
        d_sc: usize,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one witness for NC");
        assert!(
            ch.beta_m.len() == ell_m,
            "NcOracle: beta_m length mismatch (expected {}, got {})",
            ell_m,
            ch.beta_m.len()
        );
        assert!(
            ch.beta_a.len() == ell_d,
            "NcOracle: beta_a length mismatch (expected {}, got {})",
            ell_d,
            ch.beta_a.len()
        );

        #[cfg(feature = "perf-timers")]
        let t_new_total = std::time::Instant::now();

        let m_pad = 1usize << ell_m;

        // Column-domain χ_{β_m} table.
        #[cfg(feature = "perf-timers")]
        let t_eq_beta_m = std::time::Instant::now();
        let eq_beta_m_tbl = chi_tail_weights(&ch.beta_m);
        debug_assert_eq!(eq_beta_m_tbl.len(), m_pad, "chi(beta_m) length mismatch");
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "NcOracle::new: eq_beta_m table             {:.2?}",
            t_eq_beta_m.elapsed()
        );

        // Gather all Z witnesses in order: MCS first, then ME.
        #[cfg(feature = "perf-timers")]
        let t_gather = std::time::Instant::now();
        let mut all_witnesses: Vec<&Mat<F>> = Vec::with_capacity(mcs_witnesses.len() + me_witnesses.len());
        for w in mcs_witnesses {
            all_witnesses.push(&w.Z);
        }
        for z in me_witnesses {
            all_witnesses.push(z);
        }
        #[cfg(feature = "perf-timers")]
        eprintln!("NcOracle::new: gather witnesses            {:.2?}", t_gather.elapsed());
        // Precompute χ_{β_a}(rho) for rho=0..D-1.
        #[cfg(feature = "perf-timers")]
        let t_weights = std::time::Instant::now();
        let mut w_beta_a = [K::ZERO; D];
        for rho in 0..D {
            w_beta_a[rho] = eq_points_bool_mask(rho, &ch.beta_a);
        }

        // weights[i][rho] = γ^{i+1} * χ_{β_a}(rho)
        let mut weights: Vec<[K; D]> = Vec::with_capacity(all_witnesses.len());
        let mut g = ch.gamma; // γ^1
        for _ in 0..all_witnesses.len() {
            let mut wi = [K::ZERO; D];
            for rho in 0..D {
                wi[rho] = g * w_beta_a[rho];
            }
            weights.push(wi);
            g *= ch.gamma;
        }
        #[cfg(feature = "perf-timers")]
        eprintln!("NcOracle::new: weights                     {:.2?}", t_weights.elapsed());
        // Column-domain digit tables.
        #[cfg(feature = "perf-timers")]
        let t_digits = std::time::Instant::now();
        let built_digit_tables: Vec<(Vec<[K; D]>, Vec<u64>)> = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                all_witnesses
                    .par_iter()
                    .map(|Zi| {
                        crate::common::build_witness_nc_digit_table_with_masks(params, Zi, s.m)
                            .unwrap_or_else(|e| panic!("NcOracle::new: failed to build NC digit table: {e}"))
                    })
                    .collect()
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                all_witnesses
                    .iter()
                    .map(|Zi| {
                        crate::common::build_witness_nc_digit_table_with_masks(params, Zi, s.m)
                            .unwrap_or_else(|e| panic!("NcOracle::new: failed to build NC digit table: {e}"))
                    })
                    .collect()
            }
        };
        let (digits_tables, digit_lane_masks): (Vec<_>, Vec<_>) = built_digit_tables.into_iter().unzip();
        #[cfg(feature = "perf-timers")]
        eprintln!("NcOracle::new: digit tables                {:.2?}", t_digits.elapsed());

        // Symmetric range polynomial cache.
        #[cfg(feature = "perf-timers")]
        let t_range = std::time::Instant::now();
        let mut range_t_sq = Vec::new();
        if params.b > 1 {
            range_t_sq.reserve((params.b - 1) as usize);
            for t in 1..(params.b as i64) {
                let tt = F::from_i64(t);
                range_t_sq.push(K::from(tt * tt));
            }
        }
        #[cfg(feature = "perf-timers")]
        eprintln!("NcOracle::new: range cache                 {:.2?}", t_range.elapsed());

        #[cfg(feature = "perf-timers")]
        eprintln!(
            "NcOracle::new: TOTAL                       {:.2?}",
            t_new_total.elapsed()
        );
        Self {
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_m,
            d_sc,
            round_idx: 0,
            col_chals: Vec::with_capacity(ell_m),
            ajtai_chals: Vec::with_capacity(ell_d),
            cur_len: m_pad,
            eq_beta_m_tbl,
            digits_tables,
            digit_lane_masks,
            weights,
            range_t_sq,
            digit_tables_all_real: true,
        }
    }

    #[inline]
    fn num_rounds_total(&self) -> usize {
        self.ell_m + self.ell_d
    }

    #[inline]
    fn fold_table_inplace(table: &mut Vec<K>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = lo + (hi - lo) * r;
        }
        table.truncate(half);
    }

    #[inline]
    fn fold_digits_table_inplace(table: &mut Vec<[K; D]>, masks: &mut Vec<u64>, r: K) {
        debug_assert!(!table.is_empty());
        debug_assert_eq!(table.len(), masks.len(), "NC digit table/mask length mismatch");
        let half = table.len().div_ceil(2);
        // One-way invariant maintained across rounds:
        //   masks[i] == 0  =>  table[i] == [K::ZERO; D]
        // (The reverse is not true: a live lane can fold to zero by
        // cancellation. We only rely on the forward direction here.)
        // It lets us skip the zero-write when active_mask is 0 AND the slot
        // was already zero on entry (i.e., old masks[i] was already 0).
        for i in 0..half {
            let base = 2 * i;
            let active_mask = masks[base] | if base + 1 < masks.len() { masks[base + 1] } else { 0 };
            let was_zero = masks[i] == 0;
            masks[i] = active_mask;
            if active_mask == 0 {
                if !was_zero {
                    // Old masks[i] != 0 means the slot may hold non-zero
                    // data (mask is only a one-way "has live lanes" hint);
                    // explicitly clear so the forward invariant holds.
                    table[i] = [K::ZERO; D];
                }
                continue;
            }

            let lo_row = table[base];
            let hi_row = if base + 1 < table.len() {
                table[base + 1]
            } else {
                [K::ZERO; D]
            };
            if active_mask == 1 {
                let lo = lo_row[0];
                let hi = hi_row[0];
                let mut out = [K::ZERO; D];
                out[0] = if hi == lo { lo } else { lo + (hi - lo) * r };
                table[i] = out;
                continue;
            }
            let mut out = [K::ZERO; D];
            let mut lanes = active_mask;
            while lanes != 0 {
                let rho = lanes.trailing_zeros() as usize;
                lanes &= lanes - 1;
                let lo = lo_row[rho];
                let hi = hi_row[rho];
                out[rho] = if hi == lo { lo } else { lo + (hi - lo) * r };
            }
            table[i] = out;
        }
        table.truncate(half);
        masks.truncate(half);
    }

    #[inline]
    fn active_col_tail_len(&self, tail_len: usize) -> usize {
        self.digits_tables
            .first()
            .map_or(0, |tbl| tbl.len().div_ceil(2).min(tail_len))
    }

    fn evals_col_phase_generic(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        let active_tail_len = self.active_col_tail_len(tail_len);
        let xs_len = xs.len();
        if xs_len == 0 {
            return Vec::new();
        }

        // `tail_len` starts at m_pad/2 and halves each column round; parallelize only when big enough.
        const PAR_THRESHOLD: usize = 1 << 13;
        let evals_col_phase_seq = |active_tail_len: usize, xs: &[K]| -> Vec<K> {
            let xs_len = xs.len();
            let mut out = vec![K::ZERO; xs_len];
            let mut nc_sum_by_x = vec![K::ZERO; xs_len];
            let mut eq_beta_m_x = vec![K::ZERO; xs_len];

            for t in 0..active_tail_len {
                nc_sum_by_x.fill(K::ZERO);

                let idx = 2 * t;
                let e0 = self.eq_beta_m_tbl[idx];
                let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                for (x_idx, &x) in xs.iter().enumerate() {
                    eq_beta_m_x[x_idx] = e0 + e1 * x;
                }

                for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                    let lo = &tbl[idx];
                    let hi = (idx + 1 < tbl.len()).then(|| &tbl[idx + 1]);
                    let mut lane_mask =
                        self.digit_lane_masks[wit_idx][idx] | hi.map_or(0, |_| self.digit_lane_masks[wit_idx][idx + 1]);
                    let weights = &self.weights[wit_idx];

                    while lane_mask != 0 {
                        let rho = lane_mask.trailing_zeros() as usize;
                        lane_mask &= lane_mask - 1;
                        let y0 = lo[rho];
                        let y1 = hi.map_or(K::ZERO, |row| row[rho]);
                        let dy = y1 - y0;
                        let w = weights[rho];
                        for (x_idx, &x) in xs.iter().enumerate() {
                            let y = y0 + dy * x;
                            nc_sum_by_x[x_idx] += w * range_product_cached(y, &self.range_t_sq);
                        }
                    }
                }

                for x_idx in 0..xs_len {
                    out[x_idx] += eq_beta_m_x[x_idx] * nc_sum_by_x[x_idx];
                }
            }

            out
        };

        if active_tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                let (out, _scratch_nc, _scratch_eq) = (0..active_tail_len)
                    .into_par_iter()
                    .fold(
                        || (vec![K::ZERO; xs_len], vec![K::ZERO; xs_len], vec![K::ZERO; xs_len]),
                        |(mut out, mut nc_sum_by_x, mut eq_beta_m_x), t| {
                            nc_sum_by_x.fill(K::ZERO);

                            let idx = 2 * t;
                            let e0 = self.eq_beta_m_tbl[idx];
                            let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                            for (x_idx, &x) in xs.iter().enumerate() {
                                eq_beta_m_x[x_idx] = e0 + e1 * x;
                            }

                            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                                let lo = &tbl[idx];
                                let hi = (idx + 1 < tbl.len()).then(|| &tbl[idx + 1]);
                                let mut lane_mask = self.digit_lane_masks[wit_idx][idx]
                                    | hi.map_or(0, |_| self.digit_lane_masks[wit_idx][idx + 1]);
                                let weights = &self.weights[wit_idx];

                                while lane_mask != 0 {
                                    let rho = lane_mask.trailing_zeros() as usize;
                                    lane_mask &= lane_mask - 1;
                                    let y0 = lo[rho];
                                    let y1 = hi.map_or(K::ZERO, |row| row[rho]);
                                    let dy = y1 - y0;
                                    let w = weights[rho];
                                    for (x_idx, &x) in xs.iter().enumerate() {
                                        let y = y0 + dy * x;
                                        nc_sum_by_x[x_idx] += w * range_product_cached(y, &self.range_t_sq);
                                    }
                                }
                            }

                            for x_idx in 0..xs_len {
                                out[x_idx] += eq_beta_m_x[x_idx] * nc_sum_by_x[x_idx];
                            }
                            (out, nc_sum_by_x, eq_beta_m_x)
                        },
                    )
                    .reduce(
                        || (vec![K::ZERO; xs_len], vec![K::ZERO; xs_len], vec![K::ZERO; xs_len]),
                        |(mut out_a, nc_a, eq_a), (out_b, _nc_b, _eq_b)| {
                            for i in 0..xs_len {
                                out_a[i] += out_b[i];
                            }
                            (out_a, nc_a, eq_a)
                        },
                    );
                out
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                evals_col_phase_seq(active_tail_len, xs)
            }
        } else {
            evals_col_phase_seq(active_tail_len, xs)
        }
    }

    /// Per-`t` inner accumulator for `b=2`: contributes to `inner[0..4]` the sum
    /// `Σ_i Σ_ρ γ_{i,ρ} · N(a + bX)` evaluated as a degree-3 polynomial in X.
    ///
    /// When `digit_tables_all_real` is set (round 0 — tables still encode raw real
    /// witnesses), the inner kernel runs in `Fq` and lifts to `K` via `scale_base`;
    /// otherwise the generic `K` kernel runs.
    #[inline]
    fn accumulate_inner_b2_at(&self, idx: usize, inner: &mut [K; 4]) {
        if self.digit_tables_all_real {
            let three_fq = Fq::from_u64(3);
            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                let lo = &tbl[idx];
                let hi = (idx + 1 < tbl.len()).then(|| &tbl[idx + 1]);
                let mut lane_mask =
                    self.digit_lane_masks[wit_idx][idx] | hi.map_or(0, |_| self.digit_lane_masks[wit_idx][idx + 1]);
                let weights = &self.weights[wit_idx];

                while lane_mask != 0 {
                    let rho = lane_mask.trailing_zeros() as usize;
                    lane_mask &= lane_mask - 1;
                    let w = weights[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    let a = lo[rho].real();
                    let y1 = hi.map_or(Fq::ZERO, |row| row[rho].real());
                    let b = y1 - a;
                    if a == Fq::ZERO && b == Fq::ZERO {
                        continue;
                    }
                    if b == Fq::ZERO {
                        let t0 = a * a * a - a;
                        inner[0] += w.scale_base(t0);
                        continue;
                    }
                    let a2 = a * a;
                    let a3 = a2 * a;
                    let b2 = b * b;
                    let b3 = b2 * b;
                    inner[0] += w.scale_base(a3 - a);
                    inner[1] += w.scale_base(a2 * b * three_fq - b);
                    inner[2] += w.scale_base(a * b2 * three_fq);
                    inner[3] += w.scale_base(b3);
                }
            }
        } else {
            let three = K::from(F::from_u64(3));
            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                let lo = &tbl[idx];
                let hi = (idx + 1 < tbl.len()).then(|| &tbl[idx + 1]);
                let mut lane_mask =
                    self.digit_lane_masks[wit_idx][idx] | hi.map_or(0, |_| self.digit_lane_masks[wit_idx][idx + 1]);
                let weights = &self.weights[wit_idx];

                while lane_mask != 0 {
                    let rho = lane_mask.trailing_zeros() as usize;
                    lane_mask &= lane_mask - 1;
                    let w = weights[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    let a = lo[rho];
                    let y1 = hi.map_or(K::ZERO, |row| row[rho]);
                    let b = y1 - a;
                    if a == K::ZERO && b == K::ZERO {
                        continue;
                    }
                    if b == K::ZERO {
                        let t0 = (a * a * a) - a;
                        inner[0] += w * t0;
                        continue;
                    }

                    let a2 = a * a;
                    let a3 = a2 * a;
                    let b2 = b * b;
                    let b3 = b2 * b;

                    let t0 = a3 - a;
                    let t1 = (a2 * b).scale_base_k(three) - b;
                    let t2 = (a * b2).scale_base_k(three);
                    let t3 = b3;

                    inner[0] += w * t0;
                    inner[1] += w * t1;
                    inner[2] += w * t2;
                    inner[3] += w * t3;
                }
            }
        }
    }

    fn col_phase_coeffs_b2(&self) -> [K; 5] {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        let active_tail_len = self.active_col_tail_len(tail_len);

        const PAR_THRESHOLD: usize = 1 << 13;

        let coeffs_seq = |active_tail_len: usize| -> [K; 5] {
            let mut coeffs = [K::ZERO; 5];
            for t in 0..active_tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_m_tbl[idx];
                let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                let mut inner = [K::ZERO; 4];
                self.accumulate_inner_b2_at(idx, &mut inner);

                // (e0 + e1 X) * (inner0 + inner1 X + inner2 X^2 + inner3 X^3)
                coeffs[0] += e0 * inner[0];
                coeffs[1] += e0 * inner[1] + e1 * inner[0];
                coeffs[2] += e0 * inner[2] + e1 * inner[1];
                coeffs[3] += e0 * inner[3] + e1 * inner[2];
                coeffs[4] += e1 * inner[3];
            }
            coeffs
        };

        if active_tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..active_tail_len)
                    .into_par_iter()
                    .fold(
                        || [K::ZERO; 5],
                        |mut coeffs, t| {
                            let idx = 2 * t;
                            let e0 = self.eq_beta_m_tbl[idx];
                            let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                            let mut inner = [K::ZERO; 4];
                            self.accumulate_inner_b2_at(idx, &mut inner);

                            coeffs[0] += e0 * inner[0];
                            coeffs[1] += e0 * inner[1] + e1 * inner[0];
                            coeffs[2] += e0 * inner[2] + e1 * inner[1];
                            coeffs[3] += e0 * inner[3] + e1 * inner[2];
                            coeffs[4] += e1 * inner[3];
                            coeffs
                        },
                    )
                    .reduce(
                        || [K::ZERO; 5],
                        |mut a, b| {
                            for i in 0..5 {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(active_tail_len)
            }
        } else {
            coeffs_seq(active_tail_len)
        }
    }

    fn evals_col_phase_b2(&self, xs: &[K]) -> Vec<K> {
        if xs.is_empty() {
            return Vec::new();
        }
        let coeffs = self.col_phase_coeffs_b2();
        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        if xs_are_base {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                .collect()
        } else {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                .collect()
        }
    }

    fn col_phase_coeffs_b3(&self) -> [K; 7] {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        let active_tail_len = self.active_col_tail_len(tail_len);

        const PAR_THRESHOLD: usize = 1 << 13;
        let four = K::from(F::from_u64(4));
        let five = K::from(F::from_u64(5));
        let ten = K::from(F::from_u64(10));
        let fifteen = K::from(F::from_u64(15));

        let coeffs_seq = |active_tail_len: usize| -> [K; 7] {
            let mut coeffs = [K::ZERO; 7];
            for t in 0..active_tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_m_tbl[idx];
                let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                let mut inner = [K::ZERO; 6];
                for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                    let lo = &tbl[idx];
                    let hi = (idx + 1 < tbl.len()).then(|| &tbl[idx + 1]);
                    let mut lane_mask =
                        self.digit_lane_masks[wit_idx][idx] | hi.map_or(0, |_| self.digit_lane_masks[wit_idx][idx + 1]);
                    let weights = &self.weights[wit_idx];

                    while lane_mask != 0 {
                        let rho = lane_mask.trailing_zeros() as usize;
                        lane_mask &= lane_mask - 1;
                        let w = weights[rho];
                        if w == K::ZERO {
                            continue;
                        }
                        let a = lo[rho];
                        let y1 = hi.map_or(K::ZERO, |row| row[rho]);
                        let b = y1 - a;
                        if a == K::ZERO && b == K::ZERO {
                            continue;
                        }
                        if b == K::ZERO {
                            let a2 = a * a;
                            let a3 = a2 * a;
                            let a4 = a2 * a2;
                            let a5 = a4 * a;
                            let t0 = a5 - a3.scale_base_k(five) + a.scale_base_k(four);
                            inner[0] += w * t0;
                            continue;
                        }

                        let a2 = a * a;
                        let a3 = a2 * a;
                        let a4 = a2 * a2;
                        let a5 = a4 * a;

                        let b2 = b * b;
                        let b3 = b2 * b;
                        let b4 = b2 * b2;
                        let b5 = b4 * b;

                        // N(a+bX) = (a+bX)^5 - 5(a+bX)^3 + 4(a+bX)
                        let t0 = a5 - a3.scale_base_k(five) + a.scale_base_k(four);
                        let t1 = b * (a4.scale_base_k(five) - a2.scale_base_k(fifteen) + four);
                        let t2 = b2 * (a3.scale_base_k(ten) - a.scale_base_k(fifteen));
                        let t3 = b3 * (a2.scale_base_k(ten) - five);
                        let t4 = b4 * a.scale_base_k(five);
                        let t5 = b5;

                        inner[0] += w * t0;
                        inner[1] += w * t1;
                        inner[2] += w * t2;
                        inner[3] += w * t3;
                        inner[4] += w * t4;
                        inner[5] += w * t5;
                    }
                }

                // (e0 + e1 X) * Σ_{k=0..5} inner[k] X^k
                coeffs[0] += e0 * inner[0];
                coeffs[1] += e0 * inner[1] + e1 * inner[0];
                coeffs[2] += e0 * inner[2] + e1 * inner[1];
                coeffs[3] += e0 * inner[3] + e1 * inner[2];
                coeffs[4] += e0 * inner[4] + e1 * inner[3];
                coeffs[5] += e0 * inner[5] + e1 * inner[4];
                coeffs[6] += e1 * inner[5];
            }
            coeffs
        };

        if active_tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..active_tail_len)
                    .into_par_iter()
                    .fold(
                        || [K::ZERO; 7],
                        |mut coeffs, t| {
                            let idx = 2 * t;
                            let e0 = self.eq_beta_m_tbl[idx];
                            let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                            let mut inner = [K::ZERO; 6];
                            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                                let lo = &tbl[idx];
                                let hi = (idx + 1 < tbl.len()).then(|| &tbl[idx + 1]);
                                let mut lane_mask = self.digit_lane_masks[wit_idx][idx]
                                    | hi.map_or(0, |_| self.digit_lane_masks[wit_idx][idx + 1]);
                                let weights = &self.weights[wit_idx];

                                while lane_mask != 0 {
                                    let rho = lane_mask.trailing_zeros() as usize;
                                    lane_mask &= lane_mask - 1;
                                    let w = weights[rho];
                                    if w == K::ZERO {
                                        continue;
                                    }
                                    let a = lo[rho];
                                    let y1 = hi.map_or(K::ZERO, |row| row[rho]);
                                    let b = y1 - a;
                                    if a == K::ZERO && b == K::ZERO {
                                        continue;
                                    }
                                    if b == K::ZERO {
                                        let a2 = a * a;
                                        let a3 = a2 * a;
                                        let a4 = a2 * a2;
                                        let a5 = a4 * a;
                                        let t0 = a5 - a3.scale_base_k(five) + a.scale_base_k(four);
                                        inner[0] += w * t0;
                                        continue;
                                    }

                                    let a2 = a * a;
                                    let a3 = a2 * a;
                                    let a4 = a2 * a2;
                                    let a5 = a4 * a;

                                    let b2 = b * b;
                                    let b3 = b2 * b;
                                    let b4 = b2 * b2;
                                    let b5 = b4 * b;

                                    let t0 = a5 - a3.scale_base_k(five) + a.scale_base_k(four);
                                    let t1 = b * (a4.scale_base_k(five) - a2.scale_base_k(fifteen) + four);
                                    let t2 = b2 * (a3.scale_base_k(ten) - a.scale_base_k(fifteen));
                                    let t3 = b3 * (a2.scale_base_k(ten) - five);
                                    let t4 = b4 * a.scale_base_k(five);
                                    let t5 = b5;

                                    inner[0] += w * t0;
                                    inner[1] += w * t1;
                                    inner[2] += w * t2;
                                    inner[3] += w * t3;
                                    inner[4] += w * t4;
                                    inner[5] += w * t5;
                                }
                            }

                            coeffs[0] += e0 * inner[0];
                            coeffs[1] += e0 * inner[1] + e1 * inner[0];
                            coeffs[2] += e0 * inner[2] + e1 * inner[1];
                            coeffs[3] += e0 * inner[3] + e1 * inner[2];
                            coeffs[4] += e0 * inner[4] + e1 * inner[3];
                            coeffs[5] += e0 * inner[5] + e1 * inner[4];
                            coeffs[6] += e1 * inner[5];
                            coeffs
                        },
                    )
                    .reduce(
                        || [K::ZERO; 7],
                        |mut a, b| {
                            for i in 0..7 {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(active_tail_len)
            }
        } else {
            coeffs_seq(active_tail_len)
        }
    }

    fn evals_col_phase_b3(&self, xs: &[K]) -> Vec<K> {
        if xs.is_empty() {
            return Vec::new();
        }
        let coeffs = self.col_phase_coeffs_b3();
        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        if xs_are_base {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                .collect()
        } else {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                .collect()
        }
    }

    fn evals_col_phase(&self, xs: &[K]) -> Vec<K> {
        match self.params.b {
            2 => self.evals_col_phase_b2(xs),
            3 => self.evals_col_phase_b3(xs),
            _ => self.evals_col_phase_generic(xs),
        }
    }

    pub fn optimized_col_phase_round_coeffs(&self) -> Option<Vec<K>> {
        if self.round_idx >= self.ell_m {
            return None;
        }
        match self.params.b {
            2 => Some(self.col_phase_coeffs_b2().to_vec()),
            3 => Some(self.col_phase_coeffs_b3().to_vec()),
            _ => None,
        }
    }

    pub fn finalized_y_zcol_digits(&self) -> Vec<[K; D]> {
        debug_assert!(
            self.round_idx >= self.ell_m,
            "NC column point not finalized before requesting y_zcol digits"
        );
        debug_assert_eq!(
            self.cur_len, 1,
            "expected NC column tables to be fully folded before requesting y_zcol digits"
        );
        self.digits_tables
            .iter()
            .map(|tbl| {
                debug_assert_eq!(tbl.len(), 1, "expected folded NC digit table to have exactly one entry");
                tbl[0]
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn __test_col_phase_fast_vs_generic(&self, xs: &[K]) -> Option<(Vec<K>, Vec<K>)> {
        if self.round_idx >= self.ell_m {
            return None;
        }
        match self.params.b {
            2 => Some((self.evals_col_phase_b2(xs), self.evals_col_phase_generic(xs))),
            3 => Some((self.evals_col_phase_b3(xs), self.evals_col_phase_generic(xs))),
            _ => None,
        }
    }

    fn evals_ajtai_phase(&self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_m;
        debug_assert!(j < self.ell_d, "NC Ajtai phase after all Ajtai bits");
        debug_assert!(
            self.cur_len == 1,
            "NC Ajtai phase requires finalized column point (cur_len={})",
            self.cur_len
        );

        let free_a = self.ell_d - j - 1;
        let w_beta_tail = chi_tail_weights(&self.ch.beta_a[j + 1..self.ell_d]);
        let head_stride = 1usize << (j + 1);
        debug_assert_eq!(w_beta_tail.len(), 1usize << free_a);

        // Prefix factor for eq(α, β_a).
        let mut eq_beta_pref = K::ONE;
        for i in 0..j {
            eq_beta_pref *= eq_lin(self.ajtai_chals[i], self.ch.beta_a[i]);
        }
        let beta_j = self.ch.beta_a[j];

        // eq(s', β_m) is the (single) entry after folding all column bits.
        let eq_beta_m = self.eq_beta_m_tbl[0];

        // Prefold packed-coefficient rows by Ajtai prefix bits once per round.
        let mut digits_pref: Vec<[K; D]> = Vec::with_capacity(self.digits_tables.len());
        for tbl in self.digits_tables.iter() {
            let mut d = tbl[0];
            for b in 0..j {
                fold_bit_inplace(&mut d, b, self.ajtai_chals[b]);
            }
            digits_pref.push(d);
        }

        let mut out = vec![K::ZERO; xs.len()];
        for (x_idx, &x) in xs.iter().enumerate() {
            let eq_beta = eq_beta_m * (eq_beta_pref * eq_lin(x, beta_j));

            // Apply γ^{i+1} factors (witness order) explicitly.
            let mut g = self.ch.gamma;
            let mut weighted_sum = K::ZERO;
            for digits in digits_pref.iter() {
                let acc =
                    ajtai_tail_weighted_range_prefolded(digits, x, j, head_stride, &w_beta_tail, &self.range_t_sq);
                weighted_sum += g * acc;
                g *= self.ch.gamma;
            }

            out[x_idx] = eq_beta * weighted_sum;
        }

        out
    }
}

impl<'a, F> RoundOracle for NcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    fn num_rounds(&self) -> usize {
        self.num_rounds_total()
    }

    fn degree_bound(&self) -> usize {
        self.d_sc
    }

    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        if self.round_idx < self.ell_m {
            self.evals_col_phase(xs)
        } else {
            self.evals_ajtai_phase(xs)
        }
    }

    fn fold(&mut self, r_i: K) {
        if self.round_idx < self.ell_m {
            self.col_chals.push(r_i);
            Self::fold_table_inplace(&mut self.eq_beta_m_tbl, r_i);
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                self.digits_tables
                    .par_iter_mut()
                    .zip(self.digit_lane_masks.par_iter_mut())
                    .for_each(|(tbl, masks)| Self::fold_digits_table_inplace(tbl, masks, r_i));
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                for (tbl, masks) in self
                    .digits_tables
                    .iter_mut()
                    .zip(self.digit_lane_masks.iter_mut())
                {
                    Self::fold_digits_table_inplace(tbl, masks, r_i);
                }
            }
            if r_i.imag() != Fq::ZERO {
                self.digit_tables_all_real = false;
            }
            self.cur_len /= 2;
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}

#[derive(Clone, Debug)]
struct CompiledPolyTerm {
    coeff: K,
    /// (var_pos, exponent), where `var_pos` indexes the inner `Vec<Vec<K>>`
    /// of each `RowStreamState::f_var_tables_by_mcs` entry.
    vars: Vec<(usize, u32)>,
}

/// Row-phase streaming state (over the row/time hypercube).
///
/// This replaces the old `evals_row_phase` strategy of enumerating row tails and repeatedly
/// running `precompute_for_r`. Instead, we materialize row-domain tables once and fold them
/// in-place as row challenges arrive.
struct RowStreamState {
    /// Current table length = 2^(remaining row bits).
    cur_len: usize,

    /// χ_{β_r}(row) table over the padded row domain (len = cur_len).
    eq_beta_r_tbl: Vec<K>,
    /// Optional χ_{r_inputs}(row) table (len = cur_len) for Eval gating.
    eq_r_inputs_tbl: Option<Vec<K>>,

    /// γ^{i-1} weights for the MCS slots (i is 1-based).
    gamma_pow_mcs: Vec<K>,

    /// Recomposition of each MCS witness `Z_i` into row vectors:
    /// `z_i[c] = Σ_{ρ=0..D-1} b^ρ · Z_i[ρ,c]`.
    z_mcs: Vec<Vec<K>>,

    /// Per-MCS tables for the variables used by the CCS polynomial `f`.
    /// Each entry is a row-domain table of `m_j(row) = (M_j · z_i)[row]` at boolean row points.
    f_var_tables_by_mcs: Vec<Vec<Vec<K>>>,
    /// Compiled sparse polynomial terms for `f` using `f_var_tables_by_mcs[i]` indices.
    f_terms: Vec<CompiledPolyTerm>,

    /// Combined Eval block table over rows (already summed over α' and (i,j) coefficients).
    /// When present, Eval contribution is: `eq_r_inputs(r') * gamma_to_k * eval_tbl(r')`.
    eval_tbl: Option<Vec<K>>,
    gamma_to_k: K,

    b: u32,
    /// True if all streamed tables are still in the base-field embedding (imag=0).
    ///
    /// When this holds and evaluation points are also base-field, we can evaluate the hot
    /// row-phase logic entirely in `Fq` for a large speedup.
    all_base: bool,
    /// Whether row-phase tables were built through SuperNeo cached rows.
    use_superneo_rows: bool,
}

impl RowStreamState {
    fn build<Ff>(
        s: &CcsStructure<Ff>,
        b: u32,
        ch: &Challenges,
        ell_d: usize,
        ell_n: usize,
        mcs_witnesses: &[CcsWitness<Ff>],
        me_witnesses: &[Mat<Ff>],
        r_inputs: Option<&[K]>,
        _sparse: &SparseCache<Ff>,
        superneo_cache: &SuperneoEvalCache,
        witness_z_blocks: &[SuperneoZBlocks],
    ) -> Self
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        let n_pad = 1usize << ell_n;
        let n_eff = s.n;
        let t_mats = s.t();

        #[cfg(feature = "perf-timers")]
        let t_total = std::time::Instant::now();

        #[cfg(feature = "perf-timers")]
        let t_chi = std::time::Instant::now();
        // Row-domain χ tables.
        let eq_beta_r_tbl = chi_tail_weights(&ch.beta_r);
        debug_assert_eq!(
            eq_beta_r_tbl.len(),
            n_pad,
            "chi(beta_r) length mismatch (ell_n={ell_n})"
        );

        let eq_r_inputs_tbl = r_inputs.map(|r| {
            let tbl = chi_tail_weights(r);
            debug_assert_eq!(tbl.len(), n_pad, "chi(r_inputs) length mismatch");
            tbl
        });
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 1. chi tables                {:.2?}",
            t_chi.elapsed()
        );

        let all_base = ch.gamma.imag() == Fq::ZERO
            && ch.alpha.iter().all(|x| x.imag() == Fq::ZERO)
            && ch.beta_a.iter().all(|x| x.imag() == Fq::ZERO)
            && ch.beta_r.iter().all(|x| x.imag() == Fq::ZERO)
            && r_inputs
                .map(|r| r.iter().all(|x| x.imag() == Fq::ZERO))
                .unwrap_or(true);

        #[cfg(feature = "perf-timers")]
        let t_f_compile = std::time::Instant::now();
        // Compile CCS polynomial f to avoid scanning t variables per evaluation.
        if s.f.arity() != t_mats {
            panic!(
                "CCS polynomial arity mismatch: f.arity()={}, but s.t()={}",
                s.f.arity(),
                t_mats
            );
        }
        let mut used_vars = vec![false; t_mats];
        for term in s.f.terms() {
            if term.exps.len() != t_mats {
                panic!(
                    "CCS polynomial exponent vector length mismatch: got {}, expected {}",
                    term.exps.len(),
                    t_mats
                );
            }
            for (j, &exp) in term.exps.iter().enumerate() {
                if exp != 0 {
                    used_vars[j] = true;
                }
            }
        }
        let f_var_indices: Vec<usize> = used_vars
            .iter()
            .enumerate()
            .filter_map(|(j, &u)| u.then_some(j))
            .collect();

        let mut pos_by_j = vec![usize::MAX; t_mats];
        for (pos, &j) in f_var_indices.iter().enumerate() {
            pos_by_j[j] = pos;
        }

        let f_terms: Vec<CompiledPolyTerm> =
            s.f.terms()
                .iter()
                .map(|term| {
                    let mut vars = Vec::new();
                    for (j, &exp) in term.exps.iter().enumerate() {
                        if exp != 0 {
                            let pos = pos_by_j[j];
                            debug_assert_ne!(pos, usize::MAX, "missing f var mapping");
                            vars.push((pos, exp));
                        }
                    }
                    CompiledPolyTerm {
                        coeff: K::from(term.coeff),
                        vars,
                    }
                })
                .collect();
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 2. f compile / f_var_indices {:.2?} (used_vars={}, terms={})",
            t_f_compile.elapsed(),
            f_var_indices.len(),
            f_terms.len()
        );

        let k_mcs = mcs_witnesses.len();

        let k_total = k_mcs + me_witnesses.len();
        debug_assert_eq!(k_mcs + me_witnesses.len(), k_total);
        debug_assert_eq!(
            witness_z_blocks.len(),
            k_total,
            "RowStreamState::build: witness block cache length mismatch"
        );

        // Sanity: challenge vectors for Ajtai rounds must match ell_d.
        if ch.beta_a.len() != ell_d || ch.alpha.len() != ell_d {
            panic!(
                "Challenge length mismatch: alpha.len()={}, beta_a.len()={}, ell_d={ell_d}",
                ch.alpha.len(),
                ch.beta_a.len()
            );
        }
        #[cfg(feature = "perf-timers")]
        let t_decode = std::time::Instant::now();
        // Build z_i (logical field witness vectors) from each MCS witness matrix.
        let mut z_mcs: Vec<Vec<K>> = Vec::with_capacity(k_mcs);
        for (mcs_idx, Zi) in mcs_witnesses.iter().map(|w| &w.Z).enumerate() {
            let z_i = crate::common::decode_superneo_coeffs_from_witness_mat(Zi, s.m).unwrap_or_else(|e| {
                panic!(
                    "RowStreamState::new: invalid packed MCS witness[{mcs_idx}] shape for m={}: {e}",
                    s.m
                )
            });
            z_mcs.push(z_i);
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 3. decode z_mcs              {:.2?} (k_mcs={k_mcs}, s.m={})",
            t_decode.elapsed(),
            s.m
        );
        #[cfg(feature = "debug-logs")]
        for (mcs_idx, z_i) in z_mcs.iter().enumerate() {
            eprintln!(
                "RowStreamState::build: mcs[{mcs_idx}] decoded coeff len={}, s.m={}",
                z_i.len(),
                s.m
            );
        }

        let mut gamma_pow_mcs = vec![K::ONE; k_mcs];
        for i in 1..k_mcs {
            gamma_pow_mcs[i] = gamma_pow_mcs[i - 1] * ch.gamma;
        }

        // Optimized oracle now uses one canonical SuperNeo row-lifted path.
        let use_superneo_rows = true;

        #[cfg(feature = "perf-timers")]
        let t_f_var_tables = std::time::Instant::now();
        // f-var tables: m_j(row) = (M_j * z_i)[row] for each used variable and each MCS slot.
        let mut f_var_tables_by_mcs: Vec<Vec<Vec<K>>> = Vec::with_capacity(k_mcs);
        for z_i in &z_mcs {
            let z_blocks = crate::superneo_eval::SuperneoZBlocks::from_z(z_i);
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let f_tables_i: Vec<Vec<K>> = f_var_indices
                .par_iter()
                .map(|&j| {
                    let mut out = vec![K::ZERO; n_pad];
                    let mat_cache = superneo_cache
                        .matrix(j)
                        .unwrap_or_else(|| panic!("superneo cache missing matrix j={j}"));
                    out[..n_eff]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(r, out_r)| {
                            *out_r = mat_cache.row_dot_with_blocks(r, &z_blocks);
                        });
                    out
                })
                .collect();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let f_tables_i: Vec<Vec<K>> = f_var_indices
                .iter()
                .map(|&j| {
                    let mut out = vec![K::ZERO; n_pad];
                    let mat_cache = superneo_cache
                        .matrix(j)
                        .unwrap_or_else(|| panic!("superneo cache missing matrix j={j}"));
                    for (r, out_r) in out.iter_mut().take(n_eff).enumerate() {
                        *out_r = mat_cache.row_dot_with_blocks(r, &z_blocks);
                    }
                    out
                })
                .collect();
            f_var_tables_by_mcs.push(f_tables_i);
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: 4. f_var_tables_by_mcs       {:.2?} (k_mcs={k_mcs}, vars={}, n_eff={n_eff})",
            t_f_var_tables.elapsed(),
            f_var_indices.len()
        );

        // Eval table (optional): only when both (a) there are carried witnesses, and (b) r_inputs exist.
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }

        let eval_tbl = if k_total > k_mcs && eq_r_inputs_tbl.is_some() {
            let mut w_alpha = [K::ZERO; D];
            for (rho, slot) in w_alpha.iter_mut().enumerate() {
                *slot = eq_points_bool_mask(rho, &ch.alpha);
            }
            #[cfg(feature = "perf-timers")]
            let t_weighted = std::time::Instant::now();
            let weighted_mats = superneo_cache.build_weighted_matrix_caches(&w_alpha);
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "RowStreamState::build: 5. build_weighted_matrix_caches {:.2?} (t_mats={t_mats})",
                t_weighted.elapsed()
            );
            let mut gamma_pow_i = vec![K::ONE; k_total];
            for i in 1..k_total {
                gamma_pow_i[i] = gamma_pow_i[i - 1] * ch.gamma;
            }
            let mut gamma_k_pow_j = vec![K::ONE; t_mats];
            for j in 1..t_mats {
                gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
            }

            #[cfg(feature = "perf-timers")]
            let t_eval = std::time::Instant::now();
            let mut eval_tbl = vec![K::ZERO; n_pad];
            for i_abs in k_mcs..k_total {
                let coeff_i = gamma_pow_i[i_abs];
                if coeff_i == K::ZERO {
                    continue;
                }
                let z_blocks = &witness_z_blocks[i_abs];

                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    eval_tbl
                        .par_iter_mut()
                        .take(n_eff)
                        .enumerate()
                        .for_each(|(r, out_r)| {
                            let mut row_acc = K::ZERO;
                            for (j, mat_cache) in weighted_mats.iter().enumerate() {
                                let coeff = coeff_i * gamma_k_pow_j[j];
                                if coeff == K::ZERO {
                                    continue;
                                }
                                let y_alpha = mat_cache.row_dot_real_with_blocks(r, &z_blocks);
                                if y_alpha != K::ZERO {
                                    row_acc += coeff * y_alpha;
                                }
                            }
                            if row_acc != K::ZERO {
                                *out_r += row_acc;
                            }
                        });
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    for (r, out_r) in eval_tbl.iter_mut().take(n_eff).enumerate() {
                        let mut row_acc = K::ZERO;
                        for (j, mat_cache) in weighted_mats.iter().enumerate() {
                            let coeff = coeff_i * gamma_k_pow_j[j];
                            if coeff == K::ZERO {
                                continue;
                            }
                            let y_alpha = mat_cache.row_dot_real_with_blocks(r, &z_blocks);
                            if y_alpha != K::ZERO {
                                row_acc += coeff * y_alpha;
                            }
                        }
                        if row_acc != K::ZERO {
                            *out_r += row_acc;
                        }
                    }
                }
            }
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "RowStreamState::build: 6. eval_tbl loop             {:.2?} (carried={}, t_mats={t_mats}, n_eff={n_eff})",
                t_eval.elapsed(),
                k_total - k_mcs
            );

            Some(eval_tbl)
        } else {
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "RowStreamState::build: 5+6. eval_tbl skipped       (k_total={k_total}, k_mcs={k_mcs}, r_inputs={})",
                eq_r_inputs_tbl.is_some()
            );
            None
        };

        #[cfg(feature = "perf-timers")]
        eprintln!(
            "RowStreamState::build: TOTAL                       {:.2?}",
            t_total.elapsed()
        );

        Self {
            cur_len: n_pad,
            eq_beta_r_tbl,
            eq_r_inputs_tbl,
            gamma_pow_mcs,
            z_mcs,
            f_var_tables_by_mcs,
            f_terms,
            eval_tbl,
            gamma_to_k,
            b,
            all_base,
            use_superneo_rows,
        }
    }

    #[inline]
    fn fold_table_inplace(table: &mut Vec<K>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = lo + (hi - lo) * r;
        }
        table.truncate(half);
    }

    #[inline]
    fn fold_table_inplace_base(table: &mut Vec<K>, r: Fq) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i].real();
            let hi = table[2 * i + 1].real();
            table[i] = K::from(lo + (hi - lo) * r);
        }
        table.truncate(half);
    }

    fn fold_inplace(&mut self, r: K) {
        if self.all_base && r.imag() == Fq::ZERO {
            let r0 = r.real();
            Self::fold_table_inplace_base(&mut self.eq_beta_r_tbl, r0);
            if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
            for per_mcs in self.f_var_tables_by_mcs.iter_mut() {
                for tbl in per_mcs.iter_mut() {
                    Self::fold_table_inplace_base(tbl, r0);
                }
            }
            if let Some(tbl) = self.eval_tbl.as_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
        } else {
            self.all_base = false;
            Self::fold_table_inplace(&mut self.eq_beta_r_tbl, r);
            if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
                Self::fold_table_inplace(tbl, r);
            }
            for per_mcs in self.f_var_tables_by_mcs.iter_mut() {
                for tbl in per_mcs.iter_mut() {
                    Self::fold_table_inplace(tbl, r);
                }
            }
            if let Some(tbl) = self.eval_tbl.as_mut() {
                Self::fold_table_inplace(tbl, r);
            }
        }
        self.cur_len /= 2;
    }

    #[inline]
    fn poly_mul_affine_inplace_base(poly: &mut [Fq], a: Fq, b: Fq, current_deg: usize) {
        // Coeffs are low→high. Output truncates to input length:
        // new[0] = a*old[0]; new[d] = a*old[d] + b*old[d-1] (d>=1).
        let mut prev = Fq::ZERO;
        for coeff in poly.iter_mut().take(current_deg + 2) {
            let old = *coeff;
            *coeff = a * old + b * prev;
            prev = old;
        }
    }

    #[inline]
    fn poly_eval_base(coeffs: &[Fq], x: Fq) -> Fq {
        if coeffs.is_empty() {
            return Fq::ZERO;
        }
        let mut result = coeffs[coeffs.len() - 1];
        for &c in coeffs.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }

    #[inline]
    fn accumulate_weighted_f_poly_base(&self, idx: usize, deg_max: usize, inner: &mut [Fq], term_poly: &mut [Fq]) {
        inner.fill(Fq::ZERO);

        for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
            let g = self
                .gamma_pow_mcs
                .get(mcs_idx)
                .copied()
                .unwrap_or(K::ONE)
                .real();
            if g == Fq::ZERO {
                continue;
            }

            for term in &self.f_terms {
                term_poly.fill(Fq::ZERO);
                term_poly[0] = term.coeff.real() * g;
                let mut current_deg = 0usize;
                for &(var_pos, exp) in &term.vars {
                    let tbl = &per_mcs_tables[var_pos];
                    let a = tbl[idx].real();
                    let b = tbl[idx + 1].real() - a;
                    for _ in 0..exp {
                        Self::poly_mul_affine_inplace_base(term_poly, a, b, current_deg);
                        current_deg += 1;
                    }
                }
                for i in 0..=core::cmp::min(current_deg, deg_max) {
                    inner[i] += term_poly[i];
                }
            }
        }
    }

    #[inline]
    fn accumulate_weighted_f_poly(&self, idx: usize, deg_max: usize, inner: &mut [K], term_poly: &mut [K]) {
        inner.fill(K::ZERO);

        for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
            let g = self.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
            if g == K::ZERO {
                continue;
            }

            for term in &self.f_terms {
                term_poly.fill(K::ZERO);
                term_poly[0] = term.coeff * g;
                let mut current_deg = 0usize;
                for &(var_pos, exp) in &term.vars {
                    let tbl = &per_mcs_tables[var_pos];
                    let a = tbl[idx];
                    let b = tbl[idx + 1] - a;
                    for _ in 0..exp {
                        Self::poly_mul_affine_inplace(term_poly, a, b, current_deg);
                        current_deg += 1;
                    }
                }
                for i in 0..=core::cmp::min(current_deg, deg_max) {
                    inner[i] += term_poly[i];
                }
            }
        }
    }

    fn evals_row_phase_b2_base(&self, tail_len: usize, xs: &[K]) -> Vec<K> {
        let xs_base: Vec<Fq> = xs.iter().map(|&x| x.real()).collect();

        let f_max_term_deg: usize = self
            .f_terms
            .iter()
            .map(|term| {
                term.vars
                    .iter()
                    .map(|&(_, exp)| exp as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        // eq_beta_r(X) adds one degree; Eval block is quadratic.
        let deg_max = core::cmp::max(2, f_max_term_deg + 1);

        const PAR_THRESHOLD: usize = 1 << 14;
        let coeffs_seq = |tail_len: usize| -> Vec<Fq> {
            let mut coeffs = vec![Fq::ZERO; deg_max + 1];
            let mut inner = vec![Fq::ZERO; deg_max + 1];
            let mut term_poly = vec![Fq::ZERO; deg_max + 1];

            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_r_tbl[idx].real();
                let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                self.accumulate_weighted_f_poly_base(idx, deg_max, &mut inner, &mut term_poly);

                coeffs[0] += e0 * inner[0];
                for d in 1..=deg_max {
                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                }

                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[idx].real();
                    let r1 = eq_tbl[idx + 1].real() - r0;
                    let v0 = eval_tbl[idx].real();
                    let v1 = eval_tbl[idx + 1].real() - v0;

                    let g = self.gamma_to_k.real();
                    coeffs[0] += g * (r0 * v0);
                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                    coeffs[2] += g * (r1 * v1);
                }
            }

            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                            )
                        },
                        |(mut coeffs, mut inner, mut term_poly), t| {
                            let idx = 2 * t;
                            // eq_beta_r(X) = e0 + e1·X
                            let e0 = self.eq_beta_r_tbl[idx].real();
                            let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                            self.accumulate_weighted_f_poly_base(idx, deg_max, &mut inner, &mut term_poly);

                            // coeffs += eq_beta_r(X) * inner(X)
                            coeffs[0] += e0 * inner[0];
                            for d in 1..=deg_max {
                                coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                            }

                            // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                            if let (Some(eq_tbl), Some(eval_tbl)) =
                                (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                            {
                                let r0 = eq_tbl[idx].real();
                                let r1 = eq_tbl[idx + 1].real() - r0;
                                let v0 = eval_tbl[idx].real();
                                let v1 = eval_tbl[idx + 1].real() - v0;

                                let g = self.gamma_to_k.real();
                                coeffs[0] += g * (r0 * v0);
                                coeffs[1] += g * (r0 * v1 + r1 * v0);
                                coeffs[2] += g * (r1 * v1);
                            }

                            (coeffs, inner, term_poly)
                        },
                    )
                    .map(|(coeffs, _, _)| coeffs)
                    .reduce(
                        || vec![Fq::ZERO; deg_max + 1],
                        |mut a, b| {
                            for i in 0..=deg_max {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        xs_base
            .iter()
            .map(|&x| K::from(Self::poly_eval_base(&coeffs, x)))
            .collect()
    }

    fn evals_row_phase_b3_base(&self, tail_len: usize, xs: &[K]) -> Vec<K> {
        let xs_base: Vec<Fq> = xs.iter().map(|&x| x.real()).collect();

        let f_max_term_deg: usize = self
            .f_terms
            .iter()
            .map(|term| {
                term.vars
                    .iter()
                    .map(|&(_, exp)| exp as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        // eq_beta_r(X) adds one degree; Eval block is quadratic.
        let deg_max = core::cmp::max(2, f_max_term_deg + 1);

        const PAR_THRESHOLD: usize = 1 << 14;
        let coeffs_seq = |tail_len: usize| -> Vec<Fq> {
            let mut coeffs = vec![Fq::ZERO; deg_max + 1];
            let mut inner = vec![Fq::ZERO; deg_max + 1];
            let mut term_poly = vec![Fq::ZERO; deg_max + 1];

            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_r_tbl[idx].real();
                let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                self.accumulate_weighted_f_poly_base(idx, deg_max, &mut inner, &mut term_poly);

                coeffs[0] += e0 * inner[0];
                for d in 1..=deg_max {
                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                }

                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[idx].real();
                    let r1 = eq_tbl[idx + 1].real() - r0;
                    let v0 = eval_tbl[idx].real();
                    let v1 = eval_tbl[idx + 1].real() - v0;

                    let g = self.gamma_to_k.real();
                    coeffs[0] += g * (r0 * v0);
                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                    coeffs[2] += g * (r1 * v1);
                }
            }

            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                            )
                        },
                        |(mut coeffs, mut inner, mut term_poly), t| {
                            let idx = 2 * t;
                            // eq_beta_r(X) = e0 + e1·X
                            let e0 = self.eq_beta_r_tbl[idx].real();
                            let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                            self.accumulate_weighted_f_poly_base(idx, deg_max, &mut inner, &mut term_poly);

                            // coeffs += eq_beta_r(X) * inner(X)
                            coeffs[0] += e0 * inner[0];
                            for d in 1..=deg_max {
                                coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                            }

                            // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                            if let (Some(eq_tbl), Some(eval_tbl)) =
                                (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                            {
                                let r0 = eq_tbl[idx].real();
                                let r1 = eq_tbl[idx + 1].real() - r0;
                                let v0 = eval_tbl[idx].real();
                                let v1 = eval_tbl[idx + 1].real() - v0;

                                let g = self.gamma_to_k.real();
                                coeffs[0] += g * (r0 * v0);
                                coeffs[1] += g * (r0 * v1 + r1 * v0);
                                coeffs[2] += g * (r1 * v1);
                            }

                            (coeffs, inner, term_poly)
                        },
                    )
                    .map(|(coeffs, _, _)| coeffs)
                    .reduce(
                        || vec![Fq::ZERO; deg_max + 1],
                        |mut a, b| {
                            for i in 0..=deg_max {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        xs_base
            .iter()
            .map(|&x| K::from(Self::poly_eval_base(&coeffs, x)))
            .collect()
    }

    /// Multiply a polynomial by an affine `(a + b·x)` in-place.
    ///
    /// Coefficients are in low→high order. Output is truncated to the input length.
    #[inline]
    fn poly_mul_affine_inplace(poly: &mut [K], a: K, b: K, current_deg: usize) {
        let mut prev = K::ZERO;
        for coeff in poly.iter_mut().take(current_deg + 2) {
            let old = *coeff;
            *coeff = a * old + b * prev;
            prev = old;
        }
    }

    fn evals_row_phase_impl<Ff>(&self, xs: &[K], allow_base: bool) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        let xs_all_base = allow_base && self.all_base && xs_are_base;

        // Fast path for b=2: build the univariate coefficients once per round,
        // then evaluate cheaply at all requested points.
        if self.b == 2 {
            if xs_all_base {
                return self.evals_row_phase_b2_base(tail_len, xs);
            }

            let f_max_term_deg: usize = self
                .f_terms
                .iter()
                .map(|term| {
                    term.vars
                        .iter()
                        .map(|&(_, exp)| exp as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0);
            // eq_beta_r(X) adds one degree; Eval block is quadratic.
            let deg_max = core::cmp::max(2, f_max_term_deg + 1);

            // Sequential per-`t` step, factored out so seq and par paths share one body.
            let step = |coeffs: &mut [K], inner: &mut [K], term_poly: &mut [K], t: usize| {
                let e0 = self.eq_beta_r_tbl[2 * t];
                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                self.accumulate_weighted_f_poly(2 * t, deg_max, inner, term_poly);

                // coeffs += eq_beta_r(X) * inner(X)
                coeffs[0] += e0 * inner[0];
                for d in 1..=deg_max {
                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                }

                // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[2 * t];
                    let r1 = eq_tbl[2 * t + 1] - r0;
                    let v0 = eval_tbl[2 * t];
                    let v1 = eval_tbl[2 * t + 1] - v0;

                    let g = self.gamma_to_k;
                    coeffs[0] += g * (r0 * v0);
                    if deg_max >= 1 {
                        coeffs[1] += g * (r0 * v1 + r1 * v0);
                    }
                    if deg_max >= 2 {
                        coeffs[2] += g * (r1 * v1);
                    }
                }
            };

            const PAR_THRESHOLD: usize = 1 << 14;
            let coeffs = if tail_len >= PAR_THRESHOLD {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..tail_len)
                        .into_par_iter()
                        .fold(
                            || {
                                (
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                )
                            },
                            |(mut coeffs, mut inner, mut term_poly), t| {
                                step(&mut coeffs, &mut inner, &mut term_poly, t);
                                (coeffs, inner, term_poly)
                            },
                        )
                        .map(|(coeffs, _, _)| coeffs)
                        .reduce(
                            || vec![K::ZERO; deg_max + 1],
                            |mut a, b| {
                                for i in 0..=deg_max {
                                    a[i] += b[i];
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut coeffs = vec![K::ZERO; deg_max + 1];
                    let mut inner = vec![K::ZERO; deg_max + 1];
                    let mut term_poly = vec![K::ZERO; deg_max + 1];
                    for t in 0..tail_len {
                        step(&mut coeffs, &mut inner, &mut term_poly, t);
                    }
                    coeffs
                }
            } else {
                let mut coeffs = vec![K::ZERO; deg_max + 1];
                let mut inner = vec![K::ZERO; deg_max + 1];
                let mut term_poly = vec![K::ZERO; deg_max + 1];
                for t in 0..tail_len {
                    step(&mut coeffs, &mut inner, &mut term_poly, t);
                }
                coeffs
            };

            return if xs_are_base {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                    .collect()
            } else {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                    .collect()
            };
        }

        // Fast path for b=3: range polynomial is N(y) = y(y^2-1)(y^2-4) = y^5 - 5y^3 + 4y.
        // As in the b=2 case, we build the univariate coefficients once per round and then
        // evaluate at all requested points.
        if self.b == 3 {
            if xs_all_base {
                return self.evals_row_phase_b3_base(tail_len, xs);
            }

            let f_max_term_deg: usize = self
                .f_terms
                .iter()
                .map(|term| {
                    term.vars
                        .iter()
                        .map(|&(_, exp)| exp as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0);
            // eq_beta_r(X) adds one degree; Eval block is quadratic.
            let deg_max = core::cmp::max(2, f_max_term_deg + 1);

            let coeffs = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..tail_len)
                        .into_par_iter()
                        .fold(
                            || {
                                (
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                )
                            },
                            |(mut coeffs, mut inner, mut term_poly), t| {
                                // eq_beta_r(X) = e0 + e1·X
                                let e0 = self.eq_beta_r_tbl[2 * t];
                                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                                self.accumulate_weighted_f_poly(2 * t, deg_max, &mut inner, &mut term_poly);

                                // coeffs += eq_beta_r(X) * inner(X)
                                coeffs[0] += e0 * inner[0];
                                for d in 1..=deg_max {
                                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                                }

                                // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                                if let (Some(eq_tbl), Some(eval_tbl)) =
                                    (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                                {
                                    let r0 = eq_tbl[2 * t];
                                    let r1 = eq_tbl[2 * t + 1] - r0;
                                    let v0 = eval_tbl[2 * t];
                                    let v1 = eval_tbl[2 * t + 1] - v0;

                                    let g = self.gamma_to_k;
                                    coeffs[0] += g * (r0 * v0);
                                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                                    coeffs[2] += g * (r1 * v1);
                                }

                                (coeffs, inner, term_poly)
                            },
                        )
                        .map(|(coeffs, _, _)| coeffs)
                        .reduce(
                            || vec![K::ZERO; deg_max + 1],
                            |mut a, b| {
                                for i in 0..=deg_max {
                                    a[i] += b[i];
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut coeffs = vec![K::ZERO; deg_max + 1];
                    let mut inner = vec![K::ZERO; deg_max + 1];
                    let mut term_poly = vec![K::ZERO; deg_max + 1];

                    for t in 0..tail_len {
                        // eq_beta_r(X) = e0 + e1·X
                        let e0 = self.eq_beta_r_tbl[2 * t];
                        let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                        self.accumulate_weighted_f_poly(2 * t, deg_max, &mut inner, &mut term_poly);

                        // coeffs += eq_beta_r(X) * inner(X)
                        coeffs[0] += e0 * inner[0];
                        for d in 1..=deg_max {
                            coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                        }

                        // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                        if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                        {
                            let r0 = eq_tbl[2 * t];
                            let r1 = eq_tbl[2 * t + 1] - r0;
                            let v0 = eval_tbl[2 * t];
                            let v1 = eval_tbl[2 * t + 1] - v0;

                            let g = self.gamma_to_k;
                            coeffs[0] += g * (r0 * v0);
                            coeffs[1] += g * (r0 * v1 + r1 * v0);
                            coeffs[2] += g * (r1 * v1);
                        }
                    }

                    coeffs
                }
            };

            return if xs_are_base {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                    .collect()
            } else {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                    .collect()
            };
        }

        // Generic fallback: evaluate directly at each x (slower, but supports any b / K>1).
        let f_arity = self
            .f_var_tables_by_mcs
            .first()
            .map(|v| v.len())
            .unwrap_or(0);

        // `xs` is typically very small (sumcheck evaluation points), so Rayon overhead dominates here.
        xs.iter()
            .map(|&x| {
                let one_minus = K::ONE - x;
                let mut var_vals = vec![K::ZERO; f_arity];
                let mut sum_x = K::ZERO;

                for t in 0..tail_len {
                    let eq_beta_r = one_minus * self.eq_beta_r_tbl[2 * t] + x * self.eq_beta_r_tbl[2 * t + 1];

                    // f_prime = Σ_{i=1..k_mcs} γ^{i-1} · f_i(m_vals_i).
                    let mut f_prime = K::ZERO;

                    for (mcs_idx, per_mcs_tables) in self.f_var_tables_by_mcs.iter().enumerate() {
                        // f variables at (prefix, x, tail) for this MCS slot
                        for (pos, tbl) in per_mcs_tables.iter().enumerate() {
                            var_vals[pos] = one_minus * tbl[2 * t] + x * tbl[2 * t + 1];
                        }

                        let mut f_i = K::ZERO;
                        for term in &self.f_terms {
                            let mut acc = term.coeff;
                            for &(var_pos, exp) in &term.vars {
                                let xi = var_vals[var_pos];
                                let mut p = xi;
                                for _ in 1..exp {
                                    p *= xi;
                                }
                                acc *= p;
                            }
                            f_i += acc;
                        }

                        let g_i = self.gamma_pow_mcs.get(mcs_idx).copied().unwrap_or(K::ONE);
                        f_prime += g_i * f_i;
                    }

                    let mut out = eq_beta_r * f_prime;

                    // Eval: eq_r_inputs(r') * gamma_to_k * eval_tbl(r')
                    if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                        let eq_r_inputs = one_minus * eq_tbl[2 * t] + x * eq_tbl[2 * t + 1];
                        if eq_r_inputs != K::ZERO {
                            let e = one_minus * eval_tbl[2 * t] + x * eval_tbl[2 * t + 1];
                            out += eq_r_inputs * (self.gamma_to_k * e);
                        }
                    }

                    sum_x += out;
                }

                sum_x
            })
            .collect()
    }

    #[inline]
    fn evals_row_phase<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        self.evals_row_phase_impl::<Ff>(xs, true)
    }

    #[inline]
    fn evals_row_phase_force_generic<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        self.evals_row_phase_impl::<Ff>(xs, false)
    }
}

/// Symmetric range polynomial: ∏_{t=-(b-1)}^{b-1} (y - t) = y · ∏_{t=1}^{b-1} (y² - t²)
/// using cached `t²` values for `t=1..(b-1)`.
#[inline]
fn range_product_cached(y: K, range_t_sq: &[K]) -> K {
    if range_t_sq.is_empty() {
        return y;
    }
    let y2 = y * y;
    let mut prod = y;
    for &tt2 in range_t_sq {
        prod *= y2 - tt2;
    }
    prod
}

#[inline]
fn eq_lin(a: K, b: K) -> K {
    (K::ONE - a) * (K::ONE - b) + a * b
}

/// Fold one Ajtai bit into-place for a digits table (size D).
#[inline]
fn fold_bit_inplace(digits: &mut [K; D], bit: usize, a: K) {
    let stride = 1usize << bit;
    let step = stride << 1;
    let n = D;
    let mut base = 0usize;
    while base < n {
        let mut off = 0usize;
        while off < stride {
            let i0 = base + off;
            if i0 >= n {
                break;
            }
            let i1 = i0 + stride;
            let lo = digits[i0];
            let hi = if i1 < n { digits[i1] } else { K::ZERO };
            digits[i0] = lo + (hi - lo) * a;
            off += 1;
        }
        base += step;
    }
}

/// Compute `c0 + c1·x`, where that affine polynomial is the tail-weighted
/// dot after folding the current Ajtai bit into `digits_pref`.
#[inline]
fn ajtai_tail_weighted_dot_affine_prefolded(
    digits_pref: &[K; D],
    bit: usize,
    head_stride: usize,
    w_tail: &[K],
) -> (K, K) {
    let stride = 1usize << bit;
    let mut c0 = K::ZERO;
    let mut c1 = K::ZERO;
    for (t, &w) in w_tail.iter().enumerate() {
        let idx = t * head_stride;
        if idx < D {
            let lo = digits_pref[idx];
            let hi_idx = idx + stride;
            let hi = if hi_idx < D { digits_pref[hi_idx] } else { K::ZERO };
            c0 += w * lo;
            c1 += w * (hi - lo);
        }
    }
    (c0, c1)
}

/// Fold the current Ajtai bit into `digits_pref` (which already has the prefix folded),
/// then compute the tail-weighted sum of the range polynomial N(·) over the MLE heads.
#[inline]
fn ajtai_tail_weighted_range_prefolded(
    digits_pref: &[K; D],
    x: K,
    bit: usize,
    head_stride: usize,
    w_tail: &[K],
    range_t_sq: &[K],
) -> K {
    let mut tmp = *digits_pref;
    fold_bit_inplace(&mut tmp, bit, x);
    let mut acc = K::ZERO;
    for (t, &w) in w_tail.iter().enumerate() {
        let idx = t * head_stride;
        if idx < D {
            acc += w * range_product_cached(tmp[idx], range_t_sq);
        }
    }
    acc
}

#[inline]
fn chi_tail_weights(bits: &[K]) -> Vec<K> {
    let t = bits.len();
    let len = 1usize << t;
    let mut w = vec![K::ZERO; len];
    w[0] = K::ONE;
    for (i, &b) in bits.iter().enumerate() {
        let step = 1usize << i;
        let one_minus = K::ONE - b;
        for mask in 0..step {
            let v = w[mask];
            w[mask] = v * one_minus;
            w[mask + step] = v * b;
        }
    }
    w
}

/// Precomputation for a fixed r' (row assignment) - eliminates redundant v_j recomputation
struct RPrecomp {
    /// Y_eval[i][j][ρ] = (Z_i · v_j)[ρ] for Eval terms  
    y_eval: Vec<Vec<[K; D]>>,
    /// F' = f(z_1 · v_j) - independent of α'
    f_prime: K,
    /// eq(r', β_r) - independent of α'
    eq_beta_r: K,
    /// eq(r', r_inputs) if present - independent of α'
    eq_r_inputs: K,
}

#[inline]
fn materialize_y_ring_from_precomputed_digits(y_by_mat: &[[K; D]], d_pad: usize) -> (Vec<Vec<K>>, Vec<K>) {
    let mut y_ring = Vec::with_capacity(y_by_mat.len());
    let mut ct = Vec::with_capacity(y_by_mat.len());
    for digits in y_by_mat {
        let mut row = vec![K::ZERO; d_pad];
        row[..D].copy_from_slice(digits);
        ct.push(digits[0]);
        y_ring.push(row);
    }
    (y_ring, ct)
}

/// Helper: compute eq for a boolean mask against a field vector
#[inline]
fn eq_points_bool_mask(mask: usize, points: &[K]) -> K {
    let mut prod = K::ONE;
    for (bit_idx, &p) in points.iter().enumerate() {
        let is_one = ((mask >> bit_idx) & 1) == 1;
        prod *= if is_one { p } else { K::ONE - p };
    }
    prod
}

pub struct OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    // Witnesses in the same order as the engine: all MCS first, then ME
    pub mcs_witnesses: &'a [CcsWitness<F>],
    pub me_witnesses: &'a [Mat<F>],
    // Challenges (α, β, γ)
    pub ch: Challenges,
    // Shared dims and degree bound for sumcheck
    pub ell_d: usize,
    pub ell_n: usize,
    pub d_sc: usize,
    // Round tracking
    pub round_idx: usize,
    // Collected row and Ajtai challenges r' and α'
    pub row_chals: Vec<K>,
    pub ajtai_chals: Vec<K>,
    // Input ME r (if any) for Eval gating
    pub r_inputs: Option<Vec<K>>,
    // Cached sparse formats for efficient matrix-vector products
    pub sparse: Arc<SparseCache<F>>,
    // Cached SuperNeo row-lifted matrices for canonical optimized evaluation.
    superneo_cache: Arc<SuperneoEvalCache>,
    // Packed witness block views in oracle order: all MCS first, then ME.
    witness_z_blocks: Vec<SuperneoZBlocks>,

    // Streaming row-phase state (folded in-place across row rounds)
    row_stream: RowStreamState,

    // Cached row-only precomputation for Ajtai rounds (r' fixed after row phase).
    ajtai_precomp: Option<RPrecomp>,
}

impl<'a, F> OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    /// Construct with an explicit SuperNeo cache selection.
    ///
    /// `superneo_cache` must be present; optimized oracle now has a single canonical
    /// SuperNeo row-lifted evaluation path.
    #[doc(hidden)]
    pub fn new_with_sparse_and_superneo_cache(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [CcsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
        sparse: Arc<SparseCache<F>>,
        superneo_cache: Arc<SuperneoEvalCache>,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one MCS instance for F-term");
        #[cfg(feature = "perf-timers")]
        let t_z_blocks = std::time::Instant::now();
        let all_witnesses: Vec<&Mat<F>> = mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
            .collect();
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let witness_z_blocks: Vec<SuperneoZBlocks> = all_witnesses
            .par_iter()
            .enumerate()
            .map(|(idx, Zi)| {
                SuperneoZBlocks::from_witness_mat(Zi, s.m).unwrap_or_else(|e| {
                    panic!("OptimizedOracle::new: invalid packed witness block view at slot {idx}: {e}")
                })
            })
            .collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let witness_z_blocks: Vec<SuperneoZBlocks> = all_witnesses
            .iter()
            .enumerate()
            .map(|(idx, Zi)| {
                SuperneoZBlocks::from_witness_mat(Zi, s.m).unwrap_or_else(|e| {
                    panic!("OptimizedOracle::new: invalid packed witness block view at slot {idx}: {e}")
                })
            })
            .collect();
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "OptimizedOracle::new: witness z blocks     {:.2?} (witnesses={})",
            t_z_blocks.elapsed(),
            witness_z_blocks.len()
        );

        let row_stream = RowStreamState::build(
            s,
            params.b,
            &ch,
            ell_d,
            ell_n,
            mcs_witnesses,
            me_witnesses,
            r_inputs,
            sparse.as_ref(),
            superneo_cache.as_ref(),
            &witness_z_blocks,
        );

        Self {
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_n,
            d_sc,
            round_idx: 0,
            row_chals: Vec::with_capacity(ell_n),
            ajtai_chals: Vec::with_capacity(ell_d),
            r_inputs: r_inputs.map(|r| r.to_vec()),
            sparse,
            superneo_cache,
            witness_z_blocks,
            row_stream,
            ajtai_precomp: None,
        }
    }

    pub fn new_with_sparse(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [CcsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
        sparse: Arc<SparseCache<F>>,
    ) -> Self {
        let superneo_cache = crate::superneo_eval::build_superneo_eval_cache(s)
            .map(Arc::new)
            .unwrap_or_else(|| {
                panic!(
                    "OptimizedOracle requires SuperNeo-compatible CCS shape (m={}, matrices={})",
                    s.m,
                    s.matrices.len()
                )
            });
        Self::new_with_sparse_and_superneo_cache(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_n,
            d_sc,
            r_inputs,
            sparse,
            superneo_cache,
        )
    }

    #[inline]
    fn num_rounds_total(&self) -> usize {
        self.ell_n + self.ell_d
    }

    #[inline]
    fn eq_points(p: &[K], q: &[K]) -> K {
        assert_eq!(p.len(), q.len(), "eq_points: length mismatch");
        let mut acc = K::ONE;
        for i in 0..p.len() {
            let (pi, qi) = (p[i], q[i]);
            acc *= (K::ONE - pi) * (K::ONE - qi) + pi * qi;
        }
        acc
    }

    /// Precompute all data that depends only on r' (not on α') for row phase optimization.
    /// This eliminates redundant v_j recomputation across all boolean α' assignments.
    fn precompute_for_r(&self, r_prime: &[K]) -> RPrecomp {
        let t = self.s.t();

        // Build χ_r table over the Boolean row domain.
        let chi_r = chi_tail_weights(r_prime);
        let n_sz = chi_r.len();

        // Compute eq(r', β_r) and eq(r', r_inputs)
        let eq_beta_r = Self::eq_points(r_prime, &self.ch.beta_r);
        let eq_r_inputs = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };

        let n_eff = core::cmp::min(self.s.n, n_sz);
        // Compute Y_eval using the canonical SuperNeo row-lifted path.
        let superneo_cache = &self.superneo_cache;
        #[cfg(feature = "perf-timers")]
        let t_y_eval = std::time::Instant::now();
        let y_eval: Vec<Vec<[K; D]>> = if self.witness_z_blocks.len() > 1 {
            #[cfg(feature = "perf-timers")]
            let t_ring_forms = std::time::Instant::now();
            let ring_forms = superneo_cache.build_ring_linear_forms(&chi_r, n_eff);
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "OptimizedOracle::precompute_for_r: ring forms        {:.2?}",
                t_ring_forms.elapsed()
            );
            if ring_forms.len() != t {
                panic!(
                    "superneo ring-linear forms count mismatch: got {}, expected {}",
                    ring_forms.len(),
                    t
                );
            }
            let k_total = self.witness_z_blocks.len();
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                // Flatten (witness, mat) pairs so all k_total * t evaluations run
                // in parallel — the old per-witness par_iter left t-way work
                // sequential inside each task and underused cores when
                // k_total < cores.
                let flat: Vec<[K; D]> = (0..k_total * t)
                    .into_par_iter()
                    .map(|idx| {
                        let w = idx / t;
                        let m = idx % t;
                        ring_forms[m].eval_real_z_blocks(&self.witness_z_blocks[w])
                    })
                    .collect();
                (0..k_total)
                    .map(|w| flat[w * t..(w + 1) * t].to_vec())
                    .collect()
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                self.witness_z_blocks
                    .iter()
                    .map(|z_blocks| {
                        ring_forms
                            .iter()
                            .map(|form| form.eval_real_z_blocks(&z_blocks))
                            .collect()
                    })
                    .collect()
            }
        } else {
            let row_cap = core::cmp::min(n_eff, chi_r.len());
            let mut chi_re = Vec::with_capacity(row_cap);
            let mut chi_im = Vec::with_capacity(row_cap);
            for &w in chi_r.iter().take(row_cap) {
                let [re, im] = w.as_coeffs();
                chi_re.push(re);
                chi_im.push(im);
            }
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                self.witness_z_blocks
                    .par_iter()
                    .map(|z_blocks| {
                        crate::superneo_eval::eval_all_mats_ring_cached_with_split_chi(
                            superneo_cache,
                            &z_blocks,
                            &chi_re,
                            &chi_im,
                            n_eff,
                        )
                    })
                    .collect()
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                self.witness_z_blocks
                    .iter()
                    .map(|z_blocks| {
                        crate::superneo_eval::eval_all_mats_ring_cached_with_split_chi(
                            superneo_cache,
                            &z_blocks,
                            &chi_re,
                            &chi_im,
                            n_eff,
                        )
                    })
                    .collect()
            }
        };
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "OptimizedOracle::precompute_for_r: y_eval            {:.2?} (witnesses={}, mats={t})",
            t_y_eval.elapsed(),
            self.witness_z_blocks.len()
        );

        // Compute F' = Σ_{i=1..k_mcs} γ^{i-1} · f(Ẽ(M_j z_i)(r')).
        //
        // The constant lane of the ring-coefficient evaluation is the scalar
        // SuperNeo eval used by f, so this reuses `y_eval` instead of scanning
        // the matrices again to build scalar linear forms.
        #[cfg(feature = "perf-timers")]
        let t_f_prime = std::time::Instant::now();
        let mut f_prime = K::ZERO;
        for mcs_idx in 0..self.row_stream.z_mcs.len() {
            let m_vals: Vec<K> = y_eval[mcs_idx].iter().map(|coeffs| coeffs[0]).collect();
            let g_i = self
                .row_stream
                .gamma_pow_mcs
                .get(mcs_idx)
                .copied()
                .unwrap_or(K::ONE);
            f_prime += g_i * self.s.f.eval_in_ext::<K>(&m_vals);
        }
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "OptimizedOracle::precompute_for_r: f_prime           {:.2?}",
            t_f_prime.elapsed()
        );

        RPrecomp {
            y_eval,
            f_prime,
            eq_beta_r,
            eq_r_inputs,
        }
    }

    /// Compute the univariate round polynomial values at given xs for a row-bit round
    /// by summing Q over the remaining Boolean variables, with the current variable set to x.
    fn evals_row_phase(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.round_idx < self.ell_n, "row phase after all row bits");
        let expect_len = 1usize << (self.ell_n - self.round_idx);
        debug_assert_eq!(
            self.row_stream.cur_len, expect_len,
            "row_stream out of sync with round_idx"
        );
        self.row_stream.evals_row_phase::<F>(xs)
    }

    #[doc(hidden)]
    pub fn __test_row_phase_base_vs_generic(&self, xs: &[K]) -> (Vec<K>, Vec<K>) {
        debug_assert!(self.round_idx < self.ell_n, "__test_row_phase_* requires row phase");
        let base = self.row_stream.evals_row_phase::<F>(xs);
        let generic = self.row_stream.evals_row_phase_force_generic::<F>(xs);
        (base, generic)
    }

    #[doc(hidden)]
    pub fn __test_row_stream_all_base(&self) -> bool {
        self.row_stream.all_base
    }

    #[doc(hidden)]
    pub fn __test_row_stream_uses_superneo_rows(&self) -> bool {
        self.row_stream.use_superneo_rows
    }

    /// Compute the univariate round polynomial for an Ajtai-bit round.
    /// DP version: removes the 2^{free_a}·D work per x and keeps outputs bit-identical.
    fn evals_ajtai_phase(&mut self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n;
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        let r_vec = &self.row_chals;

        // r'-only precomp reused across all Ajtai rounds (r' is fixed after row phase).
        if self.ajtai_precomp.is_none() {
            self.ajtai_precomp = Some(self.precompute_for_r(r_vec));
        }
        let pre = self
            .ajtai_precomp
            .as_ref()
            .expect("ajtai_precomp just populated");

        let k_mcs = self.mcs_witnesses.len();
        let k_total = k_mcs + self.me_witnesses.len();
        let t_mats = self.s.t();

        // Tail weights (independent of x)
        let w_beta_tail = chi_tail_weights(&self.ch.beta_a[j + 1..self.ell_d]);
        let w_alpha_tail = chi_tail_weights(&self.ch.alpha[j + 1..self.ell_d]);
        let tail_len = 1usize << free_a;
        debug_assert_eq!(w_beta_tail.len(), tail_len);
        debug_assert_eq!(w_alpha_tail.len(), tail_len);
        let head_stride = 1usize << (j + 1);

        // Prefix factors (independent of x)
        let mut eq_beta_pref = K::ONE;
        let mut eq_alpha_pref = K::ONE;
        for i in 0..j {
            eq_beta_pref *= eq_lin(self.ajtai_chals[i], self.ch.beta_a[i]);
            eq_alpha_pref *= eq_lin(self.ajtai_chals[i], self.ch.alpha[i]);
        }

        // Gamma powers (independent of x)
        let mut gamma_pow_i = vec![K::ONE; k_total];
        for i in 1..k_total {
            gamma_pow_i[i] = gamma_pow_i[i - 1] * self.ch.gamma;
        }

        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= self.ch.gamma;
        }

        let mut gamma_k_pow_j = vec![K::ONE; t_mats];
        for jj in 1..t_mats {
            gamma_k_pow_j[jj] = gamma_k_pow_j[jj - 1] * gamma_to_k;
        }

        let prefix = &self.ajtai_chals[..j];
        let beta_j = self.ch.beta_a[j];
        let alpha_j = self.ch.alpha[j];
        let has_inputs = self.r_inputs.is_some();

        let eval_inner_affine = if k_total > k_mcs && has_inputs && pre.eq_r_inputs != K::ZERO {
            let mut c0 = K::ZERO;
            let mut c1 = K::ZERO;
            for j_mat in 0..t_mats {
                let gamma_j = gamma_k_pow_j[j_mat];
                for (i_abs, gamma_i) in gamma_pow_i
                    .iter()
                    .copied()
                    .enumerate()
                    .take(k_total)
                    .skip(k_mcs)
                {
                    let coeff = gamma_i * gamma_j;
                    if coeff == K::ZERO {
                        continue;
                    }
                    let mut digits = pre.y_eval[i_abs][j_mat];
                    for b in 0..j {
                        fold_bit_inplace(&mut digits, b, prefix[b]);
                    }
                    let (dot0, dot1) = ajtai_tail_weighted_dot_affine_prefolded(&digits, j, head_stride, &w_alpha_tail);
                    c0 += coeff * dot0;
                    c1 += coeff * dot1;
                }
            }
            Some((c0, c1))
        } else {
            None
        };

        let eval_at = |x: K| {
            // eq((α',r'), β) factor across α' = (prefix, x, tail)
            let eq_beta_px = eq_beta_pref * eq_lin(x, beta_j);
            let eq_beta = pre.eq_beta_r * eq_beta_px;

            // eq((α',r'), (α,r)) factor if inputs present
            let eq_ar_px = if has_inputs {
                pre.eq_r_inputs * (eq_alpha_pref * eq_lin(x, alpha_j))
            } else {
                K::ZERO
            };

            // Base: eq_beta * F'
            let mut out = eq_beta * pre.f_prime;

            // --- Eval block: γ^k · eq_ar · Σ_{j_mat,i≥2} γ^{i-1} (γ^k)^{j_mat} · Σ_tail w_alpha(tail) · ẏ_{(i,j)}(...)
            if let Some((inner0, inner1)) = eval_inner_affine {
                let inner = inner0 + inner1 * x;
                out += eq_ar_px * (gamma_to_k * inner);
            }

            out
        };

        // `xs` is typically very small (sumcheck evaluation points), so Rayon overhead dominates here.
        xs.iter().map(|&x| eval_at(x)).collect()
    }

    /// Build Π_CCS ME outputs at the finalized row point `r'` using the oracle's cached
    /// `precompute_for_r` results (no dense matrix scans).
    pub fn build_me_outputs_from_ajtai_precomp<L>(
        &mut self,
        mcs_list: &[CcsClaim<Cmt, F>],
        me_inputs: &[CeClaim<Cmt, F, K>],
        s_col: &[K],
        y_zcol_digits: Option<&[[K; D]]>,
        fold_digest: [u8; 32],
        _l: &L,
    ) -> Vec<CeClaim<Cmt, F, K>>
    where
        L: SModuleHomomorphism<F, Cmt>,
    {
        assert_eq!(
            mcs_list.len(),
            self.mcs_witnesses.len(),
            "ME output builder: mcs_list/mcs_witnesses length mismatch"
        );
        assert_eq!(
            me_inputs.len(),
            self.me_witnesses.len(),
            "ME output builder: me_inputs/me_witnesses length mismatch"
        );
        assert_eq!(
            self.row_chals.len(),
            self.ell_n,
            "ME output builder: row challenges not finalized"
        );

        let d_pad = 1usize << self.ell_d;
        assert!(
            d_pad >= D,
            "ME output builder: expected 2^ell_d >= D (2^{} = {d_pad}, D = {D})",
            self.ell_d
        );
        let row_chals = self.row_chals.clone();
        let s_col_vec = s_col.to_vec();
        let k_mcs = self.mcs_witnesses.len();

        if self.ajtai_precomp.is_none() {
            self.ajtai_precomp = Some(self.precompute_for_r(&row_chals));
        }
        let pre = self
            .ajtai_precomp
            .as_ref()
            .expect("ajtai_precomp just populated for ME output builder");

        let chi_s = if s_col.is_empty() || y_zcol_digits.is_some() {
            None
        } else {
            Some(chi_tail_weights(s_col))
        };

        let mut out = Vec::with_capacity(self.mcs_witnesses.len() + self.me_witnesses.len());

        // MCS outputs (keep order).
        for (mcs_idx, (inst, wit)) in mcs_list.iter().zip(self.mcs_witnesses.iter()).enumerate() {
            let X = crate::common::project_x_from_witness_mat(&wit.Z, self.s.m, inst.m_in)
                .unwrap_or_else(|e| panic!("ME output builder: project_x_from_witness_mat failed: {e}"));
            let (y_ring, ct) = materialize_y_ring_from_precomputed_digits(&pre.y_eval[mcs_idx], d_pad);

            let y_zcol = if let Some(y_zcol_digits) = y_zcol_digits {
                let mut row = vec![K::ZERO; d_pad];
                row[..D].copy_from_slice(&y_zcol_digits[mcs_idx]);
                row
            } else if let Some(chi_s) = chi_s.as_ref() {
                debug_assert!(chi_s.len() >= self.s.m, "chi_s too short for CCS width");
                crate::common::compute_y_zcol_from_witness_digits(self.params, &wit.Z, self.s.m, chi_s, d_pad)
                    .unwrap_or_else(|e| panic!("ME output builder: y_zcol compute failed (MCS): {e}"))
            } else {
                Vec::new()
            };

            out.push(CeClaim {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inst.c.clone(),
                X,
                r: row_chals.clone(),
                s_col: s_col_vec.clone(),
                y_ring,
                ct,
                aux_openings: Vec::new(),
                y_zcol,
                m_in: inst.m_in,
                fold_digest,
            });
        }

        // ME outputs (keep order).
        for (me_idx, inp) in me_inputs.iter().enumerate() {
            let Zi = &self.me_witnesses[me_idx];
            let (y_ring, ct) = materialize_y_ring_from_precomputed_digits(&pre.y_eval[k_mcs + me_idx], d_pad);

            let y_zcol = if let Some(y_zcol_digits) = y_zcol_digits {
                let mut row = vec![K::ZERO; d_pad];
                row[..D].copy_from_slice(&y_zcol_digits[k_mcs + me_idx]);
                row
            } else if let Some(chi_s) = chi_s.as_ref() {
                debug_assert!(chi_s.len() >= self.s.m, "chi_s too short for CCS width");
                crate::common::compute_y_zcol_from_witness_digits(self.params, Zi, self.s.m, chi_s, d_pad)
                    .unwrap_or_else(|e| panic!("ME output builder: y_zcol compute failed (ME): {e}"))
            } else {
                Vec::new()
            };

            out.push(CeClaim {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inp.c.clone(),
                X: inp.X.clone(),
                r: row_chals.clone(),
                s_col: s_col_vec.clone(),
                y_ring,
                ct,
                aux_openings: Vec::new(),
                y_zcol,
                m_in: inp.m_in,
                fold_digest,
            });
        }

        out
    }
}

impl<'a, F> RoundOracle for OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    fn num_rounds(&self) -> usize {
        self.num_rounds_total()
    }
    fn degree_bound(&self) -> usize {
        self.d_sc
    }

    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        if self.round_idx < self.ell_n {
            self.evals_row_phase(xs)
        } else {
            self.evals_ajtai_phase(xs)
        }
    }

    fn fold(&mut self, r_i: K) {
        if self.round_idx < self.ell_n {
            self.row_chals.push(r_i);
            self.row_stream.fold_inplace(r_i);
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
