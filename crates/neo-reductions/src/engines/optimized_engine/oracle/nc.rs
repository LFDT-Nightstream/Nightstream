//! Split-NC oracle for the column and Ajtai phases of Pi_CCS.

use super::*;

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
    digits_tables: Vec<NcDigitTable>,
    // Bitmask of live digit lanes for each row in `digits_tables`; table rows remain authority.
    digit_lane_masks: Vec<NcDigitMasks>,
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
        Self::new_inner(s, params, mcs_witnesses, me_witnesses, ch, ell_d, ell_m, d_sc, false)
    }

    /// [`Self::new`] without building the column-phase equality or digit
    /// tables — for callers whose device backend builds them from the challenge
    /// point and resident witness planes. Any host read of the deferred tables
    /// panics; call [`Self::materialize_deferred_col_tables`] if the backend
    /// declines.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_deferred_digit_tables(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [CcsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_m: usize,
        d_sc: usize,
    ) -> Self {
        Self::new_inner(s, params, mcs_witnesses, me_witnesses, ch, ell_d, ell_m, d_sc, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [CcsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_m: usize,
        d_sc: usize,
        defer_digit_tables: bool,
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
        let eq_beta_m_tbl = if defer_digit_tables {
            Vec::new()
        } else {
            chi_tail_weights(&ch.beta_m)
        };
        debug_assert!(
            defer_digit_tables || eq_beta_m_tbl.len() == m_pad,
            "chi(beta_m) length mismatch"
        );
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
        let built_digit_tables: Vec<(NcDigitTable, NcDigitMasks)> = if defer_digit_tables {
            all_witnesses
                .iter()
                .map(|_| (NcDigitTable::Deferred { len: s.m }, NcDigitMasks::Zero { len: s.m }))
                .collect()
        } else {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                if all_witnesses.len() > 1 {
                    all_witnesses
                        .par_iter()
                        .map(|Zi| {
                            build_nc_digit_table_compact(params, Zi, s.m)
                                .unwrap_or_else(|e| panic!("NcOracle::new: failed to build NC digit table: {e}"))
                        })
                        .collect()
                } else {
                    all_witnesses
                        .iter()
                        .map(|Zi| {
                            build_nc_digit_table_compact(params, Zi, s.m)
                                .unwrap_or_else(|e| panic!("NcOracle::new: failed to build NC digit table: {e}"))
                        })
                        .collect()
                }
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                all_witnesses
                    .iter()
                    .map(|Zi| {
                        build_nc_digit_table_compact(params, Zi, s.m)
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
                    let hi_exists = idx + 1 < tbl.len();
                    let mut lane_mask = self.digit_lane_masks[wit_idx].get(idx)
                        | if hi_exists {
                            self.digit_lane_masks[wit_idx].get(idx + 1)
                        } else {
                            0
                        };
                    let weights = &self.weights[wit_idx];

                    while lane_mask != 0 {
                        let rho = lane_mask.trailing_zeros() as usize;
                        lane_mask &= lane_mask - 1;
                        let y0 = tbl.lane(idx, rho);
                        let y1 = if hi_exists { tbl.lane(idx + 1, rho) } else { K::ZERO };
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
                                let hi_exists = idx + 1 < tbl.len();
                                let mut lane_mask = self.digit_lane_masks[wit_idx].get(idx)
                                    | if hi_exists {
                                        self.digit_lane_masks[wit_idx].get(idx + 1)
                                    } else {
                                        0
                                    };
                                let weights = &self.weights[wit_idx];

                                while lane_mask != 0 {
                                    let rho = lane_mask.trailing_zeros() as usize;
                                    lane_mask &= lane_mask - 1;
                                    let y0 = tbl.lane(idx, rho);
                                    let y1 = if hi_exists { tbl.lane(idx + 1, rho) } else { K::ZERO };
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
                let hi_exists = idx + 1 < tbl.len();
                let mut lane_mask = self.digit_lane_masks[wit_idx].get(idx)
                    | if hi_exists {
                        self.digit_lane_masks[wit_idx].get(idx + 1)
                    } else {
                        0
                    };
                let weights = &self.weights[wit_idx];

                while lane_mask != 0 {
                    let rho = lane_mask.trailing_zeros() as usize;
                    lane_mask &= lane_mask - 1;
                    let w = weights[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    let a = tbl.lane_real(idx, rho);
                    let y1 = if hi_exists {
                        tbl.lane_real(idx + 1, rho)
                    } else {
                        Fq::ZERO
                    };
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
                let hi_exists = idx + 1 < tbl.len();
                let mut lane_mask = self.digit_lane_masks[wit_idx].get(idx)
                    | if hi_exists {
                        self.digit_lane_masks[wit_idx].get(idx + 1)
                    } else {
                        0
                    };
                let weights = &self.weights[wit_idx];

                while lane_mask != 0 {
                    let rho = lane_mask.trailing_zeros() as usize;
                    lane_mask &= lane_mask - 1;
                    let w = weights[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    let a = tbl.lane(idx, rho);
                    let y1 = if hi_exists { tbl.lane(idx + 1, rho) } else { K::ZERO };
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
                    let hi_exists = idx + 1 < tbl.len();
                    let mut lane_mask = self.digit_lane_masks[wit_idx].get(idx)
                        | if hi_exists {
                            self.digit_lane_masks[wit_idx].get(idx + 1)
                        } else {
                            0
                        };
                    let weights = &self.weights[wit_idx];

                    while lane_mask != 0 {
                        let rho = lane_mask.trailing_zeros() as usize;
                        lane_mask &= lane_mask - 1;
                        let w = weights[rho];
                        if w == K::ZERO {
                            continue;
                        }
                        let a = tbl.lane(idx, rho);
                        let y1 = if hi_exists { tbl.lane(idx + 1, rho) } else { K::ZERO };
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
                                let hi_exists = idx + 1 < tbl.len();
                                let mut lane_mask = self.digit_lane_masks[wit_idx].get(idx)
                                    | if hi_exists {
                                        self.digit_lane_masks[wit_idx].get(idx + 1)
                                    } else {
                                        0
                                    };
                                let weights = &self.weights[wit_idx];

                                while lane_mask != 0 {
                                    let rho = lane_mask.trailing_zeros() as usize;
                                    lane_mask &= lane_mask - 1;
                                    let w = weights[rho];
                                    if w == K::ZERO {
                                        continue;
                                    }
                                    let a = tbl.lane(idx, rho);
                                    let y1 = if hi_exists { tbl.lane(idx + 1, rho) } else { K::ZERO };
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

    /// Read-only view of the column-phase tables, for accelerator backends
    /// that replicate the NC column rounds off-CPU. The backend must stay
    /// field-identical to `col_phase_coeffs_b2` + the column `fold`.
    pub fn col_phase_snapshot(&self) -> NcColSnapshot<'_> {
        NcColSnapshot {
            cur_len: self.cur_len,
            beta_m: &self.ch.beta_m,
            eq_beta_m_tbl: &self.eq_beta_m_tbl,
            weights: &self.weights,
            digit_tables: self
                .digits_tables
                .iter()
                .map(|tbl| match tbl {
                    NcDigitTable::Zero { len } => NcDigitTableView::Zero { len: *len },
                    NcDigitTable::Lane0(values) => NcDigitTableView::Lane0(values),
                    NcDigitTable::Strided { width, values } => NcDigitTableView::Strided { width: *width, values },
                    NcDigitTable::Dense(rows) => NcDigitTableView::Dense(rows),
                    NcDigitTable::Deferred { len } => NcDigitTableView::Deferred { len: *len },
                })
                .collect(),
        }
    }

    /// Record a column-round challenge without folding the tables — used
    /// when a device backend owns the column folds. Pair with
    /// `inject_finalized_col_state` after the last column round.
    pub fn advance_col_round_without_fold(&mut self, r_i: K) {
        debug_assert!(self.round_idx < self.ell_m, "column-phase rounds only");
        self.col_chals.push(r_i);
        if r_i.imag() != Fq::ZERO {
            self.digit_tables_all_real = false;
        }
        self.cur_len /= 2;
        self.round_idx += 1;
    }

    /// Build the equality and digit tables that
    /// `new_with_deferred_digit_tables` skipped — for the CPU fallback when a
    /// device backend declines the snapshot. No-op when they are already built.
    pub fn materialize_deferred_col_tables(&mut self) {
        if self.eq_beta_m_tbl.is_empty() {
            self.eq_beta_m_tbl = chi_tail_weights(&self.ch.beta_m);
        }
        if !matches!(self.digits_tables.first(), Some(NcDigitTable::Deferred { .. })) {
            return;
        }
        debug_assert_eq!(self.round_idx, 0, "materialize only before the first column round");
        let mut all_witnesses: Vec<&Mat<F>> = Vec::with_capacity(self.mcs_witnesses.len() + self.me_witnesses.len());
        all_witnesses.extend(self.mcs_witnesses.iter().map(|w| &w.Z));
        all_witnesses.extend(self.me_witnesses.iter());
        let built: Vec<(NcDigitTable, NcDigitMasks)> = all_witnesses
            .iter()
            .map(|Zi| {
                build_nc_digit_table_compact(self.params, Zi, self.s.m)
                    .unwrap_or_else(|e| panic!("NcOracle::materialize_deferred_col_tables: {e}"))
            })
            .collect();
        let (tables, masks): (Vec<_>, Vec<_>) = built.into_iter().unzip();
        self.digits_tables = tables;
        self.digit_lane_masks = masks;
    }

    /// Install the fully folded column state a device backend measured, so
    /// the Ajtai tail rounds (which read the folded digit rows) and
    /// `finalized_y_zcol_digits` run unchanged on the CPU.
    pub fn inject_finalized_col_state(&mut self, digit_rows: Vec<[K; D]>, eq_beta_m0: K) {
        debug_assert_eq!(self.round_idx, self.ell_m, "inject only after the last column round");
        debug_assert_eq!(
            digit_rows.len(),
            self.digits_tables.len(),
            "one folded digit row per witness"
        );
        self.cur_len = 1;
        self.eq_beta_m_tbl = vec![eq_beta_m0];
        self.digit_lane_masks = digit_rows
            .iter()
            .map(|row| {
                let mut mask = 0u64;
                for (rho, value) in row.iter().enumerate() {
                    if *value != K::ZERO {
                        mask |= 1u64 << rho;
                    }
                }
                NcDigitMasks::Dense(vec![mask])
            })
            .collect();
        self.digits_tables = digit_rows
            .into_iter()
            .map(|row| NcDigitTable::Dense(vec![row]))
            .collect();
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
                tbl.row(0)
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
            let mut d = tbl.row(0);
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
                    .for_each(|(tbl, masks)| tbl.fold_inplace(masks, r_i));
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                for (tbl, masks) in self
                    .digits_tables
                    .iter_mut()
                    .zip(self.digit_lane_masks.iter_mut())
                {
                    tbl.fold_inplace(masks, r_i);
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
