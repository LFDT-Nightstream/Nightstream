//! Top-level optimized Pi_CCS oracle and accelerator snapshots.

use super::*;

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
        Self::new_with_sparse_and_superneo_cache_and_backend(
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
            None,
        )
    }

    /// [`Self::new_with_sparse_and_superneo_cache`] with an optional device
    /// backend that may build the f-var row tables (bit-identical contract).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_sparse_and_superneo_cache_and_backend(
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
        fe_backend: Option<&mut (dyn FeSumcheckBackend + '_)>,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one MCS instance for F-term");
        #[cfg(feature = "perf-timers")]
        let t_z_blocks = std::time::Instant::now();
        let all_witnesses: Vec<&Mat<F>> = mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
            .collect();
        // When the backend serves the carried eval table from its own
        // resident planes, the running (carried) witnesses' host z blocks
        // are never read — placeholder empties keep the vec length (and the
        // length-based branches elsewhere) intact without the build cost.
        let k_mcs_count = mcs_witnesses.len();
        let defer_running = fe_backend
            .as_ref()
            .is_some_and(|b| b.serves_carried_eval_table());
        let z_block_at = |idx: usize, Zi: &Mat<F>| -> SuperneoZBlocks {
            if defer_running && idx >= k_mcs_count {
                SuperneoZBlocks::with_block_len(s.m.div_ceil(D))
            } else {
                SuperneoZBlocks::from_witness_mat(Zi, s.m).unwrap_or_else(|e| {
                    panic!("OptimizedOracle::new: invalid packed witness block view at slot {idx}: {e}")
                })
            }
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let witness_z_blocks: Vec<SuperneoZBlocks> = all_witnesses
            .par_iter()
            .enumerate()
            .map(|(idx, Zi)| z_block_at(idx, Zi))
            .collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let witness_z_blocks: Vec<SuperneoZBlocks> = all_witnesses
            .iter()
            .enumerate()
            .map(|(idx, Zi)| z_block_at(idx, Zi))
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
            fe_backend,
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
        let n_eff = core::cmp::min(self.s.n, n_sz);
        // Compute Y_eval using the canonical SuperNeo row-lifted path.
        let superneo_cache = &self.superneo_cache;
        #[cfg(feature = "perf-timers")]
        let t_y_eval = std::time::Instant::now();
        let k_total = self.witness_z_blocks.len();
        let y_eval = superneo_cache.eval_ring_linear_forms_for_real_z_blocks(&chi_r, n_eff, &self.witness_z_blocks);
        assert_eq!(y_eval.len(), k_total, "superneo witness evaluation count mismatch");
        debug_assert!(y_eval.iter().all(|by_matrix| by_matrix.len() == t));
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "OptimizedOracle::precompute_for_r: y_eval            {:.2?} (witnesses={}, mats={t})",
            t_y_eval.elapsed(),
            self.witness_z_blocks.len()
        );

        self.finish_r_precomp(r_prime, y_eval)
    }

    fn ensure_ajtai_precomp(&mut self) {
        if self.ajtai_precomp.is_none() {
            let row_point = self.row_chals.clone();
            self.ajtai_precomp = Some(self.precompute_for_r(&row_point));
        }
        self.release_witness_z_blocks();
    }

    fn release_witness_z_blocks(&mut self) {
        self.witness_z_blocks.clear();
        self.witness_z_blocks.shrink_to_fit();
    }

    /// Assemble `RPrecomp` from a computed `y_eval` (CPU or device):
    /// the eq scalars and `F'` derive from it on the host.
    fn finish_r_precomp(&self, r_prime: &[K], y_eval: Vec<Vec<[K; D]>>) -> RPrecomp {
        let eq_beta_r = Self::eq_points(r_prime, &self.ch.beta_r);
        let eq_r_inputs = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };
        // Compute F' = Σ_{i=1..k_mcs} γ^{i-1} · f(Ẽ(M_j z_i)(r')).
        //
        // The constant lane of the ring-coefficient evaluation is the scalar
        // SuperNeo eval used by f, so this reuses `y_eval` instead of scanning
        // the matrices again to build scalar linear forms.
        #[cfg(feature = "perf-timers")]
        let t_f_prime = std::time::Instant::now();
        let mut f_prime = K::ZERO;
        for mcs_idx in 0..self.row_stream.gamma_pow_mcs.len() {
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

    pub fn take_pi_dec_precompute(&mut self) -> Option<super::super::PiDecProverPrecompute> {
        self.ajtai_precomp.as_ref()?;
        Some(super::super::PiDecProverPrecompute {
            row_chals: self.row_chals.clone(),
        })
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

    /// Everything a device backend needs to compute the Ajtai-phase
    /// `Y_eval` off-CPU: the eval cache (static bar matrices), the χ table
    /// at the folded row point, and every witness matrix.
    #[allow(clippy::type_complexity)]
    pub fn ajtai_backend_context(&self) -> Option<(&SuperneoEvalCache, Vec<K>, usize, Vec<&Mat<F>>)> {
        debug_assert_eq!(self.round_idx, self.ell_n, "Ajtai context exists after the row phase");
        let chi_r = chi_tail_weights(&self.row_chals);
        let n_eff = core::cmp::min(self.s.n, chi_r.len());
        let witnesses = self
            .mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
            .collect();
        Some((self.superneo_cache.as_ref(), chi_r, n_eff, witnesses))
    }

    /// Device backends that own the row challenges do not have a host χ table
    /// yet. This returns the static/input pieces needed to build that table
    /// from the device challenge buffer instead.
    pub fn ajtai_backend_trace_context(&self) -> Option<(&SuperneoEvalCache, usize, Vec<&Mat<F>>)> {
        let chi_len = 1usize.checked_shl(self.ell_n as u32)?;
        let n_eff = core::cmp::min(self.s.n, chi_len);
        let witnesses = self
            .mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
            .collect();
        Some((self.superneo_cache.as_ref(), n_eff, witnesses))
    }

    /// Static inputs plus canonical host row challenges for backends that can
    /// construct the row-point tensor without a full host table upload.
    #[allow(clippy::type_complexity)]
    pub fn ajtai_backend_challenge_context(&self) -> Option<(&SuperneoEvalCache, &[K], usize, Vec<&Mat<F>>)> {
        debug_assert_eq!(self.round_idx, self.ell_n, "Ajtai context exists after the row phase");
        let chi_len = 1usize.checked_shl(self.row_chals.len() as u32)?;
        let n_eff = core::cmp::min(self.s.n, chi_len);
        let witnesses = self
            .mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
            .collect();
        Some((self.superneo_cache.as_ref(), &self.row_chals, n_eff, witnesses))
    }

    /// Install a device-computed Ajtai-phase `Y_eval` (indexed
    /// `[witness][matrix][lane]`, matching `ajtai_backend_inputs` order);
    /// the eq scalars and `F'` are derived on the host.
    pub fn inject_ajtai_y_eval(&mut self, y_eval: Vec<Vec<[K; D]>>) {
        debug_assert_eq!(self.round_idx, self.ell_n, "inject after the row phase");
        let r_prime = self.row_chals.clone();
        self.ajtai_precomp = Some(self.finish_r_precomp(&r_prime, y_eval));
        self.release_witness_z_blocks();
    }

    /// Record a row-round challenge without folding the row tables — used
    /// when an `FeSumcheckBackend` owns the table folds. Everything after
    /// the row phase (Ajtai precompute, tail rounds, outputs) reads only
    /// witnesses and challenges, so the unfolded tables are never touched.
    pub fn advance_row_round_without_fold(&mut self, r_i: K) {
        debug_assert!(self.round_idx < self.ell_n, "row-phase rounds only");
        self.row_chals.push(r_i);
        self.round_idx += 1;
    }

    pub fn row_phase_requires_backend(&self) -> bool {
        self.row_stream.deferred_eval_tbl
            || self
                .row_stream
                .deferred_mcs
                .iter()
                .any(|&deferred| deferred)
    }

    pub(crate) fn materialize_deferred_row_equality_tables(&mut self) {
        self.row_stream
            .materialize_deferred_equality_tables(&self.ch.beta_r, self.r_inputs.as_deref());
    }

    /// Read-only view of the row-phase sumcheck tables, for accelerator
    /// backends (e.g. CUDA) that replicate the FE round evaluation off-CPU.
    /// The backend must stay field-identical to `evals_row_phase` + `fold`.
    pub fn row_phase_snapshot(&self) -> RowPhaseSnapshot<'_> {
        let rs = &self.row_stream;
        RowPhaseSnapshot {
            cur_len: rs.cur_len,
            active_len: rs.active_len,
            beta_r: &self.ch.beta_r,
            r_inputs: self.r_inputs.as_deref(),
            eq_beta_r_tbl: &rs.eq_beta_r_tbl,
            eq_r_inputs_tbl: rs.eq_r_inputs_tbl.as_deref(),
            eval_tbl: rs.eval_tbl.as_deref(),
            deferred_eval_tbl: rs.deferred_eval_tbl,
            gamma_to_k: rs.gamma_to_k,
            gamma_pow_mcs: &rs.gamma_pow_mcs,
            zero_mcs: &rs.zero_mcs,
            deferred_mcs: &rs.deferred_mcs,
            f_at_zero: rs.f_at_zero,
            sumcheck_degree_bound: self.d_sc,
            row_phase_deg_max: rs.row_phase_deg_max,
            f_var_count: rs.f_var_count,
            f_var_tables_by_mcs: rs
                .f_var_tables_by_mcs
                .iter()
                .map(|tables| {
                    tables
                        .iter()
                        .map(|table| RowTableSnapshot {
                            real: table.real_slice(),
                            imag: table.imag_slice(),
                        })
                        .collect()
                })
                .collect(),
            f_terms: rs
                .f_terms
                .iter()
                .map(|term| (term.coeff, term.vars.clone()))
                .collect(),
        }
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

    #[doc(hidden)]
    pub fn __test_ajtai_precomp_ready(&self) -> bool {
        self.ajtai_precomp.is_some()
    }

    /// Compute the univariate round polynomial for an Ajtai-bit round.
    /// DP version: removes the 2^{free_a}·D work per x and keeps outputs bit-identical.
    fn evals_ajtai_phase(&mut self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n;
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        // r'-only precomp reused across all Ajtai rounds (r' is fixed after row phase).
        self.ensure_ajtai_precomp();
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

        self.ensure_ajtai_precomp();
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
                crate::common::compute_y_zcol_from_witness(self.params, &wit.Z, self.s.m, chi_s, d_pad)
                    .unwrap_or_else(|e| panic!("ME output builder: y_zcol compute failed (MCS): {e}"))
            } else {
                Vec::new()
            };

            out.push(CeClaim {
                adv: None,
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
                crate::common::compute_y_zcol_from_witness(self.params, Zi, self.s.m, chi_s, d_pad)
                    .unwrap_or_else(|e| panic!("ME output builder: y_zcol compute failed (ME): {e}"))
            } else {
                Vec::new()
            };

            out.push(CeClaim {
                adv: None,
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
            if self.round_idx + 1 == self.ell_n {
                self.row_stream.release_finalized_tables();
            }
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
