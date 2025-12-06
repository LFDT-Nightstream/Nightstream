//! Optimized RoundOracle for Q(X) evaluation in Π_CCS.
//!
//! This oracle uses factored algebra, precomputed terms, and cached sparse formats
//! to efficiently evaluate the Q polynomial during sumcheck rounds. Mathematically
//! equivalent to paper-exact but ~10x faster.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.

#![allow(non_snake_case)]

use neo_math::{D, K};
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::sumcheck::RoundOracle;
use neo_ccs::{CcsStructure, Mat, McsWitness};

use super::common::Challenges;
use super::sparse::CscMat;

/// Symmetric range polynomial: ∏_{t=-(b-1)}^{b-1} (y - t) = y · ∏_{t=1}^{b-1} (y² - t²)
/// This is mathematically identical but ~2x faster (b multiplications instead of 2b-1).
#[inline]
fn range_product_symmetric<Ff>(y: K, b: u32) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if b <= 1 {
        return y;
    }
    let mut prod = y;
    for t in 1..(b as i64) {
        let tt = K::from(Ff::from_i64(t));
        prod *= (y * y) - (tt * tt);
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
    let one_minus_a = K::ONE - a;
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
            digits[i0] = one_minus_a * lo + a * hi;
            off += 1;
        }
        base += step;
    }
}

/// Given Ajtai digits y[ρ] (length D), fold prefix bits and bit j (value x),
/// then return the "heads" for all tail assignments as a compact vector
/// of length 2^{ell_d - (j+1)}. Out-of-range heads are treated as 0.
#[inline]
fn mle_heads_after(digits: &[K; D], prefix: &[K], x: K, j: usize, ell_d: usize) -> Vec<K> {
    let mut tmp = *digits;
    for b in 0..j {
        fold_bit_inplace(&mut tmp, b, prefix[b]);
    }
    fold_bit_inplace(&mut tmp, j, x);
    let tail = ell_d - (j + 1);
    let len_tail = 1usize << tail;
    let head_stride = 1usize << (j + 1);
    let mut out = vec![K::ZERO; len_tail];
    for t in 0..len_tail {
        let idx = t * head_stride;
        if idx < D {
            out[t] = tmp[idx];
        }
    }
    out
}

#[inline]
fn chi_tail_weights(bits: &[K]) -> Vec<K> {
    let t = bits.len();
    let len = 1usize << t;
    let mut w = vec![K::ONE; len];
    for mask in 0..len {
        let mut prod = K::ONE;
        for i in 0..t {
            let bi = bits[i];
            let is_one = ((mask >> i) & 1) == 1;
            prod *= if is_one { bi } else { K::ONE - bi };
        }
        w[mask] = prod;
    }
    w
}

#[inline]
fn dot_weights(vals: &[K], w: &[K]) -> K {
    debug_assert_eq!(vals.len(), w.len());
    let mut acc = K::ZERO;
    for i in 0..vals.len() {
        acc += vals[i] * w[i];
    }
    acc
}

/// Precomputation for a fixed r' (row assignment) - eliminates redundant v_j recomputation
struct RPrecomp {
    /// v_j = M_j^T · χ_r' for all j (computed once per r')
    #[allow(dead_code)]
    vjs: Vec<Vec<K>>,
    /// Y_nc[i][ρ] = (Z_i · v_1)[ρ] for NC terms
    y_nc: Vec<[K; D]>,
    /// Y_eval[i][j][ρ] = (Z_i · v_j)[ρ] for Eval terms  
    y_eval: Vec<Vec<[K; D]>>,
    /// F' = f(z_1 · v_j) - independent of α'
    f_prime: K,
    /// eq(r', β_r) - independent of α'
    eq_beta_r: K,
    /// eq(r', r_inputs) if present - independent of α'
    eq_r_inputs: K,
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

/// Get sparse threshold from environment or use default.
/// Set NEO_SPARSE_THRESH to tune (e.g., 0.02 or 0.10).
fn sparse_thresh() -> f32 {
    std::env::var("NEO_SPARSE_THRESH")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| *v > 0.0 && *v < 1.0)
        .unwrap_or(0.05)
}

/// Cache of sparse matrix formats to avoid rebuilding on every eval_q_ext call.
#[derive(Clone)]
pub struct SparseCache<Ff> {
    // For each j: None (identity), or Some(CSC)
    csc: Vec<Option<CscMat<Ff>>>,
    // Density per matrix (nnz / (n*m))
    density: Vec<f32>,
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> SparseCache<Ff> {
    fn build(s: &CcsStructure<Ff>) -> Self {
        let t = s.t();

        // Parallelize sparse matrix building - happens once at setup, fully independent per matrix
        let (csc, density): (Vec<Option<CscMat<Ff>>>, Vec<f32>) = (0..t)
            .into_par_iter()
            .map(|j| {
                if j == 0 {
                    (None, 0.0f32)
                } else {
                    let mat = &s.matrices[j];
                    let mut nnz = 0usize;
                    for r in 0..mat.rows() {
                        for c in 0..mat.cols() {
                            if mat[(r, c)] != Ff::ZERO {
                                nnz += 1;
                            }
                        }
                    }
                    let d = (nnz as f32) / ((mat.rows() * mat.cols()) as f32);
                    (Some(CscMat::from_dense_row_major(mat)), d)
                }
            })
            .unzip();

        Self { csc, density }
    }
}

pub struct OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    // Witnesses in the same order as the engine: all MCS first, then ME
    pub mcs_witnesses: &'a [McsWitness<F>],
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
    pub sparse: Option<SparseCache<F>>,
}

impl<'a, F> OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub fn new(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [McsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one MCS instance for F-term");
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
            sparse: Some(SparseCache::build(s)),
        }
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

    #[inline]
    fn get_F(a: &Mat<F>, row: usize, col: usize) -> F {
        if row < a.rows() && col < a.cols() {
            a[(row, col)]
        } else {
            F::ZERO
        }
    }

    /// Precompute all data that depends only on r' (not on α') for row phase optimization.
    /// This eliminates redundant v_j recomputation across all boolean α' assignments.
    fn precompute_for_r(&self, r_prime: &[K]) -> RPrecomp {
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t = self.s.t();

        // Build χ_r table
        let n_sz = 1usize << r_prime.len();
        let mut chi_r = vec![K::ZERO; n_sz];
        for row in 0..n_sz {
            let mut w = K::ONE;
            for bit in 0..r_prime.len() {
                let r = r_prime[bit];
                let is_one = ((row >> bit) & 1) == 1;
                w *= if is_one { r } else { K::ONE - r };
            }
            chi_r[row] = w;
        }

        // Compute eq(r', β_r) and eq(r', r_inputs)
        let eq_beta_r = Self::eq_points(r_prime, &self.ch.beta_r);
        let eq_r_inputs = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };

        // Compute all v_j = M_j^T · χ_r' once
        let n_eff = core::cmp::min(self.s.n, n_sz);
        let sparse_threshold = sparse_thresh();

        let vjs: Vec<Vec<K>> = (0..t)
            .into_par_iter()
            .map(|j| {
                if j == 0 {
                    let mut v1 = vec![K::ZERO; self.s.m];
                    let cap = core::cmp::min(self.s.m, n_eff);
                    v1[..cap].copy_from_slice(&chi_r[..cap]);
                    return v1;
                }

                let mut vj = vec![K::ZERO; self.s.m];
                let use_sparse = self
                    .sparse
                    .as_ref()
                    .and_then(|sc| sc.density.get(j).copied())
                    .map(|d| d < sparse_threshold)
                    .unwrap_or(false);

                if use_sparse {
                    if let Some(ref sc) = self.sparse {
                        if let Some(ref csc) = sc.csc[j] {
                            csc.add_mul_transpose_into::<K>(&chi_r, &mut vj, n_eff);
                        }
                    }
                } else {
                    for row in 0..n_eff {
                        let wr = chi_r[row];
                        if wr == K::ZERO {
                            continue;
                        }
                        for c in 0..self.s.m {
                            vj[c] += K::from(Self::get_F(&self.s.matrices[j], row, c)) * wr;
                        }
                    }
                }
                vj
            })
            .collect();

        // Compute F' = f(z_1 · v_j) - independent of α'
        let bF = F::from_u64(self.params.b as u64);
        let mut pow_b_f = vec![F::ONE; D];
        for i in 1..D {
            pow_b_f[i] = pow_b_f[i - 1] * bF;
        }
        let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();

        let mut z1 = vec![K::ZERO; self.s.m];
        for c in 0..self.s.m {
            for rho in 0..D {
                z1[c] += K::from(self.mcs_witnesses[0].Z[(rho, c)]) * pow_b_k[rho];
            }
        }

        let mut m_vals = vec![K::ZERO; t];
        for j in 0..t {
            let mut acc = K::ZERO;
            for c in 0..self.s.m {
                acc += z1[c] * vjs[j][c];
            }
            m_vals[j] = acc;
        }
        let f_prime = self.s.f.eval_in_ext::<K>(&m_vals);

        // Precompute Y[i][j][ρ] = (Z_i · v_j)[ρ] for all instances and matrices
        let mut y_nc = vec![[K::ZERO; D]; k_total];
        let mut y_eval = vec![vec![[K::ZERO; D]; t]; k_total];

        let all_witnesses: Vec<&Mat<F>> = self
            .mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
            .collect();

        for (idx, Zi) in all_witnesses.iter().enumerate() {
            // NC uses v_1 (j=0)
            for rho in 0..D {
                let mut acc = K::ZERO;
                for c in 0..self.s.m {
                    acc += K::from(Zi[(rho, c)]) * vjs[0][c];
                }
                y_nc[idx][rho] = acc;
            }

            // Eval uses all v_j
            for j in 0..t {
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for c in 0..self.s.m {
                        acc += K::from(Zi[(rho, c)]) * vjs[j][c];
                    }
                    y_eval[idx][j][rho] = acc;
                }
            }
        }

        RPrecomp {
            vjs,
            y_nc,
            y_eval,
            f_prime,
            eq_beta_r,
            eq_r_inputs,
        }
    }

    /// Evaluate Q at a boolean α' using precomputed tables (no redundant v_j computation)
    /// Used by row phase where α' is fully boolean.
    fn eval_q_from_precomp(&self, pre: &RPrecomp, alpha_mask: usize) -> K {
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t = self.s.t();

        // eq((α',r'),β) = eq(α', β_a) * eq(r', β_r)
        let eq_beta_a = eq_points_bool_mask(alpha_mask, &self.ch.beta_a);
        let eq_beta = eq_beta_a * pre.eq_beta_r;

        // For boolean α' where alpha_mask >= D, all rows of Z_i are beyond the matrix bounds
        // so χ_α[ρ] = 0 for all ρ < D, making NC and Eval terms zero, but F' term remains
        if alpha_mask >= D {
            return eq_beta * pre.f_prime;
        }

        // For boolean α', χ_α is a one-hot vector: χ_α[ρ] = 1 if ρ == alpha_mask, else 0
        // So Σ_ρ χ_α[ρ] · Y[ρ] simplifies to Y[alpha_mask]
        let rho = alpha_mask;

        // eq((α',r'),(α,r)) = eq(α', α) * eq(r', r_inputs)
        let eq_alpha = eq_points_bool_mask(alpha_mask, &self.ch.alpha);
        let eq_ar = eq_alpha * pre.eq_r_inputs;

        // NC sum: for boolean α', y[i] = Y_nc[i][alpha_mask]
        let mut nc_sum = K::ZERO;
        let mut g = self.ch.gamma;
        for i in 0..k_total {
            let y_val = pre.y_nc[i][rho];
            let Ni = range_product_symmetric::<F>(y_val, self.params.b);
            nc_sum += g * Ni;
            g *= self.ch.gamma;
        }

        // Eval block: for boolean α', y[i][j] = Y_eval[i][j][alpha_mask]
        let mut eval_inner_sum = K::ZERO;
        if k_total >= 2 && eq_ar != K::ZERO {
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total {
                gamma_to_k *= self.ch.gamma;
            }

            let mut gamma_pow_i = vec![K::ONE; k_total];
            for i in 1..k_total {
                gamma_pow_i[i] = gamma_pow_i[i - 1] * self.ch.gamma;
            }

            let mut gamma_k_pow_j = vec![K::ONE; t];
            for j in 1..t {
                gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
            }

            for j in 0..t {
                for i_abs in 1..k_total {
                    let y_val = pre.y_eval[i_abs][j][rho];
                    eval_inner_sum += gamma_pow_i[i_abs] * gamma_k_pow_j[j] * y_val;
                }
            }

            eval_inner_sum = eq_ar * (gamma_to_k * eval_inner_sum);
        }

        eq_beta * (pre.f_prime + nc_sum) + eval_inner_sum
    }

    /// Optimized Q evaluation: factor Ajtai MLE and precompute v_j vectors.
    /// Mathematically identical but ~8-16x faster due to reduced redundant computation.
    /// Kept for potential future use or debugging.
    #[allow(dead_code)]
    fn eval_q_ext(&self, alpha_prime: &[K], r_prime: &[K]) -> K {
        use core::cmp::min;

        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t = self.s.t();

        // Build χ tables for α′ and r′
        let d_sz = 1usize << alpha_prime.len();
        let n_sz = 1usize << r_prime.len();

        let mut chi_a = vec![K::ZERO; d_sz];
        for rho in 0..d_sz {
            let mut w = K::ONE;
            for bit in 0..alpha_prime.len() {
                let a = alpha_prime[bit];
                let is_one = ((rho >> bit) & 1) == 1;
                w *= if is_one { a } else { K::ONE - a };
            }
            chi_a[rho] = w;
        }

        let mut chi_r = vec![K::ZERO; n_sz];
        for row in 0..n_sz {
            let mut w = K::ONE;
            for bit in 0..r_prime.len() {
                let r = r_prime[bit];
                let is_one = ((row >> bit) & 1) == 1;
                w *= if is_one { r } else { K::ONE - r };
            }
            chi_r[row] = w;
        }

        let eq_beta = Self::eq_points(alpha_prime, &self.ch.beta_a) * Self::eq_points(r_prime, &self.ch.beta_r);
        let eq_ar = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(alpha_prime, &self.ch.alpha) * Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };

        // ========== OPTIMIZATION: Precompute all v_j = M_j^T · χ_r' ONCE ==========
        // Guard against oversized chi_r table when n_sz > self.s.n
        let n_eff = core::cmp::min(self.s.n, n_sz);

        // Heuristic: use sparse (CSC) if matrix density < threshold (tunable via env)
        let sparse_threshold = sparse_thresh();

        // Parallelize v_j computation - each matrix-vector product is independent
        let vjs: Vec<Vec<K>> = (0..t)
            .into_par_iter()
            .map(|j| {
                if j == 0 {
                    let mut v1 = vec![K::ZERO; self.s.m];
                    let cap = core::cmp::min(self.s.m, n_eff);
                    v1[..cap].copy_from_slice(&chi_r[..cap]);
                    return v1;
                }

                let mut vj = vec![K::ZERO; self.s.m];

                let use_sparse = self
                    .sparse
                    .as_ref()
                    .and_then(|sc| sc.density.get(j).copied())
                    .map(|d| d < sparse_threshold)
                    .unwrap_or(false);

                if use_sparse {
                    if let Some(ref sc) = self.sparse {
                        if let Some(ref csc) = sc.csc[j] {
                            csc.add_mul_transpose_into::<K>(&chi_r, &mut vj, n_eff);
                        }
                    }
                } else {
                    for row in 0..n_eff {
                        let wr = chi_r[row];
                        if wr == K::ZERO {
                            continue;
                        }
                        for c in 0..self.s.m {
                            vj[c] += K::from(Self::get_F(&self.s.matrices[j], row, c)) * wr;
                        }
                    }
                }
                vj
            })
            .collect();

        // Recompose z1 once
        let bF = F::from_u64(self.params.b as u64);
        let mut pow_b_f = vec![F::ONE; D];
        for i in 1..D {
            pow_b_f[i] = pow_b_f[i - 1] * bF;
        }
        let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();

        let mut z1 = vec![K::ZERO; self.s.m];
        for c in 0..self.s.m {
            for rho in 0..D {
                z1[c] += K::from(self.mcs_witnesses[0].Z[(rho, c)]) * pow_b_k[rho];
            }
        }

        // F' using precomputed vjs
        let mut m_vals = vec![K::ZERO; t];
        for j in 0..t {
            let mut acc = K::ZERO;
            for c in 0..self.s.m {
                acc += z1[c] * vjs[j][c];
            }
            m_vals[j] = acc;
        }
        let F_prime = self.s.f.eval_in_ext::<K>(&m_vals);

        // ========== OPTIMIZATION: Precompute S_i(α') = Σ_ρ χ_α[ρ] · Z_i[ρ,·] for ALL instances ==========
        // Parallelize S_i computation - each instance is independent
        let S_cols: Vec<Vec<K>> = self
            .mcs_witnesses
            .par_iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.par_iter())
            .map(|Zi| {
                let mut sc = vec![K::ZERO; self.s.m];
                for rho in 0..min(D, d_sz) {
                    let w = chi_a[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    for c in 0..self.s.m {
                        sc[c] += K::from(Zi[(rho, c)]) * w;
                    }
                }
                sc
            })
            .collect();

        // NC sum using factored S_i and v1
        let v1 = &vjs[0];

        // Precompute gamma powers for deterministic parallel sum
        let mut g_pows = vec![K::ONE; k_total];
        for i in 1..k_total {
            g_pows[i] = g_pows[i - 1] * self.ch.gamma;
        }

        // Parallelize NC computation - each instance is independent
        let nc_sum: K = (0..k_total)
            .into_par_iter()
            .map(|i| {
                let mut y_eval = K::ZERO;
                for c in 0..self.s.m {
                    y_eval += S_cols[i][c] * v1[c];
                }
                let Ni = range_product_symmetric::<F>(y_eval, self.params.b);
                g_pows[i] * self.ch.gamma * Ni
            })
            .sum();

        // Eval block using factored S_i and precomputed vjs
        let mut eval_inner_sum = K::ZERO;
        if k_total >= 2 && eq_ar != K::ZERO {
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total {
                gamma_to_k *= self.ch.gamma;
            }

            // Precompute gamma powers
            let mut gamma_pow_i = vec![K::ONE; k_total];
            for i in 1..k_total {
                gamma_pow_i[i] = gamma_pow_i[i - 1] * self.ch.gamma;
            }

            let mut gamma_k_pow_j = vec![K::ONE; t];
            for j in 1..t {
                gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
            }

            // Parallelize over j - each j iteration computes independent dot products
            eval_inner_sum = (0..t)
                .into_par_iter()
                .map(|j| {
                    let vj = &vjs[j];
                    let mut sum_j = K::ZERO;
                    for i_abs in 1..k_total {
                        // y_eval = <S_i(α'), vj>
                        let mut y_eval = K::ZERO;
                        for c in 0..self.s.m {
                            y_eval += S_cols[i_abs][c] * vj[c];
                        }
                        sum_j += gamma_pow_i[i_abs] * gamma_k_pow_j[j] * y_eval;
                    }
                    sum_j
                })
                .sum();

            eval_inner_sum = eq_ar * (gamma_to_k * eval_inner_sum);
        }

        eq_beta * (F_prime + nc_sum) + eval_inner_sum
    }

    /// Compute the univariate round polynomial values at given xs for a row-bit round
    /// by summing Q over the remaining Boolean variables, with the current variable set to x.
    fn evals_row_phase(&self, xs: &[K]) -> Vec<K> {
        let fixed = self.round_idx; // number of fixed row bits so far
        debug_assert!(fixed < self.ell_n, "row phase after all row bits");

        let free_rows = self.ell_n - fixed - 1;
        let tail_sz = 1usize << free_rows;

        // Precompute all Ajtai boolean assignments (full {0,1}^{ell_d})
        let d_sz = 1usize << self.ell_d;

        // Parallelize over xs - each x evaluation is independent
        xs.par_iter()
            .map(|&x| {
                // Parallelize tail loop if there are enough iterations to justify overhead
                if tail_sz >= 8 {
                    (0..tail_sz)
                        .into_par_iter()
                        .map(|r_tail| {
                            let mut r_vec = vec![K::ZERO; self.ell_n];
                            for i in 0..fixed {
                                r_vec[i] = self.row_chals[i];
                            }
                            r_vec[fixed] = x;
                            for k in 0..free_rows {
                                let bit = ((r_tail >> k) & 1) == 1;
                                r_vec[fixed + 1 + k] = if bit { K::ONE } else { K::ZERO };
                            }

                            // Precompute for this r_vec once
                            let pre = self.precompute_for_r(&r_vec);

                            // Sum over all α' using precomputed tables (no redundant work!)
                            (0..d_sz)
                                .map(|alpha_mask| self.eval_q_from_precomp(&pre, alpha_mask))
                                .sum::<K>()
                        })
                        .sum()
                } else {
                    let mut sum_x = K::ZERO;
                    for r_tail in 0..tail_sz {
                        let mut r_vec = vec![K::ZERO; self.ell_n];
                        for i in 0..fixed {
                            r_vec[i] = self.row_chals[i];
                        }
                        r_vec[fixed] = x;
                        for k in 0..free_rows {
                            let bit = ((r_tail >> k) & 1) == 1;
                            r_vec[fixed + 1 + k] = if bit { K::ONE } else { K::ZERO };
                        }

                        // Precompute for this r_vec once
                        let pre = self.precompute_for_r(&r_vec);

                        // Sum over all α' using precomputed tables
                        for alpha_mask in 0..d_sz {
                            sum_x += self.eval_q_from_precomp(&pre, alpha_mask);
                        }
                    }
                    sum_x
                }
            })
            .collect()
    }

    /// Compute the univariate round polynomial for an Ajtai-bit round.
    /// DP version: removes the 2^{free_a}·D work per x and keeps outputs bit-identical.
    fn evals_ajtai_phase(&self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n;
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        let r_vec = &self.row_chals;

        // r'-only precomp reused across all x
        let pre = self.precompute_for_r(r_vec);

        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t_mats = self.s.t();

        // Tail weights (independent of x)
        let w_beta_tail = chi_tail_weights(&self.ch.beta_a[j + 1..self.ell_d]);
        let w_alpha_tail = chi_tail_weights(&self.ch.alpha[j + 1..self.ell_d]);
        let tail_len = 1usize << free_a;
        debug_assert_eq!(w_beta_tail.len(), tail_len);
        debug_assert_eq!(w_alpha_tail.len(), tail_len);

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

        xs.par_iter()
            .map(|&x| {
                // eq((α',r'), β) factor across α' = (prefix, x, tail)
                let eq_beta_px = eq_beta_pref * eq_lin(x, self.ch.beta_a[j]);
                let eq_beta = pre.eq_beta_r * eq_beta_px;

                // eq((α',r'), (α,r)) factor if inputs present
                let eq_ar_px = if self.r_inputs.is_some() {
                    pre.eq_r_inputs * (eq_alpha_pref * eq_lin(x, self.ch.alpha[j]))
                } else {
                    K::ZERO
                };

                // --- NC block: Σ_i γ^i · Σ_tail w_beta(tail) · N_i( ẏ_{(i,1)}(prefix, x, tail) )
                let mut nc_sum = K::ZERO;
                {
                    let mut g = self.ch.gamma;
                    for i_abs in 0..k_total {
                        let vals = mle_heads_after(&pre.y_nc[i_abs], prefix, x, j, self.ell_d);
                        let mut acc = K::ZERO;
                        for t in 0..tail_len {
                            let yi = vals[t];
                            let ni = range_product_symmetric::<F>(yi, self.params.b);
                            acc += w_beta_tail[t] * ni;
                        }
                        nc_sum += g * acc;
                        g *= self.ch.gamma;
                    }
                }

                // Base: eq_beta * (F' + NC')
                let mut out = eq_beta * (pre.f_prime + nc_sum);

                // --- Eval block: γ^k · eq_ar · Σ_{j_mat,i≥2} γ^{i-1} (γ^k)^{j_mat} · Σ_tail w_alpha(tail) · ẏ_{(i,j)}(...)
                if k_total >= 2 && eq_ar_px != K::ZERO {
                    let mut inner = K::ZERO;
                    for j_mat in 0..t_mats {
                        let mut sum_j = K::ZERO;
                        for i_abs in 1..k_total {
                            let vals = mle_heads_after(&pre.y_eval[i_abs][j_mat], prefix, x, j, self.ell_d);
                            let ydot = dot_weights(&vals, &w_alpha_tail);
                            sum_j += gamma_pow_i[i_abs] * gamma_k_pow_j[j_mat] * ydot;
                        }
                        inner += sum_j;
                    }
                    out += eq_ar_px * (gamma_to_k * inner);
                }

                out
            })
            .collect()
    }
}

impl<'a, F> RoundOracle for OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
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
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
