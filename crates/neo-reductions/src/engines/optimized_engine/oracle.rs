//! Optimized RoundOracle for Q(X) evaluation in Π_CCS.
//!
//! This oracle uses factored algebra, precomputed terms, and cached sparse formats
//! to efficiently evaluate the Q polynomial during sumcheck rounds. Mathematically 
//! equivalent to paper-exact but ~10x faster.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.

#![allow(non_snake_case)]

use neo_math::{K, D, KExtensions, Fq};
use p3_field::*;
use p3_field::PrimeField64;
use rayon::prelude::*;

use crate::sumcheck::RoundOracle;
use neo_ccs::{CcsStructure, McsWitness, Mat};

use super::common::Challenges;
use smallvec::SmallVec;
// Removed FromU64 trait

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
            if i0 >= n { break; }
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
    /// Temporary m values for F' (length t)
    m_vals: Vec<K>,
    /// Scratch buffer per row for support_ref branch
    row_buf: Vec<[K; D]>,
    /// Scratch buffer for dense χ_r
    chi_r_dense: Vec<K>,
    /// Whether y_eval was populated (Eval block enabled)
    need_eval: bool,
    /// F' = f(z_1 · v_j) - independent of α'
    f_prime: K,
    /// eq(r', β_r) - independent of α'
    eq_beta_r: K,
    /// eq(r', r_inputs) if present - independent of α'
    eq_r_inputs: K,
}

impl RPrecomp {
    fn new(k_total: usize, t: usize, m: usize, need_eval: bool) -> Self {
        let mut vjs = Vec::with_capacity(t);
        for _ in 0..t {
            vjs.push(vec![K::ZERO; m]);
        }

        let y_nc = vec![[K::ZERO; D]; k_total];
        let y_eval = if need_eval {
            (0..k_total).map(|_| vec![[K::ZERO; D]; t]).collect()
        } else {
            Vec::new()
        };

        Self {
            vjs,
            y_nc,
            y_eval,
            m_vals: vec![K::ZERO; t],
            row_buf: Vec::new(),
            chi_r_dense: Vec::new(),
            need_eval,
            f_prime: K::ZERO,
            eq_beta_r: K::ZERO,
            eq_r_inputs: K::ZERO,
        }
    }

    fn reset(
        &mut self,
        k_total: usize,
        t: usize,
        m: usize,
        need_eval: bool,
        rows_len: usize,
        chi_len: usize,
    ) {
        self.need_eval = need_eval;

        if self.vjs.len() < t {
            self.vjs.resize_with(t, || vec![K::ZERO; m]);
        }
        for v in self.vjs.iter_mut() {
            if v.len() < m {
                v.resize(m, K::ZERO);
            }
            v.fill(K::ZERO);
        }

        if self.y_nc.len() < k_total {
            self.y_nc.resize(k_total, [K::ZERO; D]);
        }
        for y in self.y_nc.iter_mut() {
            *y = [K::ZERO; D];
        }

        if need_eval {
            if self.y_eval.len() < k_total {
                self.y_eval
                    .resize_with(k_total, || vec![[K::ZERO; D]; t]);
            }
            for vec_j in self.y_eval.iter_mut().take(k_total) {
                if vec_j.len() < t {
                    vec_j.resize(t, [K::ZERO; D]);
                }
                for arr in vec_j.iter_mut() {
                    *arr = [K::ZERO; D];
                }
            }
        } else {
            self.y_eval.clear();
        }

        if self.row_buf.len() < rows_len {
            self.row_buf.resize(rows_len, [K::ZERO; D]);
        }
        for buf in self.row_buf.iter_mut().take(rows_len) {
            *buf = [K::ZERO; D];
        }

        if chi_len > 0 {
            if self.chi_r_dense.len() < chi_len {
                self.chi_r_dense.resize(chi_len, K::ZERO);
            } else {
                self.chi_r_dense[..chi_len].fill(K::ZERO);
            }
        }

        if self.m_vals.len() < t {
            self.m_vals.resize(t, K::ZERO);
        } else {
            self.m_vals[..t].fill(K::ZERO);
        }
    }
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

/// Non-zero entries of χ_r when the tail row bits (positions > fixed) are Boolean.
#[inline]
fn chi_support_with_tail_boolean(r_prime: &[K], fixed: usize, n_eff: usize) -> (Vec<usize>, Vec<K>) {
    debug_assert!(fixed + 1 <= r_prime.len(), "fixed out of bounds");

    let support_sz = 1usize << (fixed + 1);
    let mut rows = Vec::with_capacity(support_sz);
    let mut weights = Vec::with_capacity(support_sz);

    debug_assert!(support_sz <= n_eff, "support size exceeds n_eff");

    let mut tail_mask = 0usize;
    for i in (fixed + 1)..r_prime.len() {
        debug_assert!(
            r_prime[i] == K::ZERO || r_prime[i] == K::ONE,
            "row tail bits must be Boolean during row phase"
        );
        if r_prime[i] == K::ONE {
            tail_mask |= 1usize << i;
        }
    }

    for mask in 0..support_sz {
        let mut row_idx = tail_mask;
        let mut w = K::ONE;
        for bit in 0..=fixed {
            let is_one = ((mask >> bit) & 1) == 1;
            if is_one {
                row_idx |= 1usize << bit;
                w *= r_prime[bit];
            } else {
                w *= K::ONE - r_prime[bit];
            }
        }
        if row_idx < n_eff {
            rows.push(row_idx);
            weights.push(w);
        }
    }

    (rows, weights)
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
    // Precomputed witness weighting for F-term (independent of r')
    pub z1: Vec<K>,
    // Cached witness list for iteration order (MCS then ME)
    pub all_witnesses: Vec<&'a Mat<F>>,
    // Precomputed eq tables over all boolean alpha masks
    pub eq_beta_a_tbl: Vec<K>,
    pub eq_alpha_tbl: Vec<K>,
    // Precomputed range polynomial squares t^2
    pub range_t_sq: Vec<K>,
    // Gamma power tables reused across evaluations
    pub gamma_pow_i: Vec<K>,
    pub gamma_k_pow_j: Vec<K>,
    pub gamma_to_k: K,
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

    // Folded dense vectors for row phase optimization
    // U[j][row] = (M_j * z1)[row]
    pub current_U: Vec<Vec<K>>,
    // W[j][row][rho * k_total + i]
    // Flattened inner vector for cache locality: for fixed rho, all i are contiguous.
    pub current_W: Vec<Vec<Vec<K>>>,
    
    // Folded eq(r, beta_r)
    pub current_eq_beta: Vec<K>,
    // Folded eq(r, r_inputs)
    pub current_eq_r_inputs: Vec<K>,
}

impl<'a, F> OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync + 'static,
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
        assert!(
            !mcs_witnesses.is_empty(),
            "need at least one MCS instance for F-term"
        );

        let all_witnesses: Vec<&Mat<F>> = mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
            .collect();

        let d_sz = 1usize << ell_d;
        let mut eq_beta_a_tbl = vec![K::ZERO; d_sz];
        let mut eq_alpha_tbl = vec![K::ZERO; d_sz];
        for mask in 0..d_sz {
            eq_beta_a_tbl[mask] = eq_points_bool_mask(mask, &ch.beta_a);
            eq_alpha_tbl[mask] = eq_points_bool_mask(mask, &ch.alpha);
        }

        // Precompute z1 = Σ_ρ b^{ρ} · Z_1[ρ,·], independent of r'
        let bF = F::from_u64(params.b as u64);
        let mut pow_b_f = vec![F::ONE; D];
        for i in 1..D { pow_b_f[i] = pow_b_f[i - 1] * bF; }
        let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();

        let mut z1 = vec![K::ZERO; s.m];
        for c in 0..s.m {
            for rho in 0..D {
                z1[c] += K::from(mcs_witnesses[0].Z[(rho, c)]) * pow_b_k[rho];
            }
        }

        let k_total = all_witnesses.len();
        let t_mats = s.t();

        let mut gamma_pow_i = vec![K::ONE; k_total];
        for i in 1..k_total {
            gamma_pow_i[i] = gamma_pow_i[i - 1] * ch.gamma;
        }

        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total { gamma_to_k *= ch.gamma; }

        let mut gamma_k_pow_j = vec![K::ONE; t_mats];
        for j in 1..t_mats {
            gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
        }

        let mut range_t_sq = Vec::new();
        if params.b > 1 {
            range_t_sq.reserve((params.b - 1) as usize);
            for t in 1..(params.b as i64) {
                let tt = K::from(F::from_i64(t));
                range_t_sq.push(tt * tt);
            }
        }

        let mut oracle = Self {
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            z1,
            all_witnesses,
            eq_beta_a_tbl,
            eq_alpha_tbl,
            range_t_sq,
            gamma_pow_i,
            gamma_k_pow_j,
            gamma_to_k,
            ch,
            ell_d,
            ell_n,
            d_sc,
            round_idx: 0,
            row_chals: Vec::with_capacity(ell_n),
            ajtai_chals: Vec::with_capacity(ell_d),
            r_inputs: r_inputs.map(|r| r.to_vec()),
            current_U: Vec::new(),
            current_W: Vec::new(),
            current_eq_beta: Vec::new(),
            current_eq_r_inputs: Vec::new(),
        };

        // Initialize folded vectors for row phase
        oracle.init_folded_vectors();
        oracle
    }

    #[inline]
    fn num_rounds_total(&self) -> usize { self.ell_n + self.ell_d }

    #[inline]
    fn make_precomp(&self) -> RPrecomp {
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let need_eval = k_total >= 2 && self.r_inputs.is_some();
        RPrecomp::new(k_total, self.s.t(), self.s.m, need_eval)
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

        fn init_folded_vectors(&mut self) {
        let n_eff = 1usize << self.ell_n;
        let n_rows = core::cmp::min(self.s.n, n_eff);
        
        debug_assert!(n_eff.is_power_of_two());
        debug_assert!(n_rows <= n_eff);
        let t = self.s.t();
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let m = self.s.m;
        
        // 1. Prepare RHS matrix (dense, m x W)
        let width = 2 + k_total * D;
        let mut rhs = vec![F::ZERO; m * width];
        
        // Fill z1 columns (0 and 1)
        for c in 0..m {
            let coeffs = self.z1[c].as_coeffs();
            rhs[c * width + 0] = F::from_u64(coeffs[0].as_canonical_u64());
            rhs[c * width + 1] = F::from_u64(coeffs[1].as_canonical_u64());
        }
        
        // Fill Z columns
        for i in 0..k_total {
            let Zi = self.all_witnesses[i];
            let base_col = 2 + i * D;
            for rho in 0..D {
                let col_idx = base_col + rho;
                for c in 0..m {
                    rhs[c * width + col_idx] = Self::get_F(Zi, rho, c);
                }
            }
        }

        // Parallelize SpMM over matrices M_j
        let (U, W): (Vec<Vec<K>>, Vec<Vec<Vec<K>>>) = (0..t).into_par_iter().map(|j| {
            let csr = &self.s.sparse_matrices[j];
            let mut u_j = vec![K::ZERO; n_eff];
            let mut w_j = vec![vec![K::ZERO; D * k_total]; n_eff];
            
            // SpMM: Res = M_j * RHS
            for r in 0..csr.rows {
                if r >= n_rows { break; }
                
                let start_idx = csr.row_ptrs[r];
                let end_idx = csr.row_ptrs[r + 1];
                
                let mut row_acc: SmallVec<[F; 64]> = smallvec::smallvec![F::ZERO; width];
                
                for idx in start_idx..end_idx {
                    let c = csr.col_indices[idx];
                    let val = csr.values[idx];
                    
                    let rhs_row_start = c * width;
                    for k in 0..width {
                        row_acc[k] += val * rhs[rhs_row_start + k];
                    }
                }
                
                let re = Fq::from_u64(row_acc[0].as_canonical_u64());
                let im = Fq::from_u64(row_acc[1].as_canonical_u64());
                u_j[r] = K::from_coeffs([re, im]);
                
                for i in 0..k_total {
                    for rho in 0..D {
                        let rhs_col = 2 + i * D + rho;
                        let w_idx = rho * k_total + i;
                        // Direct access, no conversion needed as row_acc is F and K::from takes F
                        w_j[r][w_idx] = K::from(row_acc[rhs_col]);
                    }
                }
            }
            (u_j, w_j)
        }).unzip();
        
        debug_assert_eq!(U[0].len(), n_eff, "U length mismatch");
        self.current_U = U;
        self.current_W = W;

        // Initialize eq vectors
        self.current_eq_beta = vec![K::ZERO; n_eff];
        for r in 0..n_eff {
            let mut w = K::ONE;
            for bit in 0..self.ell_n {
                let b = self.ch.beta_r[bit];
                let is_one = ((r >> bit) & 1) == 1;
                w *= if is_one { b } else { K::ONE - b };
            }
            self.current_eq_beta[r] = w;
        }

        if let Some(ref r_in) = self.r_inputs {
            self.current_eq_r_inputs = vec![K::ZERO; n_eff];
            for r in 0..n_eff {
                let mut w = K::ONE;
                for bit in 0..self.ell_n {
                    let b = r_in[bit];
                    let is_one = ((r >> bit) & 1) == 1;
                    w *= if is_one { b } else { K::ONE - b };
                }
                self.current_eq_r_inputs[r] = w;
            }
        }
    }

    #[inline]
    fn get_F(a: &Mat<F>, row: usize, col: usize) -> F {
        if row < a.rows() && col < a.cols() { a[(row, col)] } else { F::ZERO }
    }

    #[inline]
    fn range_product_cached(&self, y: K) -> K {
        if self.range_t_sq.is_empty() {
            return y;
        }
        let y2 = y * y;
        let mut prod = y;
        for &tt2 in &self.range_t_sq {
            prod *= y2 - tt2;
        }
        prod
    }

    /// Precompute all data that depends only on r' (not on α') for row phase optimization.
    /// This eliminates redundant v_j recomputation across all boolean α' assignments.
    ///
    /// `fixed_row_bits` is `Some(fixed)` during the row phase (positions ≥ fixed are Boolean),
    /// and `None` during the Ajtai phase (full dense evaluation).
    fn precompute_for_r(&self, r_prime: &[K], fixed_row_bits: Option<usize>, pre: &mut RPrecomp) {
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t = self.s.t();
        let need_eval = k_total >= 2 && self.r_inputs.is_some();

        // Compute eq(r', β_r) and eq(r', r_inputs)
        pre.eq_beta_r = Self::eq_points(r_prime, &self.ch.beta_r);
        pre.eq_r_inputs = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };

        // Guard against oversized chi_r table when n_sz > self.s.n
        let n_sz = 1usize << r_prime.len();
        let n_eff = core::cmp::min(self.s.n, n_sz);

        let rows_len = fixed_row_bits.map(|fixed| 1usize << (fixed + 1)).unwrap_or(0);
        let chi_len = if fixed_row_bits.is_none() { n_eff } else { 0 };
        pre.reset(k_total, t, self.s.m, need_eval, rows_len, chi_len);

        // Optional sparse representation when row tail bits are Boolean
        let support_rows = fixed_row_bits.map(|fixed| chi_support_with_tail_boolean(r_prime, fixed, n_eff));
        let support_ref = support_rows.as_ref();

        // Dense χ_r fallback (Ajtai phase)
        let mut chi_r_dense = if support_ref.is_none() {
            let mut chi = std::mem::take(&mut pre.chi_r_dense);
            if chi.len() < n_eff {
                chi.resize(n_eff, K::ZERO);
            } else {
                chi[..n_eff].fill(K::ZERO);
            }
            for row in 0..n_eff {
                let mut w = K::ONE;
                for bit in 0..r_prime.len() {
                    let r = r_prime[bit];
                    let is_one = ((row >> bit) & 1) == 1;
                    w *= if is_one { r } else { K::ONE - r };
                }
                chi[row] = w;
            }
            Some(chi)
        } else {
            None
        };


        // Compute all v_j = M_j^T · χ_r' once
        (0..t)
            .into_par_iter()
            .zip(pre.vjs.par_iter_mut())
            .for_each(|(j, vj)| {
                if j == 0 {
                    if let Some((rows, weights)) = support_ref {
                        for (&row, &w) in rows.iter().zip(weights.iter()) {
                            if row < self.s.m {
                                vj[row] = w;
                            }
                        }
                    } else if let Some(ref chi_r) = chi_r_dense {
                        let cap = core::cmp::min(self.s.m, n_eff);
                        vj[..cap].copy_from_slice(&chi_r[..cap]);
                    }
                    return;
                }

                if let Some((rows, weights)) = support_ref {
                    let mat = &self.s.matrices[j];
                    for (&row, &w) in rows.iter().zip(weights.iter()) {
                        for c in 0..self.s.m {
                            vj[c] += K::from(Self::get_F(mat, row, c)) * w;
                        }
                    }
                } else if let Some(ref chi_r) = chi_r_dense {
                    let use_sparse = true;

                    if use_sparse {
                        // Use precomputed CSR from CcsStructure
                        let csr = &self.s.sparse_matrices[j];
                        
                        // vj = M_j^T * chi_r
                        // chi_r is Vec<K>, split into re/im
                        let mut chi_re = vec![F::ZERO; n_eff];
                        let mut chi_im = vec![F::ZERO; n_eff];
                        for (r, val) in chi_r.iter().enumerate().take(n_eff) {
                            let coeffs = val.as_coeffs();
                            chi_re[r] = F::from_u64(coeffs[0].as_canonical_u64());
                            chi_im[r] = F::from_u64(coeffs[1].as_canonical_u64());
                        }
                        
                        let v_re = csr.spmv_transpose(&chi_re[..csr.rows]);
                        let v_im = csr.spmv_transpose(&chi_im[..csr.rows]);
                        
                        for c in 0..self.s.m {
                            let re = Fq::from_u64(v_re[c].as_canonical_u64());
                            let im = Fq::from_u64(v_im[c].as_canonical_u64());
                            vj[c] += K::from_coeffs([re, im]);
                        }
                    } else {
                        // Dense fallback - no M_0 identity assumption
                        for row in 0..n_eff {
                            let wr = chi_r[row];
                            if wr == K::ZERO { continue; }
                            for c in 0..self.s.m {
                                vj[c] += K::from(Self::get_F(&self.s.matrices[j], row, c)) * wr;
                            }
                        }
                    }
                }
            });

        if let Some(buf) = chi_r_dense.take() {
            pre.chi_r_dense = buf;
        }

        // Compute F' = f(z_1 · v_j) - independent of α'
        for j in 0..t {
            let mut acc = K::ZERO;
            for c in 0..self.s.m {
                acc += self.z1[c] * pre.vjs[j][c];
            }
            pre.m_vals[j] = acc;
        }
        pre.f_prime = self.s.f.eval_in_ext::<K>(&pre.m_vals);

        // Precompute Y[i][j][ρ] = (Z_i · v_j)[ρ] for all instances and matrices
        if let Some((rows, weights)) = support_ref {
            let row_buf = &mut pre.row_buf[..rows.len()];
            for (idx, Zi) in self.all_witnesses.iter().enumerate() {
                // NC uses the sparse v_1 (identity matrix)
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for (&row, &w) in rows.iter().zip(weights.iter()) {
                        if row < self.s.m {
                            acc += K::from(Zi[(rho, row)]) * w;
                        }
                    }
                    pre.y_nc[idx][rho] = acc;
                }

                if need_eval {
                    for j in 0..t {
                        for (r_idx, &row) in rows.iter().enumerate() {
                            let buf_row = &mut row_buf[r_idx];
                            for rho in 0..D {
                                let mut acc = K::ZERO;
                                for c in 0..self.s.m {
                                    acc +=
                                        K::from(Zi[(rho, c)]) * K::from(Self::get_F(&self.s.matrices[j], row, c));
                                }
                                buf_row[rho] = acc;
                            }
                        }

                        for rho in 0..D {
                            let mut acc = K::ZERO;
                            for (r_idx, &w) in weights.iter().enumerate() {
                                acc += w * row_buf[r_idx][rho];
                            }
                            pre.y_eval[idx][j][rho] = acc;
                        }
                    }
                }
            }
        } else {
            for (idx, Zi) in self.all_witnesses.iter().enumerate() {
                // NC uses v_1 (j=0)
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for c in 0..self.s.m {
                        acc += K::from(Zi[(rho, c)]) * pre.vjs[0][c];
                    }
                    pre.y_nc[idx][rho] = acc;
                }

                if need_eval {
                    // Eval uses all v_j
                    for j in 0..t {
                        for rho in 0..D {
                            let mut acc = K::ZERO;
                            for c in 0..self.s.m {
                                acc += K::from(Zi[(rho, c)]) * pre.vjs[j][c];
                            }
                            pre.y_eval[idx][j][rho] = acc;
                        }
                    }
                }
            }
        }
    }

    /// Evaluate Q at a boolean α' using precomputed tables (no redundant v_j computation)
    /// Used by row phase where α' is fully boolean.
    #[allow(dead_code)]
    fn eval_q_from_precomp(&self, pre: &RPrecomp, alpha_mask: usize) -> K {
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t = self.s.t();
        
        // eq((α',r'),β) = eq(α', β_a) * eq(r', β_r)
        let eq_beta_a = self.eq_beta_a_tbl[alpha_mask];
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
        let eq_alpha = self.eq_alpha_tbl[alpha_mask];
        let eq_ar = eq_alpha * pre.eq_r_inputs;
        
        // NC sum: for boolean α', y[i] = Y_nc[i][alpha_mask]
        let mut nc_sum = K::ZERO;
        let mut g = self.ch.gamma;
        for i in 0..k_total {
            let y_val = pre.y_nc[i][rho];
            let Ni = self.range_product_cached(y_val);
            nc_sum += g * Ni;
            g *= self.ch.gamma;
        }
        
        // Eval block: for boolean α', y[i][j] = Y_eval[i][j][alpha_mask]
        let mut eval_inner_sum = K::ZERO;
        if pre.need_eval && k_total >= 2 && eq_ar != K::ZERO {
            for j in 0..t {
                let gamma_k = self.gamma_k_pow_j[j];
                for i_abs in 1..k_total {
                    let y_val = pre.y_eval[i_abs][j][rho];
                    eval_inner_sum += self.gamma_pow_i[i_abs] * gamma_k * y_val;
                }
            }
            
            eval_inner_sum = eq_ar * (self.gamma_to_k * eval_inner_sum);
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
                
                let use_sparse = true;
                
                if use_sparse {
                     // vj = M_j^T * chi_r
                    let csr = &self.s.sparse_matrices[j];
                    
                    let mut chi_re = vec![F::ZERO; n_eff];
                    let mut chi_im = vec![F::ZERO; n_eff];
                    for (r, val) in chi_r.iter().enumerate().take(n_eff) {
                        let coeffs = val.as_coeffs();
                            chi_re[r] = F::from_u64(coeffs[0].as_canonical_u64());
                            chi_im[r] = F::from_u64(coeffs[1].as_canonical_u64());
                    }
                    
                    let v_re = csr.spmv_transpose(&chi_re[..csr.rows]);
                    let v_im = csr.spmv_transpose(&chi_im[..csr.rows]);
                    
                    for c in 0..self.s.m {
                        let re = Fq::from_u64(v_re[c].as_canonical_u64());
                        let im = Fq::from_u64(v_im[c].as_canonical_u64());
                        vj[c] += K::from_coeffs([re, im]);
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
        for i in 1..D { pow_b_f[i] = pow_b_f[i - 1] * bF; }
        let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();
        
        let mut z1 = vec![K::ZERO; self.s.m];
        for c in 0..self.s.m {
            for rho in 0..D { 
                z1[c] += K::from(self.mcs_witnesses[0].Z[(rho, c)]) * pow_b_k[rho]; 
            }
        }

        // F' using precomputed vjs
        let mut m_vals: SmallVec<[K; 16]> = SmallVec::from_elem(K::ZERO, t);
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
        let S_cols: Vec<Vec<K>> = self.mcs_witnesses
            .par_iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.par_iter())
            .map(|Zi| {
                let mut sc = vec![K::ZERO; self.s.m];
                for rho in 0..min(D, d_sz) {
                    let w = chi_a[rho];
                    if w == K::ZERO { continue; }
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
                let Ni = self.range_product_cached(y_eval);
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
    fn fold_state(&mut self, r: K) {

        let n_curr = self.current_U[0].len();
        debug_assert!(n_curr.is_power_of_two(), "n_curr must be power of 2");
        debug_assert!(n_curr >= 2, "n_curr must be >= 2");
        
        let n_next = n_curr / 2;
        
        // Fold U
        self.current_U.par_iter_mut().for_each(|u_j| {
            for k in 0..n_next {
                let even = u_j[2 * k];
                let odd = u_j[2 * k + 1];
                u_j[k] = even + r * (odd - even);
            }
            u_j.truncate(n_next);
        });
        
        // Fold W
        self.current_W.par_iter_mut().for_each(|w_j| {
            for k in 0..n_next {
                let odd = std::mem::take(&mut w_j[2*k+1]);
                let mut even = std::mem::take(&mut w_j[2*k]);
                for (e, o) in even.iter_mut().zip(odd.iter()) {
                    *e = *e + r * (*o - *e);
                }
                w_j[k] = even;
            }
            w_j.truncate(n_next);
        });

        // Fold eq_beta
        // We must fold with r to preserve the prefix factor eq(r_prev, beta_prev)
        let n_next = n_curr / 2;
        for k in 0..n_next {
            let even = self.current_eq_beta[2*k];
            let odd = self.current_eq_beta[2*k+1];
            self.current_eq_beta[k] = even + r * (odd - even);
        }
        self.current_eq_beta.truncate(n_next);

        // Fold eq_r_inputs
        if !self.current_eq_r_inputs.is_empty() {
            for k in 0..n_next {
                let even = self.current_eq_r_inputs[2*k];
                let odd = self.current_eq_r_inputs[2*k+1];
                self.current_eq_r_inputs[k] = even + r * (odd - even);
            }
            self.current_eq_r_inputs.truncate(n_next);
        }

    }

    /// Compute the univariate round polynomial values at given xs for a row-bit round
    /// using the folded dense vectors (linear-time sumcheck).
    fn evals_row_phase(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.round_idx < self.ell_n, "round_idx out of bounds");
        debug_assert!(D <= self.eq_beta_a_tbl.len(), "D exceeds table size");
        
        let n_curr = self.current_U[0].len();
        debug_assert!(n_curr.is_power_of_two());
        let n_next = n_curr / 2;
        
        // Precompute folded eq weights for the tail (sum over next round's bit)
        let eq_beta_tail: Vec<K> = (0..n_next).into_par_iter().map(|k| {
            self.current_eq_beta[2*k] + self.current_eq_beta[2*k+1]
        }).collect();
        
        let eq_r_tail: Option<Vec<K>> = if !self.current_eq_r_inputs.is_empty() {
            Some((0..n_next).into_par_iter().map(|k| {
                self.current_eq_r_inputs[2*k] + self.current_eq_r_inputs[2*k+1]
            }).collect())
        } else {
            None
        };
        
        let t = self.s.t();
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let beta_j = self.ch.beta_r[self.round_idx];
        let r_in_j = self.r_inputs.as_ref().map(|r| r[self.round_idx]);
        
        // Precompute gamma powers
        let mut gamma_pow_i = vec![K::ONE; k_total];
        for i in 1..k_total { gamma_pow_i[i] = gamma_pow_i[i-1] * self.ch.gamma; }
        
        let mut gamma_k_pow_j = vec![K::ONE; t];
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total { gamma_to_k *= self.ch.gamma; }
        for j in 1..t { gamma_k_pow_j[j] = gamma_k_pow_j[j-1] * gamma_to_k; }
        
        if self.round_idx == 0 {

        }
        

        let res: Vec<K> = xs.par_iter().map(|&x| {
            let eq_bx = eq_lin(x, beta_j);
            let eq_rx = r_in_j.map(|r| eq_lin(x, r)).unwrap_or(K::ZERO);
            
            let sum: K = (0..n_next).into_par_iter().map(|k| {
                // Fold U -> m_vals
                let mut m_vals: SmallVec<[K; 16]> = SmallVec::from_elem(K::ZERO, t);
                for j in 0..t {
                    let even = self.current_U[j][2*k];
                    let odd = self.current_U[j][2*k+1];
                    m_vals[j] = even + x * (odd - even);
                }
                
                let f_prime = self.s.f.eval_in_ext::<K>(&m_vals);
                
                let mut f_nc_part = f_prime;
                let mut eval_part = K::ZERO;
                
                for rho in 0..D {
                    let eq_b_a = self.eq_beta_a_tbl[rho];
                    let eq_a = self.eq_alpha_tbl[rho];
                    
                    if eq_b_a == K::ZERO && eq_a == K::ZERO { continue; }
                    
                    // NC part
                    let mut nc_val = K::ZERO;
                    let mut g = self.ch.gamma;
                    let offset_base = rho * k_total;
                    
                    for i in 0..k_total {
                        let offset = offset_base + i;
                        let even = self.current_W[0][2*k][offset];
                        let odd = self.current_W[0][2*k+1][offset];
                        let val = even + x * (odd - even);
                        
                        let Ni = self.range_product_cached(val);
                        nc_val += g * Ni;
                        g *= self.ch.gamma;
                    }
                    
                    f_nc_part += eq_b_a * nc_val;
                    
                    // Eval part
                    if k_total >= 2 && self.r_inputs.is_some() {
                        let mut eval_val = K::ZERO;
                        for j in 0..t {
                            let gk = gamma_k_pow_j[j];
                            for i in 1..k_total {
                                let offset = offset_base + i;
                                let even = self.current_W[j][2*k][offset];
                                let odd = self.current_W[j][2*k+1][offset];
                                let val = even + x * (odd - even);
                                
                                eval_val += gamma_pow_i[i] * gk * val;
                            }
                        }
                        eval_val *= gamma_to_k;
                        eval_part += eq_a * eval_val;
                    }
                }
                
                let res = eq_beta_tail[k] * f_nc_part * eq_bx;
                if let Some(ref eq_r_t) = eq_r_tail {
                    res + eq_r_t[k] * eq_rx * eval_part
                } else {
                    res
                }
            }).sum();
            
            sum
        }).collect();
        

        res
    }

    /// Compute the univariate round polynomial for an Ajtai-bit round.
    /// DP version: removes the 2^{free_a}·D work per x and keeps outputs bit-identical.
    fn evals_ajtai_phase(&self, xs: &[K]) -> Vec<K> {

        let j = self.round_idx - self.ell_n;
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        
        // Optimization: if row phase is complete, current_U has length 1 (folded to single value per matrix)
        // We can reuse these values instead of recomputing them via precompute_for_r.
        let use_folded = self.current_U[0].len() == 1;
        
        let pre = if use_folded {
            // Construct RPrecomp from folded values
            let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
            let t = self.s.t();
            let need_eval = k_total >= 2 && self.r_inputs.is_some();
            
            let mut p = RPrecomp::new(k_total, t, self.s.m, need_eval);
            
            // F' term: m_vals[j] = current_U[j][0]
            for j_idx in 0..t {
                p.m_vals[j_idx] = self.current_U[j_idx][0];
            }
            p.f_prime = self.s.f.eval_in_ext::<K>(&p.m_vals);
            
            // NC term: y_nc[i][rho] = current_W[0][0][rho * k_total + i] (j=0 is identity/NC)
            for i in 0..k_total {
                for rho in 0..D {
                    p.y_nc[i][rho] = self.current_W[0][0][rho * k_total + i];
                }
            }
            
            // Eval term: y_eval[i][j][rho] = current_W[j][0][rho * k_total + i]
            if need_eval {
                for i in 0..k_total {
                    for j_idx in 0..t {
                        for rho in 0..D {
                            p.y_eval[i][j_idx][rho] = self.current_W[j_idx][0][rho * k_total + i];
                        }
                    }
                }
            }
            
            // eq(r', β_r) and eq(r', r_inputs) are already folded into single values
            p.eq_beta_r = self.current_eq_beta[0];
            if !self.current_eq_r_inputs.is_empty() {
                p.eq_r_inputs = self.current_eq_r_inputs[0];
            }
            
            p
        } else {
            // Fallback (should not happen if row phase ran correctly)
            let r_vec = &self.row_chals;
            let mut p = self.make_precomp();
            self.precompute_for_r(r_vec, None, &mut p);
            p
        };

        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t_mats = self.s.t();

        // Tail weights (independent of x)
        let w_beta_tail = chi_tail_weights(&self.ch.beta_a[j + 1 .. self.ell_d]);
        let w_alpha_tail = chi_tail_weights(&self.ch.alpha [j + 1 .. self.ell_d]);
        let tail_len = 1usize << free_a;
        debug_assert_eq!(w_beta_tail.len(), tail_len);
        debug_assert_eq!(w_alpha_tail.len(), tail_len);

        // Prefix factors (independent of x)
        let mut eq_beta_pref = K::ONE;
        let mut eq_alpha_pref = K::ONE;
        for i in 0..j {
            eq_beta_pref  *= eq_lin(self.ajtai_chals[i], self.ch.beta_a[i]);
            eq_alpha_pref *= eq_lin(self.ajtai_chals[i], self.ch.alpha [i]);
        }

        // Gamma powers (independent of x)
        let gamma_pow_i = &self.gamma_pow_i;
        let gamma_k_pow_j = &self.gamma_k_pow_j;
        let gamma_to_k = self.gamma_to_k;

        let prefix = &self.ajtai_chals[..j];
        
        let res: Vec<K> = xs.par_iter().map(|&x| {
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
                        let ni = self.range_product_cached(yi);
                        acc += w_beta_tail[t] * ni;
                    }
                    nc_sum += g * acc;
                    g *= self.ch.gamma;
                }
            }

            // Base: eq_beta * (F' + NC')
            let mut out = eq_beta * (pre.f_prime + nc_sum);

            // --- Eval block: γ^k · eq_ar · Σ_{j_mat,i≥2} γ^{i-1} (γ^k)^{j_mat} · Σ_tail w_alpha(tail) · ẏ_{(i,j)}(...)
            if pre.need_eval && k_total >= 2 && eq_ar_px != K::ZERO {
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
        }).collect();
        

        res
    }
}

impl<'a, F> RoundOracle for OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync + 'static,
    K: From<F>,
{
    fn num_rounds(&self) -> usize { self.num_rounds_total() }
    fn degree_bound(&self) -> usize { self.d_sc }

    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        if self.round_idx < self.ell_n {
            self.evals_row_phase(xs)
        } else {
            self.evals_ajtai_phase(xs)
        }
    }

    fn fold(&mut self, r_i: K) {
        if self.round_idx < self.ell_n {
            self.fold_state(r_i);
            self.row_chals.push(r_i);
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
