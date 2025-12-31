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
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::{D, K};
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;
use std::sync::Arc;

use crate::sumcheck::RoundOracle;

use super::common::Challenges;
use super::sparse::CscMat;

#[derive(Clone, Debug)]
struct CompiledPolyTerm {
    coeff: K,
    /// (var_pos, exponent), where `var_pos` indexes `RowStreamState::f_var_tables`.
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

    /// Tables for the variables used by the CCS polynomial `f`.
    /// Each entry is a row-domain table of `m_j(row) = (M_j · z1)[row]` at boolean row points.
    f_var_tables: Vec<Vec<K>>,
    /// Compiled sparse polynomial terms for `f` using `f_var_tables` indices.
    f_terms: Vec<CompiledPolyTerm>,

    /// NC tables: for each witness i, a row-domain table where each entry holds Ajtai digits
    /// `[Z_i[0,row], ..., Z_i[D-1,row]]`.
    nc_tables: Vec<Vec<[K; D]>>,

    /// Weights w_beta_a[rho] = eq(bool_mask=rho, beta_a) for rho in 0..D.
    w_beta_a: Vec<K>,
    /// NC gamma coefficients: gamma_nc[i] = gamma^(i+1), matching existing oracle semantics.
    gamma_nc: Vec<K>,

    /// Combined Eval block table over rows (already summed over α' and (i,j) coefficients).
    /// When present, Eval contribution is: `eq_r_inputs(r') * gamma_to_k * eval_tbl(r')`.
    eval_tbl: Option<Vec<K>>,
    gamma_to_k: K,

    b: u32,
}

impl RowStreamState {
    fn build<Ff>(
        s: &CcsStructure<Ff>,
        b: u32,
        ch: &Challenges,
        ell_d: usize,
        ell_n: usize,
        mcs_witnesses: &[McsWitness<Ff>],
        me_witnesses: &[Mat<Ff>],
        r_inputs: Option<&[K]>,
        sparse: &SparseCache<Ff>,
    ) -> Self
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        let n_pad = 1usize << ell_n;
        let n_eff = s.n;
        let t_mats = s.t();

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

        let f_terms: Vec<CompiledPolyTerm> = s
            .f
            .terms()
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

        // Gather witnesses in oracle order: all MCS first, then ME.
        let all_witnesses: Vec<&Mat<Ff>> = mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
            .collect();
        let k_total = all_witnesses.len();

        // NC: w_beta_a weights and gamma coefficients.
        if ch.beta_a.len() != ell_d || ch.alpha.len() != ell_d {
            panic!(
                "Challenge length mismatch: alpha.len()={}, beta_a.len()={}, ell_d={ell_d}",
                ch.alpha.len(),
                ch.beta_a.len()
            );
        }
        let w_beta_a: Vec<K> = (0..D).map(|rho| eq_points_bool_mask(rho, &ch.beta_a)).collect();
        let mut gamma_nc = vec![K::ZERO; k_total];
        {
            let mut g = ch.gamma;
            for i in 0..k_total {
                gamma_nc[i] = g;
                g *= ch.gamma;
            }
        }

        // Build z1 = recomposition of Z_1 (first MCS witness).
        let mut z1: Vec<K> = vec![K::ZERO; s.m];
        {
            let b_k = K::from(Ff::from_u64(b as u64));
            let mut pow_b = [K::ZERO; D];
            let mut cur = K::ONE;
            for rho in 0..D {
                pow_b[rho] = cur;
                cur *= b_k;
            }

            let Z1 = all_witnesses
                .first()
                .copied()
                .expect("need at least one witness for z1");
            if Z1.rows() != D || Z1.cols() != s.m {
                panic!(
                    "Z1 shape mismatch: expected {}×{}, got {}×{}",
                    D,
                    s.m,
                    Z1.rows(),
                    Z1.cols()
                );
            }
            for c in 0..s.m {
                let mut acc = K::ZERO;
                for rho in 0..D {
                    acc += K::from(Z1[(rho, c)]) * pow_b[rho];
                }
                z1[c] = acc;
            }
        }

        // f-var tables: m_j(row) = (M_j * z1)[row] for each variable used by f.
        if sparse.csc.len() != t_mats {
            panic!(
                "sparse cache matrix count mismatch: got {}, expected {}",
                sparse.csc.len(),
                t_mats
            );
        }
        let mut f_var_tables: Vec<Vec<K>> = Vec::with_capacity(f_var_indices.len());
        for &j in &f_var_indices {
            let mut out = vec![K::ZERO; n_pad];
            if j == 0 {
                let cap = core::cmp::min(n_eff, z1.len());
                out[..cap].copy_from_slice(&z1[..cap]);
            } else {
                let csc = sparse.csc[j]
                    .as_ref()
                    .unwrap_or_else(|| panic!("missing CSC for matrix j={j}"));
                if csc.ncols != z1.len() {
                    panic!(
                        "matrix-vector dim mismatch for j={j}: csc.ncols={} != z1.len()={}",
                        csc.ncols,
                        z1.len()
                    );
                }
                for c in 0..csc.ncols {
                    let s0 = csc.col_ptr[c];
                    let e0 = csc.col_ptr[c + 1];
                    if s0 == e0 {
                        continue;
                    }
                    let x_c = z1[c];
                    if x_c == K::ZERO {
                        continue;
                    }
                    for k in s0..e0 {
                        let r = csc.row_idx[k];
                        if r < n_eff {
                            out[r] += K::from(csc.vals[k]) * x_c;
                        }
                    }
                }
            }
            f_var_tables.push(out);
        }

        // NC tables: digits at each boolean row index.
        let mut nc_tables: Vec<Vec<[K; D]>> = Vec::with_capacity(k_total);
        for &Zi in &all_witnesses {
            if Zi.rows() != D || Zi.cols() != s.m {
                panic!(
                    "Z shape mismatch: expected {}×{}, got {}×{}",
                    D,
                    s.m,
                    Zi.rows(),
                    Zi.cols()
                );
            }
            let mut tbl = vec![[K::ZERO; D]; n_pad];
            let cap = core::cmp::min(n_eff, n_pad);
            for row in 0..cap {
                for rho in 0..D {
                    tbl[row][rho] = K::from(Zi[(rho, row)]);
                }
            }
            nc_tables.push(tbl);
        }

        // Eval table (optional): only when both (a) there are carried witnesses, and (b) r_inputs exist.
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }

        let eval_tbl = if k_total >= 2 && eq_r_inputs_tbl.is_some() {
            let w_alpha: Vec<K> = (0..D).map(|rho| eq_points_bool_mask(rho, &ch.alpha)).collect();

            let mut gamma_pow_i = vec![K::ONE; k_total];
            for i in 1..k_total {
                gamma_pow_i[i] = gamma_pow_i[i - 1] * ch.gamma;
            }
            let mut gamma_k_pow_j = vec![K::ONE; t_mats];
            for j in 1..t_mats {
                gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
            }

            let mut eval_tbl = vec![K::ZERO; n_pad];
            for i_abs in 1..k_total {
                let Zi = all_witnesses[i_abs];
                let coeff_i = gamma_pow_i[i_abs];
                if coeff_i == K::ZERO {
                    continue;
                }

                // S_i(α) = Σ_ρ χ_α[ρ] · Z_i[ρ,·]
                let mut s_alpha = vec![K::ZERO; s.m];
                for rho in 0..D {
                    let w = w_alpha[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    for c in 0..s.m {
                        s_alpha[c] += w * K::from(Zi[(rho, c)]);
                    }
                }

                for j in 0..t_mats {
                    let coeff = coeff_i * gamma_k_pow_j[j];
                    if coeff == K::ZERO {
                        continue;
                    }

                    if j == 0 {
                        let cap = core::cmp::min(n_eff, s_alpha.len());
                        for r in 0..cap {
                            eval_tbl[r] += coeff * s_alpha[r];
                        }
                        continue;
                    }

                    let csc = sparse.csc[j]
                        .as_ref()
                        .unwrap_or_else(|| panic!("missing CSC for matrix j={j}"));
                    if csc.ncols != s_alpha.len() {
                        panic!(
                            "matrix-vector dim mismatch for eval j={j}: csc.ncols={} != s_alpha.len()={}",
                            csc.ncols,
                            s_alpha.len()
                        );
                    }
                    for c in 0..csc.ncols {
                        let s0 = csc.col_ptr[c];
                        let e0 = csc.col_ptr[c + 1];
                        if s0 == e0 {
                            continue;
                        }
                        let x_c = s_alpha[c];
                        if x_c == K::ZERO {
                            continue;
                        }
                        let scaled_x = coeff * x_c;
                        for k in s0..e0 {
                            let r = csc.row_idx[k];
                            if r < n_eff {
                                eval_tbl[r] += K::from(csc.vals[k]) * scaled_x;
                            }
                        }
                    }
                }
            }

            Some(eval_tbl)
        } else {
            None
        };

        Self {
            cur_len: n_pad,
            eq_beta_r_tbl,
            eq_r_inputs_tbl,
            f_var_tables,
            f_terms,
            nc_tables,
            w_beta_a,
            gamma_nc,
            eval_tbl,
            gamma_to_k,
            b,
        }
    }

    #[inline]
    fn fold_table_inplace(table: &mut Vec<K>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        let one_minus = K::ONE - r;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = one_minus * lo + r * hi;
        }
        table.truncate(half);
    }

    #[inline]
    fn fold_digits_table_inplace(table: &mut Vec<[K; D]>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        let one_minus = K::ONE - r;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            let mut out = [K::ZERO; D];
            for rho in 0..D {
                out[rho] = one_minus * lo[rho] + r * hi[rho];
            }
            table[i] = out;
        }
        table.truncate(half);
    }

    fn fold_inplace(&mut self, r: K) {
        Self::fold_table_inplace(&mut self.eq_beta_r_tbl, r);
        if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
            Self::fold_table_inplace(tbl, r);
        }
        for tbl in self.f_var_tables.iter_mut() {
            Self::fold_table_inplace(tbl, r);
        }
        for tbl in self.nc_tables.iter_mut() {
            Self::fold_digits_table_inplace(tbl, r);
        }
        if let Some(tbl) = self.eval_tbl.as_mut() {
            Self::fold_table_inplace(tbl, r);
        }
        self.cur_len /= 2;
    }

    /// Multiply a polynomial by an affine `(a + b·x)` in-place.
    ///
    /// Coefficients are in low→high order. Output is truncated to the input length.
    #[inline]
    fn poly_mul_affine_inplace(poly: &mut [K], a: K, b: K) {
        for d in (0..poly.len()).rev() {
            let prev = if d == 0 { K::ZERO } else { poly[d - 1] };
            poly[d] = a * poly[d] + b * prev;
        }
    }

    fn evals_row_phase<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;

        // Fast path for b=2: build the univariate coefficients once per round,
        // then evaluate cheaply at all requested points.
        if self.b == 2 {
            let three = K::from(Ff::from_u64(3));

            let f_max_term_deg: usize = self
                .f_terms
                .iter()
                .map(|term| term.vars.iter().map(|&(_, exp)| exp as usize).sum::<usize>())
                .max()
                .unwrap_or(0);
            // NC contributes degree 4 after multiplying by eq_beta_r(X).
            let deg_max = core::cmp::max(4, f_max_term_deg + 1);

            let mut coeffs = vec![K::ZERO; deg_max + 1];
            let mut inner = vec![K::ZERO; deg_max + 1];
            let mut term_poly = vec![K::ZERO; deg_max + 1];

            for t in 0..tail_len {
                // eq_beta_r(X) = e0 + e1·X
                let e0 = self.eq_beta_r_tbl[2 * t];
                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                // inner(X) = f_prime(X) + nc_total(X)
                inner.fill(K::ZERO);

                // f_prime(X): expand sparse polynomial with affine substitutions.
                for term in &self.f_terms {
                    term_poly.fill(K::ZERO);
                    term_poly[0] = term.coeff;
                    for &(var_pos, exp) in &term.vars {
                        if exp == 0 {
                            continue;
                        }
                        let tbl = &self.f_var_tables[var_pos];
                        // v(X) = a + b·X
                        let a = tbl[2 * t];
                        let b = tbl[2 * t + 1] - a;
                        for _ in 0..exp {
                            Self::poly_mul_affine_inplace(&mut term_poly, a, b);
                        }
                    }
                    for (dst, src) in inner.iter_mut().zip(term_poly.iter()) {
                        *dst += *src;
                    }
                }

                // NC: degree-3 in X before multiplying by eq_beta_r(X).
                for rho in 0..D {
                    let w = self.w_beta_a[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    let mut c0 = K::ZERO;
                    let mut c1 = K::ZERO;
                    let mut c2 = K::ZERO;
                    let mut c3 = K::ZERO;

                    for (i, gamma_i) in self.gamma_nc.iter().enumerate() {
                        // y(X) = a + b·X
                        let a = self.nc_tables[i][2 * t][rho];
                        let b = self.nc_tables[i][2 * t + 1][rho] - a;

                        // For b=2, range_product_symmetric(y) = y(y^2-1) = y^3 - y.
                        let a2 = a * a;
                        let a3 = a2 * a;
                        let b2 = b * b;
                        let b3 = b2 * b;

                        // (a + bX)^3 - (a + bX)
                        let t0 = a3 - a;
                        let t1 = (three * a2 * b) - b;
                        let t2 = three * a * b2;
                        let t3 = b3;

                        c0 += *gamma_i * t0;
                        c1 += *gamma_i * t1;
                        c2 += *gamma_i * t2;
                        c3 += *gamma_i * t3;
                    }

                    inner[0] += w * c0;
                    if deg_max >= 1 {
                        inner[1] += w * c1;
                    }
                    if deg_max >= 2 {
                        inner[2] += w * c2;
                    }
                    if deg_max >= 3 {
                        inner[3] += w * c3;
                    }
                }

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
                    if deg_max >= 1 {
                        coeffs[1] += g * (r0 * v1 + r1 * v0);
                    }
                    if deg_max >= 2 {
                        coeffs[2] += g * (r1 * v1);
                    }
                }
            }

            return xs
                .iter()
                .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                .collect();
        }

        // Generic fallback: evaluate directly at each x (slower).
        let f_arity = self.f_var_tables.len();

        xs.par_iter()
            .map(|&x| {
                let one_minus = K::ONE - x;
                let mut var_vals = vec![K::ZERO; f_arity];
                let mut sum_x = K::ZERO;

                for t in 0..tail_len {
                    let eq_beta_r =
                        one_minus * self.eq_beta_r_tbl[2 * t] + x * self.eq_beta_r_tbl[2 * t + 1];

                    // f variables at (prefix, x, tail)
                    for (pos, tbl) in self.f_var_tables.iter().enumerate() {
                        var_vals[pos] = one_minus * tbl[2 * t] + x * tbl[2 * t + 1];
                    }

                    // f_prime = f(m_vals)
                    let mut f_prime = K::ZERO;
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
                        f_prime += acc;
                    }

                    // NC: Σ_rho w_beta_a[rho] · Σ_i gamma^(i+1) · N_i(y_i_rho)
                    let mut nc_total = K::ZERO;
                    for rho in 0..D {
                        let w = self.w_beta_a[rho];
                        if w == K::ZERO {
                            continue;
                        }
                        let mut inner = K::ZERO;
                        for (i, gamma_i) in self.gamma_nc.iter().enumerate() {
                            let lo = self.nc_tables[i][2 * t][rho];
                            let hi = self.nc_tables[i][2 * t + 1][rho];
                            let y = one_minus * lo + x * hi;
                            let ni = range_product_symmetric::<Ff>(y, self.b);
                            inner += *gamma_i * ni;
                        }
                        nc_total += w * inner;
                    }

                    let mut out = eq_beta_r * (f_prime + nc_total);

                    // Eval: eq_r_inputs(r') * gamma_to_k * eval_tbl(r')
                    if let (Some(eq_tbl), Some(eval_tbl)) =
                        (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                    {
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
}

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

/// Cache of CSC matrix formats used by the optimized oracle.
#[derive(Clone)]
pub struct SparseCache<Ff> {
    // For each j: None (identity), or Some(CSC)
    csc: Vec<Option<CscMat<Ff>>>,
}

impl<Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync> SparseCache<Ff> {
    pub fn build(s: &CcsStructure<Ff>) -> Self {
        let t = s.t();

        // Parallelize sparse matrix building - happens once at setup, fully independent per matrix
        let csc: Vec<Option<CscMat<Ff>>> = (0..t)
            .into_par_iter()
            .map(|j| {
                if j == 0 {
                    None
                } else {
                    let mat = &s.matrices[j];
                    Some(CscMat::from_dense_row_major(mat))
                }
            })
            .collect();

        Self { csc }
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
    pub sparse: Arc<SparseCache<F>>,

    // Streaming row-phase state (folded in-place across row rounds)
    row_stream: RowStreamState,

    // Cached row-only precomputation for Ajtai rounds (r' fixed after row phase).
    ajtai_precomp: Option<RPrecomp>,
}

impl<'a, F> OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub fn new_with_sparse(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [McsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
        sparse: Arc<SparseCache<F>>,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one MCS instance for F-term");

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
            row_stream,
            ajtai_precomp: None,
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

        // Compute all v_j = M_j^T · χ_r' once (sparse representation).
        //
        // NOTE: `evals_row_phase` already parallelizes over the remaining Boolean row assignments,
        // so this is intentionally sequential to avoid nested rayon parallelism.
        let n_eff = core::cmp::min(self.s.n, n_sz);
        let sparse = self.sparse.as_ref();
        if sparse.csc.len() != t {
            panic!("optimized oracle sparse cache: matrix count mismatch");
        }

        // vjs_nz[j] = list of (col, vj[col]) pairs where vj[col] != 0.
        let mut vjs_nz: Vec<Vec<(usize, K)>> = Vec::with_capacity(t);

        // j=0 is identity: v1[c] = χ_r[c] for c < min(m, n_eff).
        let cap = core::cmp::min(self.s.m, n_eff);
        let mut v1_nz = Vec::with_capacity(cap);
        for c in 0..cap {
            let v = chi_r[c];
            if v != K::ZERO {
                v1_nz.push((c, v));
            }
        }
        vjs_nz.push(v1_nz);

        for j in 1..t {
            let csc = sparse.csc[j]
                .as_ref()
                .expect("optimized oracle: missing CSC matrix for j>0");
            let mut nz = Vec::<(usize, K)>::new();
            for c in 0..csc.ncols {
                let s = csc.col_ptr[c];
                let e = csc.col_ptr[c + 1];
                if s == e {
                    continue;
                }
                let mut acc = K::ZERO;
                for k in s..e {
                    let r = csc.row_idx[k];
                    if r < n_eff {
                        acc += K::from(csc.vals[k]) * chi_r[r];
                    }
                }
                if acc != K::ZERO {
                    nz.push((c, acc));
                }
            }
            vjs_nz.push(nz);
        }

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
        for (j, vj) in vjs_nz.iter().enumerate() {
            let mut acc = K::ZERO;
            for &(c, v) in vj {
                acc += z1[c] * v;
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
                for &(c, v) in &vjs_nz[0] {
                    acc += K::from(Zi[(rho, c)]) * v;
                }
                y_nc[idx][rho] = acc;
            }

            // Eval uses all v_j
            for j in 0..t {
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for &(c, v) in &vjs_nz[j] {
                        acc += K::from(Zi[(rho, c)]) * v;
                    }
                    y_eval[idx][j][rho] = acc;
                }
            }
        }

        RPrecomp {
            y_nc,
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

    /// Build Π_CCS ME outputs at the finalized row point `r'` using the oracle's cached
    /// `precompute_for_r` results (no dense matrix scans).
    pub fn build_me_outputs_from_ajtai_precomp<L>(
        &mut self,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        fold_digest: [u8; 32],
        l: &L,
    ) -> Vec<MeInstance<Cmt, F, K>>
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

        // Ensure r'-only precomputation is available.
        if self.ajtai_precomp.is_none() {
            self.ajtai_precomp = Some(self.precompute_for_r(&self.row_chals));
        }
        let pre = self
            .ajtai_precomp
            .as_ref()
            .expect("ajtai_precomp just populated");

        let d_pad = 1usize << self.ell_d;
        assert!(
            d_pad >= D,
            "ME output builder: expected 2^ell_d >= D (2^{} = {d_pad}, D = {D})",
            self.ell_d
        );

        // Base-b recomposition cache for y_scalars.
        let base = K::from(F::from_u64(self.params.b as u64));
        let mut pow_cache = vec![K::ONE; D];
        for rho in 1..D {
            pow_cache[rho] = pow_cache[rho - 1] * base;
        }
        let recompose = |digits: &[K; D]| -> K {
            let mut acc = K::ZERO;
            for rho in 0..D {
                acc += digits[rho] * pow_cache[rho];
            }
            acc
        };

        let t_mats = self.s.t();
        let mut out = Vec::with_capacity(self.mcs_witnesses.len() + self.me_witnesses.len());

        // MCS outputs (keep order).
        for (idx, (inst, wit)) in mcs_list.iter().zip(self.mcs_witnesses.iter()).enumerate() {
            let X = l.project_x(&wit.Z, inst.m_in);
            let mut y = Vec::with_capacity(t_mats);
            let mut y_scalars = Vec::with_capacity(t_mats);

            for j in 0..t_mats {
                let digits = &pre.y_eval[idx][j];
                let mut yj = vec![K::ZERO; d_pad];
                yj[..D].copy_from_slice(digits);
                y.push(yj);
                y_scalars.push(recompose(digits));
            }

            out.push(MeInstance {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inst.c.clone(),
                X,
                r: self.row_chals.clone(),
                y,
                y_scalars,
                m_in: inst.m_in,
                fold_digest,
            });
        }

        // ME outputs (keep order).
        let base_idx = self.mcs_witnesses.len();
        for (me_idx, inp) in me_inputs.iter().enumerate() {
            let idx = base_idx + me_idx;
            let mut y = Vec::with_capacity(t_mats);
            let mut y_scalars = Vec::with_capacity(t_mats);

            for j in 0..t_mats {
                let digits = &pre.y_eval[idx][j];
                let mut yj = vec![K::ZERO; d_pad];
                yj[..D].copy_from_slice(digits);
                y.push(yj);
                y_scalars.push(recompose(digits));
            }

            out.push(MeInstance {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inp.c.clone(),
                X: inp.X.clone(),
                r: self.row_chals.clone(),
                y,
                y_scalars,
                m_in: inp.m_in,
                fold_digest,
            });
        }

        out
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
            self.row_stream.fold_inplace(r_i);
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
