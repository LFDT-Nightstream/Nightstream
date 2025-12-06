//! Paper-exact RoundOracle: literal, slow reference for Q(X) in Π_CCS.
//!
//! This oracle evaluates the paper's Q(X) by brute-force summing over the
//! remaining Boolean variables each round. It is suitable for testing and
//! cross-checking correctness against the optimized engine.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.
//! All γ exponents, eq-gating, and range products follow §4.4 exactly.

#![allow(non_snake_case)]

use neo_math::{D, K};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::optimized_engine::Challenges;
use crate::sumcheck::RoundOracle;
use neo_ccs::{CcsStructure, Mat, McsWitness};

#[cfg(feature = "paper-exact")]
pub struct PaperExactOracle<'a, F>
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
}

#[cfg(feature = "paper-exact")]
impl<'a, F> PaperExactOracle<'a, F>
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

    /// Evaluate the literal Q at extension point (α′, r′), including Eval block.
    /// Matches §4.4 exactly with:
    ///   Q = eq((α′,r′),β)·(F' + Σ γ^i N_i') + γ^k Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
    /// and E_{(i,j)} = eq((α′,r′),(α,r)) · ẏ'_{(i,j)}(α′).
    fn eval_q_ext(&self, alpha_prime: &[K], r_prime: &[K]) -> K {
        use core::cmp::min;

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

        // eq((α′,r′), β) and eq((α′,r′),(α,r))
        let eq_beta = Self::eq_points(alpha_prime, &self.ch.beta_a) * Self::eq_points(r_prime, &self.ch.beta_r);

        let eq_ar = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(alpha_prime, &self.ch.alpha) * Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };

        // ---------------------------
        // F' := f( Ẽ(M_j z_1)(r') ) using z_1 from the first MCS instance
        // ---------------------------
        let mut z1 = vec![K::ZERO; self.s.m];
        {
            // base-b powers in K for recomposition of digits
            let bF = F::from_u64(self.params.b as u64);
            let mut pow_b_f = vec![F::ONE; D];
            for i in 1..D {
                pow_b_f[i] = pow_b_f[i - 1] * bF;
            }
            let pow_b_k: Vec<K> = pow_b_f.iter().copied().map(K::from).collect();
            for c in 0..self.s.m {
                let mut acc = K::ZERO;
                for rho in 0..D {
                    acc += K::from(self.mcs_witnesses[0].Z[(rho, c)]) * pow_b_k[rho];
                }
                z1[c] = acc;
            }
        }

        let mut m_vals = vec![K::ZERO; self.s.t()];
        for j in 0..self.s.t() {
            // Ẽ( (M_j z_1) )(r′) = Σ_row χ_r[row] · Σ_c M_j[row,c]·z1[c]
            let mut y_eval = K::ZERO;
            for row in 0..n_sz {
                let wr = if row < self.s.n { chi_r[row] } else { K::ZERO };
                if wr == K::ZERO {
                    continue;
                }
                let mut y_row = K::ZERO;
                for c in 0..self.s.m {
                    y_row += K::from(Self::get_F(&self.s.matrices[j], row, c)) * z1[c];
                }
                y_eval += wr * y_row;
            }
            m_vals[j] = y_eval;
        }
        let F_prime = self.s.f.eval_in_ext::<K>(&m_vals);

        // ---------------------------------------
        // v1 := M_1^T · χ_{r'}  (K^m), used for N_i'
        // ---------------------------------------
        let mut v1 = vec![K::ZERO; self.s.m];
        for row in 0..n_sz {
            let wr = if row < self.s.n { chi_r[row] } else { K::ZERO };
            if wr == K::ZERO {
                continue;
            }
            for c in 0..self.s.m {
                v1[c] += K::from(Self::get_F(&self.s.matrices[0], row, c)) * wr;
            }
        }

        // ---------------------------------------
        // Σ γ^i · N_i' with Ajtai MLE at α′
        // ---------------------------------------
        let mut nc_sum = K::ZERO;
        {
            let mut g = self.ch.gamma; // γ^1
                                       // MCS instances
            for w in self.mcs_witnesses {
                let mut y_digits = vec![K::ZERO; D];
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for c in 0..self.s.m {
                        acc += K::from(w.Z[(rho, c)]) * v1[c];
                    }
                    y_digits[rho] = acc;
                }
                let mut y_eval = K::ZERO;
                for rho in 0..min(D, d_sz) {
                    y_eval += y_digits[rho] * chi_a[rho];
                }

                // Range product ∏_{t=-(b-1)}^{b-1} (y_eval - t)
                let lo = -((self.params.b as i64) - 1);
                let hi = (self.params.b as i64) - 1;
                let mut prod = K::ONE;
                for t in lo..=hi {
                    prod *= y_eval - K::from(F::from_i64(t));
                }

                nc_sum += g * prod;
                g *= self.ch.gamma;
            }
            // ME witnesses
            for Z in self.me_witnesses {
                let mut y_digits = vec![K::ZERO; D];
                for rho in 0..D {
                    let mut acc = K::ZERO;
                    for c in 0..self.s.m {
                        acc += K::from(Z[(rho, c)]) * v1[c];
                    }
                    y_digits[rho] = acc;
                }
                let mut y_eval = K::ZERO;
                for rho in 0..min(D, d_sz) {
                    y_eval += y_digits[rho] * chi_a[rho];
                }

                let lo = -((self.params.b as i64) - 1);
                let hi = (self.params.b as i64) - 1;
                let mut prod = K::ONE;
                for t in lo..=hi {
                    prod *= y_eval - K::from(F::from_i64(t));
                }

                nc_sum += g * prod;
                g *= self.ch.gamma;
            }
        }

        // ---------------------------------------
        // Eval block: compute Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · ẏ'_{(i,j)}(α′)
        // and then multiply once by outer γ^k and by eq_ar.
        // ---------------------------------------
        let mut eval_inner_sum = K::ZERO;
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        if k_total >= 2 && eq_ar != K::ZERO {
            // γ^k
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total {
                gamma_to_k *= self.ch.gamma;
            }

            for j in 0..self.s.t() {
                // vj := M_j^T · χ_{r'}
                let mut vj = vec![K::ZERO; self.s.m];
                for row in 0..n_sz {
                    let wr = if row < self.s.n { chi_r[row] } else { K::ZERO };
                    if wr == K::ZERO {
                        continue;
                    }
                    for c in 0..self.s.m {
                        vj[c] += K::from(Self::get_F(&self.s.matrices[j], row, c)) * wr;
                    }
                }

                // i starts from the second instance (skip index 0)
                for (i_abs, Zi) in self
                    .mcs_witnesses
                    .iter()
                    .map(|w| &w.Z)
                    .chain(self.me_witnesses.iter())
                    .enumerate()
                    .skip(1)
                {
                    // y_digits = Z_i · vj
                    let mut y_digits = vec![K::ZERO; D];
                    for rho in 0..D {
                        let mut acc = K::ZERO;
                        for c in 0..self.s.m {
                            acc += K::from(Zi[(rho, c)]) * vj[c];
                        }
                        y_digits[rho] = acc;
                    }
                    // ẏ'_{(i,j)}(α′) = ⟨ y_digits, χ_{α′} ⟩
                    let mut y_eval = K::ZERO;
                    for rho in 0..min(D, d_sz) {
                        y_eval += y_digits[rho] * chi_a[rho];
                    }

                    // inner weight = γ^{i-1} · (γ^k)^j  (0-based j)
                    let mut weight = K::ONE;
                    for _ in 0..i_abs {
                        weight *= self.ch.gamma;
                    } // γ^{i-1}
                    for _ in 0..j {
                        weight *= gamma_to_k;
                    } // (γ^k)^j

                    eval_inner_sum += weight * y_eval;
                }
            }

            // Multiply by the outer γ^k and by eq_ar
            eval_inner_sum = eq_ar * (gamma_to_k * eval_inner_sum);
        } else {
            eval_inner_sum = K::ZERO;
        }

        // Assemble Q(α′, r′)
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
        let mut alphas_bool: Vec<Vec<K>> = Vec::with_capacity(d_sz);
        for a_mask in 0..d_sz {
            let mut a = vec![K::ZERO; self.ell_d];
            for bit in 0..self.ell_d {
                a[bit] = if ((a_mask >> bit) & 1) == 1 { K::ONE } else { K::ZERO };
            }
            alphas_bool.push(a);
        }

        xs.iter()
            .map(|&x| {
                let mut sum_x = K::ZERO;
                for r_tail in 0..tail_sz {
                    let mut r_vec = vec![K::ZERO; self.ell_n];
                    // prefix fixed
                    for i in 0..fixed {
                        r_vec[i] = self.row_chals[i];
                    }
                    // current variable
                    r_vec[fixed] = x;
                    // remaining bits as boolean mask
                    for k in 0..free_rows {
                        let bit = ((r_tail >> k) & 1) == 1;
                        r_vec[fixed + 1 + k] = if bit { K::ONE } else { K::ZERO };
                    }

                    // sum over all Ajtai boolean assignments
                    for a in alphas_bool.iter() {
                        sum_x += self.eval_q_ext(a, &r_vec);
                    }
                }
                sum_x
            })
            .collect()
    }

    /// Compute the univariate round polynomial values at given xs for an Ajtai-bit round
    /// by summing Q over the remaining Ajtai Boolean variables, with the current variable set to x.
    fn evals_ajtai_phase(&self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n; // number of fixed Ajtai bits so far
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        let tail_sz = 1usize << free_a;

        // Fixed row vector is the fully collected row_chals
        let r_vec = self.row_chals.clone();

        xs.iter()
            .map(|&x| {
                let mut sum_x = K::ZERO;
                for a_tail in 0..tail_sz {
                    let mut a_vec = vec![K::ZERO; self.ell_d];
                    // prefix fixed
                    for i in 0..j {
                        a_vec[i] = self.ajtai_chals[i];
                    }
                    // current var
                    a_vec[j] = x;
                    // remaining bits (Boolean)
                    for k in 0..free_a {
                        let bit = ((a_tail >> k) & 1) == 1;
                        a_vec[j + 1 + k] = if bit { K::ONE } else { K::ZERO };
                    }
                    sum_x += self.eval_q_ext(&a_vec, &r_vec);
                }
                sum_x
            })
            .collect()
    }
}

#[cfg(feature = "paper-exact")]
impl<'a, F> RoundOracle for PaperExactOracle<'a, F>
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
