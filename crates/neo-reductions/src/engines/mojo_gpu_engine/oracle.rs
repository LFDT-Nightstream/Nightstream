use crate::sumcheck::RoundOracle;
use crate::engines::mojo_gpu_engine::linops::{LinOps, CpuLinOps};
use neo_ccs::{CcsStructure, McsWitness, Mat};
use neo_params::NeoParams;
use neo_math::{K, D};
use p3_field::{Field, PrimeCharacteristicRing};
use crate::engines::optimized_engine::Challenges;

/// Utility for eq(p, q)
#[inline]
pub fn eq_points(p: &[K], q: &[K]) -> K {
    assert_eq!(p.len(), q.len(), "eq_points: length mismatch");
    let mut acc = K::ONE;
    for i in 0..p.len() {
        let (pi, qi) = (p[i], q[i]);
        acc *= (K::ONE - pi) * (K::ONE - qi) + pi * qi;
    }
    acc
}

pub struct MojoOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync + 'static,
    K: From<F>,
{
    s: &'a CcsStructure<F>,
    params: &'a NeoParams,
    mcs_witnesses: &'a [McsWitness<F>],
    me_witnesses: &'a [Mat<F>],
    ch: Challenges,
    ell_d: usize,
    ell_n: usize,
    d_sc: usize,
    r_inputs: Option<Vec<K>>,
    // Backend
    backend: CpuLinOps, // Hardcoded for now
    
    // Round tracking
    round_idx: usize,
    row_chals: Vec<K>,
    ajtai_chals: Vec<K>,
}

impl<'a, F> MojoOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync + 'static,
    K: From<F>,
{
    pub fn new(
        s: &'a CcsStructure<F>,
        params: &'a NeoParams,
        mcs_witnesses: &'a [McsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
    ) -> Self {
        Self {
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_n,
            d_sc,
            r_inputs: r_inputs.map(|r| r.to_vec()),
            backend: CpuLinOps,
            round_idx: 0,
            row_chals: Vec::with_capacity(ell_n),
            ajtai_chals: Vec::with_capacity(ell_d),
        }
    }
}

impl<'a, F> RoundOracle for MojoOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync + 'static,
    K: From<F>,
{
    fn num_rounds(&self) -> usize {
        self.ell_n + self.ell_d
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

impl<'a, F> MojoOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync + 'static,
    K: From<F>,
{
    /// Evaluate Q(r') at an extension point r' = [row_bits, ajtai_bits].
    /// r' is interpreted as [row_bits | ajtai_bits] (ell_n row bits, then ell_d Ajtai bits).
    /// This is a permutation of the paper's (alpha, r) order but all eq-gating uses the same permutation on both arguments, so Q is unchanged.
    pub fn eval_q_ext(&mut self, r_prime: &[K]) -> K {
        debug_assert_eq!(r_prime.len(), self.ell_n + self.ell_d);
        let (r_row, r_ajtai) = r_prime.split_at(self.ell_n);
        
        // Compute chi_row (size n)
        let n_sz = 1 << self.ell_n;
        let mut chi_row = vec![K::ZERO; n_sz];
        for row in 0..n_sz {
            let mut val = K::ONE;
            for (bit, &r_i) in r_row.iter().enumerate() {
                let is_one = ((row >> bit) & 1) == 1;
                val *= if is_one { r_i } else { K::ONE - r_i };
            }
            chi_row[row] = val;
        }

        // Compute chi_ajtai (size d)
        let d_sz = 1 << self.ell_d;
        let mut chi_ajtai = vec![K::ZERO; d_sz];
        for rho in 0..d_sz {
            let mut val = K::ONE;
            for (bit, &r_i) in r_ajtai.iter().enumerate() {
                let is_one = ((rho >> bit) & 1) == 1;
                val *= if is_one { r_i } else { K::ONE - r_i };
            }
            chi_ajtai[rho] = val;
        }

        // 2. Compute v_j = M_j^T * chi_row
        let mut v_js = Vec::with_capacity(self.s.t());
        for j in 0..self.s.t() {
            let m_j = &self.s.matrices[j];
            let mut v_j = vec![K::ZERO; self.s.m];
            // Slice chi_row to n (rows of M_j)
            self.backend.gemv_transpose_FK(m_j, &chi_row[0..self.s.n], &mut v_j);
            v_js.push(v_j);
        }

        // 3. Compute F' term
        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let mut all_y_digits = Vec::with_capacity(k_total); // [i][j][rho]
        
        let d = D;
        let bK = K::from(F::from_u64(self.params.b as u64));
        let mut pow_b = vec![K::ONE; d];
        for i in 1..d { pow_b[i] = pow_b[i-1] * bK; }

        for wit_Z in self.mcs_witnesses.iter().map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
        {
            let mut y_digits_i = Vec::with_capacity(self.s.t());
            for j in 0..self.s.t() {
                let mut y_ij = vec![K::ZERO; d];
                self.backend.gemv_FK(wit_Z, &v_js[j], &mut y_ij);
                y_digits_i.push(y_ij);
            }
            all_y_digits.push(y_digits_i);
        }

        // F' calculation
        let mut m_vals = vec![K::ZERO; self.s.t()];
        for j in 0..self.s.t() {
            let y_1j = &all_y_digits[0][j]; // Z1
            let mut acc = K::ZERO;
            for rho in 0..d {
                acc += y_1j[rho] * pow_b[rho];
            }
            m_vals[j] = acc;
        }
        let F_prime = self.s.f.eval_in_ext::<K>(&m_vals);

        // NC' calculation
        let mut nc_sum = K::ZERO;
        let mut g = self.ch.gamma;
        for i in 0..k_total {
            let y_i1 = &all_y_digits[i][0]; // j=0 (M1=I)
            let mut y_eval = K::ZERO;
            for rho in 0..core::cmp::min(d, d_sz) {
                y_eval += y_i1[rho] * chi_ajtai[rho];
            }
            
            let mut prod = K::ONE;
            let b_int = self.params.b as i64;
            for t in -(b_int-1)..=b_int-1 {
                prod *= y_eval - K::from(F::from_i64(t));
            }
            
            nc_sum += g * prod;
            g *= self.ch.gamma;
        }

        // Eval' calculation
        let mut eval_sum = K::ZERO;
        if k_total >= 2 && self.r_inputs.is_some() {
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total { gamma_to_k *= self.ch.gamma; }
            
            for j in 0..self.s.t() {
                for i in 1..k_total { // skip i=0 (Z1)
                    let y_ij = &all_y_digits[i][j];
                    let mut y_eval = K::ZERO;
                    for rho in 0..core::cmp::min(d, d_sz) {
                        y_eval += y_ij[rho] * chi_ajtai[rho];
                    }
                    
                    let mut weight = K::ONE;
                    for _ in 0..i { weight *= self.ch.gamma; }
                    for _ in 0..j { weight *= gamma_to_k; }
                    
                    eval_sum += weight * y_eval;
                }
            }
        }

        // Final assembly
        let eq_beta_r = eq_points(r_row, &self.ch.beta_r);
        let eq_beta_a = eq_points(r_ajtai, &self.ch.beta_a);
        let eq_beta = eq_beta_r * eq_beta_a;

        let eq_ar = if let Some(r_in) = &self.r_inputs {
             let eq_r = eq_points(r_row, r_in);
             let eq_a = eq_points(r_ajtai, &self.ch.alpha);
             eq_r * eq_a
        } else {
            K::ZERO
        };

        let mut gamma_to_k_outer = K::ONE;
        for _ in 0..k_total { gamma_to_k_outer *= self.ch.gamma; }

        eq_beta * (F_prime + nc_sum) + eq_ar * gamma_to_k_outer * eval_sum
    }

    fn evals_row_phase(&mut self, xs: &[K]) -> Vec<K> {
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

        xs.iter().map(|&x| {
            let mut sum_x = K::ZERO;
            for r_tail in 0..tail_sz {
                let mut r_vec = vec![K::ZERO; self.ell_n];
                // prefix fixed
                for i in 0..fixed { r_vec[i] = self.row_chals[i]; }
                // current variable
                r_vec[fixed] = x;
                // remaining bits as boolean mask
                for k in 0..free_rows {
                    let bit = ((r_tail >> k) & 1) == 1;
                    r_vec[fixed + 1 + k] = if bit { K::ONE } else { K::ZERO };
                }

                // sum over all Ajtai boolean assignments
                for a in alphas_bool.iter() {
                    // Concatenate r_vec and a
                    let mut full_r = Vec::with_capacity(self.ell_n + self.ell_d);
                    full_r.extend_from_slice(&r_vec);
                    full_r.extend_from_slice(a);
                    sum_x += self.eval_q_ext(&full_r);
                }
            }
            sum_x
        }).collect()
    }

    fn evals_ajtai_phase(&mut self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n; // number of fixed Ajtai bits so far
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        let tail_sz = 1usize << free_a;

        // Fixed row vector is the fully collected row_chals
        let r_vec = self.row_chals.clone();

        xs.iter().map(|&x| {
            let mut sum_x = K::ZERO;
            for a_tail in 0..tail_sz {
                let mut a_vec = vec![K::ZERO; self.ell_d];
                // prefix fixed
                for i in 0..j { a_vec[i] = self.ajtai_chals[i]; }
                // current var
                a_vec[j] = x;
                // remaining bits (Boolean)
                for k in 0..free_a {
                    let bit = ((a_tail >> k) & 1) == 1;
                    a_vec[j + 1 + k] = if bit { K::ONE } else { K::ZERO };
                }
                
                // Concatenate r_vec and a_vec
                let mut full_r = Vec::with_capacity(self.ell_n + self.ell_d);
                full_r.extend_from_slice(&r_vec);
                full_r.extend_from_slice(&a_vec);
                sum_x += self.eval_q_ext(&full_r);
            }
            sum_x
        }).collect()
    }
}
