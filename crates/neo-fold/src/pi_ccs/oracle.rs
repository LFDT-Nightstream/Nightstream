//! Sum-check oracle for Generic CCS (Paper Section 4.4)
//!
//! This module provides the `GenericCcsOracle` that implements the Q polynomial
//! evaluation during the sum-check protocol for CCS reduction.

use neo_ccs::CcsStructure;
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::pi_ccs::precompute::{MlePartials, pad_to_pow2_k};
use crate::pi_ccs::sparse_matrix::Csr;
use crate::pi_ccs::eq_weights::{HalfTableEq, spmv_csr_t_weighted_fk};
use crate::sumcheck::RoundOracle;

#[cfg(feature = "debug-logs")]
use crate::pi_ccs::format_ext;

/// NC state after row rounds: y_{i,1}(r') Ajtai partials & γ weights
pub struct NcState {
    pub y_partials: Vec<Vec<K>>,
    pub gamma_pows: Vec<K>,
    pub f_at_rprime: Option<K>,
}

/// Sum-check oracle for Generic CCS (Paper Section 4.4)
/// 
/// **Paper Reference**: Section 4.4, Equation for Q polynomial:
/// ```text
/// Q(X_{[1,log(dn)]}) := eq(X,β)·(F(X_{[log(d)+1,log(dn)]}) + Σ_{i∈[k]} γ^i·NC_i(X))
///                        + γ^k·Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1}·Eval_{(i,j)}(X)
/// ```
/// where:
/// - **F**: CCS constraint polynomial f(M̃_1·z_1,...,M̃_t·z_1)
/// - **NC_i**: Norm/decomposition constraints ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)
/// - **Eval_{i,j}**: Evaluation ties eq(X,(α,r))·M̃_{i,j}(X)
/// 
/// **Implementation Note**: Uses two-axis decomposition (row-first, then Ajtai) to
/// avoid materializing the full 2^{log(dn)} tensor.
pub struct GenericCcsOracle<'a, F> 
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    #[allow(dead_code)]
    pub s: &'a CcsStructure<F>,
    // F term partials for instance 1 (row-domain MLE, LIVE in row phase)
    pub partials_first_inst: MlePartials,
    // Equality gates split by axes (two-axis structure for log(dn) domain):
    pub w_beta_a_partial: Vec<K>,   // size 2^ell_d (Ajtai dimension) for β_a
    pub w_alpha_a_partial: Vec<K>,  // size 2^ell_d (Ajtai dimension) for α
    pub w_beta_r_partial: Vec<K>,   // size 2^ell_n (row dimension)
    pub w_eval_r_partial: Vec<K>,   // eq against r_input (row dimension only, Ajtai pre-collapsed)
    // Z witnesses for all k instances (needed for NC in Ajtai phase)
    #[allow(dead_code)]  // Used only if dynamic NC computation is enabled
    pub z_witnesses: Vec<&'a neo_ccs::Mat<F>>,
    // Parameters
    #[allow(dead_code)]  // Used only if dynamic NC computation is enabled
    pub gamma: K,
    #[allow(dead_code)]  // Used for gamma_pows in NcState
    pub k_total: usize,
    #[allow(dead_code)]
    pub b: u32,
    pub ell_d: usize,  // log d (Ajtai dimension bits)
    pub ell_n: usize,  // log n (row dimension bits)
    pub d_sc: usize,
    pub round_idx: usize,  // track which round we're in (for phase detection)
    // Claimed initial sum for this sum-check instance (for diagnostics)
    #[allow(dead_code)]
    pub initial_sum_claim: K,
    // Precomputed constants for eq(·,β) block (kept for logging/initial_sum only)
    #[allow(dead_code)]
    pub f_at_beta_r: K,      // F(β_r) precomputed
    #[allow(dead_code)]  // Only used for initial_sum, not in oracle evaluation
    pub nc_sum_beta: K,      // Σ_i γ^i N_i(β) precomputed
    // Eval block (pre-collapsed over Ajtai at α), row-domain polynomial
    pub eval_row_partial: Vec<K>,
    // Row-phase bookkeeping for NC/F after rows are bound
    #[allow(dead_code)]  // Used only if dynamic NC computation is enabled
    pub row_chals: Vec<K>,           // Row challenges collected during row rounds
    #[allow(dead_code)]  // Used only if dynamic NC computation is enabled
    pub csr_m1: &'a Csr<F>,          // CSR for M_1 to build M_1^T * χ_{r'}
    pub csrs: &'a [Csr<F>],          // All M_j CSRs to build M_j^T * χ_{r'}
    // Ajtai-phase Eval aggregation and ME offset
    pub eval_ajtai_partial: Option<Vec<K>>, // length 2^ell_d
    pub me_offset: usize,                    // index where ME witnesses start in z_witnesses
    // Row-phase NC: Full y_{i,1} matrices (d×n_rows) for exact Ajtai sum computation
    // y_matrices[i] = Z_i · M_1^T, where rows are Ajtai dimension (ρ ∈ [d]),
    // columns are row dimension (folded during row rounds)
    pub nc_y_matrices: Vec<Vec<Vec<K>>>,  // [instance][ρ][row] - folded columnwise
    #[allow(dead_code)]
    pub nc_row_gamma_pows: Vec<K>,
    // NC after row binding: y_{i,1}(r') Ajtai partials & γ weights
    #[allow(dead_code)]  // Used only if dynamic NC computation is enabled
    pub nc: Option<NcState>,
}

impl<'a, F> GenericCcsOracle<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Lazily prepare NC state at first Ajtai round:
    /// Build y_{i,1}(r') = Z_i * M_1^T * χ_{r'} for all instances
    #[allow(dead_code)]  // Only used if dynamic NC computation is enabled
    fn ensure_nc_precomputed(&mut self) {
        if self.nc.is_some() { return; }
        
        // Build χ_{r'} over rows from collected row challenges
        // Use HalfTableEq + CSR to avoid materializing χ_r explicitly
        let w_r = HalfTableEq::new(&self.row_chals);
        // v1 = M_1^T · χ_{r'} ∈ K^m
        let v1 = spmv_csr_t_weighted_fk(self.csr_m1, &w_r);
        
        // For each Z_i, y_{i,1}(r') = Z_i · v1 ∈ K^d, then pad to 2^ell_d
        let mut y_partials = Vec::with_capacity(self.z_witnesses.len());
        for Zi in &self.z_witnesses {
            let z_ref = neo_ccs::MatRef::from_mat(Zi);
            let yi = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &v1);
            y_partials.push(pad_to_pow2_k(yi, self.ell_d).expect("pad y_i"));
        }
        
        // γ^i weights, i starts at 1
        let mut gamma_pows = Vec::with_capacity(self.z_witnesses.len());
        let mut g = self.gamma;
        for _ in 0..self.z_witnesses.len() { gamma_pows.push(g); g *= self.gamma; }
        
        // Also compute F(r') from already-folded row partials (instance 1 only)
        let mut m_vals_rp = Vec::with_capacity(self.partials_first_inst.s_per_j.len());
        for v in &self.partials_first_inst.s_per_j {
            debug_assert_eq!(v.len(), 1, "row partials must be folded before Ajtai");
            m_vals_rp.push(v[0]);
        }
        let f_at_rprime = self.s.f.eval_in_ext::<K>(&m_vals_rp);

        // Build Eval Ajtai partial once: Σ_{i≥2,j} γ^{(i_off+1) + j*k_total} · y'_{(i,j)}(·)
        if self.eval_ajtai_partial.is_none() {
            let d = 1usize << self.ell_d;
            let mut G_eval = vec![K::ZERO; d];

            // Compute all v_j = M_j^T · χ_{r'} ∈ K^m
            let mut vjs: Vec<Vec<K>> = Vec::with_capacity(self.csrs.len());
            for csr in self.csrs.iter() {
                vjs.push(spmv_csr_t_weighted_fk(csr, &w_r));
            }

            // Precompute γ^{i_off+1} for ME witnesses (i_off = 0..me_count-1)
            let me_count = self.z_witnesses.len().saturating_sub(self.me_offset);
            let mut gamma_pow_i = vec![K::ONE; me_count];
            let mut cur = self.gamma;
            for i_off in 0..me_count { gamma_pow_i[i_off] = cur; cur *= self.gamma; }

            // γ^{k_total}
            let mut gamma_to_k = K::ONE;
            for _ in 0..self.k_total { gamma_to_k *= self.gamma; }

            // Accumulate Ajtai vector over ME witnesses only (i≥2)
            for (i_off, Zi) in self.z_witnesses.iter().skip(self.me_offset).enumerate() {
                let z_ref = neo_ccs::MatRef::from_mat(Zi);
                for j in 0..self.s.t() {
                    let y_ij = neo_ccs::utils::mat_vec_mul_fk::<F,K>(
                        z_ref.data, z_ref.rows, z_ref.cols, &vjs[j]
                    );
                    // Paper: γ^k · γ^{i+(j-1)k-1} = γ^{i-1 + j·k}; our j runs 0..t-1
                    // Initialize with the constant γ^k, then multiply j times by γ^k
                    let mut w_pow = gamma_to_k * gamma_pow_i[i_off];
                    for _ in 0..j { w_pow *= gamma_to_k; }
                    let rho_lim = core::cmp::min(d, y_ij.len());
                    for rho in 0..rho_lim { G_eval[rho] += w_pow * y_ij[rho]; }
                }
            }

            self.eval_ajtai_partial = Some(G_eval);
        }

        #[cfg(feature = "debug-logs")]
        {
            let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg_oracle { eprintln!("[oracle][ajtai-pre] f_at_r' = {}", format_ext(f_at_rprime)); }
        }
        self.nc = Some(NcState { y_partials, gamma_pows, f_at_rprime: Some(f_at_rprime) });
    }
}

impl<'a, F> RoundOracle for GenericCcsOracle<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Total rounds = ell_n (row rounds first) + ell_d (Ajtai rounds second)
    /// Row-first ordering matches paper's two-axis sum-check decomposition
    fn num_rounds(&self) -> usize { self.ell_d + self.ell_n }
    fn degree_bound(&self) -> usize { self.d_sc }
    
    /// Evaluate Q at sample points in current variable
    /// 
    /// **Paper Reference**: Section 4.4, Q polynomial evaluation
    /// Two-phase oracle matching paper's two-axis structure:
    /// - **Row phase** (rounds 0..ell_n-1): Process X_r bits, evaluate all terms
    /// - **Ajtai phase** (rounds ell_n..ell): Process X_a bits, fold gates only
    fn evals_at(&mut self, sample_xs: &[K]) -> Vec<K> {
        #[cfg(feature = "debug-logs")]
        use crate::pi_ccs::format_ext;
        
        // Two-axis oracle: Row-first approach (rounds 0..ell_n-1 are rows, ell_n..ell are Ajtai)
        // Q(X_a, X_r) = eq(·,β)·[F(X_r)+ΣNC_i(·)] + eq(·,(α,r))·Eval
        
        let mut sample_ys = vec![K::ZERO; sample_xs.len()];
        
        // Helper: per-pair equality gate evaluation (not sum over pairs!)
        // Computes (1-X)·w[2k] + X·w[2k+1] for the k-th pair
        #[allow(dead_code)]
        let _eq_gate_pair = |weights: &Vec<K>, k: usize, X: K| -> K {
            debug_assert!(weights.len() >= 2 * (k + 1), "weights must have even length with at least {} elements", 2*(k+1));
            let w0 = weights[2 * k];
            let w1 = weights[2 * k + 1];
            (K::ONE - X) * w0 + X * w1
        };
        
        if self.round_idx < self.ell_n {
            // ===== ROW PHASE =====
            // Q(X_a,X_r) row view:
            //   Σ_k [(1-X)wβr[2k] + X wβr[2k+1]] · F_k(X)
            // + Σ_k [(1-X)wβr[2k] + X wβr[2k+1]] · (Σ_i γ^i · [Σ_Xa eq_a(Xa,βa)·NC_i(Xa,X_r)])
            // + Σ_k [(1-X)weval[2k] + X weval[2k+1]] · Geval_k(X)
            //
            // NOTE: NC requires computing the Ajtai sum EXACTLY:
            // Σ_Xa eq(Xa,βa)·NC(Xa,Xr) = Σ_ρ χ_βa[ρ] · ∏_t (y_{i,1}[ρ,Xr] - t)
            // We CANNOT pull NC out of the sum because it's non-multilinear (degree 2b-1).
            //
            // where for each pair k and each j:
            //   m_j,k(X) = (1-X)·s_j[2k] + X·s_j[2k+1]
            //   F_k(X)   = f(m_1,k(X),...,m_t,k(X))
            //   H_i,k(X) = Σ_ρ χ_βa[ρ] · ∏_t (y_{i,1}[ρ]((X,remaining_k)) - t)  [exact Ajtai sum]
            //   Geval_k(X) = (1-X)·G_eval[2k] + X·G_eval[2k+1]

            #[cfg(feature = "debug-logs")]
            {
                let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                if dbg_oracle {
                    eprintln!("[oracle][row{}] computing row-univariates (per-pair with exact NC Ajtai sum)", self.round_idx);
                }
            }

            let half_rows = self.w_beta_r_partial.len() >> 1;
            let half_eval = self.eval_row_partial.len() >> 1;
            debug_assert_eq!(self.w_beta_r_partial.len() % 2, 0);
            debug_assert_eq!(self.eval_row_partial.len() % 2, 0);
            
            for (sx, &X) in sample_xs.iter().enumerate() {
                // (A) F block: Σ_k gate_r(k,X) * f(m_k(X))
                let mut f_contribution = K::ZERO;
                for k in 0..half_rows {
                    let w0 = self.w_beta_r_partial[2*k];
                    let w1 = self.w_beta_r_partial[2*k + 1];
                    let gate_r = (K::ONE - X) * w0 + X * w1;

                    let mut m_vals = vec![K::ZERO; self.partials_first_inst.s_per_j.len()];
                    for j in 0..m_vals.len() {
                        let a = self.partials_first_inst.s_per_j[j][2*k];
                        let b = self.partials_first_inst.s_per_j[j][2*k + 1];
                        m_vals[j] = (K::ONE - X) * a + X * b;
                    }
                    let f_val = self.s.f.eval_in_ext::<K>(&m_vals);
                    let contrib = gate_r * f_val;
                    sample_ys[sx] += contrib;
                    f_contribution += contrib;
                }
                
                #[cfg(feature = "debug-logs")]
                {
                    let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                    if dbg_oracle && sx <= 1 && self.round_idx <= 1 {
                        eprintln!("[row{}][sample{}] F contribution: {} (X={})", 
                                 self.round_idx, sx, format_ext(f_contribution), format_ext(X));
                    }
                }

                // (A2) NC row block: exact Ajtai hypercube sum per pair (matches paper)
                // IMPORTANT: NC is non-linear in X_r. We must compute Ni at the two
                // discrete row assignments (bit=0 and bit=1) separately, then combine
                // them linearly via the gate weights. Interpolating yi and then taking
                // the product is incorrect.
                if !self.nc_y_matrices.is_empty() {
                    let d_ell = 1usize << self.ell_d;
                    for k in 0..half_rows {
                        let w0 = self.w_beta_r_partial[2*k];
                        let w1 = self.w_beta_r_partial[2*k + 1];
                        let gate0 = (K::ONE - X) * w0;      // weight for bit=0 branch
                        let gate1 = X * w1;                  // weight for bit=1 branch

                        let mut sum_over_i = K::ZERO;
                        for (i, y_mat) in self.nc_y_matrices.iter().enumerate() {
                            let rows_len = y_mat.len();

                            // Preload y at the two discrete row assignments for this pair
                            let mut y0_at_pair: Vec<K> = Vec::with_capacity(rows_len);
                            let mut y1_at_pair: Vec<K> = Vec::with_capacity(rows_len);
                            for rho in 0..rows_len {
                                debug_assert!(y_mat[rho].len() >= 2*k + 2);
                                y0_at_pair.push(y_mat[rho][2*k]);
                                y1_at_pair.push(y_mat[rho][2*k + 1]);
                            }

                            // Compute Ajtai sums separately for bit=0 and bit=1 branches
                            let mut ajtai_sum0 = K::ZERO;
                            let mut ajtai_sum1 = K::ZERO;
                            for xa_idx in 0..d_ell {
                                // ẑ_i(X_a, X_r=branch)
                                let mut zi0 = K::ZERO;
                                let mut zi1 = K::ZERO;
                                for (rho, (&y0, &y1)) in y0_at_pair.iter().zip(y1_at_pair.iter()).enumerate() {
                                    let mut chi_xa_rho = K::ONE;
                                    for bit_pos in 0..self.ell_d {
                                        let xa_bit = if (xa_idx >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                                        let rho_bit = if (rho >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                                        chi_xa_rho *= xa_bit * rho_bit + (K::ONE - xa_bit) * (K::ONE - rho_bit);
                                    }
                                    zi0 += chi_xa_rho * y0;
                                    zi1 += chi_xa_rho * y1;
                                }
                                // Range polynomials at the two discrete branches
                                // ∏_{t=-(b-1)}^{b-1} (· - t)
                                let mut N0 = K::ONE;
                                let mut N1 = K::ONE;
                                let low  = -((self.b as i64) - 1);
                                let high =  (self.b as i64) - 1;
                                for t in low..=high {
                                    let t_k = K::from(F::from_i64(t));
                                    N0 *= zi0 - t_k;
                                    N1 *= zi1 - t_k;
                                }
                                let eq_xa_beta = self.w_beta_a_partial[xa_idx];
                                ajtai_sum0 += eq_xa_beta * N0;
                                ajtai_sum1 += eq_xa_beta * N1;
                            }
                            // Combine branches linearly with gate weights
                            sum_over_i += self.nc_row_gamma_pows[i] * (gate0 * ajtai_sum0 + gate1 * ajtai_sum1);
                        }
                        sample_ys[sx] += sum_over_i;
                    }
                }

                // (B) Eval row block: Σ_k gate_eval(k,X) * Geval_k(X)
                let mut eval_contribution = K::ZERO;
                for k in 0..half_eval {
                    let w0 = self.w_eval_r_partial[2*k];
                    let w1 = self.w_eval_r_partial[2*k + 1];
                    let gate_eval = (K::ONE - X) * w0 + X * w1;
                    let a = self.eval_row_partial[2*k];
                    let b = self.eval_row_partial[2*k + 1];
                    let g_ev = (K::ONE - X) * a + X * b;
                    let contrib = gate_eval * g_ev;
                    sample_ys[sx] += contrib;
                    eval_contribution += contrib;
                }
                
                #[cfg(feature = "debug-logs")]
                {
                    let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                    if dbg_oracle && sx <= 1 && self.round_idx <= 1 {
                        eprintln!("[row{}][sample{}] Eval contribution: {}, TOTAL: {} (X={})", 
                                 self.round_idx, sx, format_ext(eval_contribution), 
                                 format_ext(sample_ys[sx]), format_ext(X));
                    }
                }
            }

        } else {
            // ===== AJTAI PHASE =====
            // After row rounds, only eq_a(X_a,β_a)·eq_r(r',β_r)·F(r') contributes (Option B).
            // Eval block has been collapsed already over Ajtai at α; NC is excluded in Option B.

            // Row equality gate after all row folds
            let wr_scalar = if !self.w_beta_r_partial.is_empty() {
                self.w_beta_r_partial[0]
            } else {
                K::ONE
            };

            // Ensure F(r') and Ajtai y-partials are computed
            self.ensure_nc_precomputed();
            let half_beta_a = self.w_beta_a_partial.len() >> 1;
            let _half_alpha_a = self.w_alpha_a_partial.len() >> 1;

            // (A) β-block for F: Ajtai-gated F(r') (only if available)
            if let Some(f_rp) = self.nc.as_ref().and_then(|st| st.f_at_rprime) {
                for k in 0..half_beta_a {
                    let w0b = self.w_beta_a_partial[2 * k];
                    let w1b = self.w_beta_a_partial[2 * k + 1];
                    for (sx, &X) in sample_xs.iter().enumerate() {
                        let gate_beta = (K::ONE - X) * w0b + X * w1b;
                        sample_ys[sx] += wr_scalar * gate_beta * f_rp;
                    }
                }
            }

            // (B) β-block for NC at r' (Ajtai univariate)
            if let Some(ref nc) = self.nc {
                for k in 0..half_beta_a {
                    let w0b = self.w_beta_a_partial[2 * k];
                    let w1b = self.w_beta_a_partial[2 * k + 1];
                    for (sx, &X) in sample_xs.iter().enumerate() {
                        let gate_beta = (K::ONE - X) * w0b + X * w1b;
                        let mut nc_sum = K::ZERO;
                        for (i, y) in nc.y_partials.iter().enumerate() {
                            let y0 = y[2 * k];
                            let y1 = y[2 * k + 1];
                            let yi = (K::ONE - X) * y0 + X * y1;
                            // Range polynomial: ∏_{t=-(b-1)}^{b-1} (yi - t)
                            let mut Ni = K::ONE;
                            let low  = -((self.b as i64) - 1);
                            let high =  (self.b as i64) - 1;
                            for t in low..=high {
                                Ni *= yi - K::from(F::from_i64(t));
                            }
                            nc_sum += nc.gamma_pows[i] * Ni; // γ^i
                        }
                        sample_ys[sx] += wr_scalar * gate_beta * nc_sum;
                    }
                }
            }

            // (C) EVAL(X_a, r') with Ajtai gating at α (used in unit tests and when provided)
            let wr_eval_scalar = if !self.w_eval_r_partial.is_empty() { self.w_eval_r_partial[0] } else { K::ONE };
            if let Some(ref eval_vec) = self.eval_ajtai_partial {
                let half_alpha_a = self.w_alpha_a_partial.len() >> 1;
                debug_assert_eq!(half_alpha_a, half_beta_a);
                for k in 0..half_alpha_a {
                    let a0 = eval_vec[2 * k];
                    let a1 = eval_vec[2 * k + 1];
                    for (sx, &X) in sample_xs.iter().enumerate() {
                        let w0 = self.w_alpha_a_partial[2 * k];
                        let w1 = self.w_alpha_a_partial[2 * k + 1];
                        let gate_alpha = (K::ONE - X) * w0 + X * w1;
                        let eval_x = a0 + (a1 - a0) * X;
                        sample_ys[sx] += gate_alpha * wr_eval_scalar * eval_x;
                    }
                }
            }

            #[cfg(feature = "debug-logs")]
            {
                let ajtai_round = self.round_idx - self.ell_n;
                eprintln!("[oracle][ajtai{}] Ajtai-gating F(r') and Eval via s_weighted", ajtai_round);
            }
        }
        
        // Debug: compare s(0)+s(1) vs claim for early rounds when enabled
        #[cfg(feature = "debug-logs")]
        if std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1") && self.round_idx <= 2 {
            // Try to locate indices for X=0 and X=1 for quick invariant check
            let mut idx0: Option<usize> = None;
            let mut idx1: Option<usize> = None;
            for (i, &x) in sample_xs.iter().enumerate() {
                if x == K::ZERO { idx0 = Some(i); }
                if x == K::ONE  { idx1 = Some(i); }
            }
            if let (Some(i0), Some(i1)) = (idx0, idx1) {
                let s0 = sample_ys[i0];
                let s1 = sample_ys[i1];
                let sum01 = s0 + s1;
                eprintln!("[oracle][round{}] s(0)={}, s(1)={}, s(0)+s(1)={}, claim={}",
                    self.round_idx, format_ext(s0), format_ext(s1), format_ext(sum01), format_ext(self.initial_sum_claim));
            } else if self.round_idx < self.ell_n {
                // In skip-at-one engine, we do not receive X=1. Compute s(1) ad hoc for diagnostics.
                let X1 = K::ONE;
                let mut s1 = K::ZERO;
                // F block at X=1
                let half_rows = self.w_beta_r_partial.len() >> 1;
                for k in 0..half_rows {
                    let w0 = self.w_beta_r_partial[2*k];
                    let w1 = self.w_beta_r_partial[2*k + 1];
                    let gate_r = (K::ONE - X1) * w0 + X1 * w1;
                    let mut m_vals = vec![K::ZERO; self.partials_first_inst.s_per_j.len()];
                    for j in 0..m_vals.len() {
                        let a = self.partials_first_inst.s_per_j[j][2*k];
                        let b = self.partials_first_inst.s_per_j[j][2*k + 1];
                        m_vals[j] = (K::ONE - X1) * a + X1 * b;
                    }
                    s1 += gate_r * self.s.f.eval_in_ext::<K>(&m_vals);
                }
                // NC row block at X=1 (exact Ajtai sum, branch=1 only)
                if !self.nc_y_matrices.is_empty() {
                    for k in 0..half_rows {
                        let _w0 = self.w_beta_r_partial[2*k];
                        let w1 = self.w_beta_r_partial[2*k + 1];
                        let gate1 = X1 * w1; // only branch 1 contributes at X=1
                        let mut nc_sum = K::ZERO;
                        for (i, y_mat) in self.nc_y_matrices.iter().enumerate() {
                            // Branch 1 values only
                            let rows_len = y_mat.len();
                            let mut y1_at_pair: Vec<K> = Vec::with_capacity(rows_len);
                            for rho in 0..rows_len { y1_at_pair.push(y_mat[rho][2*k + 1]); }
                            let d_ell = 1usize << self.ell_d;
                            // Ajtai sum for branch 1
                            let mut ajtai_sum1 = K::ZERO;
                            for xa_idx in 0..d_ell {
                                let mut zi1 = K::ZERO;
                                for (rho, &y1) in y1_at_pair.iter().enumerate() {
                                    let mut chi_xa_rho = K::ONE;
                                    for bit_pos in 0..self.ell_d {
                                        let xa_bit = if (xa_idx >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                                        let rho_bit = if (rho >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                                        chi_xa_rho *= xa_bit * rho_bit + (K::ONE - xa_bit) * (K::ONE - rho_bit);
                                    }
                                    zi1 += chi_xa_rho * y1;
                                }
                                let mut N1 = K::ONE;
                                // Range polynomial: ∏_{t=-(b-1)}^{b-1} (· - t)
                                let low  = -((self.b as i64) - 1);
                                let high =  (self.b as i64) - 1;
                                for t in low..=high { N1 *= zi1 - K::from(F::from_i64(t)); }
                                let eq_xa_beta = self.w_beta_a_partial[xa_idx];
                                ajtai_sum1 += eq_xa_beta * N1;
                            }
                            nc_sum += self.nc_row_gamma_pows[i] * (gate1 * ajtai_sum1);
                        }
                        s1 += nc_sum;
                    }
                }
                // Eval row block at X=1
                let half_eval = self.eval_row_partial.len() >> 1;
                for k in 0..half_eval {
                    let w0 = self.w_eval_r_partial[2*k];
                    let w1 = self.w_eval_r_partial[2*k + 1];
                    let gate_eval = (K::ONE - X1) * w0 + X1 * w1;
                    let a = self.eval_row_partial[2*k];
                    let b = self.eval_row_partial[2*k + 1];
                    let g_ev = (K::ONE - X1) * a + X1 * b;
                    s1 += gate_eval * g_ev;
                }
                // s0 from current returned samples if present
                if let Some(i0) = idx0 {
                    let s0 = sample_ys[i0];
                    let sum01 = s0 + s1;
                    eprintln!("[oracle][round{}] s(0)={}, s(1)[recomp]={}, s(0)+s(1)={}, claim={}",
                        self.round_idx, format_ext(s0), format_ext(s1), format_ext(sum01), format_ext(self.initial_sum_claim));
                } else {
                    eprintln!("[oracle][round{}] s(1)[recomp]={}, claim={}",
                        self.round_idx, format_ext(s1), format_ext(self.initial_sum_claim));
                }
            }
            eprintln!("[oracle][round{}] Returning {} samples: {:?}", 
                self.round_idx, sample_ys.len(), 
                if sample_ys.len() <= 4 { format!("{:?}", sample_ys) } else { format!("[{} values]", sample_ys.len()) });
        }
        
        sample_ys
    }
    
    /// Fold oracle state after binding one variable to challenge r_i
    /// 
    /// **Paper Reference**: Standard MLE folding: f(X) → (1-r)·f(0,X) + r·f(1,X)
    /// 
    /// **Implementation**: Folds all partial vectors in current phase:
    /// - Row phase: F partials, Eval partials, β/r equality gates
    /// - Ajtai phase: NC partials, β_a equality gates
    fn fold(&mut self, r_i: K) {
        // Fold based on current phase (row or Ajtai)
        if self.round_idx < self.ell_n {
            // Row phase: collect challenge and fold row partials (gates, F partials, Eval)
            self.row_chals.push(r_i);
            
            let fold_partial = |partial: &mut Vec<K>, r: K| {
                let n2 = partial.len() >> 1;
                for k in 0..n2 {
                    let a0 = partial[2*k];
                    let b0 = partial[2*k + 1];
                    partial[k] = (K::ONE - r) * a0 + r * b0;
                }
                partial.truncate(n2);
            };
            
            fold_partial(&mut self.w_beta_r_partial, r_i);
            fold_partial(&mut self.w_eval_r_partial, r_i);
            fold_partial(&mut self.eval_row_partial, r_i);
            
            // Fold F partials (instance 1 only)
            for v in &mut self.partials_first_inst.s_per_j {
                fold_partial(v, r_i);
            }
            // Fold NC y_matrices columnwise (over row dimension)
            if !self.nc_y_matrices.is_empty() {
                for y_mat in &mut self.nc_y_matrices {
                    for y_row in y_mat.iter_mut() {
                        let len2 = y_row.len() >> 1;
                        for k in 0..len2 {
                            y_row[k] = (K::ONE - r_i) * y_row[2*k] + r_i * y_row[2*k + 1];
                        }
                        y_row.truncate(len2);
                    }
                }
            }

        } else {
            // Ajtai phase: fold Ajtai partials (β gates and y-partials for NC)
            let fold_partial = |partial: &mut Vec<K>, r: K| {
                let n2 = partial.len() >> 1;
                for k in 0..n2 {
                    let a0 = partial[2*k];
                    let b0 = partial[2*k + 1];
                    partial[k] = (K::ONE - r) * a0 + r * b0;
                }
                partial.truncate(n2);
            };
            
            fold_partial(&mut self.w_beta_a_partial, r_i);
            fold_partial(&mut self.w_alpha_a_partial, r_i);
            
            // Fold NC Ajtai partials if they exist
            if let Some(ref mut nc) = self.nc {
                for y in &mut nc.y_partials {
                    fold_partial(y, r_i);
                }
            }
            if let Some(ref mut v) = self.eval_ajtai_partial { fold_partial(v, r_i); }
        }
        
        // Increment round index
        self.round_idx += 1;
    }
}
