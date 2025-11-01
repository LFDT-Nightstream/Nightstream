//! Main oracle delegator implementing RoundOracle trait
//!
//! This orchestrates the two-phase sum-check oracle: row phase 
//! (rounds 0..ell_n-1) and Ajtai phase (rounds ell_n..ell_n+ell_d-1).

use neo_ccs::{CcsStructure, MatRef, utils::mat_vec_mul_fk};
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::pi_ccs::precompute::{MlePartials, pad_to_pow2_k};
use crate::pi_ccs::sparse_matrix::Csr;
use crate::pi_ccs::eq_weights::{HalfTableEq, RowWeight, spmv_csr_t_weighted_fk};
use neo_math::KExtensions; // for K::as_coeffs in diagnostics
use crate::sumcheck::RoundOracle;
use crate::pi_ccs::oracle::NcState;

#[cfg(feature = "debug-logs")]
use crate::pi_ccs::format_ext;

/// Sum-check oracle for Generic CCS (Paper Section 4.4)
/// 
/// **Paper Reference**: Section 4.4, Equation for Q polynomial:
/// ```text
/// Q(X_{[1,log(dn)]}) := eq(X,β)·(F(X_{[log(d)+1,log(dn)]}) + Σ_{i∈[k]} γ^i·NC_i(X))
///                        + γ^k·Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1}·Eval_{(i,j)}(X)
/// ```
pub struct GenericCcsOracle<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    // Core structure and parameters
    pub s: &'a CcsStructure<F>,
    pub gamma: K,
    pub k_total: usize,
    pub b: u32,
    pub ell_d: usize,
    pub ell_n: usize,
    pub d_sc: usize,
    pub round_idx: usize,
    pub me_offset: usize,
    
    // Shared mutable state (needed for phase transitions)
    pub partials_first_inst: MlePartials,
    pub w_beta_a_partial: Vec<K>,
    pub w_alpha_a_partial: Vec<K>,
    pub w_beta_r_partial: Vec<K>,
    pub w_eval_r_partial: Vec<K>,
    pub eval_row_partial: Vec<K>,
    pub eval_ajtai_partial: Option<Vec<K>>,
    pub row_chals: Vec<K>,
    
    // Resources for Ajtai precomputation
    pub z_witnesses: Vec<&'a neo_ccs::Mat<F>>,
    pub csr_m1: &'a Csr<F>,
    pub csrs: &'a [Csr<F>],
    pub nc_y_matrices: Vec<Vec<Vec<K>>>,
    pub nc_row_gamma_pows: Vec<K>,
    pub nc_state: Option<NcState>,
    
    // Debug/diagnostic fields
    pub initial_sum_claim: K,
    pub f_at_beta_r: K,
    pub nc_sum_beta: K,
}

impl<'a, F> GenericCcsOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Send + Sync + Copy,
    K: From<F>,
{
    /// Create a new oracle instance
    pub fn new(
        s: &'a CcsStructure<F>,
        partials_first_inst: MlePartials,
        w_beta_a_partial: Vec<K>,
        w_alpha_a_partial: Vec<K>,
        w_beta_r_partial: Vec<K>,
        w_eval_r_partial: Vec<K>,
        eval_row_partial: Vec<K>,
        z_witnesses: Vec<&'a neo_ccs::Mat<F>>,
        csr_m1: &'a Csr<F>,
        csrs: &'a [Csr<F>],
        nc_y_matrices: Vec<Vec<Vec<K>>>,
        nc_row_gamma_pows: Vec<K>,
        gamma: K,
        k_total: usize,
        b: u32,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        me_offset: usize,
        initial_sum_claim: K,
        f_at_beta_r: K,
        nc_sum_beta: K,
    ) -> Self {
        #[cfg(feature = "debug-logs")]
        {
            eprintln!("[GenericCcsOracle::new] nc_y_matrices.len() = {}", nc_y_matrices.len());
            eprintln!("[GenericCcsOracle::new] nc_row_gamma_pows = {:?}", 
                     nc_row_gamma_pows.iter().map(|g| crate::pi_ccs::format_ext(*g)).collect::<Vec<_>>());
            eprintln!("[GenericCcsOracle::new] k_total = {}", k_total);
            eprintln!("[GenericCcsOracle::new] nc_sum_beta = {}", crate::pi_ccs::format_ext(nc_sum_beta));
        }

        // Debug-only sanity checks for schedule, sizes, and eq tables
        #[cfg(debug_assertions)]
        {
            // Ajtai eq tables must be full size 2^ell_d and sum to 1
            let d_a = 1usize << ell_d;
            debug_assert_eq!(w_beta_a_partial.len(), d_a, "w_beta_a_partial wrong length; expected 2^ell_d");
            debug_assert_eq!(w_alpha_a_partial.len(), d_a, "w_alpha_a_partial wrong length; expected 2^ell_d");
            let sum_beta_a: K = w_beta_a_partial.iter().copied().sum();
            let sum_alpha_a: K = w_alpha_a_partial.iter().copied().sum();
            debug_assert_eq!(sum_beta_a, K::ONE, "Sum of chi_beta_a table must be 1");
            debug_assert_eq!(sum_alpha_a, K::ONE, "Sum of chi_alpha table must be 1");

            // NC resources must align with instances
            debug_assert_eq!(z_witnesses.len(), nc_y_matrices.len(), "#Z != #NC y-matrices (instance misalignment)");
            for (i, y) in nc_y_matrices.iter().enumerate() {
                debug_assert!(y.len() <= d_a, "NC y_matrices[{}] Ajtai rows ({}) exceed 2^ell_d ({})", i, y.len(), d_a);
            }

            // Gamma schedule must be γ^{1..k}
            debug_assert_eq!(nc_row_gamma_pows.len(), z_witnesses.len(),
                "γ^i count does not match #instances (MCS+ME)");
            let mut g_chk = gamma;
            for (i, &got) in nc_row_gamma_pows.iter().enumerate() {
                debug_assert_eq!(got, g_chk, "NC gamma power mismatch at i={} (instance order?)", i);
                g_chk *= gamma;
            }
        }
        
        Self {
            s,
            gamma,
            k_total,
            b,
            ell_d,
            ell_n,
            d_sc,
            round_idx: 0,
            me_offset,
            partials_first_inst,
            w_beta_a_partial,
            w_alpha_a_partial,
            w_beta_r_partial,
            w_eval_r_partial,
            eval_row_partial,
            eval_ajtai_partial: None,
            row_chals: Vec::new(),
            z_witnesses,
            csr_m1,
            csrs,
            nc_y_matrices,
            nc_row_gamma_pows,
            nc_state: None,
            initial_sum_claim,
            f_at_beta_r,
            nc_sum_beta,
        }
    }
    
    /// Transition from row phase to Ajtai phase
    /// Precomputes all necessary Ajtai state exactly once
    fn enter_ajtai_phase(&mut self) {
        // Build χ_{r'} over rows from collected row challenges
        let w_r = HalfTableEq::new(&self.row_chals);
        
        #[cfg(feature = "debug-logs")]
        {
            let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg_oracle {
                eprintln!("[enter_ajtai] M1: {}×{}, instances: {}", 
                    self.csr_m1.rows, self.csr_m1.cols, self.z_witnesses.len());
            }
        }
        
        // v1 = M_1^T · χ_{r'} ∈ K^m
        let v1 = spmv_csr_t_weighted_fk(self.csr_m1, &w_r);
        
        // For each Z_i, y_{i,1}(r') = Z_i · v1 ∈ K^d, then pad to 2^ell_d
        // Note: Instance 1 is the MCS instance, which only contributes to NC, not Eval
        let mut y_partials = Vec::with_capacity(self.z_witnesses.len());
        for (_idx, Zi) in self.z_witnesses.iter().enumerate() {
            let z_ref = MatRef::from_mat(Zi);
            let yi = mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &v1);
            y_partials.push(pad_to_pow2_k(yi, self.ell_d).expect("pad y_i"));
        }
        
        // γ^i weights, i starts at 1
        let mut gamma_pows = Vec::with_capacity(self.z_witnesses.len());
        let mut g = self.gamma;
        for _ in 0..self.z_witnesses.len() {
            gamma_pows.push(g);
            g *= self.gamma;
        }
        
        // Compute F(r') from already-folded row partials
        let mut m_vals_rp = Vec::with_capacity(self.partials_first_inst.s_per_j.len());
        for v in &self.partials_first_inst.s_per_j {
            debug_assert_eq!(v.len(), 1, "row partials must be folded before Ajtai");
            m_vals_rp.push(v[0]);
        }
        let f_at_rprime = self.s.f.eval_in_ext::<K>(&m_vals_rp);
        
        // Build Eval Ajtai partial if not already computed
        if self.eval_ajtai_partial.is_none() {
            let d = 1usize << self.ell_d;
            let mut G_eval = vec![K::ZERO; d];
            
            // Compute all v_j = M_j^T · χ_{r'} ∈ K^m
            let mut vjs: Vec<Vec<K>> = Vec::with_capacity(self.csrs.len());
            for csr in self.csrs.iter() {
                vjs.push(spmv_csr_t_weighted_fk(csr, &w_r));
            }
            
            // Precompute γ^k
            let mut gamma_to_k = K::ONE;
            for _ in 0..self.k_total {
                gamma_to_k *= self.gamma;
            }
            
            // Precompute γ^{i_abs-1} for ME witnesses (i_abs starts at me_offset+1)
            // Allow multiple leading MCS instances: ME witnesses begin at index `me_offset`.
            // In chained IVC, we may have me_offset > 1 (e.g., folding prev+current MCS).
            debug_assert!(
                self.me_offset >= 1 && self.me_offset <= self.z_witnesses.len(),
                "invalid me_offset: expected 1..=#z_witnesses, got {} (z_witnesses={})",
                self.me_offset,
                self.z_witnesses.len()
            );
            let me_count = self.z_witnesses.len().saturating_sub(self.me_offset);
            let mut gamma_pow_i_abs = vec![K::ONE; me_count];
            {
                let mut g = K::ONE;
                for _ in 0..self.me_offset { g *= self.gamma; }
                for i_off in 0..me_count {
                    gamma_pow_i_abs[i_off] = g; // γ^{i_abs-1}
                    g *= self.gamma;
                }
            }
            
            // Accumulate Ajtai vector over ME witnesses only (i≥2)
            for (i_off, Zi) in self.z_witnesses.iter().skip(self.me_offset).enumerate() {
                let z_ref = MatRef::from_mat(Zi);
                for j in 0..self.s.t() {
                    let y_ij = mat_vec_mul_fk::<F,K>(
                        z_ref.data, z_ref.rows, z_ref.cols, &vjs[j]
                    );
                    
                    // weight = γ^{i_abs-1} * (γ^k)^(j+1)  // j is 0-based here
                    let mut w_pow = gamma_pow_i_abs[i_off];
                    for _ in 0..=j { // include baseline γ^k
                        w_pow *= gamma_to_k;
                    }
                    
                    let rho_lim = core::cmp::min(d, y_ij.len());
                    for rho in 0..rho_lim {
                        G_eval[rho] += w_pow * y_ij[rho];
                    }
                }
            }
            
            self.eval_ajtai_partial = Some(G_eval);
        }
        
        #[cfg(feature = "debug-logs")]
        {
            let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg_oracle {
                eprintln!("[oracle][ajtai-pre] f_at_r' = {}", format_ext(f_at_rprime));
            }
        }
        
        self.nc_state = Some(NcState {
            y_partials,
            gamma_pows,
            f_at_rprime: Some(f_at_rprime),
        });
    }
}

impl<'a, F> RoundOracle for GenericCcsOracle<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn num_rounds(&self) -> usize {
        self.ell_d + self.ell_n
    }
    
    fn degree_bound(&self) -> usize {
        self.d_sc
    }
    
    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        #[cfg(feature = "debug-logs")]
        {
            let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg_oracle {
                eprintln!("[oracle][evals_at] round_idx={}, xs_len={}", self.round_idx, xs.len());
            }
        }
        use crate::pi_ccs::oracle::gate::PairGate;
        use crate::pi_ccs::oracle::blocks::{
            RowBlock, AjtaiBlock,
            NcRowBlock, NcAjtaiBlock, FAjtaiBlock,
        };
        
        if self.round_idx < self.ell_n {
            // ===== ROW PHASE =====
            #[cfg(feature = "debug-logs")]
            {
                let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                if dbg_oracle {
                    eprintln!("[oracle][row{}] {} samples", self.round_idx, xs.len());
                    eprintln!("  - k_total: {}, me_offset: {}, #nc_y_matrices: {}", 
                        self.k_total, self.me_offset, self.nc_y_matrices.len());
                    eprintln!("  - F at beta_r: {}", format_ext(self.f_at_beta_r));
                    eprintln!("  - NC sum beta: {}", format_ext(self.nc_sum_beta));
                    eprintln!("  - gamma: {}", format_ext(self.gamma));
                    eprintln!("  - b: {}, ell_d: {}, ell_n: {}", self.b, self.ell_d, self.ell_n);
                    eprintln!("  - xs: {:?}", xs.iter().map(|&x| format_ext(x)).collect::<Vec<_>>());
                }
            }
            
            // Create gates
            let g_beta = PairGate::new(&self.w_beta_r_partial);
            let g_eval = PairGate::new(&self.w_eval_r_partial);
            
            #[cfg(debug_assertions)]
            {
                debug_assert!(self.w_beta_r_partial.len().is_power_of_two());
                debug_assert_eq!(self.partials_first_inst.s_per_j.len(), self.s.t());
                for v in &self.partials_first_inst.s_per_j {
                    debug_assert_eq!(v.len() >> 1, g_beta.half);
                }
                debug_assert_eq!(self.eval_row_partial.len() >> 1, g_eval.half);
            }
            
            // Create NC block if needed
            let nc_block = if !self.nc_y_matrices.is_empty() {
                Some(NcRowBlock {
                    w_beta_a: &self.w_beta_a_partial,
                    ell_d: self.ell_d,
                    b: self.b,
                    y_matrices: &self.nc_y_matrices,
                    gamma_row_pows: &self.nc_row_gamma_pows,
                    _phantom: core::marker::PhantomData,
                })
            } else {
                None
            };

            // Collect/compute round 0 components for debugging (force x in {0,1})
            #[cfg(feature = "debug-logs")]
            let round0_components = if self.round_idx == 0 {
                // Evaluate F, NC, Eval at x=0 and x=1 explicitly regardless of xs
                let mut nc0 = K::ZERO; let mut nc1 = K::ZERO;

                // Helper to eval F and Eval at a chosen x
                let eval_f_ev = |x: K| -> (K, K) {
                    let mut f_contrib = K::ZERO;
                    let mut eval_contrib = K::ZERO;

                    // F
                    let half = g_beta.half;
                    let mut m_vals = vec![K::ZERO; self.partials_first_inst.s_per_j.len()];
                    for k in 0..half {
                        let gate = g_beta.eval(k, x);
                        for (j, partials) in self.partials_first_inst.s_per_j.iter().enumerate() {
                            let a = partials[2*k];
                            let b = partials[2*k+1];
                            m_vals[j] = (K::ONE - x) * a + x * b;
                        }
                        f_contrib += gate * self.s.f.eval_in_ext::<K>(&m_vals);
                    }

                    // Eval
                    let half_e = g_eval.half;
                    for k in 0..half_e {
                        let gate = g_eval.eval(k, x);
                        let a = self.eval_row_partial[2*k];
                        let b = self.eval_row_partial[2*k+1];
                        let g_ev = (K::ONE - x) * a + x * b;
                        eval_contrib += gate * g_ev;
                    }
                    (f_contrib, eval_contrib)
                };

                let (f0, ev0) = eval_f_ev(K::ZERO);
                let (f1, ev1) = eval_f_ev(K::ONE);

                if let Some(ref nc) = nc_block {
                    nc0 = nc.eval_at(K::ZERO, g_beta);
                    nc1 = nc.eval_at(K::ONE,  g_beta);
                }

                Some((f0, f1, nc0, nc1, ev0, ev1))
            } else { None };
            
            // Evaluate each sample point
            let results: Vec<K> = xs.iter().map(|&x| {
                let mut y = K::ZERO;
                
                // F block evaluation
                let mut f_contrib = K::ZERO;
                {
                    let half = g_beta.half;
                    // Reuse m_vals buffer for performance
                    let mut m_vals = vec![K::ZERO; self.partials_first_inst.s_per_j.len()];
                    
                    for k in 0..half {
                        let gate = g_beta.eval(k, x);
                        
                        // Evaluate m_j,k(X) = (1-X)*s_j[2k] + X*s_j[2k+1] for each j
                        for (j, partials) in self.partials_first_inst.s_per_j.iter().enumerate() {
                            let a = partials[2*k];
                            let b = partials[2*k+1];
                            m_vals[j] = (K::ONE - x) * a + x * b;
                        }
                        
                        // Evaluate f(m_1,...,m_t) and accumulate with gate
                        f_contrib += gate * self.s.f.eval_in_ext::<K>(&m_vals);
                    }
                    y += f_contrib;
                }
                
                // NC block evaluation
                let mut _nc_contrib = K::ZERO;
                if let Some(ref nc) = nc_block {
                    _nc_contrib = nc.eval_at(x, g_beta);
                    y += _nc_contrib;
                }
                
                // Eval block evaluation
                let mut eval_contrib = K::ZERO;
                {
                    let half = g_eval.half;
                    for k in 0..half {
                        let gate = g_eval.eval(k, x);
                        let a = self.eval_row_partial[2*k];
                        let b = self.eval_row_partial[2*k+1];
                        let g_ev = (K::ONE - x) * a + x * b;
                        eval_contrib += gate * g_ev;
                    }
                    y += eval_contrib;
                }
                
                // Inline per-x logging suppressed; we log aggregated sums after
                
                #[cfg(feature = "debug-logs")]
                {
                    let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                    if dbg_oracle && (x == K::ZERO || x == K::ONE) {
                        eprintln!("    row_eval_at({:?}) = {}", 
                            if x == K::ZERO { "0" } else { "1" }, 
                            format_ext(y));
                    }
                }
                
                y
            }).collect();
            
            // Check round 0 block-by-block identity
            #[cfg(feature = "debug-logs")]
            if let Some((f0, f1, nc0, nc1, ev0, ev1)) = round0_components {
                eprintln!("[oracle][round0] about to print block sums");
                    let f_sum = f0 + f1;
                    let nc_sum = nc0 + nc1;
                    let ev_sum = ev0 + ev1;
                    let p_sum = f_sum + nc_sum + ev_sum;
                    
                    eprintln!("\n[oracle] Round 0 block sums:");
                    eprintln!("  F(0)+F(1)    = {} (expected f_at_β_r = {})",
                             format_ext(f_sum), format_ext(self.f_at_beta_r));
                    eprintln!("  NC(0)+NC(1)  = {} (expected nc_sum_β = {})",
                             format_ext(nc_sum), format_ext(self.nc_sum_beta));
                    eprintln!("  Eval(0)+Eval(1) = {}", format_ext(ev_sum));
                    eprintln!("  TOTAL p(0)+p(1) = {}\n", format_ext(p_sum));
                    
                    #[cfg(debug_assertions)]
                    {
                        // Hard assertions for invariants at round 0
                        debug_assert_eq!(f_sum,  self.f_at_beta_r,  "F block mismatch at round 0");

                        // If NC mismatch occurs, print per-instance breakdown before asserting
                        if nc_sum != self.nc_sum_beta {
                            if let Some(ref nc_block) = nc_block {
                                let nc0_per_inst = nc_block.eval_at_per_instance(K::ZERO, g_beta);
                                let nc1_per_inst = nc_block.eval_at_per_instance(K::ONE, g_beta);
                                eprintln!("\n  Per-instance NC contributions:");
                                for (i, (nc0, nc1)) in nc0_per_inst.iter().zip(nc1_per_inst.iter()).enumerate() {
                                    let nc_sum_i = *nc0 + *nc1;
                                    eprintln!("    Instance {}: NC(0)+NC(1) = {}", i, format_ext(nc_sum_i));
                                    eprintln!("              γ^{} = {}", i+1, format_ext(nc_block.gamma_row_pows[i]));
                                }
                                eprintln!("\n  NC is using challenges:");
                                eprintln!("    β_a[0] = {}", format_ext(nc_block.w_beta_a[0]));
                                eprintln!("    γ = {}", format_ext(self.gamma));
                                eprintln!("");
                            }
                        }

                        // Defer NC and TOTAL assertions until after reference per-instance check
                    }
                    // Optional: ground-truth per-instance check vs full hypercube (debug only)
                    #[cfg(all(debug_assertions, feature = "debug-logs"))]
                    if let Some(ref nc_block) = nc_block {
                        // Row-round per-instance via NcRowBlock at x=0,1
                        let nc0_i = nc_block.eval_at_per_instance(K::ZERO, g_beta);
                        let nc1_i = nc_block.eval_at_per_instance(K::ONE,  g_beta);
                        let row_per_i: Vec<K> = nc0_i.iter().zip(nc1_i.iter()).map(|(a,b)| *a + *b).collect();

                        // Reference per-instance from full hypercube (no γ)
                        let z_owned: Vec<neo_ccs::Mat<F>> = self.z_witnesses.iter().map(|m| (*m).clone()).collect();
                        let per_i_ref = crate::pi_ccs::nc_constraints::compute_nc_hypercube_sum_per_i(
                            self.s, &z_owned, &self.w_beta_a_partial, &self.w_beta_r_partial,
                            self.b, self.ell_d, self.ell_n,
                        );

                        eprintln!("[round0][NC per-i] comparing row vs ref·γ^i:");
                        let mut sum_ref_weighted = K::ZERO;
                        for (i, (row_i, ref_i)) in row_per_i.iter().zip(per_i_ref.iter()).enumerate() {
                            let g = self.nc_row_gamma_pows[i];
                            let rhs = *ref_i * g;
                            eprintln!("  i={}: row={}  ref*γ^i={}", i+1, format_ext(*row_i), format_ext(rhs));
                            debug_assert_eq!(*row_i, rhs, "NC per-instance mismatch at i={}", i+1);
                            sum_ref_weighted += rhs;
                        }
                        eprintln!("  Σ_i ref*γ^i = {}", format_ext(sum_ref_weighted));
                        eprintln!("  nc_sum_beta  = {}", format_ext(self.nc_sum_beta));
                    }

                    // Finally assert NC and total sums
                    #[cfg(debug_assertions)]
                    {
                        debug_assert_eq!(nc_sum, self.nc_sum_beta,  "NC block mismatch at round 0");
                        debug_assert_eq!(p_sum,  self.initial_sum_claim, "Round-0 invariant violated");
                    }
            }
            
            results
            
        } else {
            // ===== AJTAI PHASE =====
            if self.nc_state.is_none() {
                self.enter_ajtai_phase();
            }
            
            // Compute row-equality scalars for Ajtai phase by contracting
            // χ_{β_r} and χ_{r'} tables: eq_r(r', β_r) = ⟨χ_{β_r}, χ_{r'}⟩.
            // Similarly for Eval gate with r_input when present.
            let (wr_scalar, wr_eval_scalar) = {
                let w_r = HalfTableEq::new(&self.row_chals);
                let rows = 1usize << self.ell_n;

                // wr_scalar: prefer full contraction when sizes match; accept scalar fallback
                let wr_scalar = if self.w_beta_r_partial.is_empty() {
                    K::ONE
                } else if self.w_beta_r_partial.len() == rows {
                    let mut s = K::ZERO;
                    for row in 0..rows { s += self.w_beta_r_partial[row] * w_r.w(row); }
                    s
                } else if self.w_beta_r_partial.len() == 1 {
                    self.w_beta_r_partial[0]
                } else {
                    K::ONE
                };

                // wr_eval_scalar: contraction when sizes match; scalar fallback; else zero (no ME inputs)
                let wr_eval_scalar = if self.w_eval_r_partial.is_empty() {
                    K::ZERO
                } else if self.w_eval_r_partial.len() == rows {
                    let mut s = K::ZERO;
                    for row in 0..rows { s += self.w_eval_r_partial[row] * w_r.w(row); }
                    s
                } else if self.w_eval_r_partial.len() == 1 {
                    self.w_eval_r_partial[0]
                } else {
                    K::ZERO
                };

                (wr_scalar, wr_eval_scalar)
            };

            // Optional diagnostics: show Ajtai row-equality scalars vs expectations
            if std::env::var("NEO_PAPER_CROSSCHECK").ok().as_deref() == Some("1") {
                let fmt = |v: K| { let c = v.as_coeffs(); format!("{} + {}·u", c[0], c[1]) };
                eprintln!(
                    "[ajtai-gates] eq_r(r',β_r)={}  eq_r(r',r_in)={}  (w_beta_r_len={}, w_eval_r_len={}, rows={})",
                    fmt(wr_scalar), fmt(wr_eval_scalar),
                    self.w_beta_r_partial.len(), self.w_eval_r_partial.len(), 1usize << self.ell_n
                );
            }
            
            // Create gates
            let g_beta = PairGate::new(&self.w_beta_a_partial);
            let g_alpha = PairGate::new(&self.w_alpha_a_partial);
            
            #[cfg(debug_assertions)]
            {
                debug_assert_eq!(self.w_beta_a_partial.len() % 2, 0);
                debug_assert_eq!(self.w_alpha_a_partial.len() % 2, 0);
            }
            
            // Create blocks
            let f_block = self.nc_state.as_ref().and_then(|nc| nc.f_at_rprime).map(|f_rp| {
                FAjtaiBlock { f_at_rprime: f_rp }
            });
            
            #[cfg(feature = "debug-logs")]
            {
                let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                if dbg_oracle {
                    let ajtai_round = self.round_idx - self.ell_n;
                    eprintln!("[oracle][ajtai{}] {} samples", ajtai_round, xs.len());
                }
            }
            
            // Evaluate
            xs.iter().map(|&x| {
                let mut y = K::ZERO;
                
                // F contribution
                if let Some(ref f_block) = f_block {
                    y += f_block.eval_at(x, g_beta, wr_scalar);
                }
                
                // NC contribution
                if let Some(ref ncstate) = self.nc_state {
                    let nc_block = NcAjtaiBlock::<F> {
                        y_partials: &ncstate.y_partials,
                        gamma_pows: &ncstate.gamma_pows,
                        b: self.b,
                        _phantom: core::marker::PhantomData,
                    };
                    y += nc_block.eval_at(x, g_beta, wr_scalar);
                }
                
                // Eval contribution - directly compute without creating block
                if let Some(ref eval_vec) = self.eval_ajtai_partial {
                    let half = g_alpha.half;
                    for k in 0..half {
                        let gate = g_alpha.eval(k, x);
                        let a0 = eval_vec[2*k];
                        let a1 = eval_vec[2*k+1];
                        let eval_x = a0 + (a1 - a0) * x;
                        y += gate * wr_eval_scalar * eval_x;
                    }
                }
                
                y
            }).collect()
        }
    }
    
    fn fold(&mut self, r_i: K) {
        use crate::pi_ccs::oracle::gate::fold_partial_in_place;
        
        if self.round_idx < self.ell_n {
            // Row phase folding
            self.row_chals.push(r_i);
            
            fold_partial_in_place(&mut self.w_beta_r_partial, r_i);
            self.w_beta_r_partial.truncate(self.w_beta_r_partial.len() >> 1);
            
            fold_partial_in_place(&mut self.w_eval_r_partial, r_i);
            self.w_eval_r_partial.truncate(self.w_eval_r_partial.len() >> 1);
            
            fold_partial_in_place(&mut self.eval_row_partial, r_i);
            self.eval_row_partial.truncate(self.eval_row_partial.len() >> 1);
            
            for v in &mut self.partials_first_inst.s_per_j {
                fold_partial_in_place(v, r_i);
                v.truncate(v.len() >> 1);
            }
            
            // Fold NC y_matrices
            for y_mat in &mut self.nc_y_matrices {
                for row in y_mat {
                    fold_partial_in_place(row, r_i);
                    row.truncate(row.len() >> 1);
                }
            }

            // Ajtai eq tables must remain untouched during row phase
            #[cfg(debug_assertions)]
            {
                let d_a = 1usize << self.ell_d;
                debug_assert_eq!(self.w_beta_a_partial.len(),  d_a, "β_a table mutated in row phase");
                debug_assert_eq!(self.w_alpha_a_partial.len(), d_a, "α table mutated in row phase");
            }
        } else {
            // Ajtai phase folding
            fold_partial_in_place(&mut self.w_beta_a_partial, r_i);
            self.w_beta_a_partial.truncate(self.w_beta_a_partial.len() >> 1);
            
            fold_partial_in_place(&mut self.w_alpha_a_partial, r_i);
            self.w_alpha_a_partial.truncate(self.w_alpha_a_partial.len() >> 1);
            
            if let Some(ref mut nc) = self.nc_state {
                for y in &mut nc.y_partials {
                    fold_partial_in_place(y, r_i);
                    y.truncate(y.len() >> 1);
                }
            }
            
            if let Some(ref mut v) = self.eval_ajtai_partial {
                fold_partial_in_place(v, r_i);
                v.truncate(v.len() >> 1);
            }
        }
        
        self.round_idx += 1;
    }
}
