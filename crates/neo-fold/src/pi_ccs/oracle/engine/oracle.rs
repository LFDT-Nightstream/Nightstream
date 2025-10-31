//! Main oracle delegator implementing RoundOracle trait
//!
//! This orchestrates the two-phase sum-check oracle: row phase 
//! (rounds 0..ell_n-1) and Ajtai phase (rounds ell_n..ell_n+ell_d-1).

use neo_ccs::{CcsStructure, MatRef, utils::mat_vec_mul_fk};
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::pi_ccs::precompute::{MlePartials, pad_to_pow2_k};
use crate::pi_ccs::sparse_matrix::Csr;
use crate::pi_ccs::eq_weights::{HalfTableEq, spmv_csr_t_weighted_fk};
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
            debug_assert!(self.me_offset == 1, "code assumes ME witnesses start at instance 2, but me_offset={}", self.me_offset);
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
                    
                    // weight = γ^{i_abs-1} * (γ^k)^j
                    let mut w_pow = gamma_pow_i_abs[i_off];
                    for _ in 0..j {
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
            
            // Evaluate each sample point
            xs.iter().map(|&x| {
                let mut y = K::ZERO;
                
                // F block evaluation
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
                        y += gate * self.s.f.eval_in_ext::<K>(&m_vals);
                    }
                }
                
                // NC block evaluation
                if let Some(ref nc) = nc_block {
                    y += nc.eval_at(x, g_beta);
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
                
                #[cfg(feature = "debug-logs")]
                if self.round_idx == 0 && (x == K::ZERO || x == K::ONE) {
                    let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                    if dbg_oracle {
                        eprintln!("    [oracle] Round 0, x={}: Eval contrib = {}, g_eval.half = {}", 
                                 if x == K::ZERO { "0" } else { "1" }, 
                                 format_ext(eval_contrib),
                                 g_eval.half);
                    }
                }
                
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
            }).collect()
            
        } else {
            // ===== AJTAI PHASE =====
            if self.nc_state.is_none() {
                self.enter_ajtai_phase();
            }
            
            let wr_scalar = if !self.w_beta_r_partial.is_empty() { 
                self.w_beta_r_partial[0] 
            } else { 
                K::ONE 
            };
            let wr_eval_scalar = if !self.w_eval_r_partial.is_empty() { 
                self.w_eval_r_partial[0] 
            } else { 
                K::ONE 
            };
            
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