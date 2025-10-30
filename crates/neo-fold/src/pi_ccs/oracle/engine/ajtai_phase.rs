//! Ajtai phase implementation for sum-check oracle
//!
//! Handles rounds ell_n..ell_n+ell_d-1, processing X_a bits

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::pi_ccs::oracle::gate::{PairGate, fold_partial_in_place};
use crate::pi_ccs::oracle::blocks::{
    UnivariateBlock, AjtaiBlock,
    FAjtaiBlock, NcAjtaiBlock, EvalAjtaiBlock,
};

/// Ajtai phase state and operations
pub struct AjtaiPhase<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Equality gate weights (folded during Ajtai rounds)
    pub w_beta_a: &'a mut Vec<K>,
    pub w_alpha_a: &'a mut Vec<K>,
    
    /// Row equality scalars after all row folds
    pub wr_scalar: K,         // eq_r(r', β_r)
    pub wr_eval_scalar: K,    // eq_r(r', r_input)
    
    /// F block for Ajtai phase
    pub f_block: Option<FAjtaiBlock>,
    
    /// Optional NC block (if witnesses provided)
    pub nc_block: Option<NcAjtaiBlock<'a, F>>,
    
    /// Optional Eval block (if ME witnesses provided)
    pub eval_block: Option<EvalAjtaiBlock<'a>>,
}

impl<'a, F> AjtaiPhase<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Evaluate Q at sample points for current Ajtai round
    pub fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        let g_beta = PairGate::new(self.w_beta_a);
        let g_alpha = PairGate::new(self.w_alpha_a);
        
        xs.iter().map(|&x| {
            let mut y = K::ZERO;
            
            // F block contribution: eq_a(·,βa)*eq_r(r',βr)*F(r')
            if let Some(ref f_block) = self.f_block {
                y += f_block.eval_at(x, g_beta, self.wr_scalar);
            }
            
            // NC block contribution
            if let Some(ref nc) = self.nc_block {
                y += nc.eval_at(x, g_beta, self.wr_scalar);
            }
            
            // Eval block contribution
            if let Some(ref ev) = self.eval_block {
                y += ev.eval_at(x, g_alpha, self.wr_eval_scalar);
            }
            
            y
        }).collect()
    }
    
    /// Fold oracle state with challenge r
    pub fn fold(&mut self, r: K) {
        // Fold equality gates
        fold_partial_in_place(self.w_beta_a, r);
        self.w_beta_a.truncate(self.w_beta_a.len() >> 1);
        
        fold_partial_in_place(self.w_alpha_a, r);
        self.w_alpha_a.truncate(self.w_alpha_a.len() >> 1);
        
        // Fold blocks
        if let Some(ref mut nc) = self.nc_block {
            nc.fold(r);
        }
        if let Some(ref mut ev) = self.eval_block {
            ev.fold(r);
        }
        // F block doesn't need folding (constant F(r'))
    }
}