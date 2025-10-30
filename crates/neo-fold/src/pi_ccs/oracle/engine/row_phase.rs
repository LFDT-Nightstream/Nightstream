//! Row phase implementation for sum-check oracle
//!
//! Handles rounds 0..ell_n-1, processing X_r bits

use neo_ccs::CcsStructure;
use neo_math::K;
use p3_field::Field;
use crate::pi_ccs::oracle::gate::{PairGate, fold_partial_in_place};
use crate::pi_ccs::oracle::blocks::{
    UnivariateBlock, RowBlock,
    FRowBlock, NcRowBlock, EvalRowBlock,
};

/// Row phase state and operations
pub struct RowPhase<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// CCS structure reference
    pub s: &'a CcsStructure<F>,
    
    /// F block state
    pub f_block: FRowBlock<'a, F>,
    
    /// Optional NC block (if witnesses provided)
    pub nc_block: Option<NcRowBlock<'a, F>>,
    
    /// Eval block state
    pub eval_block: EvalRowBlock<'a>,
    
    /// Equality gate weights (folded during rounds)
    pub w_beta_r: &'a mut Vec<K>,
    pub w_eval_r: &'a mut Vec<K>,
    
    /// Current round index
    pub round_idx: usize,
    
    /// Total row rounds
    pub ell_n: usize,
    
    /// Row challenges collected during row rounds
    pub row_chals: &'a mut Vec<K>,
}

impl<'a, F> RowPhase<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// Evaluate Q at sample points for current row round
    pub fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        let g_beta = PairGate::new(self.w_beta_r);
        let g_eval = PairGate::new(self.w_eval_r);
        
        // Evaluate each block and sum contributions
        xs.iter().map(|&x| {
            let mut y = self.f_block.eval_at(x, g_beta);
            
            if let Some(ref mut nc) = self.nc_block {
                y += nc.eval_at(x, g_beta);
            }
            
            y += self.eval_block.eval_at(x, g_eval);
            y
        }).collect()
    }
    
    /// Fold oracle state with challenge r
    pub fn fold(&mut self, r: K) {
        // Collect row challenge
        self.row_chals.push(r);
        
        // Fold equality gates
        fold_partial_in_place(self.w_beta_r, r);
        self.w_beta_r.truncate(self.w_beta_r.len() >> 1);
        
        fold_partial_in_place(self.w_eval_r, r);
        self.w_eval_r.truncate(self.w_eval_r.len() >> 1);
        
        // Fold blocks
        self.f_block.fold(r);
        self.eval_block.fold(r);
        // NC row block uses unfolded rows; no fold needed
        
        // Update NC block round index if present
        if let Some(ref mut nc) = self.nc_block {
            nc.round_idx += 1;
        }
        
        self.round_idx += 1;
    }
    
    /// Check if row phase is complete
    pub fn done(&self) -> bool {
        self.round_idx == self.ell_n
    }
}