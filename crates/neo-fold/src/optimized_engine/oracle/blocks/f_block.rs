//! F block: CCS constraint polynomial evaluation
//!
//! This module implements the F term in the Q polynomial:
//! F(X_{[log d + 1, log dn]}) = f(M̃_1·z_1,...,M̃_t·z_1)

use neo_ccs::CcsStructure;
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::optimized_engine::oracle::gate::{PairGate, fold_partial_in_place};
use crate::optimized_engine::oracle::blocks::{UnivariateBlock, RowBlock, AjtaiBlock};

/// F block for row phase evaluation
pub struct FRowBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    /// Per-j row partials for instance 1 (folded during row rounds)
    pub s_per_j: &'a mut Vec<Vec<K>>,
}

impl<'a, F> UnivariateBlock for FRowBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn fold(&mut self, r: K) {
        // Fold each per-j partial vector
        for v in self.s_per_j.iter_mut() {
            fold_partial_in_place(v, r);
            v.truncate(v.len() >> 1);
        }
    }
}

impl<'a, F> RowBlock for FRowBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn eval_at(&self, x: K, w_beta_r: PairGate) -> K {
        let half = w_beta_r.half;
        let mut acc = K::ZERO;
        
        // Reuse m_vals buffer for performance
        let mut m_vals = vec![K::ZERO; self.s_per_j.len()];
        
        for k in 0..half {
            let gate = w_beta_r.eval(k, x);
            
            // Evaluate m_j,k(X) = (1-X)*s_j[2k] + X*s_j[2k+1] for each j
            for (j, partials) in self.s_per_j.iter().enumerate() {
                let a = partials[2*k];
                let b = partials[2*k+1];
                m_vals[j] = (K::ONE - x) * a + x * b;
            }
            
            // Evaluate f(m_1,...,m_t) and accumulate with gate
            acc += gate * self.s.f.eval_in_ext::<K>(&m_vals);
        }
        
        acc
    }
}

/// F block for Ajtai phase (just scaling by constants)
pub struct FAjtaiBlock {
    /// F(r') precomputed after row rounds
    pub f_at_rprime: K,
}

impl UnivariateBlock for FAjtaiBlock {
    fn fold(&mut self, _r: K) {
        // No-op: F(r') is constant in Ajtai phase
    }
}

impl AjtaiBlock for FAjtaiBlock {
    fn eval_at(&self, x: K, w_beta_a: PairGate, wr_scalar: K) -> K {
        let mut acc = K::ZERO;
        for k in 0..w_beta_a.half {
            let gate = w_beta_a.eval(k, x);
            acc += gate * wr_scalar * self.f_at_rprime;
        }
        acc
    }
}