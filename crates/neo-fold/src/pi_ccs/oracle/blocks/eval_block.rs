//! Eval block: Evaluation ties
//!
//! This module implements the Eval terms in the Q polynomial:
//! Eval_{(i,j)}(X) = eq(X,(α,r))·M̃_{i,j}(X)

use neo_math::K;
use p3_field::PrimeCharacteristicRing;
use crate::pi_ccs::oracle::gate::{PairGate, fold_partial_in_place};
use crate::pi_ccs::oracle::blocks::{UnivariateBlock, RowBlock, AjtaiBlock};

/// Eval block for row phase
pub struct EvalRowBlock<'a> {
    /// Row-domain Eval partial (pre-collapsed over Ajtai at α)
    pub eval_row_partial: &'a mut Vec<K>,
}

impl<'a> UnivariateBlock for EvalRowBlock<'a> {
    fn fold(&mut self, r: K) {
        fold_partial_in_place(self.eval_row_partial, r);
        self.eval_row_partial.truncate(self.eval_row_partial.len() >> 1);
    }
}

impl<'a> RowBlock for EvalRowBlock<'a> {
    fn eval_at(&self, x: K, w_eval_r: PairGate) -> K {
        let half = w_eval_r.half;
        debug_assert_eq!(self.eval_row_partial.len() >> 1, half);

        let mut acc = K::ZERO;
        for k in 0..half {
            let gate = w_eval_r.eval(k, x);
            let a = self.eval_row_partial[2*k];
            let b = self.eval_row_partial[2*k+1];
            let g_ev = (K::ONE - x) * a + x * b;
            acc += gate * g_ev;
        }

        acc
    }
}

/// Eval block for Ajtai phase
pub struct EvalAjtaiBlock<'a> {
    /// Ajtai-domain Eval partial (ME witnesses aggregated)
    pub eval_ajtai_partial: &'a mut Vec<K>,
}

impl<'a> UnivariateBlock for EvalAjtaiBlock<'a> {
    fn fold(&mut self, r: K) {
        fold_partial_in_place(self.eval_ajtai_partial, r);
        self.eval_ajtai_partial.truncate(self.eval_ajtai_partial.len() >> 1);
    }
}

impl<'a> AjtaiBlock for EvalAjtaiBlock<'a> {
    fn eval_at(&self, x: K, w_alpha_a: PairGate, wr_eval_scalar: K) -> K {
        let half = w_alpha_a.half;
        let mut acc = K::ZERO;

        for k in 0..half {
            let gate = w_alpha_a.eval(k, x);
            let a0 = self.eval_ajtai_partial[2*k];
            let a1 = self.eval_ajtai_partial[2*k+1];
            let eval_x = a0 + (a1 - a0) * x;
            acc += gate * wr_eval_scalar * eval_x;
        }

        acc
    }
}