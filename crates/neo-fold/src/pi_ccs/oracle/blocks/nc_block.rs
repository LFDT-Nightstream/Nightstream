//! NC block: Norm/decomposition constraints
//!
//! This module implements the NC terms in the Q polynomial:
//! NC_i(X) = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::pi_ccs::oracle::gate::{PairGate, fold_partial_in_place, pair_to_full_indices};
use crate::pi_ccs::oracle::blocks::{UnivariateBlock, RowBlock, AjtaiBlock};

/// NC block for row phase with exact Ajtai sum computation
pub struct NcRowBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// eq_a(X_a, β_a) table length 2^ell_d
    pub w_beta_a: &'a [K],
    pub ell_d: usize,
    pub b: u32,
    pub round_idx: usize,
    /// y_matrices[i][ρ][row] - unfolded for dynamic stride computation
    pub y_matrices: &'a [Vec<Vec<K>>],
    /// γ^{i+1} weights where i is 0-based index
    pub gamma_row_pows: &'a [K],
    pub _phantom: core::marker::PhantomData<F>,
}

impl<'a, F> UnivariateBlock for NcRowBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn fold(&mut self, _r: K) {
        // No-op: uses unfolded rows for dynamic stride computation
    }
}

impl<'a, F> RowBlock for NcRowBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn eval_at(&self, x: K, w_beta_r: PairGate) -> K {
        if self.y_matrices.is_empty() {
            return K::ZERO;
        }

        let half = w_beta_r.half;
        let d = 1usize << self.ell_d;
        let low = -((self.b as i64) - 1);
        let high = (self.b as i64) - 1;
        let mut total = K::ZERO;

        for k in 0..half {
            let (j0, j1) = pair_to_full_indices(k, self.round_idx);
            let (w0, w1) = w_beta_r.pair(k);

            let mut sum_over_i = K::ZERO;
            for (i_inst, y_mat) in self.y_matrices.iter().enumerate() {
                let rows_len = y_mat.len();
                let mut ajtai_sum0 = K::ZERO;
                let mut ajtai_sum1 = K::ZERO;

                // Exact Ajtai sum: Σ_xa w_beta_a[xa] · ∏_{t=-(b-1)}^{b-1} (z(xa) - t)
                for xa in 0..d {
                    let mut zi0 = K::ZERO;
                    let mut zi1 = K::ZERO;

                    // χ(xa, ρ) · y[ρ, j?]
                    for rho in 0..rows_len {
                        // Bit-product equality (χ)
                        let mut chi = K::ONE;
                        for bit in 0..self.ell_d {
                            let xb = if (xa >> bit) & 1 == 1 { K::ONE } else { K::ZERO };
                            let rb = if (rho >> bit) & 1 == 1 { K::ONE } else { K::ZERO };
                            chi *= xb * rb + (K::ONE - xb) * (K::ONE - rb);
                        }
                        zi0 += chi * y_mat[rho][j0];
                        zi1 += chi * y_mat[rho][j1];
                    }

                    let mut n0 = K::ONE;
                    let mut n1 = K::ONE;
                    for t in low..=high {
                        let tk = K::from(F::from_i64(t));
                        n0 *= zi0 - tk;
                        n1 *= zi1 - tk;
                    }

                    let wxa = self.w_beta_a[xa];
                    ajtai_sum0 += wxa * n0;
                    ajtai_sum1 += wxa * n1;
                }

                let gate = (K::ONE - x) * w0 * ajtai_sum0 + x * w1 * ajtai_sum1;
                sum_over_i += self.gamma_row_pows[i_inst] * gate;
            }
            total += sum_over_i;
        }

        total
    }
}

/// NC block for Ajtai phase
pub struct NcAjtaiBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// y_partials[i] length 2^ell_d, folded across Ajtai rounds
    pub y_partials: &'a mut Vec<Vec<K>>,
    /// γ^i, i starts at 1
    pub gamma_pows: &'a [K],
    pub b: u32,
    pub _phantom: core::marker::PhantomData<F>,
}

impl<'a, F> UnivariateBlock for NcAjtaiBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn fold(&mut self, r: K) {
        for y in self.y_partials.iter_mut() {
            fold_partial_in_place(y, r);
            let len2 = y.len() >> 1;
            y.truncate(len2);
        }
    }
}

impl<'a, F> AjtaiBlock for NcAjtaiBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    fn eval_at(&self, x: K, w_beta_a: PairGate, wr_scalar: K) -> K {
        let half = w_beta_a.half;
        let low = -((self.b as i64) - 1);
        let high = (self.b as i64) - 1;
        let mut acc = K::ZERO;

        for k in 0..half {
            let (w0, w1) = w_beta_a.pair(k);
            let mut nc0 = K::ZERO;
            let mut nc1 = K::ZERO;

            for (i, y) in self.y_partials.iter().enumerate() {
                let y0 = y[2*k];
                let y1 = y[2*k+1];
                let mut n0 = K::ONE;
                let mut n1 = K::ONE;

                for t in low..=high {
                    let tk = K::from(F::from_i64(t));
                    n0 *= y0 - tk;
                    n1 *= y1 - tk;
                }

                nc0 += self.gamma_pows[i] * n0;
                nc1 += self.gamma_pows[i] * n1;
            }

            acc += wr_scalar * ((K::ONE - x) * w0 * nc0 + x * w1 * nc1);
        }

        acc
    }
}