//! NC block: Norm/decomposition constraints
//!
//! This module implements the NC terms in the Q polynomial:
//! NC_i(X) = ∏_{t=-(b-1)}^{b-1} (Ẑ_i(X) - t)

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use crate::pi_ccs::oracle::gate::PairGate;
use crate::pi_ccs::oracle::blocks::{UnivariateBlock, RowBlock, AjtaiBlock};
use crate::pi_ccs::nc_core;

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
    /// y_matrices[i][ρ][row] - folded each round
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
        // No-op: folding is handled by the oracle engine on y_matrices
    }
}

impl<'a, F> RowBlock for NcRowBlock<'a, F>
where
    F: Field + PrimeCharacteristicRing + Send + Sync + Copy,
    K: From<F>,
{
    fn eval_at(&self, x: K, w_beta_r: PairGate) -> K {
        if self.y_matrices.is_empty() { return K::ZERO; }

        // We must aggregate across *all* rows with eq_r before applying NC,
        // because NC is non-linear. See tests: round 0 block sums must match
        // the hypercube sum computed in precompute.
        //
        // For each instance i and for each Ajtai index xa, build:
        //   y_i(xa; x) = Σ_k wr_k(x) * ((1-x)*<y[:,2k],χ_xa> + x*<y[:,2k+1],χ_xa>)
        // then sum w_beta_a[xa] * NC(y_i(xa; x)) and finally weight by γ^i.

        let d = 1usize << self.ell_d;
        let half = w_beta_r.half;
        let mut total = K::ZERO;

        for (i_inst, y_mat) in self.y_matrices.iter().enumerate() {
            let rows_len = y_mat.len(); // = D (unpadded Ajtai rows)

            // y_agg[xa] will hold y_i(xa; x) after aggregating all row pairs with eq_r gate
            let mut y_agg = vec![K::ZERO; d];

            // Accumulate row contribution into y_agg **before** NC
            for k in 0..half {
                let j0 = 2 * k;
                let j1 = j0 + 1;
                let wrx = w_beta_r.eval(k, x); // affine in x

                // For each Ajtai index xa compute <y[:,j?], χ_xa>
                for xa in 0..d {
                    let mut ya0 = K::ZERO;
                    let mut ya1 = K::ZERO;

                    // χ(xa, ρ) against each Ajtai row ρ
                    for rho in 0..rows_len {
                        let mut chi = K::ONE;
                        for bit in 0..self.ell_d {
                            let xb = if (xa >> bit) & 1 == 1 { K::ONE } else { K::ZERO };
                            let rb = if (rho >> bit) & 1 == 1 { K::ONE } else { K::ZERO };
                            chi *= xb * rb + (K::ONE - xb) * (K::ONE - rb);
                        }
                        ya0 += chi * y_mat[rho][j0];
                        ya1 += chi * y_mat[rho][j1];
                    }

                    // Interpolate the VALUE first (row-bit variable x), then add with wrx
                    let y_pair_x = (K::ONE - x) * ya0 + x * ya1;
                    y_agg[xa] += wrx * y_pair_x;
                }
            }

            // Now apply NC to the *aggregated* y and mix with eq_a weights
            let mut ajtai_sum = K::ZERO;
            for xa in 0..d {
                let ni = nc_core::range_product::<F>(y_agg[xa], self.b);
                
                #[cfg(feature = "debug-logs")]
                {
                    let dbg = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
                    if dbg && i_inst == 0 && xa < 4 {
                        eprintln!("[NC block] xa={}: y_agg={}, NC={}, w_beta_a={}", 
                                xa, 
                                crate::pi_ccs::format_ext(y_agg[xa]),
                                crate::pi_ccs::format_ext(ni),
                                crate::pi_ccs::format_ext(self.w_beta_a[xa]));
                    }
                }
                
                ajtai_sum += self.w_beta_a[xa] * ni;
            }

            total += self.gamma_row_pows[i_inst] * ajtai_sum;
        }

        total
    }
}

// Debug-only implementation to get per-instance NC contributions
#[cfg(feature = "debug-logs")]
impl<'a, F> NcRowBlock<'a, F> 
where
    F: Field + PrimeCharacteristicRing + Send + Sync + Copy,
    K: From<F>,
{
    pub fn eval_at_per_instance(&self, x: K, w_beta_r: PairGate) -> Vec<K> {
        if self.y_matrices.is_empty() { return vec![]; }

        let d = 1usize << self.ell_d;
        let half = w_beta_r.half;
        let mut out = vec![K::ZERO; self.y_matrices.len()];

        for (i_inst, y_mat) in self.y_matrices.iter().enumerate() {
            let rows_len = y_mat.len();
            let mut y_agg = vec![K::ZERO; d];

            for k in 0..half {
                let j0 = 2 * k;
                let j1 = j0 + 1;
                let wrx = w_beta_r.eval(k, x);

                for xa in 0..d {
                    let mut ya0 = K::ZERO;
                    let mut ya1 = K::ZERO;
                    for rho in 0..rows_len {
                        let mut chi = K::ONE;
                        for bit in 0..self.ell_d {
                            let xb = if (xa >> bit) & 1 == 1 { K::ONE } else { K::ZERO };
                            let rb = if (rho >> bit) & 1 == 1 { K::ONE } else { K::ZERO };
                            chi *= xb * rb + (K::ONE - xb) * (K::ONE - rb);
                        }
                        ya0 += chi * y_mat[rho][j0];
                        ya1 += chi * y_mat[rho][j1];
                    }
                    let y_pair_x = (K::ONE - x) * ya0 + x * ya1;
                    y_agg[xa] += wrx * y_pair_x;
                }
            }

            let mut ajtai_sum = K::ZERO;
            for xa in 0..d {
                let ni = nc_core::range_product::<F>(y_agg[xa], self.b);
                ajtai_sum += self.w_beta_a[xa] * ni;
            }

            out[i_inst] = self.gamma_row_pows[i_inst] * ajtai_sum;
        }

        out
    }
}

/// NC block for Ajtai phase
pub struct NcAjtaiBlock<'a, F>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    /// y_partials[i] length 2^ell_d, folded across Ajtai rounds
    pub y_partials: &'a [Vec<K>],
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
    fn fold(&mut self, _r: K) {
        // Folding is handled externally since y_partials is immutable
        // This is a no-op for the immutable block
    }
}

impl<'a, F> AjtaiBlock for NcAjtaiBlock<'a, F>
where
    F: Field + PrimeCharacteristicRing + Send + Sync + Copy,
    K: From<F>,
{
    fn eval_at(&self, x: K, w_beta_a: PairGate, wr_scalar: K) -> K {
        let half = w_beta_a.half;
        let mut acc = K::ZERO;

        for k in 0..half {
            // eq_a gate for this pair and this x
            let gate = w_beta_a.eval(k, x);

            // Sum over instances: γ^i · ∏_t ( z(x) - t ), where z(x) = (1-x)·y0 + x·y1
            let mut sum_i = K::ZERO;
            for (i, y) in self.y_partials.iter().enumerate() {
                let y0 = y[2 * k];
                let y1 = y[2 * k + 1];
                // Use core function for interpolation and NC evaluation
                let Ni = nc_core::nc_interpolated::<F>(y0, y1, x, self.b);
                sum_i += self.gamma_pows[i] * Ni;
            }

            acc += wr_scalar * gate * sum_i;
        }

        acc
    }
}