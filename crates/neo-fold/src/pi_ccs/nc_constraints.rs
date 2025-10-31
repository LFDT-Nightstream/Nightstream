/// Normalization constraint (NC) computation for CCS reduction
///
/// This module implements the NC_i terms in the Q polynomial:
///   NC_i(X) = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)
///
/// These constraints enforce:
/// 1. Decomposition correctness: Z = Decomp_b(z)
/// 2. Digit range bounds: ||Z||_∞ < b

use neo_ccs::{Mat, MatRef};
use p3_field::{Field, PrimeCharacteristicRing};
use neo_math::K;

use crate::pi_ccs::{CcsStructure, McsWitness};

/// Compute the full hypercube sum of NC terms: Σ_{X∈{0,1}^{log(dn)}} eq(X,β) · Σ_i γ^i · NC_i(X)
///
/// **Paper Reference**: Section 4.4, NC_i term contribution to Q polynomial sum
///
/// Since NC is NON-MULTILINEAR (degree 2b-1), we CANNOT use the identity Σ_X eq(X,β)·NC(X) = NC(β).
/// Instead, we must compute the ACTUAL sum over the hypercube that the oracle will verify.
///
/// # Arguments
/// * `s` - CCS structure containing matrices M_j
/// * `witnesses` - MCS witness Z matrices
/// * `me_witnesses` - Additional ME witness Z matrices
/// * `beta_a` - Challenge vector for Ajtai dimension (length ell_d)
/// * `beta_r` - Challenge vector for row dimension (length ell_n)
/// * `gamma` - Challenge scalar for instance weighting
/// * `params` - NeoParams containing base b
/// * `ell_d` - Log of Ajtai dimension
/// * `ell_n` - Log of row dimension
///
/// # Returns
/// The weighted sum: Σ_i γ^i · (Σ_X eq(X,β) · NC_i(X))
pub fn compute_nc_hypercube_sum<F>(
    s: &CcsStructure<F>,
    witnesses: &[McsWitness<F>],
    me_witnesses: &[Mat<F>],
    beta_a: &[K],
    beta_r: &[K],
    gamma: K,
    params: &neo_params::NeoParams,
    ell_d: usize,
    ell_n: usize,
) -> K
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[compute_nc_hypercube_sum] Called with:");
        eprintln!("  witnesses.len() = {}", witnesses.len());
        eprintln!("  me_witnesses.len() = {}", me_witnesses.len());
        eprintln!("  gamma = {}", crate::pi_ccs::format_ext(gamma));
        eprintln!("  beta_a = {:?}", beta_a.iter().map(|b| crate::pi_ccs::format_ext(*b)).collect::<Vec<_>>());
        eprintln!("  beta_r = {:?}", beta_r.iter().map(|b| crate::pi_ccs::format_ext(*b)).collect::<Vec<_>>());
        eprintln!("  ell_d = {}, ell_n = {}", ell_d, ell_n);
    }
    // Build equality tables and compute per-instance contributions explicitly, then sum with γ^i
    let w_beta_a_table = neo_ccs::utils::tensor_point::<K>(beta_a);
    let w_beta_r_table = neo_ccs::utils::tensor_point::<K>(beta_r);

    let mut per_i = vec![K::ZERO; witnesses.len() + me_witnesses.len()];
    let d_rows = 1usize << ell_d;
    let n_rows = 1usize << ell_n;

    for xa in 0..d_rows {
        for xr in 0..n_rows {
            let eq_x_beta = w_beta_a_table[xa] * w_beta_r_table[xr];

            // v1_x = M_1^T · χ_{X_r}
            let mut v1_x = vec![K::ZERO; s.m];
            for row in 0..s.n {
                let mut chi_x_r_row = K::ONE;
                for bit_pos in 0..ell_n {
                    let xrb = if (xr >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    let rb = if (row     >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    chi_x_r_row *= xrb*rb + (K::ONE - xrb)*(K::ONE - rb);
                }
                for col in 0..s.m {
                    v1_x[col] += K::from(s.matrices[0][(row, col)]) * chi_x_r_row;
                }
            }

            for (i, Zi) in witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()).enumerate() {
                let z_ref = MatRef::from_mat(Zi);
                let y_i1_x = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &v1_x);
                let mut y_mle_x = K::ZERO;
                for (rho, &y_rho) in y_i1_x.iter().enumerate() {
                    let mut chi_xa_rho = K::ONE;
                    for bit_pos in 0..ell_d {
                        let xab = if (xa   >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                        let rb   = if (rho >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                        chi_xa_rho *= xab*rb + (K::ONE - xab)*(K::ONE - rb);
                    }
                    y_mle_x += chi_xa_rho * y_rho;
                }
                let Ni_x = crate::pi_ccs::nc_core::range_product::<F>(y_mle_x, params.b);
                per_i[i] += eq_x_beta * Ni_x;
            }
        }
    }

    let mut nc_sum_hypercube = K::ZERO;
    let mut gamma_pow_i = gamma;
    for v in per_i {
        nc_sum_hypercube += gamma_pow_i * v;
        gamma_pow_i *= gamma;
    }
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("[compute_nc_hypercube_sum] Result = {}", crate::pi_ccs::format_ext(nc_sum_hypercube));
    }
    
    nc_sum_hypercube
}

// Debug-only ground-truth: per-instance NC hypercube sums (without γ weighting)
// Mirrors compute_nc_hypercube_sum but returns a Vec with one entry per instance.
#[cfg(debug_assertions)]
pub fn compute_nc_hypercube_sum_per_i<F>(
    s: &CcsStructure<F>,
    z_all: &[Mat<F>], // Instances ordered: MCS first, then ME
    w_beta_a_table: &[K],
    w_beta_r_table: &[K],
    b: u32,
    ell_d: usize,
    ell_n: usize,
) -> Vec<K>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    let mut per_i = vec![K::ZERO; z_all.len()];
    let d_rows = 1usize << ell_d;
    let n_rows = 1usize << ell_n;

    for xa in 0..d_rows {
        for xr in 0..n_rows {
            let eq_x_beta = w_beta_a_table[xa] * w_beta_r_table[xr];

            // v1_x = M_1^T · χ_{X_r}
            let mut v1_x = vec![K::ZERO; s.m];
            for row in 0..s.n {
                let mut chi_x_r_row = K::ONE;
                for bit_pos in 0..ell_n {
                    let xrb = if (xr >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    let rb = if (row     >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    chi_x_r_row *= xrb*rb + (K::ONE - xrb)*(K::ONE - rb);
                }
                for col in 0..s.m {
                    v1_x[col] += K::from(s.matrices[0][(row, col)]) * chi_x_r_row;
                }
            }

            for (i, Zi) in z_all.iter().enumerate() {
                let z_ref = MatRef::from_mat(Zi);
                let y_i1_x = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &v1_x);
                let mut y_mle_x = K::ZERO;
                for (rho, &y_rho) in y_i1_x.iter().enumerate() {
                    let mut chi_xa_rho = K::ONE;
                    for bit_pos in 0..ell_d {
                        let xab = if (xa   >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                        let rb   = if (rho >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                        chi_xa_rho *= xab*rb + (K::ONE - xab)*(K::ONE - rb);
                    }
                    y_mle_x += chi_xa_rho * y_rho;
                }

                let Ni_x = crate::pi_ccs::nc_core::range_product::<F>(y_mle_x, b);
                per_i[i] += eq_x_beta * Ni_x;
            }
        }
    }

    per_i
}
