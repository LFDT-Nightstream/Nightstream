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
    let chi_beta_full: Vec<K> = {
        let mut beta_full = beta_a.to_vec();
        beta_full.extend_from_slice(beta_r);
        neo_ccs::utils::tensor_point::<K>(&beta_full)
    };
    
    let mut nc_sum_hypercube = K::ZERO;
    let dn = (1usize << ell_d) * (1usize << ell_n);
    
    for x_idx in 0..dn {
        let eq_x_beta = chi_beta_full[x_idx];
        
        // For each witness, compute NC_i at this hypercube point X
        let mut gamma_pow_i = gamma;
        for (_i_idx, Zi) in witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()).enumerate() {
            #[cfg(feature = "debug-logs")]
            if x_idx == 0 {
                eprintln!("[compute_nc_hypercube_sum] x_idx=0: Instance {}, gamma_pow = {}", 
                         _i_idx, crate::pi_ccs::format_ext(gamma_pow_i));
            }
            // Compute y_{i,1}(X) = Z̃_i(X) where X = (X_a, X_r)
            // First, compute M_1^T · χ_X_r
            let x_r_idx = x_idx % (1 << ell_n);
            let x_a_idx = x_idx >> ell_n;
            
            // Build χ_X_r and compute M_1^T · χ_X_r
            let mut v1_x = vec![K::ZERO; s.m];
            for row in 0..s.n {
                let mut chi_x_r_row = K::ONE;
                for bit_pos in 0..ell_n {
                    let x_r_bit = if (x_r_idx >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    let row_bit = if (row >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    chi_x_r_row *= x_r_bit * row_bit + (K::ONE - x_r_bit) * (K::ONE - row_bit);
                }
                for col in 0..s.m {
                    v1_x[col] += K::from(s.matrices[0][(row, col)]) * chi_x_r_row;
                }
            }
            
            // Compute y_{i,1}(X_r) = Z_i · v1_x (vector of length d)
            let z_ref = MatRef::from_mat(Zi);
            let y_i1_x = neo_ccs::utils::mat_vec_mul_fk::<F,K>(
                z_ref.data, z_ref.rows, z_ref.cols, &v1_x
            );
            
            // Compute MLE at X_a: y_mle_x = ⟨y_{i,1}(X_r), χ_{X_a}⟩
            let mut y_mle_x = K::ZERO;
            for (rho, &y_rho) in y_i1_x.iter().enumerate() {
                let mut chi_xa_rho = K::ONE;
                for bit_pos in 0..ell_d {
                    let xa_bit = if (x_a_idx >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    let rho_bit = if (rho >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    chi_xa_rho *= xa_bit * rho_bit + (K::ONE - xa_bit) * (K::ONE - rho_bit);
                }
                y_mle_x += chi_xa_rho * y_rho;
            }
            
            // Apply range polynomial: NC_i = ∏_{t=-(b-1)}^{b-1} (y_mle_x - t)
            let Ni_x = crate::pi_ccs::nc_core::range_product::<F>(y_mle_x, params.b);
            
            #[cfg(feature = "debug-logs")]
            {
                if x_idx < 4 && _i_idx == 0 {
                    eprintln!("[compute_nc_hypercube_sum] x_idx={}: y_mle_x={}, NC={}, eq_x_beta={}", 
                             x_idx, 
                             crate::pi_ccs::format_ext(y_mle_x),
                             crate::pi_ccs::format_ext(Ni_x),
                             crate::pi_ccs::format_ext(eq_x_beta));
                }
            }
            
            nc_sum_hypercube += eq_x_beta * gamma_pow_i * Ni_x;
            gamma_pow_i *= gamma;
        }
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

    beta_a: &[K],
    beta_r: &[K],
    b: u32,
    ell_d: usize,
    ell_n: usize,
) -> Vec<K>
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    // Build chi over the full (Ajtai+row) domain at β
    let chi_beta_full: Vec<K> = {
        let mut beta_full = beta_a.to_vec();
        beta_full.extend_from_slice(beta_r);
        neo_ccs::utils::tensor_point::<K>(&beta_full)
    };

    let mut per_i = vec![K::ZERO; z_all.len()];
    let dn = (1usize << ell_d) * (1usize << ell_n);

    for x_idx in 0..dn {
        let eq_x_beta = chi_beta_full[x_idx];
        let x_r_idx = x_idx % (1 << ell_n);
        let x_a_idx = x_idx >> ell_n;

        // v1_x = M_1^T · χ_{X_r}
        let mut v1_x = vec![K::ZERO; s.m];
        for row in 0..s.n {
            let mut chi_x_r_row = K::ONE;
            for bit_pos in 0..ell_n {
                let xr = if (x_r_idx >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                let rb = if (row     >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                chi_x_r_row *= xr*rb + (K::ONE - xr)*(K::ONE - rb);
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
                    let xa = if (x_a_idx >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    let rb = if (rho     >> bit_pos) & 1 == 1 { K::ONE } else { K::ZERO };
                    chi_xa_rho *= xa*rb + (K::ONE - xa)*(K::ONE - rb);
                }
                y_mle_x += chi_xa_rho * y_rho;
            }

            let Ni_x = crate::pi_ccs::nc_core::range_product::<F>(y_mle_x, b);
            per_i[i] += eq_x_beta * Ni_x;
        }
    }

    per_i
}
