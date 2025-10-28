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
        for Zi in witnesses.iter().map(|w| &w.Z).chain(me_witnesses.iter()) {
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
            
            // Apply range polynomial: NC_i = ∏_{t=-b+1}^{b-1} (y_mle_x - t)
            let mut Ni_x = K::ONE;
            for t in -(params.b as i64 - 1)..=(params.b as i64 - 1) {
                Ni_x *= y_mle_x - K::from(F::from_i64(t));
            }
            
            nc_sum_hypercube += eq_x_beta * gamma_pow_i * Ni_x;
            gamma_pow_i *= gamma;
        }
    }
    
    nc_sum_hypercube
}

/// Evaluate range/decomposition constraint polynomials NC_i(z,Z) at point u.
/// 
/// **Paper Reference**: Section 4.4, NC_i term in Q polynomial
/// NC_i(X) = ∏_{j=-b+1}^{b-1} (Z̃_i(X) - j)
/// These assert: Z = Decomp_b(z) and ||Z||_∞ < b
/// 
/// NOTE: For honest instances where Z == Decomp_b(z) and ||Z||_∞ < b, 
///       this MUST return zero to make the composed polynomial Q sum to zero.
pub fn eval_range_decomp_constraints<F>(
    z: &[F],
    Z: &Mat<F>,
    _u: &[K],
    params: &neo_params::NeoParams,
) -> K
where
    F: Field + Send + Sync + Copy,
    K: From<F>,
{
    // REAL CONSTRAINT EVALUATION (degree 0 in u)
    // Enforces two facts:
    // 1. Decomposition correctness: z[c] = Σ_{i=0}^{d-1} b^i * Z[i,c] 
    // 2. Digit range (balanced): R_b(x) = x * ∏_{t=1}^{b-1} (x-t)(x+t) = 0
    
    let d = Z.rows();
    let m = Z.cols();

    // Sanity: shapes
    if z.len() != m {
        // Treat shape mismatch as a hard violation: contribute a non-zero sentinel.
        return K::from(F::ONE);
    }

    // Precompute base powers in F for recomposition
    let b_f = F::from_u64(params.b as u64);
    let mut pow_b = vec![F::ONE; d];
    for i in 1..d { 
        pow_b[i] = pow_b[i - 1] * b_f; 
    }

    // === (A) Decomposition correctness residual: sum of squares in K ===
    let mut decomp_residual = K::ZERO;
    for c in 0..m {
        // z_rec = Σ_{i=0..d-1} (b^i * Z[i,c])
        let mut z_rec_f = F::ZERO;
        for i in 0..d {
            z_rec_f += Z[(i, c)] * pow_b[i];
        }
        // Residual (in K): (z_rec - z[c])^2
        let diff_k = K::from(z_rec_f) - K::from(z[c]);
        decomp_residual += diff_k * diff_k;
    }

    // === (B) Range residual: R_b(x) = x * ∏_{t=1}^{b-1} (x - t)(x + t) for every digit ===
    let mut range_residual = K::ZERO;

    // Precompute constants in F for 1..(b-1)
    let mut t_vals = Vec::with_capacity((params.b - 1) as usize);
    for t in 1..params.b {
        t_vals.push(F::from_u64(t as u64));
    }

    for c in 0..m {
        for i in 0..d {
            let digit_f = Z[(i, c)];
            let digit_k = K::from(digit_f);

            // R_b(x) = x * ∏_{t=1}^{b-1} (x - t)(x + t)
            let mut rb_val = digit_k;
            for &t_f in &t_vals {
                let t_k = K::from(t_f);
                rb_val *= (digit_k - t_k) * (digit_k + t_k);
            }
            range_residual += rb_val * rb_val;
        }
    }

    // Return combined residual: if both are zero, constraints are satisfied
    decomp_residual + range_residual
}

