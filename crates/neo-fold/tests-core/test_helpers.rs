/// Test utility functions for neo-fold tests
///
/// These are NOT part of the protocol - only for testing purposes

use neo_ccs::Mat;
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};

/// Evaluate range/decomposition constraint polynomials NC_i(z,Z) at point u.
/// 
/// **Test Utility Only** - This is NOT used in the actual protocol.
/// 
/// Checks two properties:
/// 1. Decomposition correctness: z[c] = Σ_{i=0}^{d-1} b^i * Z[i,c]
/// 2. Digit range (balanced): R_b(x) = x * ∏_{t=1}^{b-1} (x-t)(x+t) = 0
/// 
/// For honest instances where Z == Decomp_b(z) and ||Z||_∞ < b, 
/// this returns zero (or near-zero due to floating point).
#[allow(non_snake_case)]
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

