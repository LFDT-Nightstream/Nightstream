//! Gradient-based blame analysis (mathematically correct sensitivity)

use super::types::GradientContribution;
use super::witness::field_to_canonical_hex;
use neo_ccs::{CcsStructure, SparsePoly};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Compute sensitivity ∂r/∂z_i for CCS constraint
/// 
/// For CCS with f(y_1,...,y_t) and y_j = <M_j[row,:], z>:
/// ∂r/∂z_i = Σ_j (∂f/∂y_j) * M_j[row, i]
pub fn compute_gradient_blame(
    structure: &CcsStructure<F>,
    row: usize,
    witness: &[F],
    top_k: usize,
) -> Vec<GradientContribution> {
    let t = structure.t();
    
    // Compute y_j = M_j[row, :] · z for each matrix
    let y_vals: Vec<F> = structure.matrices.iter()
        .map(|mat| {
            let mut sum = F::ZERO;
            for col in 0..mat.cols() {
                sum += mat[(row, col)] * witness[col];
            }
            sum
        })
        .collect();
    
    // Compute ∂f/∂y_j at (y_1,...,y_t)
    let df_dy = compute_poly_gradient(&structure.f, &y_vals);
    
    // Compute ∂r/∂z_i = Σ_j (∂f/∂y_j) * M_j[row, i]
    let mut gradients = Vec::new();
    for i in 0..structure.m {
        let mut grad = F::ZERO;
        for j in 0..t {
            grad += df_dy[j] * structure.matrices[j][(row, i)];
        }
        
        if grad != F::ZERO {
            let contrib = grad * witness[i];
            gradients.push(GradientContribution {
                i,
                name: None,  // Filled by symbol table if available
                gradient: field_to_canonical_hex(grad),
                gradient_abs: grad.as_canonical_u64().to_string(),
                z_value: field_to_canonical_hex(witness[i]),
                contribution: field_to_canonical_hex(contrib),
            });
        }
    }
    
    // Sort by absolute gradient (descending), take top K
    gradients.sort_by_key(|g| {
        std::cmp::Reverse(g.gradient_abs.parse::<u64>().unwrap_or(0))
    });
    gradients.truncate(top_k);
    gradients
}

/// Compute gradient of sparse polynomial
/// For f = Σ c_i * Π_j y_j^{e_{i,j}}, we have:
/// ∂f/∂y_k = Σ_i (e_{i,k} * c_i * y_k^{e_{i,k}-1} * Π_{j≠k} y_j^{e_{i,j}})
fn compute_poly_gradient(poly: &SparsePoly<F>, y: &[F]) -> Vec<F> {
    let t = y.len();
    let mut grad = vec![F::ZERO; t];
    
    // For each term in the polynomial
    for term in poly.terms() {
        let coeff = term.coeff;
        let exps = &term.exps;  // Exponents for each variable
        
        // For each variable y_k
        for k in 0..t {
            if exps[k] == 0 {
                continue;  // Variable doesn't appear in this term
            }
            
            // Compute ∂(term)/∂y_k = e_k * coeff * y_k^{e_k-1} * Π_{j≠k} y_j^{e_j}
            let exp_k = exps[k];
            
            // SECURITY FIX: Use field multiplication by exponent, not repeated addition
            // Previous bug: for _ in 1..exp_k { prod += coeff; } is WRONG
            let mut prod = coeff * F::from_u64(exp_k as u64);
            
            // Multiply by y_k^{e_k-1}
            if exp_k > 1 {
                prod *= pow_u32(y[k], exp_k - 1);
            }
            // If exp_k == 1, y_k^0 = 1 (no multiplication needed)
            
            // Multiply by Π_{j≠k} y_j^{e_j}
            for j in 0..t {
                if j == k {
                    continue;
                }
                let exp_j = exps[j];
                if exp_j > 0 {
                    prod *= pow_u32(y[j], exp_j);
                }
            }
            
            grad[k] += prod;
        }
    }
    
    grad
}

/// Fast exponentiation for field elements
fn pow_u32(base: F, exp: u32) -> F {
    if exp == 0 {
        return F::ONE;
    }
    if exp == 1 {
        return base;
    }
    
    let mut result = F::ONE;
    let mut base = base;
    let mut exp = exp;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result *= base;
        }
        base *= base;
        exp /= 2;
    }
    
    result
}

/// Special case: R1CS gradient (for reference and testing)
/// For r = (Az)(Bz) - Cz:
/// ∂r/∂z_i = a_i(Bz) + b_i(Az) - c_i
#[allow(dead_code)]
pub fn compute_r1cs_gradient_blame(
    a_row: &[F],
    b_row: &[F],
    c_row: &[F],
    witness: &[F],
    top_k: usize,
) -> Vec<GradientContribution> {
    use crate::diagnostic::witness::field_to_signed_i128;
    
    let az: F = a_row.iter().zip(witness).map(|(a, z)| *a * *z).sum();
    let bz: F = b_row.iter().zip(witness).map(|(b, z)| *b * *z).sum();
    
    let mut gradients = Vec::new();
    for i in 0..witness.len() {
        let grad = a_row[i] * bz + b_row[i] * az - c_row[i];
        
        if grad != F::ZERO {
            let contrib = grad * witness[i];
            gradients.push(GradientContribution {
                i,
                name: None,
                gradient: field_to_canonical_hex(grad),
                gradient_abs: field_to_signed_i128(grad).abs().to_string(),
                z_value: field_to_canonical_hex(witness[i]),
                contribution: field_to_canonical_hex(contrib),
            });
        }
    }
    
    gradients.sort_by_key(|g| {
        std::cmp::Reverse(g.gradient_abs.parse::<u128>().unwrap_or(0))
    });
    gradients.truncate(top_k);
    gradients
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    
    #[test]
    fn test_r1cs_gradient() {
        // Simple R1CS: z[0] * z[1] = z[2]
        // With z = [2, 3, 7], residual = 2*3 - 7 = -1
        // az = 2, bz = 3
        // ∂r/∂z_0 = a_0*bz + b_0*az - c_0 = 1*3 + 0*2 - 0 = 3
        // ∂r/∂z_1 = a_1*bz + b_1*az - c_1 = 0*3 + 1*2 - 0 = 2  
        // ∂r/∂z_2 = a_2*bz + b_2*az - c_2 = 0*3 + 0*2 - 1 = -1
        
        let a_row = vec![F::ONE, F::ZERO, F::ZERO];
        let b_row = vec![F::ZERO, F::ONE, F::ZERO];
        let c_row = vec![F::ZERO, F::ZERO, F::ONE];
        let witness = vec![
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(7),
        ];
        
        let blame = compute_r1cs_gradient_blame(&a_row, &b_row, &c_row, &witness, 10);
        
        assert_eq!(blame.len(), 3);
        assert_eq!(blame[0].gradient_abs, "3");  // Highest gradient is 3
        assert_eq!(blame[0].i, 0);  // At index 0
        assert_eq!(blame[1].gradient_abs, "2");  // Second highest is 2
        assert_eq!(blame[1].i, 1);  // At index 1
        assert_eq!(blame[2].gradient_abs, "1");  // Third is 1 (abs of -1)
        assert_eq!(blame[2].i, 2);  // At index 2
    }
}

