//! Core NC (Norm Constraint) polynomial evaluation functions
//!
//! This module provides the fundamental building blocks for evaluating
//! NC polynomials: NC_i(y) = ∏_{t=-(b-1)}^{b-1} (y - t)
//!
//! These functions are used across different phases of the sum-check protocol
//! to ensure consistent evaluation of the range constraint polynomial.

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};

/// Compute the range product: ∏_{t=-(b-1)}^{b-1} (value - t)
/// 
/// This is the core NC polynomial that enforces the digit bound constraint.
/// For a value to have valid b-ary decomposition, it must satisfy NC(value) = 0.
///
/// # Arguments
/// * `value` - The value to evaluate the range polynomial at
/// * `b` - The base for decomposition (digit bound)
#[inline]
pub fn range_product<F: Field + PrimeCharacteristicRing>(value: K, b: u32) -> K 
where 
    K: From<F> 
{
    let low = -((b as i64) - 1);
    let high = (b as i64) - 1;
    let mut product = K::ONE;
    for t in low..=high {
        product *= value - K::from(F::from_i64(t));
    }
    product
}

/// Compute NC polynomial at an interpolated point: NC((1-x)·y0 + x·y1)
/// 
/// This function is critical for the Ajtai phase where we must:
/// 1. First interpolate the value: z(x) = (1-x)·y0 + x·y1
/// 2. Then compute the range product: NC(z(x))
/// 
/// # Arguments
/// * `y0` - Value at x=0
/// * `y1` - Value at x=1
/// * `x` - Interpolation point
/// * `b` - Base for decomposition
#[inline]
pub fn nc_interpolated<F: Field + PrimeCharacteristicRing>(
    y0: K, 
    y1: K, 
    x: K, 
    b: u32
) -> K 
where 
    K: From<F> 
{
    let z = (K::ONE - x) * y0 + x * y1;
    range_product::<F>(z, b)
}

/// Compute weighted NC sum: Σ_i γ^i · NC_i(values[i])
/// 
/// Used when we need to combine multiple NC evaluations with gamma powers.
/// 
/// # Arguments
/// * `values` - Values to evaluate NC at
/// * `gamma_pows` - Powers of gamma for weighting (γ^1, γ^2, ...)
/// * `b` - Base for decomposition
pub fn weighted_nc_sum<F: Field + PrimeCharacteristicRing>(
    values: &[K],
    gamma_pows: &[K],
    b: u32
) -> K 
where 
    K: From<F> 
{
    debug_assert_eq!(values.len(), gamma_pows.len(), 
        "values and gamma_pows must have same length");
    
    values.iter()
        .zip(gamma_pows)
        .map(|(&val, &gamma_pow)| gamma_pow * range_product::<F>(val, b))
        .fold(K::ZERO, |acc, term| acc + term)
}

/// Compute NC at multiple interpolated points in parallel
/// 
/// Useful for batch evaluation during sum-check rounds.
/// 
/// # Arguments
/// * `y0_vec` - Values at x=0 for each instance
/// * `y1_vec` - Values at x=1 for each instance  
/// * `x` - Common interpolation point
/// * `gamma_pows` - Powers of gamma for weighting
/// * `b` - Base for decomposition
pub fn nc_interpolated_batch<F: Field + PrimeCharacteristicRing>(
    y0_vec: &[K],
    y1_vec: &[K], 
    x: K,
    gamma_pows: &[K],
    b: u32
) -> K 
where 
    K: From<F> 
{
    debug_assert_eq!(y0_vec.len(), y1_vec.len(), 
        "y0_vec and y1_vec must have same length");
    debug_assert_eq!(y0_vec.len(), gamma_pows.len(),
        "y vectors and gamma_pows must have same length");
    
    y0_vec.iter()
        .zip(y1_vec)
        .zip(gamma_pows)
        .map(|((&y0, &y1), &gamma_pow)| {
            gamma_pow * nc_interpolated::<F>(y0, y1, x, b)
        })
        .fold(K::ZERO, |acc, term| acc + term)
}
