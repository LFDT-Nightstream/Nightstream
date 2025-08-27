//! Pay-per-bit Embedding and Decomposition
//!
//! Implements Neo's core pay-per-bit optimization through:
//! - decomp_b: map z ∈ F_q^m to Z ∈ F_q^{d × m} by placing base-b digits into coefficients
//! - split_b: reduce norms by splitting high-norm vectors into low-norm components
//! - Bit-sparse ring multiplication for cost scaling with number of 1-bits

use neo_math::F;
use neo_math::{Coeff, ModInt};
use neo_math::RingElement;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

/// Pay-per-bit embedding: map z ∈ F_q^m to Z ∈ F_q^{d × m}
/// Places base-b digits into coefficient vectors for low-norm matrices
pub fn decomp_b(z: &[F], b: u64, d: usize) -> RowMajorMatrix<F> {
    let m = z.len();
    let mut data = vec![F::ZERO; d * m];
    
    for (col, &val) in z.iter().enumerate() {
        let mut x = val;
        for row in 0..d {
            let digit = F::from_u64(x.as_canonical_u64() % b);
            data[row * m + col] = digit;
            x = F::from_u64(x.as_canonical_u64() / b);
        }
    }
    
    RowMajorMatrix::new(data, m)
}

/// Split high-norm vector into k low-norm components: split_b(Z) = (Z₁, ..., Zₖ)
/// Used in Π_DEC to reduce norms from B to b while preserving the relation
/// Z = ∑ᵢ bⁱ⁻¹ Zᵢ
pub fn split_b<C: Coeff + Into<i128> + From<i128> + Send + Sync>(
    z: &[C],
    b: u64,
    k: usize,
) -> (Vec<RowMajorMatrix<C>>, Vec<C>) {
    let base = b as i128;
    let r = (base - 1) / 2; // Balanced representation range [-r, r]
    let m = z.len();
    
    // Gadget vector g = [1, b, b², ..., bᵏ⁻¹]
    let mut g = vec![C::zero(); k];
    let mut pow = C::one();
    for g_item in g.iter_mut() {
        *g_item = pow;
        pow *= C::from(b as i128);
    }
    
    // Split each element of z into k components
    let mut components = vec![vec![C::zero(); m]; k];
    
    for (col, &val) in z.iter().enumerate() {
        let q: i128 = C::modulus() as i128;
        let mut x = val.into();
        
        // Convert to signed representation
        if x > q / 2 {
            x -= q;
        }
        
        // Decompose in balanced base-b representation
        for row in 0..k {
            let mut rem = x % base;
            let mut quot = x / base;
            
            // Balance the remainder to [-r, r]
            if rem > r {
                rem -= base;
                quot += 1;
            } else if rem < -r {
                rem += base;
                quot -= 1;
            }
            
            components[row][col] = C::from(rem);
            x = quot;
        }
        
        // Ensure complete decomposition
        assert_eq!(
            x, 0,
            "Decomposition incomplete: k={} too small for value {} with b={}",
            k, val.into(), b
        );
    }
    
    // Convert to matrices
    let matrices: Vec<RowMajorMatrix<C>> = components
        .into_iter()
        .map(|comp| RowMajorMatrix::new(comp, m))
        .collect();
    
    (matrices, g)
}

/// Reconstruct original vector from split components: z = ∑ᵢ gᵢ Zᵢ
pub fn reconstruct_split<C: Coeff + Send + Sync>(
    matrices: &[RowMajorMatrix<C>], 
    g: &[C]
) -> Vec<C> {
    assert_eq!(matrices.len(), g.len(), "Gadget vector length mismatch");
    
    if matrices.is_empty() {
        return Vec::new();
    }
    
    let m = matrices[0].width();
    let mut z = vec![C::zero(); m];
    
    for (col, z_item) in z.iter_mut().enumerate() {
        let mut acc = C::zero();
        for (matrix, &g_val) in matrices.iter().zip(g.iter()) {
            acc += g_val * matrix.get(0, col).unwrap(); // All matrices have height 1 for vector splitting
        }
        *z_item = acc;
    }
    
    z
}

/// Pack decomposed matrix into ring elements for commitment
/// Maps F_q^{d × m} matrix to Vec<RingElement<ModInt>> by placing columns into coefficient vectors
pub fn pack_decomp_matrix(mat: &RowMajorMatrix<F>, params_n: usize) -> Vec<RingElement<ModInt>> {
    let d = mat.height();
    let m = mat.width();
    
    // Each row becomes a ring element with columns as coefficients
    (0..d)
        .map(|row| {
            let mut coeffs: Vec<ModInt> = (0..m)
                .map(|col| ModInt::from_u64(mat.get(row, col).unwrap().as_canonical_u64()))
                .collect();
            
            // Pad to ring degree
            if coeffs.len() > params_n {
                panic!("Matrix width {} exceeds ring degree {}", coeffs.len(), params_n);
            }
            coeffs.resize(params_n, ModInt::zero());
            
            RingElement::from_coeffs(coeffs, params_n)
        })
        .collect()
}

/// Unpack ring elements back to matrix form
pub fn unpack_to_matrix(
    ring_elements: &[RingElement<ModInt>], 
    m: usize
) -> RowMajorMatrix<F> {
    let d = ring_elements.len();
    let mut data = vec![F::ZERO; d * m];
    
    for (row, ring_elem) in ring_elements.iter().enumerate() {
        for (col, &coeff) in ring_elem.coeffs().iter().enumerate().take(m) {
            data[row * m + col] = F::from_u64(coeff.as_canonical_u64());
        }
    }
    
    RowMajorMatrix::new(data, m)
}

/// Compute pay-per-bit cost: scales with actual bit usage, not field size
/// This is Neo's key optimization - cost grows with witness sparsity
pub fn pay_per_bit_cost(witness_bits: usize, params_k: usize, params_d: usize, params_q: u64) -> u64 {
    let bit_width = (params_q as f64).log2().ceil() as u64;
    
    // Commitment cost: κ × d × bit_width (for matrix A)
    let commitment_cost = params_k as u64 * params_d as u64 * bit_width;
    
    // Witness cost: scales with actual bits used
    let witness_cost = witness_bits as u64 * bit_width;
    
    commitment_cost + witness_cost
}

/// Bit-sparse ring multiplication: cost scales with number of 1-bits
/// Optimizes ring multiplication when coefficient vectors have few non-zero bits
pub fn bit_sparse_ring_multiply(
    a: &RingElement<ModInt>,
    b_sparse_coeffs: &[(usize, ModInt)], // (index, coefficient) pairs for sparse b
    n: usize,
) -> RingElement<ModInt> {
    let mut result_coeffs = vec![ModInt::zero(); n];
    
    // Only multiply by non-zero coefficients of b
    for &(b_idx, b_coeff) in b_sparse_coeffs {
        if b_coeff == ModInt::zero() {
            continue;
        }
        
        // Multiply a by X^b_idx * b_coeff (with negacyclic reduction)
        for (a_idx, &a_coeff) in a.coeffs().iter().enumerate() {
            let result_idx = (a_idx + b_idx) % n;
            let sign = if (a_idx + b_idx) >= n { 
                // Negacyclic: X^n = -1
                ModInt::from_u64(<ModInt as Coeff>::modulus() - 1) // -1 mod q
            } else { 
                ModInt::one() 
            };
            
            result_coeffs[result_idx] += a_coeff * b_coeff * sign;
        }
    }
    
    RingElement::from_coeffs(result_coeffs, n)
}

/// Extract sparse representation from ring element (for bit-sparse operations)
pub fn extract_sparse_coeffs(ring: &RingElement<ModInt>) -> Vec<(usize, ModInt)> {
    ring.coeffs()
        .iter()
        .enumerate()
        .filter_map(|(idx, &coeff)| {
            if coeff != ModInt::zero() {
                Some((idx, coeff))
            } else {
                None
            }
        })
        .collect()
}

/// Compute bit-width of a field element (for pay-per-bit calculations)
pub fn bit_width(val: F) -> usize {
    if val == F::ZERO {
        return 0;
    }
    
    let canonical = val.as_canonical_u64();
    64 - canonical.leading_zeros() as usize
}

/// Count total bits used in a vector (for pay-per-bit cost analysis)
pub fn count_witness_bits(witness: &[F]) -> usize {
    witness.iter().map(|&val| bit_width(val)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_decomp_b_basic() {
        let z = vec![F::from_u64(5), F::from_u64(13)];
        let mat = decomp_b(&z, 2, 4); // Base-2, 4 digits
        
        // 5 = 1*1 + 0*2 + 1*4 + 0*8 = [1,0,1,0]
        // 13 = 1*1 + 0*2 + 1*4 + 1*8 = [1,0,1,1]
        assert_eq!(mat.get(0, 0), Some(F::ONE));   // 5 & 1
        assert_eq!(mat.get(1, 0), Some(F::ZERO));  // (5 >> 1) & 1
        assert_eq!(mat.get(2, 0), Some(F::ONE));   // (5 >> 2) & 1
        assert_eq!(mat.get(3, 0), Some(F::ZERO));  // (5 >> 3) & 1
        
        assert_eq!(mat.get(0, 1), Some(F::ONE));   // 13 & 1
        assert_eq!(mat.get(1, 1), Some(F::ZERO));  // (13 >> 1) & 1
        assert_eq!(mat.get(2, 1), Some(F::ONE));   // (13 >> 2) & 1
        assert_eq!(mat.get(3, 1), Some(F::ONE));   // (13 >> 3) & 1
    }

    #[test]
    fn test_split_b_reconstruct() {
        let z = vec![ModInt::from_u64(42), ModInt::from_u64(100)];
        let (matrices, g) = split_b(&z, 3, 8); // Base-3, 8 components (enough for larger values)
        
        // Reconstruct should give original
        let reconstructed = reconstruct_split(&matrices, &g);
        assert_eq!(reconstructed, z);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let data = vec![F::ONE, F::from_u64(2), F::from_u64(3), F::from_u64(4)];
        let mat = RowMajorMatrix::new(data, 2); // 2x2 matrix
        
        let packed = pack_decomp_matrix(&mat, 8); // Ring degree 8
        let unpacked = unpack_to_matrix(&packed, 2);
        
        // Should match original (up to matrix dimensions)
        assert_eq!(unpacked.height(), mat.height());
        assert_eq!(unpacked.width(), mat.width());
        for i in 0..mat.height() {
            for j in 0..mat.width() {
                assert_eq!(unpacked.get(i, j), mat.get(i, j));
            }
        }
    }

    #[test]
    fn test_pay_per_bit_cost() {
        let cost = pay_per_bit_cost(100, 16, 32, 1u64 << 61);
        assert!(cost > 0);
        
        // Cost should scale with witness bits
        let cost_double = pay_per_bit_cost(200, 16, 32, 1u64 << 61);
        assert!(cost_double > cost);
    }

    #[test]
    fn test_bit_sparse_multiply() {
        let a = RingElement::from_coeffs(vec![ModInt::one(), ModInt::from_u64(2)], 4);
        let b_sparse = vec![(0, ModInt::one()), (2, ModInt::from_u64(3))]; // X^0 + 3*X^2
        
        let result = bit_sparse_ring_multiply(&a, &b_sparse, 4);
        
        // Should have correct degree
        assert_eq!(result.coeffs().len(), 4);
    }

    #[test]
    fn test_bit_width_calculation() {
        assert_eq!(bit_width(F::ZERO), 0);
        assert_eq!(bit_width(F::ONE), 1);
        assert_eq!(bit_width(F::from_u64(7)), 3); // 111 in binary
        assert_eq!(bit_width(F::from_u64(8)), 4); // 1000 in binary
    }

    #[test]
    fn test_count_witness_bits() {
        let witness = vec![F::ZERO, F::ONE, F::from_u64(7), F::from_u64(8)];
        let total_bits = count_witness_bits(&witness);
        assert_eq!(total_bits, 0 + 1 + 3 + 4); // Sum of individual bit widths
    }
}
