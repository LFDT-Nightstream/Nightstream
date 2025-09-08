//! Shared utilities for Neo benchmarks

use neo_ccs::{CcsStructure, Mat};
use neo::F;
use p3_field::PrimeCharacteristicRing;

/// Pad all CCS matrices with zero-rows so that ccs.n is power-of-two.
/// 
/// This is required for proper CCS evaluation over the Boolean hypercube {0,1}^ℓ
/// which requires dimensions aligned to powers of two for clean multilinear 
/// evaluation and sum-check domains.
pub fn pad_ccs_rows_to_pow2(ccs: CcsStructure<F>) -> CcsStructure<F> {
    let n = ccs.n;
    let n_pad = n.next_power_of_two();
    if n_pad == n { 
        return ccs; 
    }

    let m = ccs.m;
    let mut padded_mats = Vec::with_capacity(ccs.matrices.len());
    
    for mat in &ccs.matrices {
        let mut out = Mat::zero(n_pad, m, F::ZERO);
        // Copy existing rows
        for r in 0..n {
            for c in 0..m {
                out[(r, c)] = mat[(r, c)];
            }
        }
        padded_mats.push(out);
    }

    // Rebuild CCS with identical polynomial f, just with padded matrices
    CcsStructure::new(padded_mats, ccs.f.clone())
        .expect("valid CCS after row padding")
}

/// Helper function to convert sparse triplets to dense row-major format
pub fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        data[row * cols + col] = val;
    }
    data
}

/// Helper to get proof size in bytes
/// Note: Neo's Proof type has a stable .size() method, so we use it directly
pub fn proof_size_bytes(proof: &neo::Proof) -> usize {
    proof.size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_ccs::{r1cs_to_ccs, Mat};
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_padding_power_of_two() {
        // Create a simple 3x3 CCS (not power of two)
        let rows = 3;
        let cols = 3;
        
        let a_data = vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE];
        let b_data = vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ONE];
        let c_data = vec![F::ZERO; 9];

        let a = Mat::from_row_major(rows, cols, a_data);
        let b = Mat::from_row_major(rows, cols, b_data);
        let c = Mat::from_row_major(rows, cols, c_data);

        let ccs = r1cs_to_ccs(a, b, c);
        assert_eq!(ccs.n, 3); // Original size

        let padded_ccs = pad_ccs_rows_to_pow2(ccs);
        assert_eq!(padded_ccs.n, 4); // Padded to next power of two
        assert_eq!(padded_ccs.m, 3); // Columns unchanged
    }

    #[test]
    fn test_padding_already_power_of_two() {
        // Create a 4x3 CCS (already power of two)
        let rows = 4;
        let cols = 3;
        
        let a_data = vec![F::ONE; 12];
        let b_data = vec![F::ONE; 12];
        let c_data = vec![F::ZERO; 12];

        let a = Mat::from_row_major(rows, cols, a_data);
        let b = Mat::from_row_major(rows, cols, b_data);
        let c = Mat::from_row_major(rows, cols, c_data);

        let ccs = r1cs_to_ccs(a, b, c);
        let padded_ccs = pad_ccs_rows_to_pow2(ccs);
        
        assert_eq!(padded_ccs.n, 4); // Should remain unchanged
        assert_eq!(padded_ccs.m, 3);
    }
}
