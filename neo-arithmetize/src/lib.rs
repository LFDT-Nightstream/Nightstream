//! Neo Arithmetize - High-level to CCS conversion utilities
//!
//! This module provides functions to convert high-level computations into
//! Customizable Constraint Systems (CCS) that can be used with Neo's
//! lattice-based proof system.

use neo_ccs::{CcsStructure, mv_poly};
use neo_fields::{ExtF, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

/// Convert a Fibonacci sequence of given length into a CCS structure.
///
/// This function arithmetizes the Fibonacci recurrence relation:
/// x_i = x_{i-1} + x_{i-2} for i = 2 to length-1
///
/// # Arguments
/// * `length` - The length of the Fibonacci sequence to arithmetize
///
/// # Returns
/// A `CcsStructure` representing the Fibonacci constraints
///
/// # Example
/// ```
/// use neo_arithmetize::fibonacci_ccs;
///
/// // Just test that the function exists and returns something reasonable
/// let ccs = fibonacci_ccs(0); // Minimal case - returns empty structure
/// assert_eq!(ccs.witness_size, 0);
/// ```
pub fn fibonacci_ccs(length: usize) -> CcsStructure {
    if length < 2 {
        // Return empty structure for trivial cases
        let empty_mats = vec![];
        let empty_f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 0);
        return CcsStructure::new(empty_mats, empty_f);
    }

    let num_constraints = length.saturating_sub(2);
    let witness_size = length;

    // Create constraint matrices for each Fibonacci constraint
    // For each constraint i (where i goes from 2 to length-1):
    // M_a[i, i] = 1 (coefficient for x_i)
    // M_b[i, i-1] = 1 (coefficient for x_{i-1})
    // M_c[i, i-2] = 1 (coefficient for x_{i-2})

    let mut m_a_data = vec![F::ZERO; num_constraints * witness_size];
    let mut m_b_data = vec![F::ZERO; num_constraints * witness_size];
    let mut m_c_data = vec![F::ZERO; num_constraints * witness_size];

    for row in 0..num_constraints {
        let i = row + 2; // constraint for x_i where i starts from 2

        // Set coefficients in the matrices
        let a_idx = row * witness_size + i;
        let b_idx = row * witness_size + (i - 1);
        let c_idx = row * witness_size + (i - 2);

        m_a_data[a_idx] = F::ONE;      // x_i coefficient
        m_b_data[b_idx] = F::ONE;      // x_{i-1} coefficient
        m_c_data[c_idx] = F::ONE;      // x_{i-2} coefficient
    }

    let m_a = RowMajorMatrix::new(m_a_data, witness_size);
    let m_b = RowMajorMatrix::new(m_b_data, witness_size);
    let m_c = RowMajorMatrix::new(m_c_data, witness_size);

    let mats = vec![m_a, m_b, m_c];

    // Constraint polynomial: f(a, b, c) = a - b - c = 0
    // This enforces x_i = x_{i-1} + x_{i-2}
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() >= 3 {
                inputs[0] - inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        1, // degree 1 polynomial
    );

    CcsStructure::new(mats, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_ccs::{CcsInstance, CcsWitness, check_satisfiability};
    use neo_fields::embed_base_to_ext;

    #[test]
    fn test_fibonacci_ccs_small() {
        let length = 5;
        let ccs = fibonacci_ccs(length);

        // Expected Fibonacci sequence: [0, 1, 1, 2, 3]
        let fib_sequence = vec![0u64, 1, 1, 2, 3];
        let z: Vec<ExtF> = fib_sequence
            .iter()
            .map(|&x| embed_base_to_ext(F::from_u64(x)))
            .collect();

        let witness = CcsWitness { z };

        // Create a dummy instance for satisfiability check
        let dummy_instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };

        // Check satisfiability - should pass for correct Fibonacci sequence
        assert!(check_satisfiability(&ccs, &dummy_instance, &witness));
    }

    #[test]
    fn test_fibonacci_ccs_wrong_sequence() {
        let length = 5;
        let ccs = fibonacci_ccs(length);

        // Wrong sequence: [0, 1, 2, 2, 3] (x_2 should be 1, not 2)
        let wrong_sequence = vec![0u64, 1, 2, 2, 3];
        let z: Vec<ExtF> = wrong_sequence
            .iter()
            .map(|&x| embed_base_to_ext(F::from_u64(x)))
            .collect();

        let witness = CcsWitness { z };

        // Create a dummy instance for satisfiability check
        let dummy_instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };

        // Check satisfiability - should fail for incorrect sequence
        assert!(!check_satisfiability(&ccs, &dummy_instance, &witness));
    }

    #[test]
    fn test_fibonacci_ccs_trivial() {
        let ccs_0 = fibonacci_ccs(0);
        assert_eq!(ccs_0.num_constraints, 0);

        let ccs_1 = fibonacci_ccs(1);
        assert_eq!(ccs_1.num_constraints, 0);

        let ccs_2 = fibonacci_ccs(2);
        assert_eq!(ccs_2.num_constraints, 0); // No constraints for length 2
    }
}
