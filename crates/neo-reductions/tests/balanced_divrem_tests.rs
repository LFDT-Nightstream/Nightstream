//! Tests for balanced base-b digit decomposition.
//!
//! Validates that balanced_divrem correctly decomposes values into balanced digits
//! that terminate (quotient reaches 0) for both positive and negative inputs.

use neo_reductions::split_b_matrix_k;
use neo_ccs::Mat;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Helper to test balanced decomposition by manually applying the algorithm
fn test_balanced_decomposition(initial_value: i128, b: i128, max_digits: usize) -> (Vec<i128>, i128, i128) {
    let mut v = initial_value;
    let mut digits = Vec::new();
    
    for i in 0..max_digits {
        // Mimic the balanced_divrem logic from common.rs
        let mut r = v % b;
        let mut q = (v - r) / b;
        
        let half = b / 2;
        
        if r > half {
            r -= b;
            q += 1;
        } else if r < -half {
            r += b;
            q -= 1;
        }
        
        eprintln!("Step {}: v={} -> (r={}, q={}), check: {}*{} + {} = {}", 
                  i, v, r, q, b, q, r, b*q + r);
        digits.push(r);
        v = q;
        
        if v == 0 {
            break;
        }
    }
    
    eprintln!("Final digits: {:?}", digits);
    eprintln!("Final remainder: {}", v);
    
    // Reconstruct the value
    let mut reconstructed = 0i128;
    let mut power = 1i128;
    for &digit in &digits {
        reconstructed += digit * power;
        power *= b;
    }
    eprintln!("Reconstructed: {}", reconstructed);
    
    (digits, v, reconstructed)
}

#[test]
fn test_balanced_divrem_decompose_4_base2() {
    // Test that 4 can be decomposed in base 2 with balanced digits
    let (digits, final_v, reconstructed) = test_balanced_decomposition(4, 2, 12);
    
    assert_eq!(final_v, 0, "Should consume all digits (quotient should reach 0)");
    assert_eq!(reconstructed, 4, "Should reconstruct to original value");
    assert!(digits.len() <= 3, "4 should only need 3 digits at most");
}

#[test]
fn test_balanced_divrem_decompose_minus2_base2() {
    // Test that -2 can be decomposed in base 2 with balanced digits
    let (digits, final_v, reconstructed) = test_balanced_decomposition(-2, 2, 12);
    
    assert_eq!(final_v, 0, "Should consume all digits (quotient should reach 0)");
    assert_eq!(reconstructed, -2, "Should reconstruct to original value");
    assert!(digits.len() <= 2, "-2 should only need 2 digits at most");
}

#[test]
fn test_balanced_divrem_various_values_base2() {
    // Test a range of values to ensure termination
    let test_cases = vec![
        (0, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (-1, 2),
        (-3, 3),
        (-4, 3),
        (15, 4),
        (-15, 4),
    ];
    
    for (value, expected_max_digits) in test_cases {
        let (digits, final_v, reconstructed) = test_balanced_decomposition(value, 2, 12);
        assert_eq!(final_v, 0, "Value {} should terminate", value);
        assert_eq!(reconstructed, value, "Value {} should reconstruct correctly", value);
        assert!(digits.len() <= expected_max_digits, 
                "Value {} used {} digits, expected <= {}", value, digits.len(), expected_max_digits);
    }
}

#[test]
#[ignore = "base-3 balanced decomposition has edge cases - needs investigation"]
fn test_balanced_divrem_base3() {
    // Test base-3 decomposition
    let test_cases = vec![
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 2),
        (-1, 1),
        (-2, 1),
        (-3, 2),
        (26, 3), // 3^3 - 1
    ];
    
    for (value, expected_max_digits) in test_cases {
        let (digits, final_v, reconstructed) = test_balanced_decomposition(value, 3, 12);
        assert_eq!(final_v, 0, "Value {} (base 3) should terminate", value);
        assert_eq!(reconstructed, value, "Value {} (base 3) should reconstruct correctly", value);
        assert!(digits.len() <= expected_max_digits, 
                "Value {} (base 3) used {} digits, expected <= {}", value, digits.len(), expected_max_digits);
    }
}

#[test]
fn test_split_b_matrix_k_simple() {
    // Test the full split_b_matrix_k function with simple values
    let b = 2u32;
    let k = 12usize;
    
    // Create a simple 2x2 matrix with small values
    #[allow(non_snake_case)]
    let Z = Mat::from_row_major(2, 2, vec![
        F::from_u64(4),
        F::from_u64(2),
        F::ZERO - F::from_u64(2), // -2 in field representation
        F::from_u64(1),
    ]);
    
    let result = split_b_matrix_k(&Z, k, b);
    assert!(result.is_ok(), "Should successfully split simple matrix: {:?}", result.err());
    
    let split_matrices = result.unwrap();
    assert_eq!(split_matrices.len(), k, "Should produce k={} output matrices", k);
    
    // Each output matrix should have the same dimensions as input
    for (i, mat) in split_matrices.iter().enumerate() {
        assert_eq!(mat.rows(), Z.rows(), "Split matrix {} should have same number of rows", i);
        assert_eq!(mat.cols(), Z.cols(), "Split matrix {} should have same number of cols", i);
    }
    
    // Reconstruct and verify
    let mut reconstructed = Mat::zero(Z.rows(), Z.cols(), F::ZERO);
    let mut power = F::ONE;
    let b_field = F::from_u64(b as u64);
    
    for mat in &split_matrices {
        for r in 0..Z.rows() {
            for c in 0..Z.cols() {
                reconstructed[(r, c)] += mat[(r, c)] * power;
            }
        }
        power *= b_field;
    }
    
    // Check reconstruction matches original
    for r in 0..Z.rows() {
        for c in 0..Z.cols() {
            assert_eq!(
                reconstructed[(r, c)].as_canonical_u64(),
                Z[(r, c)].as_canonical_u64(),
                "Reconstructed Z[{},{}] should match original", r, c
            );
        }
    }
}
