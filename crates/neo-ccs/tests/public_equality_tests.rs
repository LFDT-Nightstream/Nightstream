//! Tests for public vector equality gadgets

use neo_ccs::gadgets::public_equality::{public_equality_ccs, build_public_vec_eq_witness, multiple_public_equality_constraints};
use neo_ccs::relations::check_ccs_rowwise_zero;
use p3_goldilocks::Goldilocks;
use p3_field::PrimeCharacteristicRing;

type F = Goldilocks;

#[test]
fn test_public_vec_equality_basic() {
    let len = 3;
    let ccs = public_equality_ccs(len);
    
    // Test case: vectors are equal
    let a = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
    let b = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
    
    let mut public_input = Vec::with_capacity(2 * len);
    public_input.extend_from_slice(&a);
    public_input.extend_from_slice(&b);
    
    let witness = build_public_vec_eq_witness();
    
    // Should satisfy the constraints
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_ok());
}

#[test] 
fn test_public_vec_equality_different() {
    let len = 2;
    let ccs = public_equality_ccs(len);
    
    // Test case: vectors are different
    let a = vec![F::from_u64(1), F::from_u64(2)];
    let b = vec![F::from_u64(1), F::from_u64(3)]; // different!
    
    let mut public_input = Vec::with_capacity(2 * len);
    public_input.extend_from_slice(&a);
    public_input.extend_from_slice(&b);
    
    let witness = build_public_vec_eq_witness();
    
    // Should NOT satisfy the constraints
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_err());
}

#[test]
fn test_public_vec_equality_single_element() {
    let len = 1;
    let ccs = public_equality_ccs(len);
    
    // Single element vectors - equal
    let a = vec![F::from_u64(42)];
    let b = vec![F::from_u64(42)];
    
    let mut public_input = Vec::with_capacity(2 * len);
    public_input.extend_from_slice(&a);
    public_input.extend_from_slice(&b);
    
    let witness = build_public_vec_eq_witness();
    
    // Should satisfy the constraints
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_ok());
}

#[test]
fn test_public_vec_equality_large_vectors() {
    let len = 8;
    let ccs = public_equality_ccs(len);
    
    // Large vectors - equal
    let a = vec![F::from_u64(1), F::from_u64(4), F::from_u64(9), F::from_u64(16),
                 F::from_u64(25), F::from_u64(36), F::from_u64(49), F::from_u64(64)];
    let b = a.clone(); // identical
    
    let mut public_input = Vec::with_capacity(2 * len);
    public_input.extend_from_slice(&a);
    public_input.extend_from_slice(&b);
    
    let witness = build_public_vec_eq_witness();
    
    // Should satisfy the constraints
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_ok());
}

#[test]
fn test_multiple_public_equality_constraints() {
    // Test the multiple constraint binding function
    let bindings = vec![(0, 0), (2, 1)]; // public[0] = witness[0], public[2] = witness[1]
    let witness_cols = 3; // witness has 3 columns including constant
    let public_cols = 3;  // 3 public inputs
    
    let ccs = multiple_public_equality_constraints(&bindings, witness_cols, public_cols);
    
    // For bindings (0, 0) and (2, 1):
    // public[0] should equal witness[0] = 1 (the constant)
    // public[2] should equal witness[1] = 5
    let corrected_witness = vec![F::from_u64(1), F::from_u64(5), F::from_u64(9)];
    let corrected_public = vec![F::from_u64(1), F::from_u64(7), F::from_u64(5)];
    
    assert!(check_ccs_rowwise_zero(&ccs, &corrected_public, &corrected_witness).is_ok());
}

#[test]
fn test_multiple_public_equality_constraints_mismatch() {
    let bindings = vec![(0, 1), (1, 2)]; // public[0] = witness[1], public[1] = witness[2]
    let witness_cols = 4;
    let public_cols = 2;
    
    let ccs = multiple_public_equality_constraints(&bindings, witness_cols, public_cols);
    
    // Mismatched values - should fail
    let public_input = vec![F::from_u64(10), F::from_u64(20)];
    let witness = vec![F::from_u64(1), F::from_u64(99), F::from_u64(88), F::from_u64(77)]; // witness[1]=99 ≠ public[0]=10
    
    assert!(check_ccs_rowwise_zero(&ccs, &public_input, &witness).is_err());
}
