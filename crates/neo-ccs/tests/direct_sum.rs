//! Direct sum composition test
//! 
//! This ensures direct-sum (block-diagonal) wiring is correct: 
//! witness and public inputs must concatenate left||right and each block is enforced independently.

use neo_ccs::{check_ccs_rowwise_zero, direct_sum_transcript_mixed, CcsStructure};
use neo_ccs::gadgets::commitment_opening::commitment_lincomb_ccs;
use neo_ccs::gadgets::public_equality::public_equality_ccs;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn fe(x: u64) -> F { F::from_u64(x) }

#[test]
fn direct_sum_composes_public_and_witness_concatenation() {
    // Left: public equality with len = 2
    let ccs_left = public_equality_ccs(2);
    let lhs = vec![fe(10), fe(20)];
    let rhs = vec![fe(10), fe(20)]; // same values for equality
    let mut pub_left = lhs.clone();
    pub_left.extend_from_slice(&rhs);
    let wit_left = vec![F::ONE]; // public equality just needs constant witness

    // Right: commitment lincomb len = 2
    let ccs_right = commitment_lincomb_ccs(2);
    let rho_r = fe(5);
    let c_prev = vec![fe(1), fe(2)];
    let c_step = vec![fe(7), fe(8)];
    let u: Vec<F> = c_step.iter().map(|&x| rho_r * x).collect();
    let c_next: Vec<F> = c_prev.iter().zip(&u).map(|(a,b)| *a + *b).collect();

    let mut wit_right = vec![F::ONE];
    wit_right.extend_from_slice(&u);

    let mut pub_right = vec![rho_r];
    pub_right.extend_from_slice(&c_prev);
    pub_right.extend_from_slice(&c_step);
    pub_right.extend_from_slice(&c_next);

    // Direct-sum
    let ccs_sum: CcsStructure<F> = direct_sum_transcript_mixed(&ccs_left, &ccs_right, [0u8;32]).unwrap();

    // Safe direct_sum expects simple concatenation: [left_all || right_all]
    let mut left_input = pub_left.clone();
    left_input.extend_from_slice(&wit_left);
    
    let mut right_input = pub_right.clone();
    right_input.extend_from_slice(&wit_right);
    
    // Input vector for safe direct_sum: [left_all || right_all]
    let mut combined_input = left_input.clone();
    combined_input.extend_from_slice(&right_input);
    
    // For CCS check: all input is "witness", no separate public
    let public = vec![]; // No separate public input
    let witness = combined_input;

    assert!(check_ccs_rowwise_zero(&ccs_sum, &public, &witness).is_ok());

    // Tamper right side input -> must fail
    let mut witness_bad = witness.clone();
    let last = witness_bad.len() - 1;
    witness_bad[last] = witness_bad[last] + fe(1);
    assert!(check_ccs_rowwise_zero(&ccs_sum, &public, &witness_bad).is_err());

    // Tamper left side input -> must also fail  
    let mut witness_left_bad = witness.clone();
    witness_left_bad[0] = witness_left_bad[0] + fe(1); // tamper first lhs element
    assert!(check_ccs_rowwise_zero(&ccs_sum, &public, &witness_left_bad).is_err());
}

#[test]
fn direct_sum_independent_constraints() {
    // Test that left and right constraints are enforced independently
    // Left: public equality that should pass
    let ccs_left = public_equality_ccs(1);
    let pub_left = vec![fe(42), fe(42)]; // equal values
    let wit_left = vec![F::ONE];

    // Right: public equality that should fail if we tamper
    let ccs_right = public_equality_ccs(1);
    let pub_right = vec![fe(100), fe(101)]; // different values - should fail alone
    let wit_right = vec![F::ONE];

    // Check that right side fails on its own
    assert!(check_ccs_rowwise_zero(&ccs_right, &pub_right, &wit_right).is_err());

    // But left side passes on its own
    assert!(check_ccs_rowwise_zero(&ccs_left, &pub_left, &wit_left).is_ok());

    // Direct sum should fail because right side fails
    let ccs_sum = direct_sum_transcript_mixed(&ccs_left, &ccs_right, [0u8;32]).unwrap();
    
    // Safe direct_sum expects simple concatenation: [left_all || right_all]
    let mut left_input = pub_left.clone();
    left_input.extend_from_slice(&wit_left);
    
    let mut right_input = pub_right.clone();
    right_input.extend_from_slice(&wit_right);
    
    let mut combined_input = left_input.clone();
    combined_input.extend_from_slice(&right_input);
    
    let public_combined = vec![]; // No separate public input
    let witness_combined = combined_input;

    assert!(check_ccs_rowwise_zero(&ccs_sum, &public_combined, &witness_combined).is_err());
}

#[test]
fn direct_sum_both_sides_pass() {
    // Test case where both sides should pass
    let ccs_left = public_equality_ccs(1);
    let pub_left = vec![fe(123), fe(123)]; // equal
    let wit_left = vec![F::ONE];

    let ccs_right = public_equality_ccs(1); 
    let pub_right = vec![fe(456), fe(456)]; // equal
    let wit_right = vec![F::ONE];

    // Both should pass individually
    assert!(check_ccs_rowwise_zero(&ccs_left, &pub_left, &wit_left).is_ok());
    assert!(check_ccs_rowwise_zero(&ccs_right, &pub_right, &wit_right).is_ok());

    // Direct sum should also pass
    let ccs_sum = direct_sum_transcript_mixed(&ccs_left, &ccs_right, [0u8;32]).unwrap();
    
    // Safe direct_sum expects simple concatenation: [left_all || right_all]
    let mut left_input = pub_left.clone();
    left_input.extend_from_slice(&wit_left);
    
    let mut right_input = pub_right.clone();
    right_input.extend_from_slice(&wit_right);
    
    let mut combined_input = left_input.clone();
    combined_input.extend_from_slice(&right_input);
    
    let public_combined = vec![]; // No separate public input
    let witness_combined = combined_input;

    assert!(check_ccs_rowwise_zero(&ccs_sum, &public_combined, &witness_combined).is_ok());
}

#[test]
fn direct_sum_preserves_matrix_structure() {
    // Test that the direct sum creates the correct block-diagonal structure
    let ccs_left = public_equality_ccs(1);
    let ccs_right = public_equality_ccs(1);
    
    let ccs_sum = direct_sum_transcript_mixed(&ccs_left, &ccs_right, [0u8;32]).unwrap();
    
    // Check dimensions
    assert_eq!(ccs_sum.n, ccs_left.n + ccs_right.n, "row count should be sum");
    assert_eq!(ccs_sum.m, ccs_left.m + ccs_right.m, "column count should be sum");
    assert_eq!(ccs_sum.matrices.len(), ccs_left.matrices.len() + ccs_right.matrices.len(), "matrix count should be t1 + t2");
}

#[test]
fn direct_sum_empty_step_digest() {
    // Test that step digest parameter doesn't affect the CCS structure
    let ccs1 = public_equality_ccs(1);
    let ccs2 = public_equality_ccs(1);
    
    let digest1 = [0u8; 32];
    let digest2 = [1u8; 32]; // different digest
    
    let sum1 = direct_sum_transcript_mixed(&ccs1, &ccs2, digest1).unwrap();
    let sum2 = direct_sum_transcript_mixed(&ccs1, &ccs2, digest2).unwrap();
    
    // Results should be identical (digest is only for binding at higher layers)
    assert_eq!(sum1.n, sum2.n);
    assert_eq!(sum1.m, sum2.m);
    assert_eq!(sum1.matrices.len(), sum2.matrices.len());
}
