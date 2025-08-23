use neo_ccs::{mv_poly, CcsStructure};
use neo_fields::{embed_base_to_ext, ExtF, F};
use p3_matrix::dense::RowMajorMatrix;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_multilinearity_check() {
    // Test that multilinear polynomials are correctly identified
    let f_multilinear = mv_poly(|ys: &[ExtF]| ys[0] + ys[1], 1);
    assert_eq!(f_multilinear.max_individual_degree(), 1);
    
    let f_zero = mv_poly(|_ys: &[ExtF]| ExtF::ZERO, 0);
    assert_eq!(f_zero.max_individual_degree(), 0);
    
    let f_nonlinear = mv_poly(|ys: &[ExtF]| ys[0] * ys[0], 2);
    assert_eq!(f_nonlinear.max_individual_degree(), 2);
}

#[test]
fn test_alpha_twisted_computation() {
    // Test the α-twisted computation logic directly
    
    // Simple test: α-twisted MLE should work correctly
    let alpha = embed_base_to_ext(F::from_u64(2));
    let alpha_pows = vec![ExtF::ONE, alpha, alpha * alpha, alpha * alpha * alpha];
    
    // Test data: [1, 2, 3, 4]
    let data = vec![
        embed_base_to_ext(F::from_u64(1)), 
        embed_base_to_ext(F::from_u64(2)), 
        embed_base_to_ext(F::from_u64(3)), 
        embed_base_to_ext(F::from_u64(4))
    ];
    
    // α-twisted data should be [1*1, 2*2, 3*4, 4*8] = [1, 4, 12, 32]
    let mut twisted = vec![ExtF::ZERO; 4];
    for i in 0..4 {
        twisted[i] = alpha_pows[i] * data[i];
    }
    
    assert_eq!(twisted[0], embed_base_to_ext(F::from_u64(1)));
    assert_eq!(twisted[1], embed_base_to_ext(F::from_u64(4)));
    assert_eq!(twisted[2], embed_base_to_ext(F::from_u64(12)));
    assert_eq!(twisted[3], embed_base_to_ext(F::from_u64(32)));
}

#[test]
fn test_final_binding_logic() {
    // Test the final binding check logic directly
    let f = mv_poly(|ys: &[ExtF]| ys[0] + ys[1], 1); // Multilinear
    
    // Test values
    let ys = vec![
        embed_base_to_ext(F::from_u64(3)), 
        embed_base_to_ext(F::from_u64(4))
    ];
    let expected_result = embed_base_to_ext(F::from_u64(7));
    
    // The binding check should pass when f(ys) equals the expected result
    let actual_result = f.evaluate(&ys);
    assert_eq!(actual_result, expected_result);
    
    // Test that multilinearity check works
    assert!(f.max_individual_degree() <= 1);
}

#[test]
#[should_panic(expected = "CcsStructure: f must be multilinear")]
fn strict_multilinear_enforcement() {
    // This should panic since multilinearity is now always enforced
    let mats = vec![RowMajorMatrix::new(vec![F::ONE, F::ZERO], 2)];
    let f = mv_poly(|ys: &[ExtF]| ys[0] * ys[0], 2); // Non-multilinear: degree 2 in first variable
    let _ccs = CcsStructure::new(mats, f); // Should panic
}

#[test]
fn test_ccs_structure_creation() {
    // Test that CcsStructure can be created with multilinear polynomials
    let mats = vec![RowMajorMatrix::new(vec![F::ONE, F::ZERO], 2)];
    let f = mv_poly(|ys: &[ExtF]| ys[0], 1); // Multilinear
    let ccs = CcsStructure::new(mats, f);
    
    assert_eq!(ccs.mats.len(), 1);
    assert_eq!(ccs.num_constraints, 1);
    assert_eq!(ccs.witness_size, 2);
    assert_eq!(ccs.f.max_individual_degree(), 1);
}
