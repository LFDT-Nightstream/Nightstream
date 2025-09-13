//! Step output extractor tests (no placeholders!)
//! These tests ensure extractors return real step outputs, not placeholder values.

use neo::F;
use neo::ivc::{LastNExtractor, IndexExtractor, StepOutputExtractor};
use p3_field::PrimeCharacteristicRing;

#[test]
fn last_n_extractor_works() {
    let w: Vec<F> = (0..10).map(|i| F::from_u64(i as u64)).collect();
    let ex = LastNExtractor { n: 3 };
    let y = ex.extract_y_step(&w);
    let exp: Vec<F> = (7..10).map(|i| F::from_u64(i)).collect();
    assert_eq!(y, exp);
}

#[test]
fn last_n_extractor_handles_edge_cases() {
    let w: Vec<F> = (0..3).map(|i| F::from_u64(i as u64)).collect();
    
    // Extract more than available - should return all
    let ex = LastNExtractor { n: 5 };
    let y = ex.extract_y_step(&w);
    assert_eq!(y, w);
    
    // Extract zero elements  
    let ex_zero = LastNExtractor { n: 0 };
    let y_empty = ex_zero.extract_y_step(&w);
    assert_eq!(y_empty, vec![]);
    
    // Extract from empty witness
    let empty_w: Vec<F> = vec![];
    let ex_normal = LastNExtractor { n: 2 };
    let y_from_empty = ex_normal.extract_y_step(&empty_w);
    assert_eq!(y_from_empty, vec![]);
}

#[test]
fn index_extractor_works() {
    let w: Vec<F> = (0..6).map(|i| F::from_u64(i as u64)).collect();
    let ex = IndexExtractor { indices: vec![0, 2, 5] };
    let y = ex.extract_y_step(&w);
    let exp = vec![F::from_u64(0), F::from_u64(2), F::from_u64(5)];
    assert_eq!(y, exp);
}

#[test]
fn index_extractor_handles_out_of_bounds() {
    let w: Vec<F> = (0..4).map(|i| F::from_u64(i as u64)).collect();
    
    // Some indices out of bounds - should skip them
    let ex = IndexExtractor { indices: vec![1, 10, 2, 20] };
    let y = ex.extract_y_step(&w);
    let exp = vec![F::from_u64(1), F::from_u64(2)];
    assert_eq!(y, exp);
    
    // All indices out of bounds - should return empty
    let ex_oob = IndexExtractor { indices: vec![10, 20, 30] };
    let y_empty = ex_oob.extract_y_step(&w);
    assert_eq!(y_empty, vec![]);
}

#[test]
fn index_extractor_empty_indices() {
    let w: Vec<F> = (0..5).map(|i| F::from_u64(i as u64)).collect();
    let ex = IndexExtractor { indices: vec![] };
    let y = ex.extract_y_step(&w);
    assert_eq!(y, vec![]);
}

#[test]
fn index_extractor_duplicate_indices() {
    let w: Vec<F> = (0..4).map(|i| F::from_u64(i as u64)).collect();
    let ex = IndexExtractor { indices: vec![1, 1, 2, 1] };
    let y = ex.extract_y_step(&w);
    // Should extract each index as specified (including duplicates)
    let exp = vec![F::from_u64(1), F::from_u64(1), F::from_u64(2), F::from_u64(1)];
    assert_eq!(y, exp);
}

#[test]
fn extractors_return_real_values_not_placeholders() {
    // This test ensures extractors don't return hardcoded placeholder values
    let witness1: Vec<F> = vec![F::from_u64(100), F::from_u64(200), F::from_u64(300)];
    let witness2: Vec<F> = vec![F::from_u64(400), F::from_u64(500), F::from_u64(600)];
    
    let extractor = LastNExtractor { n: 2 };
    
    let y1 = extractor.extract_y_step(&witness1);
    let y2 = extractor.extract_y_step(&witness2);
    
    // Results should be different for different witnesses (not placeholders)
    assert_ne!(y1, y2, "Extractor must return actual witness values, not placeholders");
    
    // Results should match expected values from actual witness
    assert_eq!(y1, vec![F::from_u64(200), F::from_u64(300)]);
    assert_eq!(y2, vec![F::from_u64(500), F::from_u64(600)]);
}
