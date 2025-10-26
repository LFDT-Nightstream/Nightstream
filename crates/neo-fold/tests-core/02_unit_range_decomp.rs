use neo_fold::eval_range_decomp_constraints;
use neo_ccs::Mat;
use neo_math::{F, K, D};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn get_test_params() -> NeoParams {
    NeoParams::goldilocks_for_circuit(3, 2, 2)
}

#[test]
#[allow(non_snake_case)]
fn range_decomp_zero_for_all_zero_witness() {
    let params = get_test_params();
    let d = D;
    let m = 2usize;
    
    let z = vec![F::ZERO; m];
    let Z = Mat::<F>::zero(d, m, F::ZERO);
    let u_dummy = vec![];
    
    let res = eval_range_decomp_constraints(&z, &Z, &u_dummy, &params);
    assert_eq!(res, K::ZERO, "honest all-zero range+decomp residual must be 0");
}

#[test]
#[allow(non_snake_case)]
fn range_decomp_zero_for_simple_honest_witness() {
    let params = get_test_params();
    let b = params.b;
    let d = D;
    let m = 2usize;
    
    let z = vec![F::from_u64(1), F::from_u64(2)];
    
    let z_digits = neo_ajtai::decomp_b(&z, b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { 
        for row in 0..d { 
            row_major[row * m + col] = z_digits[col * d + row]; 
        } 
    }
    let Z = Mat::from_row_major(d, m, row_major);
    let u_dummy = vec![];
    
    let res = eval_range_decomp_constraints(&z, &Z, &u_dummy, &params);
    assert_eq!(res, K::ZERO, "honest decomposition with small values should have residual 0");
}

#[test]
#[allow(non_snake_case)]
fn range_decomp_detects_invalid_digit_out_of_range() {
    let params = get_test_params();
    let b = params.b as i64;
    let d = D;
    let m = 2usize;
    
    let z = vec![F::ZERO; m];
    let mut Z_data = vec![F::ZERO; d * m];
    Z_data[0] = F::from_i64(b + 1);
    
    let Z = Mat::from_row_major(d, m, Z_data);
    let u_dummy = vec![];
    
    let res = eval_range_decomp_constraints(&z, &Z, &u_dummy, &params);
    assert_ne!(res, K::ZERO, "digit out of range should produce non-zero residual");
}

#[test]
#[allow(non_snake_case)]
fn range_decomp_detects_recomposition_mismatch() {
    let params = get_test_params();
    let d = D;
    let m = 2usize;
    
    let z = vec![F::from_u64(5), F::from_u64(7)];
    
    let mut z_digits = neo_ajtai::decomp_b(&z, params.b, d, neo_ajtai::DecompStyle::Balanced);
    z_digits[0] = F::from_u64(99);
    
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m { 
        for row in 0..d { 
            row_major[row * m + col] = z_digits[col * d + row]; 
        } 
    }
    let Z = Mat::from_row_major(d, m, row_major);
    let u_dummy = vec![];
    
    let res = eval_range_decomp_constraints(&z, &Z, &u_dummy, &params);
    assert_ne!(res, K::ZERO, "invalid recomposition should produce non-zero residual");
}

