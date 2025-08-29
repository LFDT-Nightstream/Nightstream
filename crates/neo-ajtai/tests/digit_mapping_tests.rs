use neo_ajtai::util::to_balanced_i128;
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn balanced_digit_maps_exactly() {
    let digs = [-3i32, -2i32, -1i32, 0i32, 1i32, 2i32, 3i32, 17i32, -17i32];
    for &d in &digs {
        let f = if d >= 0 { 
            F::from_u64(d as u64) 
        } else { 
            F::ZERO - F::from_u64((-d) as u64) 
        };
        // round-trip via to_balanced_i128()
        let back = to_balanced_i128(f);
        let expected = d as i128 % ((1i128 << 64) - (1i128 << 32) + 1);
        assert_eq!(back, expected, "Failed for digit {d}");
    }
}

#[test]
fn digit_mapping_is_consistent_with_field_ops() {

    
    // Test that our mapping preserves basic field operations
    let a_val = 42i32;
    let b_val = -17i32;
    
    let a = if a_val >= 0 { 
        F::from_u64(a_val as u64) 
    } else { 
        F::ZERO - F::from_u64((-a_val) as u64) 
    };
    
    let b = if b_val >= 0 { 
        F::from_u64(b_val as u64) 
    } else { 
        F::ZERO - F::from_u64((-b_val) as u64) 
    };
    
    // Check that a + b matches expected field behavior
    let sum = a + b;
    let back_sum = to_balanced_i128(sum);
    let expected_sum = (a_val + b_val) as i128;
    let field_expected = expected_sum % ((1i128 << 64) - (1i128 << 32) + 1);
    
    assert_eq!(back_sum, field_expected);
}

#[test] 
fn large_negative_digits() {
    // Test edge cases with larger negative values
    let large_neg = -12345i32;
    let f = F::ZERO - F::from_u64(12345u64);
    let back = to_balanced_i128(f);
    let expected = large_neg as i128 % ((1i128 << 64) - (1i128 << 32) + 1);
    assert_eq!(back, expected);
}
