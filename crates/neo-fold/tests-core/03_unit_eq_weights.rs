use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

fn naive_eq(row: usize, r: &[K]) -> K {
    let mut result = K::ONE;
    for (i, &ri) in r.iter().enumerate() {
        let bit = (row >> i) & 1;
        result *= if bit == 0 { K::ONE - ri } else { ri };
    }
    result
}

#[test]
fn eq_weights_sum_to_one_simple_cases() {
    for ell in [1usize, 2, 3, 4] {
        let r: Vec<K> = (0..ell)
            .map(|i| if i % 2 == 0 { K::ZERO } else { K::ONE })
            .collect();
        
        let mut sum = K::ZERO;
        let n = 1usize << ell;
        for row in 0..n {
            sum += naive_eq(row, &r);
        }
        
        assert_eq!(sum, K::ONE, "sum of eq(·, r) over {{0,1}}^{{{}}} must be 1", ell);
    }
}

#[test]
fn eq_weights_corner_cases() {
    let r_all_zero = vec![K::ZERO, K::ZERO, K::ZERO];
    assert_eq!(naive_eq(0, &r_all_zero), K::ONE, "eq(000, 000) = 1");
    assert_eq!(naive_eq(1, &r_all_zero), K::ZERO, "eq(001, 000) = 0");
    assert_eq!(naive_eq(7, &r_all_zero), K::ZERO, "eq(111, 000) = 0");
    
    let r_all_one = vec![K::ONE, K::ONE, K::ONE];
    assert_eq!(naive_eq(0, &r_all_one), K::ZERO, "eq(000, 111) = 0");
    assert_eq!(naive_eq(7, &r_all_one), K::ONE, "eq(111, 111) = 1");
}

#[test]
fn eq_weights_specific_values() {
    let r = vec![
        K::from(F::from_u64(2)),
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
    ];
    
    let n = 1usize << r.len();
    let mut sum = K::ZERO;
    for row in 0..n {
        sum += naive_eq(row, &r);
    }
    
    assert_eq!(sum, K::ONE, "sum of eq weights with arbitrary r must still be 1");
}

#[test]
fn eq_weights_single_bit() {
    let r = vec![K::from(F::from_u64(7))];
    
    let w0 = naive_eq(0, &r);
    let w1 = naive_eq(1, &r);
    
    assert_eq!(w0 + w1, K::ONE, "single bit: eq(0,r) + eq(1,r) = 1");
}

#[test]
fn eq_weights_two_bits() {
    let r = vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))];
    
    let mut sum = K::ZERO;
    for row in 0..4 {
        sum += naive_eq(row, &r);
    }
    
    assert_eq!(sum, K::ONE, "two bits: sum over all 4 rows = 1");
}

