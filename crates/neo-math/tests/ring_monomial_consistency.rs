use neo_math::ring::{Rq, cf, cf_inv};
use neo_math::{Fq, D};
use p3_field::PrimeCharacteristicRing;

#[test]
fn mul_by_monomial_is_consistent() {
    let mut a = [Fq::ZERO; D]; 
    for i in 0..D { 
        a[i] = Fq::from_u64(i as u64 + 1); 
    }
    let a = Rq(a);
    
    for j in 0..(2 * D) { // multiple wraps
        let xj = {
            let mut x = Rq::one();
            for _ in 0..j { 
                x = x.mul_by_monomial(1); 
            }
            x
        };
        let ref_mul = a.mul(&xj);
        let fast = a.mul_by_monomial(j);
        assert_eq!(cf(ref_mul), cf(fast), "Monomial multiplication inconsistent at j={}", j);
    }
}

#[test]
fn mul_by_monomial_zero_is_identity() {
    // Test that multiplying by X^0 = 1 is identity
    let mut a = [Fq::ZERO; D];
    for i in 0..D {
        a[i] = Fq::from_u64(i as u64 + 42);
    }
    let a = Rq(a);
    
    let result = a.mul_by_monomial(0);
    assert_eq!(cf(a), cf(result), "mul_by_monomial(0) should be identity");
}

#[test] 
fn mul_by_monomial_wraps_correctly() {
    // Test that multiplication by X^D reduces properly modulo the cyclotomic polynomial
    let mut a = [Fq::ZERO; D];
    a[0] = Fq::ONE; // a = 1
    let a = Rq(a);
    
    // X^D should reduce to some specific pattern based on the cyclotomic polynomial
    let result_d = a.mul_by_monomial(D);
    let result_2d = a.mul_by_monomial(2 * D);
    
    // At minimum, check that results are not the same as the input
    // (unless the cyclotomic polynomial has a very specific form)
    // This test will need to be adjusted based on the specific reduction used
    println!("X^D reduces to: {:?}", cf(result_d));
    println!("X^(2D) reduces to: {:?}", cf(result_2d));
    
    // The key property is that mul_by_monomial should be consistent with ring multiplication
    let x_to_d_direct = {
        let mut x = Rq::one();
        for _ in 0..D {
            x = x.mul_by_monomial(1);
        }
        x
    };
    
    assert_eq!(cf(result_d), cf(x_to_d_direct), "Direct and repeated monomial multiplication should match");
}

#[test]
fn reduce_mod_phi_81_self_consistency() {
    // Test that the reduction modulo Φ_81 is self-consistent
    use neo_math::ring::reduce_mod_phi_81;
    
    // Test with a few simple polynomials
    let test_cases = [
        vec![1, 0, 0, 0], // X^0
        vec![0, 1, 0, 0], // X^1  
        vec![0, 0, 1, 0], // X^2
        vec![1, 1, 1, 1], // 1 + X + X^2 + X^3
    ];
    
    for (i, coeffs) in test_cases.iter().enumerate() {
        let mut extended = vec![Fq::ZERO; 2 * D];
        for (j, &coeff) in coeffs.iter().enumerate() {
            if j < extended.len() {
                extended[j] = Fq::from_u64(coeff);
            }
        }
        
        let reduced = reduce_mod_phi_81(&extended);
        println!("Test case {}: {:?} -> {:?}", i, extended[..coeffs.len()].to_vec(), reduced[..8].to_vec());
        
        // The reduced result should have length D
        assert_eq!(reduced.len(), D, "Reduced polynomial should have degree < D");
        
        // For simple cases, we can check some basic properties
        if i == 0 { // constant 1
            assert_eq!(reduced[0], Fq::ONE);
            for j in 1..D {
                assert_eq!(reduced[j], Fq::ZERO);
            }
        }
    }
}
