//! Critical property tests for ring arithmetic and S-action correctness.
//! These tests anchor the algebra to the paper's specification and catch regressions.

use neo_math::{Rq, Fq, D, cf, cf_inv, SAction};
use neo_math::ring::{rot_apply_vec, test_reduce_mod_phi_81};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use rand_chacha::rand_core::RngCore;

/// Test the fundamental ring/S-action isomorphism: cf(a*b) == rot(a)·cf(b)
#[test]
fn ring_s_action_isomorphism() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    for _ in 0..20 {
        // Generate random ring elements
        let a = Rq::random_uniform(&mut rng);
        let b = Rq::random_uniform(&mut rng);
        
        // Compute a*b in the ring
        let ab = a.mul(&b);
        let cf_ab = cf(ab);
        
        // Compute rot(a) · cf(b) using S-action
        let cf_b = cf(b);
        let rot_a_cf_b = rot_apply_vec(&a, &cf_b);
        
        // They must be equal (this is the fundamental isomorphism)
        assert_eq!(cf_ab, rot_a_cf_b, 
            "Ring isomorphism failed: cf(a*b) != rot(a)·cf(b)");
    }
}

/// Test power-of-X property: cf(a * X^j) equals column j of rot(a)
#[test]
fn power_of_x_columns() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    for _ in 0..10 {
        let a = Rq::random_uniform(&mut rng);
        let s_action = SAction::from_ring(a);
        let rot_a_matrix = s_action.to_matrix();
        
        for j in 0..D {
            // Compute a * X^j
            let x_j = {
                let mut coeffs = [Fq::ZERO; D];
                coeffs[j] = Fq::ONE; // X^j as a polynomial
                cf_inv(coeffs)
            };
            let a_times_x_j = a.mul(&x_j);
            let cf_result = cf(a_times_x_j);
            
            // Extract column j from rot(a) matrix
            let mut col_j = [Fq::ZERO; D];
            for i in 0..D {
                col_j[i] = rot_a_matrix.get(i, j).unwrap_or(Fq::ZERO);
            }
            
            // They must match
            assert_eq!(cf_result, col_j, 
                "Power of X property failed: cf(a*X^{}) != column {} of rot(a)", j, j);
        }
    }
}

/// Test monomial multiplication correctness with known values
#[test]
fn monomial_multiplication_regression() {
    // Test with known values to catch regressions
    let mut coeffs = [Fq::ZERO; D];
    coeffs[0] = Fq::ONE; // polynomial "1"
    coeffs[1] = Fq::from_u64(2); // polynomial "1 + 2X"
    let p = Rq::from_field_coeffs(coeffs.to_vec());
    
    // Multiply by X (should shift coefficients)
    let px = p.mul_by_monomial(1);
    let px_coeffs = px.field_coeffs();
    
    // Check that coefficients shifted correctly
    assert_eq!(px_coeffs[0], Fq::ZERO);
    assert_eq!(px_coeffs[1], Fq::ONE);
    assert_eq!(px_coeffs[2], Fq::from_u64(2));
    
    // Test multiplication by X^27 (valid since 27 < D=54)
    let px27 = p.mul_by_monomial(27);
    // Note: X^54 ≡ -X^27 - 1 in cyclotomic ring, not X^0 (tested in cyclotomic_phi_81_relation)
    
    // This operation should not panic and should produce valid results
    assert!(px27.field_coeffs().iter().any(|&c| c != Fq::ZERO));
}

/// Test cyclotomic polynomial relation: X^54 ≡ -X^27 - 1 (mod Φ_81)
#[test]
fn cyclotomic_phi_81_relation() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    // Test the fundamental cyclotomic relation for random polynomials
    for _ in 0..10 {
        let p = Rq::random_uniform(&mut rng);
        
        // Cannot test p * X^54 directly since mul_by_monomial only works for j < D
        // Instead, test the relation via manual construction
        
        // Construct polynomial X^54 by building coefficients and reducing
        let x54 = {
            let mut tmp = [Fq::ZERO; 2*D - 1];
            tmp[54] = Fq::ONE; // X^54 coefficient
            test_reduce_mod_phi_81(&mut tmp);
            let mut out = [Fq::ZERO; D];
            out.copy_from_slice(&tmp[..D]);
            Rq::from_field_coeffs(out.to_vec())
        };
        
        // X^54 ≡ -X^27 - 1 (mod Φ_81)
        let minus_x27_minus_1 = {
            let mut coeffs = [Fq::ZERO; D];
            coeffs[0] = -Fq::ONE;    // -1
            coeffs[27] = -Fq::ONE;   // -X^27
            Rq::from_field_coeffs(coeffs.to_vec())
        };
        
        assert_eq!(x54.field_coeffs(), minus_x27_minus_1.field_coeffs(),
            "X^54 must equal -X^27 - 1 in the cyclotomic ring");
        
        // Test that the relation holds for multiplication: p * X^54 ≡ p * (-X^27 - 1)
        let p_times_x54 = p.mul(&x54);
        let p_times_x27 = p.mul_by_monomial(27);
        let expected = Rq::zero() - p_times_x27 - p; // -p*X^27 - p
        
        assert_eq!(p_times_x54.field_coeffs(), expected.field_coeffs(),
            "p * X^54 must equal -p * X^27 - p for the cyclotomic relation");
    }
}

/// Test that sparse bit multiplication is consistent with dense multiplication
#[test]
fn sparse_vs_dense_multiplication() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    for _ in 0..10 {
        let a = Rq::random_small(&mut rng, 100);
        
        // Create a sparse polynomial representation
        let mut sparse_bits = Vec::new();
        let mut dense_poly = Rq::zero();
        
        for i in 0..D {
            if (rng.next_u32() % 2) == 0 && sparse_bits.len() < 5 {
                sparse_bits.push((i, true));
                // Build equivalent dense polynomial
                let mut x_i_coeffs = [Fq::ZERO; D];
                x_i_coeffs[i] = Fq::ONE;
                dense_poly = dense_poly + cf_inv(x_i_coeffs);
            }
        }
        
        if !sparse_bits.is_empty() {
            let sparse_result = a.mul_sparse_bits(&sparse_bits);
            let dense_result = a.mul(&dense_poly);
            
            assert_eq!(sparse_result.field_coeffs(), dense_result.field_coeffs(),
                "Sparse multiplication doesn't match dense multiplication");
        }
    }
}

/// Test ring arithmetic properties: associativity, distributivity, etc.
#[test]
fn ring_arithmetic_properties() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    for _ in 0..10 {
        let a = Rq::random_uniform(&mut rng);
        let b = Rq::random_uniform(&mut rng);
        let c = Rq::random_uniform(&mut rng);
        
        // Test associativity: (a*b)*c == a*(b*c)
        let ab_c = a.mul(&b).mul(&c);
        let a_bc = a.mul(&b.mul(&c));
        assert_eq!(cf(ab_c), cf(a_bc), "Multiplication not associative");
        
        // Test distributivity: a*(b+c) == a*b + a*c
        let bc = b + c;
        let a_bc_dist = a.mul(&bc);
        let ab_ac = a.mul(&b) + a.mul(&c);
        assert_eq!(cf(a_bc_dist), cf(ab_ac), "Multiplication not distributive over addition");
        
        // Test commutativity: a*b == b*a
        let ab = a.mul(&b);
        let ba = b.mul(&a);
        assert_eq!(cf(ab), cf(ba), "Multiplication not commutative");
    }
}

/// Test infinity norm properties
#[test]
fn infinity_norm_properties() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    // Test that zero has norm 0
    assert_eq!(Rq::zero().norm_inf(), 0);
    
    // Test that one has norm 1
    assert_eq!(Rq::one().norm_inf(), 1);
    
    // Test triangle inequality for addition: ||a + b|| <= ||a|| + ||b||
    for _ in 0..10 {
        let a = Rq::random_small(&mut rng, 1000);
        let b = Rq::random_small(&mut rng, 1000);
        let sum = a + b;
        
        let norm_sum = sum.norm_inf();
        let norm_a = a.norm_inf();
        let norm_b = b.norm_inf();
        
        // Triangle inequality
        assert!(norm_sum <= norm_a + norm_b, 
            "Triangle inequality violated: ||a+b|| = {} > {} + {} = ||a|| + ||b||",
            norm_sum, norm_a, norm_b);
    }
}

/// Test that rotation step agrees with ring multiply by X
#[test]
fn rotation_step_vs_ring_multiply() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x5eed_u64);
    
    for _ in 0..10 {
        let a = Rq::random_uniform(&mut rng);
        
        // Method 1: Use rotation matrix application
        // rot(a) · cf(X) should equal cf(a · X)
        let x_poly = {
            let mut coeffs = vec![Fq::ZERO; neo_math::D];
            coeffs[1] = Fq::ONE; // X = 0 + 1*X + 0*X^2 + ...
            Rq::from_field_coeffs(coeffs)
        };
        let cf_x = cf(x_poly); // cf of X (monomial degree 1)
        let rot_a_cf_x = rot_apply_vec(&a, &cf_x);
        
        // Method 2: Ring multiply by X (monomial degree 1) then cf
        let a_times_x = a.mul_by_monomial(1);
        let cf_a_times_x = cf(a_times_x);
        
        // They should be equivalent: rot(a)·cf(X) = cf(a·X)
        assert_eq!(rot_a_cf_x, cf_a_times_x,
            "rot(a)·cf(X) should equal cf(a·X)");
    }
    
    println!("✅ Rotation step agrees with ring multiply by X");
}
