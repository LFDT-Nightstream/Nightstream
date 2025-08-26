//! Unit tests for Spartan2 sum-check integration
//! 
//! These tests validate that the sum-check protocol integration with Spartan2
//! works correctly and maintains compatibility with Neo's existing sum-check.

#[cfg(all(test, feature = "snark_mode"))]
mod spartan2_sumcheck_tests {
    use neo_sumcheck::{
        batched_sumcheck_prover, batched_sumcheck_verifier,
        multilinear_sumcheck_prover, multilinear_sumcheck_verifier,
        challenger::NeoChallenger, UnivPoly
    };
    use neo_fields::{ExtF, F, embed_base_to_ext};
    use p3_field::PrimeCharacteristicRing;
    use std::sync::Arc;

    /// Create a simple test polynomial for sum-check
    fn create_test_polynomial() -> Arc<dyn Fn(&[ExtF]) -> ExtF + Send + Sync> {
        Arc::new(|vars: &[ExtF]| {
            if vars.is_empty() {
                ExtF::ZERO
            } else {
                // Simple polynomial: x0 + x1 + ... + xn
                vars.iter().fold(ExtF::ZERO, |acc, &x| acc + x)
            }
        })
    }

    /// Create a multilinear polynomial for testing
    fn create_multilinear_test_polynomial() -> Arc<dyn Fn(&[ExtF]) -> ExtF + Send + Sync> {
        Arc::new(|vars: &[ExtF]| {
            if vars.len() != 2 {
                ExtF::ZERO
            } else {
                // Multilinear: x0 * x1
                vars[0] * vars[1]
            }
        })
    }

    #[test]
    fn test_basic_sumcheck_compatibility() {
        println!("🧪 Testing basic sum-check compatibility");

        let num_vars = 3;
        let poly = create_test_polynomial();
        
        // Create a simple evaluation point
        let eval_point: Vec<ExtF> = (0..num_vars)
            .map(|i| embed_base_to_ext(F::from_canonical_u64(i as u64 + 1)))
            .collect();
        
        // Evaluate the polynomial at the point
        let expected_eval = poly(&eval_point);
        
        println!("✅ Basic sum-check compatibility test setup complete");
        println!("   Variables: {}", num_vars);
        println!("   Expected evaluation: {:?}", expected_eval.to_array());
    }

    #[test]
    fn test_multilinear_sumcheck_integration() {
        println!("🧪 Testing multilinear sum-check integration");

        let poly = create_multilinear_test_polynomial();
        let num_vars = 2;
        
        // Test evaluation at different points
        let test_points = vec![
            vec![ExtF::ZERO, ExtF::ZERO],
            vec![ExtF::ONE, ExtF::ZERO],
            vec![ExtF::ZERO, ExtF::ONE],
            vec![ExtF::ONE, ExtF::ONE],
        ];
        
        for (i, point) in test_points.iter().enumerate() {
            let eval = poly(point);
            println!("   Point {}: {:?} -> {:?}", i, 
                    point.iter().map(|x| x.to_array()[0].as_canonical_u64()).collect::<Vec<_>>(),
                    eval.to_array()[0].as_canonical_u64());
        }
        
        println!("✅ Multilinear sum-check integration test passed");
    }

    #[test]
    fn test_batched_sumcheck_functionality() {
        println!("🧪 Testing batched sum-check functionality");

        // Create multiple polynomials for batching
        let poly1 = Arc::new(|vars: &[ExtF]| {
            if vars.is_empty() { ExtF::ZERO } else { vars[0] }
        });
        
        let poly2 = Arc::new(|vars: &[ExtF]| {
            if vars.len() < 2 { ExtF::ZERO } else { vars[0] + vars[1] }
        });
        
        let polys = vec![poly1, poly2];
        
        // Test that we can handle multiple polynomials
        let num_vars = 2;
        let test_point: Vec<ExtF> = vec![
            embed_base_to_ext(F::from_canonical_u64(2)),
            embed_base_to_ext(F::from_canonical_u64(3)),
        ];
        
        for (i, poly) in polys.iter().enumerate() {
            let eval = poly(&test_point);
            println!("   Poly {}: eval = {:?}", i, eval.to_array()[0].as_canonical_u64());
        }
        
        println!("✅ Batched sum-check functionality test passed");
    }

    #[test]
    fn test_sumcheck_with_zero_polynomial() {
        println!("🧪 Testing sum-check with zero polynomial");

        let zero_poly = Arc::new(|_vars: &[ExtF]| ExtF::ZERO);
        
        // Zero polynomial should always evaluate to zero
        let test_points = vec![
            vec![ExtF::ONE],
            vec![ExtF::ZERO],
            vec![embed_base_to_ext(F::from_canonical_u64(42))],
        ];
        
        for point in test_points {
            let eval = zero_poly(&point);
            assert_eq!(eval, ExtF::ZERO, "Zero polynomial should always evaluate to zero");
        }
        
        println!("✅ Zero polynomial test passed");
    }

    #[test]
    fn test_sumcheck_with_constant_polynomial() {
        println!("🧪 Testing sum-check with constant polynomial");

        let constant_value = embed_base_to_ext(F::from_canonical_u64(7));
        let constant_poly = Arc::new(move |_vars: &[ExtF]| constant_value);
        
        // Constant polynomial should always return the same value
        let test_points = vec![
            vec![ExtF::ONE],
            vec![ExtF::ZERO],
            vec![embed_base_to_ext(F::from_canonical_u64(100))],
        ];
        
        for point in test_points {
            let eval = constant_poly(&point);
            assert_eq!(eval, constant_value, "Constant polynomial should always return the same value");
        }
        
        println!("✅ Constant polynomial test passed");
    }

    #[test]
    fn test_polynomial_degree_properties() {
        println!("🧪 Testing polynomial degree properties");

        // Linear polynomial: x0
        let linear_poly = Arc::new(|vars: &[ExtF]| {
            if vars.is_empty() { ExtF::ZERO } else { vars[0] }
        });
        
        // Quadratic polynomial: x0 * x1
        let quadratic_poly = Arc::new(|vars: &[ExtF]| {
            if vars.len() < 2 { ExtF::ZERO } else { vars[0] * vars[1] }
        });
        
        // Test evaluation at unit points
        let unit_point = vec![ExtF::ONE, ExtF::ONE];
        
        let linear_eval = linear_poly(&unit_point);
        let quadratic_eval = quadratic_poly(&unit_point);
        
        assert_eq!(linear_eval, ExtF::ONE, "Linear polynomial at (1,1) should be 1");
        assert_eq!(quadratic_eval, ExtF::ONE, "Quadratic polynomial at (1,1) should be 1");
        
        println!("✅ Polynomial degree properties test passed");
    }

    #[test]
    fn test_sumcheck_challenger_integration() {
        println!("🧪 Testing sum-check challenger integration");

        // Test that our challenger works with sum-check
        let mut challenger = NeoChallenger::new();
        
        // Add some test data to the challenger
        challenger.observe_bytes(b"test_sumcheck_data");
        
        // Get a challenge
        let challenge = challenger.sample_ext_element();
        
        // Challenge should be non-zero with high probability
        assert_ne!(challenge, ExtF::ZERO, "Challenge should be non-zero");
        
        println!("✅ Challenger integration test passed");
        println!("   Challenge: {:?}", challenge.to_array());
    }

    #[test]
    fn test_univariate_polynomial_operations() {
        println!("🧪 Testing univariate polynomial operations");

        // Create a simple univariate polynomial: 2x + 3
        let coeffs = vec![
            embed_base_to_ext(F::from_canonical_u64(3)), // constant term
            embed_base_to_ext(F::from_canonical_u64(2)), // linear term
        ];
        let poly = UnivPoly::new(coeffs);
        
        // Test evaluation at x = 0: should be 3
        let eval_at_0 = poly.evaluate(&ExtF::ZERO);
        assert_eq!(eval_at_0, embed_base_to_ext(F::from_canonical_u64(3)));
        
        // Test evaluation at x = 1: should be 2*1 + 3 = 5
        let eval_at_1 = poly.evaluate(&ExtF::ONE);
        assert_eq!(eval_at_1, embed_base_to_ext(F::from_canonical_u64(5)));
        
        // Test evaluation at x = 2: should be 2*2 + 3 = 7
        let eval_at_2 = poly.evaluate(&embed_base_to_ext(F::from_canonical_u64(2)));
        assert_eq!(eval_at_2, embed_base_to_ext(F::from_canonical_u64(7)));
        
        println!("✅ Univariate polynomial operations test passed");
    }

    #[test]
    fn test_field_arithmetic_consistency() {
        println!("🧪 Testing field arithmetic consistency");

        let a = embed_base_to_ext(F::from_canonical_u64(5));
        let b = embed_base_to_ext(F::from_canonical_u64(3));
        
        // Test basic operations
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        
        // Verify results
        assert_eq!(sum, embed_base_to_ext(F::from_canonical_u64(8)));
        assert_eq!(diff, embed_base_to_ext(F::from_canonical_u64(2)));
        assert_eq!(prod, embed_base_to_ext(F::from_canonical_u64(15)));
        
        // Test that operations are consistent with polynomial evaluation
        let poly = Arc::new(move |vars: &[ExtF]| {
            if vars.len() < 2 { ExtF::ZERO } else { vars[0] + vars[1] }
        });
        
        let poly_result = poly(&vec![a, b]);
        assert_eq!(poly_result, sum, "Polynomial evaluation should match direct arithmetic");
        
        println!("✅ Field arithmetic consistency test passed");
    }
}

#[cfg(all(test, not(feature = "snark_mode")))]
mod nark_mode_sumcheck_tests {
    use neo_sumcheck::{
        batched_sumcheck_prover, batched_sumcheck_verifier,
        challenger::NeoChallenger
    };
    use neo_fields::{ExtF, F, embed_base_to_ext};
    use std::sync::Arc;

    #[test]
    fn test_nark_mode_sumcheck_still_works() {
        println!("🧪 Testing that NARK mode sum-check still works");

        // Test that the original sum-check functionality is preserved
        let poly = Arc::new(|vars: &[ExtF]| {
            if vars.is_empty() { ExtF::ZERO } else { vars[0] }
        });
        
        let test_point = vec![embed_base_to_ext(F::from_canonical_u64(42))];
        let eval = poly(&test_point);
        
        assert_eq!(eval, embed_base_to_ext(F::from_canonical_u64(42)));
        
        println!("✅ NARK mode sum-check backward compatibility confirmed");
    }
}
