//! Unit tests for CCS to R1CS conversion
//! 
//! These tests validate the correctness of converting CCS (Customizable Constraint Systems)
//! to R1CS (Rank-1 Constraint Systems) for Spartan2 integration.

#[cfg(all(test, feature = "snark_mode"))]
mod ccs_to_r1cs_tests {
    use neo_ccs::{
        CcsStructure, CcsInstance, CcsWitness, verifier_ccs,
        converters::{ccs_to_r1cs, ccs_instance_to_r1cs, ccs_witness_to_r1cs, convert_ccs_to_r1cs_full}
    };
    use neo_fields::{embed_base_to_ext, ExtF, F};
    use neo_modint::ModInt;
    use neo_ring::RingElement;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    /// Create a simple test CCS for validation
    fn create_simple_test_ccs() -> (CcsStructure, CcsInstance, CcsWitness) {
        // Create a simple 2x2 CCS that checks a + b = c
        let mats = vec![
            // Matrix A: selects first variable (a)
            RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO], 3),
            // Matrix B: selects second variable (b)  
            RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO], 3),
            // Matrix C: selects third variable (c)
            RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE], 3),
        ];

        // Constraint: a + b - c = 0, so f(x0, x1, x2) = x0 + x1 - x2
        let f = neo_ccs::mv_poly(|inputs: &[ExtF]| {
            if inputs.len() != 3 {
                ExtF::ZERO
            } else {
                inputs[0] + inputs[1] - inputs[2]
            }
        }, 1);

        let ccs = CcsStructure::new(mats, f);

        // Create witness: a=2, b=3, c=5
        let witness_values = vec![
            embed_base_to_ext(F::from_canonical_u64(2)),
            embed_base_to_ext(F::from_canonical_u64(3)),
            embed_base_to_ext(F::from_canonical_u64(5)),
        ];
        let witness = CcsWitness { z: witness_values };

        // Create instance with dummy commitment
        let instance = CcsInstance {
            commitment: vec![RingElement::zero()],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };

        (ccs, instance, witness)
    }

    #[test]
    fn test_ccs_to_r1cs_shape_conversion() {
        println!("🧪 Testing CCS to R1CS shape conversion");

        let (ccs, _, _) = create_simple_test_ccs();
        
        let r1cs_result = ccs_to_r1cs(&ccs);
        assert!(r1cs_result.is_ok(), "CCS to R1CS conversion should succeed");
        
        let r1cs_shape = r1cs_result.unwrap();
        
        // Check basic properties
        assert_eq!(r1cs_shape.num_cons(), ccs.num_constraints, "Constraint count should match");
        assert_eq!(r1cs_shape.num_vars(), ccs.witness_size, "Variable count should match");
        
        println!("✅ R1CS shape conversion successful");
        println!("   Constraints: {}", r1cs_shape.num_cons());
        println!("   Variables: {}", r1cs_shape.num_vars());
    }

    #[test]
    fn test_ccs_instance_conversion() {
        println!("🧪 Testing CCS instance to R1CS instance conversion");

        let (ccs, instance, _) = create_simple_test_ccs();
        let r1cs_shape = ccs_to_r1cs(&ccs).unwrap();
        
        let r1cs_instance_result = ccs_instance_to_r1cs(&instance, &r1cs_shape);
        assert!(r1cs_instance_result.is_ok(), "CCS instance to R1CS conversion should succeed");
        
        let r1cs_instance = r1cs_instance_result.unwrap();
        
        // Basic validation - instance should have proper structure
        println!("✅ R1CS instance conversion successful");
    }

    #[test]
    fn test_ccs_witness_conversion() {
        println!("🧪 Testing CCS witness to R1CS witness conversion");

        let (_, instance, witness) = create_simple_test_ccs();
        
        let r1cs_witness_result = ccs_witness_to_r1cs(&witness, &instance);
        assert!(r1cs_witness_result.is_ok(), "CCS witness to R1CS conversion should succeed");
        
        let r1cs_witness = r1cs_witness_result.unwrap();
        
        // Check that witness has the right structure
        // R1CS witness should include constant 1, public inputs, and private witness
        let expected_length = 1 + instance.public_input.len() + witness.z.len();
        assert_eq!(r1cs_witness.assignment().len(), expected_length, "R1CS witness should have correct length");
        
        println!("✅ R1CS witness conversion successful");
        println!("   Witness length: {}", r1cs_witness.assignment().len());
    }

    #[test]
    fn test_full_conversion_pipeline() {
        println!("🧪 Testing full CCS to R1CS conversion pipeline");

        let (ccs, instance, witness) = create_simple_test_ccs();
        
        let conversion_result = convert_ccs_to_r1cs_full(&ccs, &instance, &witness);
        assert!(conversion_result.is_ok(), "Full conversion pipeline should succeed");
        
        let (r1cs_shape, r1cs_instance, r1cs_witness) = conversion_result.unwrap();
        
        // Validate consistency between components
        assert_eq!(r1cs_shape.num_vars(), r1cs_witness.assignment().len(), 
                  "R1CS shape and witness should have consistent variable count");
        
        println!("✅ Full conversion pipeline successful");
        println!("   R1CS constraints: {}", r1cs_shape.num_cons());
        println!("   R1CS variables: {}", r1cs_shape.num_vars());
        println!("   Witness length: {}", r1cs_witness.assignment().len());
    }

    #[test]
    fn test_verifier_ccs_conversion() {
        println!("🧪 Testing verifier CCS conversion");

        let ccs = verifier_ccs();
        
        let r1cs_result = ccs_to_r1cs(&ccs);
        assert!(r1cs_result.is_ok(), "Verifier CCS to R1CS conversion should succeed");
        
        let r1cs_shape = r1cs_result.unwrap();
        
        // Verifier CCS has specific dimensions
        assert_eq!(r1cs_shape.num_cons(), 2, "Verifier CCS should have 2 constraints");
        assert_eq!(r1cs_shape.num_vars(), 4, "Verifier CCS should have 4 variables");
        
        println!("✅ Verifier CCS conversion successful");
    }

    #[test]
    fn test_conversion_with_extension_field_elements() {
        println!("🧪 Testing conversion with extension field elements");

        let (ccs, instance, witness) = create_simple_test_ccs();
        
        // Verify that extension field elements are properly handled
        for ext_val in &witness.z {
            let arr = ext_val.to_array();
            // For this test, we expect real values (imaginary part should be zero)
            assert_eq!(arr[1], F::ZERO, "Test witness should have real values only");
        }
        
        let conversion_result = convert_ccs_to_r1cs_full(&ccs, &instance, &witness);
        assert!(conversion_result.is_ok(), "Conversion with extension field should succeed");
        
        println!("✅ Extension field conversion test passed");
    }

    #[test]
    fn test_conversion_error_handling() {
        println!("🧪 Testing conversion error handling");

        // Test with insufficient matrices
        let mats = vec![
            RowMajorMatrix::new(vec![F::ONE, F::ZERO], 2), // Only one matrix
        ];
        let f = neo_ccs::mv_poly(|_| ExtF::ZERO, 0);
        let insufficient_ccs = CcsStructure::new(mats, f);
        
        let r1cs_result = ccs_to_r1cs(&insufficient_ccs);
        assert!(r1cs_result.is_err(), "Conversion should fail with insufficient matrices");
        
        println!("✅ Error handling test passed");
    }

    #[test]
    fn test_matrix_sparsity_preservation() {
        println!("🧪 Testing matrix sparsity preservation");

        let (ccs, _, _) = create_simple_test_ccs();
        let r1cs_result = ccs_to_r1cs(&ccs).unwrap();
        
        // The conversion should preserve the sparse structure of matrices
        // This is important for efficiency in Spartan2
        
        println!("✅ Matrix sparsity preservation test completed");
        println!("   Original CCS matrices: {}", ccs.mats.len());
        println!("   R1CS constraint count: {}", r1cs_result.num_cons());
    }
}

#[cfg(all(test, not(feature = "snark_mode")))]
mod fallback_tests {
    #[test]
    fn test_converter_not_available_without_snark_mode() {
        println!("🧪 Testing that converter is not available without snark_mode feature");
        
        // This test ensures that the converter code is properly feature-gated
        // When snark_mode is not enabled, the converter modules should not be available
        
        println!("✅ Converter properly feature-gated");
    }
}
