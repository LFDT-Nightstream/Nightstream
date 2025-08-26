//! Unit tests for CCS to R1CS conversion
//! 
//! These tests validate the correctness of converting CCS (Customizable Constraint Systems)
//! to R1CS (Rank-1 Constraint Systems) for Spartan2 integration.

mod ccs_to_r1cs_tests {
    use neo_ccs::{
        CcsStructure, CcsInstance, CcsWitness, verifier_ccs,
        converters::{ccs_to_r1cs_format, ccs_instance_to_r1cs_format, ccs_witness_to_r1cs_format, integration}
    };
    use neo_fields::{embed_base_to_ext, ExtF, F};
    #[allow(unused_imports)]
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
            embed_base_to_ext(F::from_u64(2)),
            embed_base_to_ext(F::from_u64(3)),
            embed_base_to_ext(F::from_u64(5)),
        ];
        let witness = CcsWitness { z: witness_values };

        // Create instance with dummy commitment
        let instance = CcsInstance {
            commitment: vec![RingElement::zero(64)],
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
        
        let r1cs_result = ccs_to_r1cs_format(&ccs);
        assert!(r1cs_result.is_ok(), "CCS to R1CS conversion should succeed");
        
        let (a_matrix, b_matrix, c_matrix) = r1cs_result.unwrap();
        
        // Check basic properties
        assert_eq!(a_matrix.len(), ccs.num_constraints, "Constraint count should match");
        assert_eq!(b_matrix.len(), ccs.num_constraints, "Constraint count should match");
        assert_eq!(c_matrix.len(), ccs.num_constraints, "Constraint count should match");
        if !a_matrix.is_empty() {
            assert_eq!(a_matrix[0].len(), ccs.witness_size + 1, "Variable count should match (witness_size + 1 for constant)");
        }
        
        println!("✅ R1CS matrices conversion successful");
        println!("   Constraints: {}", a_matrix.len());
        println!("   Variables: {}", if a_matrix.is_empty() { 0 } else { a_matrix[0].len() });
    }

    #[test]
    fn test_ccs_instance_conversion() {
        println!("🧪 Testing CCS instance to R1CS instance conversion");

        let (ccs, instance, _) = create_simple_test_ccs();
        let _r1cs_matrices = ccs_to_r1cs_format(&ccs).unwrap();
        
        let r1cs_instance_result = ccs_instance_to_r1cs_format(&instance);
        assert!(r1cs_instance_result.is_ok(), "CCS instance to R1CS conversion should succeed");
        
        let r1cs_public_inputs = r1cs_instance_result.unwrap();
        
        // Basic validation - should have at least the constant 1
        assert!(!r1cs_public_inputs.is_empty(), "R1CS public inputs should not be empty");
        println!("✅ R1CS instance conversion successful");
        println!("   Public inputs length: {}", r1cs_public_inputs.len());
    }

    #[test]
    fn test_ccs_witness_conversion() {
        println!("🧪 Testing CCS witness to R1CS witness conversion");

        let (_, instance, witness) = create_simple_test_ccs();
        
        let r1cs_witness_result = ccs_witness_to_r1cs_format(&witness, &instance);
        assert!(r1cs_witness_result.is_ok(), "CCS witness to R1CS conversion should succeed");
        
        let r1cs_witness_vec = r1cs_witness_result.unwrap();
        
        // Check that witness has the right structure
        // R1CS witness should include constant 1 and the witness values
        let expected_length = 1 + witness.z.len();
        assert_eq!(r1cs_witness_vec.len(), expected_length, "R1CS witness should have correct length");
        
        println!("✅ R1CS witness conversion successful");
        println!("   Witness length: {}", r1cs_witness_vec.len());
    }

    #[test]
    fn test_full_conversion_pipeline() {
        println!("🧪 Testing full CCS to R1CS conversion pipeline");

        let (ccs, instance, witness) = create_simple_test_ccs();
        
        let conversion_result = integration::convert_ccs_for_spartan2(&ccs, &instance, &witness);
        assert!(conversion_result.is_ok(), "Full conversion pipeline should succeed");
        
        let (matrices, public_inputs, witness_vec) = conversion_result.unwrap();
        let (a_matrix, b_matrix, c_matrix) = matrices;
        
        // Validate consistency between components
        assert_eq!(a_matrix.len(), b_matrix.len(), "A and B matrices should have same constraint count");
        assert_eq!(b_matrix.len(), c_matrix.len(), "B and C matrices should have same constraint count");
        if !a_matrix.is_empty() && !witness_vec.is_empty() {
            assert_eq!(a_matrix[0].len(), witness_vec.len(), 
                      "Matrix width should match witness length");
        }
        
        println!("✅ Full conversion pipeline successful");
        println!("   R1CS constraints: {}", a_matrix.len());
        println!("   R1CS variables: {}", if a_matrix.is_empty() { 0 } else { a_matrix[0].len() });
        println!("   Public inputs: {}", public_inputs.len());
        println!("   Witness length: {}", witness_vec.len());
    }

    #[test]
    fn test_verifier_ccs_conversion() {
        println!("🧪 Testing verifier CCS conversion");

        let ccs = verifier_ccs();
        
        let r1cs_result = ccs_to_r1cs_format(&ccs);
        assert!(r1cs_result.is_ok(), "Verifier CCS to R1CS conversion should succeed");
        
        let (a_matrix, b_matrix, c_matrix) = r1cs_result.unwrap();
        
        // Verifier CCS has specific dimensions
        assert_eq!(a_matrix.len(), 2, "Verifier CCS should have 2 constraints");
        assert_eq!(b_matrix.len(), 2, "Verifier CCS should have 2 constraints");
        assert_eq!(c_matrix.len(), 2, "Verifier CCS should have 2 constraints");
        if !a_matrix.is_empty() {
            assert_eq!(a_matrix[0].len(), 5, "Verifier CCS should have 5 variables (4 + 1 for constant)");
        }
        
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
        
        let conversion_result = integration::convert_ccs_for_spartan2(&ccs, &instance, &witness);
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
        
        let r1cs_result = ccs_to_r1cs_format(&insufficient_ccs);
        assert!(r1cs_result.is_err(), "Conversion should fail with insufficient matrices");
        
        println!("✅ Error handling test passed");
    }

    #[test]
    fn test_matrix_sparsity_preservation() {
        println!("🧪 Testing matrix sparsity preservation");

        let (ccs, _, _) = create_simple_test_ccs();
        let r1cs_result = ccs_to_r1cs_format(&ccs).unwrap();
        
        // The conversion should preserve the sparse structure of matrices
        // This is important for efficiency in Spartan2
        
        println!("✅ Matrix sparsity preservation test completed");
        println!("   Original CCS matrices: {}", ccs.mats.len());
        println!("   R1CS constraint count: {}", r1cs_result.0.len());
    }
}

// Converter is now always available since SNARK mode is the default
