#[cfg(test)]
mod tests {

    use neo_fields::*;
    use neo_poly::Polynomial;
    use std::io::Cursor;
    use byteorder::{BigEndian, ReadBytesExt};
    use p3_field::PrimeCharacteristicRing;
    use neo_sumcheck::sumcheck::{serialize_uni, serialize_ext};

    #[test]
    fn test_serialize_uni_includes_degree_prefix() {
        // Test with zero polynomial
        let zero_poly = Polynomial::new(vec![ExtF::ZERO]);
        let serialized = serialize_uni(&zero_poly);
        
        // Should start with degree as u8
        assert_eq!(serialized[0], zero_poly.degree() as u8);
        println!("Zero poly degree: {}, serialized[0]: {}", zero_poly.degree(), serialized[0]);
        
        // Should have 1 + (degree+1)*16 bytes total (degree + coefficients)
        let expected_len = 1 + (zero_poly.degree() + 1) * 16;
        assert_eq!(serialized.len(), expected_len);
    }

    #[test]
    fn test_serialize_uni_various_degrees() {
        // Test polynomials of different degrees
        let test_cases = vec![
            // Degree 0: constant polynomial
            vec![ExtF::from_u64(42)],
            // Degree 1: linear polynomial  
            vec![ExtF::from_u64(1), ExtF::from_u64(2)],
            // Degree 2: quadratic polynomial
            vec![ExtF::from_u64(1), ExtF::from_u64(2), ExtF::from_u64(3)],
        ];

        for (i, coeffs) in test_cases.iter().enumerate() {
            let poly = Polynomial::new(coeffs.clone());
            let serialized = serialize_uni(&poly);
            
            println!("Test case {}: degree={}, coeffs.len()={}", i, poly.degree(), coeffs.len());
            
            // Check degree prefix
            assert_eq!(serialized[0], poly.degree() as u8, "Degree prefix mismatch for case {}", i);
            
            // Check total length: 1 byte (degree) + (degree+1) * 16 bytes (coefficients)
            let expected_len = 1 + (poly.degree() + 1) * 16;
            assert_eq!(serialized.len(), expected_len, "Length mismatch for case {}", i);
            
            // Verify we can deserialize the coefficients correctly
            let mut cursor = Cursor::new(&serialized[1..]); // Skip degree byte
            for (j, expected_coeff) in coeffs.iter().enumerate() {
                let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
                let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
                let deserialized_coeff = ExtF::new_complex(real, imag);
                assert_eq!(deserialized_coeff, *expected_coeff, "Coefficient {} mismatch for case {}", j, i);
            }
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        // Create a test polynomial
        let coeffs = vec![
            ExtF::from_u64(123),
            ExtF::from_u64(456), 
            ExtF::from_u64(789)
        ];
        let original_poly = Polynomial::new(coeffs);
        
        // Serialize using our function
        let serialized = serialize_uni(&original_poly);
        
        // Deserialize manually (simulating extract_msgs_ccs logic)
        let mut cursor = Cursor::new(&serialized);
        let deg = cursor.read_u8().unwrap() as usize;
        
        assert_eq!(deg, original_poly.degree(), "Deserialized degree should match original");
        
        let mut deserialized_coeffs = Vec::new();
        for _ in 0..=deg {
            let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            deserialized_coeffs.push(ExtF::new_complex(real, imag));
        }
        
        let deserialized_poly = Polynomial::new(deserialized_coeffs);
        
        // Verify polynomials are equivalent by checking evaluations at multiple points
        let test_points = vec![ExtF::ZERO, ExtF::ONE, ExtF::from_u64(42), ExtF::from_u64(999)];
        for point in test_points {
            assert_eq!(
                original_poly.eval(point),
                deserialized_poly.eval(point),
                "Polynomial evaluation mismatch at point {:?}", point
            );
        }
    }

    #[test]
    fn test_serialize_ext_format() {
        let test_values = vec![
            ExtF::ZERO,
            ExtF::ONE,
            ExtF::from_u64(12345),
            ExtF::new_complex(F::from_u64(123), F::from_u64(456))
        ];

        for (i, value) in test_values.iter().enumerate() {
            let serialized = serialize_ext(*value);
            
            // Should be exactly 16 bytes (8 for real + 8 for imaginary)
            assert_eq!(serialized.len(), 16, "ExtF serialization should be 16 bytes for case {}", i);
            
            // Deserialize and verify
            let mut cursor = Cursor::new(&serialized);
            let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            let deserialized = ExtF::new_complex(real, imag);
            
            assert_eq!(deserialized, *value, "ExtF roundtrip failed for case {}", i);
        }
    }

    #[test]
    fn test_zero_polynomial_special_case() {
        // This is crucial for our zero polynomial handling
        let zero_poly = Polynomial::new(vec![ExtF::ZERO]);
        let serialized = serialize_uni(&zero_poly);
        
        println!("Zero polynomial:");
        println!("  Degree: {}", zero_poly.degree());
        println!("  Coeffs: {:?}", zero_poly.coeffs());
        println!("  Serialized length: {}", serialized.len());
        println!("  Degree prefix: {}", serialized[0]);
        
        // The degree should be 0
        assert_eq!(serialized[0], 0, "Zero polynomial should have degree 0");
        
        // Total length should be 1 + 1*16 = 17 bytes
        assert_eq!(serialized.len(), 17, "Zero polynomial serialization should be 17 bytes");
        
        // Verify the zero coefficient is correctly serialized
        let mut cursor = Cursor::new(&serialized[1..]);
        let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
        let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
        let coeff = ExtF::new_complex(real, imag);
        assert_eq!(coeff, ExtF::ZERO, "Zero polynomial coefficient should deserialize to zero");
    }

    #[test]
    fn test_compatibility_with_extract_msgs_ccs() {
        // This test simulates the exact flow from prover to verifier
        
        // Create a test polynomial (like what sumcheck prover would create)
        let coeffs = vec![ExtF::from_u64(100), ExtF::from_u64(200)];
        let poly = Polynomial::new(coeffs);
        let blind_eval = ExtF::from_u64(999);
        
        // Serialize the polynomial using our function
        let poly_serialized = serialize_uni(&poly);
        let blind_serialized = serialize_ext(blind_eval);
        
        // Create a mock transcript with one message
        let mut transcript = Vec::new();
        transcript.push(1u8); // Number of messages
        transcript.extend(&poly_serialized);
        transcript.extend(&blind_serialized);
        
        // Now deserialize using the same logic as extract_msgs_ccs
        let mut cursor = Cursor::new(&transcript[..]);
        let len = cursor.read_u8().unwrap() as usize;
        assert_eq!(len, 1, "Should have exactly one message");
        
        let deg = cursor.read_u8().unwrap() as usize;
        assert_eq!(deg, poly.degree(), "Degree should match");
        
        let mut deserialized_coeffs = Vec::new();
        for _ in 0..=deg {
            let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            deserialized_coeffs.push(ExtF::new_complex(real, imag));
        }
        
        let deserialized_poly = Polynomial::new(deserialized_coeffs);
        
        let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
        let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
        let deserialized_blind = ExtF::new_complex(real, imag);
        
        // Verify everything matches
        for test_point in [ExtF::ZERO, ExtF::ONE, ExtF::from_u64(42)] {
            assert_eq!(
                poly.eval(test_point),
                deserialized_poly.eval(test_point),
                "Polynomial evaluation should match at {:?}", test_point
            );
        }
        assert_eq!(blind_eval, deserialized_blind, "Blind eval should match");
        
        println!("✅ Compatibility test passed - serialization format works with extract_msgs_ccs");
    }

    #[test]
    fn test_transcript_consistency() {
        // This test ensures that our serialization changes don't break transcript building
        
        let poly1 = Polynomial::new(vec![ExtF::from_u64(1), ExtF::from_u64(2)]);
        let poly2 = Polynomial::new(vec![ExtF::ZERO]);
        
        // Build transcript the way the prover does
        let mut transcript1 = Vec::new();
        transcript1.extend(b"test_prefix");
        transcript1.extend(serialize_uni(&poly1));
        transcript1.extend(b"middle");
        transcript1.extend(serialize_uni(&poly2));
        transcript1.extend(b"test_suffix");
        
        // Build the same transcript again
        let mut transcript2 = Vec::new();
        transcript2.extend(b"test_prefix");
        transcript2.extend(serialize_uni(&poly1));
        transcript2.extend(b"middle");
        transcript2.extend(serialize_uni(&poly2));
        transcript2.extend(b"test_suffix");
        
        // They should be identical
        assert_eq!(transcript1, transcript2, "Transcript building should be deterministic");
        
        println!("✅ Transcript consistency test passed - {} bytes", transcript1.len());
    }

    #[test]
    fn test_prover_verifier_simulation() {
        // This is the most important test - it simulates the full prover-verifier flow
        
        // === PROVER SIDE ===
        let test_polynomial = Polynomial::new(vec![ExtF::from_u64(42), ExtF::from_u64(99)]);
        let test_blind = ExtF::from_u64(123);
        
        // Prover builds transcript
        let mut prover_transcript = Vec::new();
        prover_transcript.extend(b"sumcheck_start");
        prover_transcript.extend(serialize_uni(&test_polynomial)); // This is what we fixed
        prover_transcript.extend(b"sumcheck_end");
        
        // Prover creates message
        let mut message_data = Vec::new();
        message_data.push(1u8); // One message
        message_data.extend(serialize_uni(&test_polynomial));
        message_data.extend(serialize_ext(test_blind));
        
        // === VERIFIER SIDE ===
        // Verifier receives the message and extracts polynomial
        let mut cursor = Cursor::new(&message_data[..]);
        let len = cursor.read_u8().unwrap() as usize;
        assert_eq!(len, 1);
        
        let deg = cursor.read_u8().unwrap() as usize;
        let mut coeffs = Vec::new();
        for _ in 0..=deg {
            let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
            coeffs.push(ExtF::new_complex(real, imag));
        }
        let verifier_polynomial = Polynomial::new(coeffs);
        
        let real = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
        let imag = F::from_u64(cursor.read_u64::<BigEndian>().unwrap());
        let verifier_blind = ExtF::new_complex(real, imag);
        
        // Verifier rebuilds transcript (using pre-sumcheck snapshot approach)
        let mut verifier_transcript = Vec::new();
        verifier_transcript.extend(b"sumcheck_start");
        verifier_transcript.extend(serialize_uni(&verifier_polynomial)); // Should match prover
        verifier_transcript.extend(b"sumcheck_end");
        
        // === VERIFICATION ===
        assert_eq!(prover_transcript, verifier_transcript, "Transcripts should be identical");
        assert_eq!(test_blind, verifier_blind, "Blind values should match");
        
        // Most importantly - polynomial evaluations should match
        let test_points = vec![ExtF::ZERO, ExtF::ONE, ExtF::from_u64(999)];
        for point in test_points {
            assert_eq!(
                test_polynomial.eval(point),
                verifier_polynomial.eval(point),
                "Polynomial evaluations should match at {:?}", point
            );
        }
        
        println!("✅ Full prover-verifier simulation passed!");
        println!("   Transcript length: {} bytes", prover_transcript.len());
        println!("   Polynomial degree: {}", test_polynomial.degree());
    }
}
