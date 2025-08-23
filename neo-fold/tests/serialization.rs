use neo_fold::*;
use std::io::Cursor;
use neo_poly::Polynomial;
use neo_fields::ExtF;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_sumcheck_msgs_serialization_round_trip() {
    // Test the serialization/deserialization round-trip for sumcheck messages
    // This ensures our fix for the transcript format issue is robust
    
    // Create test polynomials with different degrees
    let poly1 = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE, ExtF::from_u64(42)]);
    let blind1 = ExtF::from_u64(123);
    
    let poly2 = Polynomial::new(vec![ExtF::from_u64(100), ExtF::from_u64(200)]);
    let blind2 = ExtF::from_u64(456);
    
    let poly3 = Polynomial::new(vec![ExtF::ZERO]); // Zero polynomial
    let blind3 = ExtF::ZERO;
    
    let original_msgs = vec![
        (poly1.clone(), blind1),
        (poly2.clone(), blind2),
        (poly3.clone(), blind3),
    ];
    
    // Serialize the messages
    let mut transcript = Vec::new();
    serialize_sumcheck_msgs(&mut transcript, &original_msgs);
    
    // Deserialize the messages
    let mut cursor = Cursor::new(&transcript[..]);
    let extracted_msgs = extract_msgs_ccs(&mut cursor, 3); // max_deg=3
    
    // Verify round-trip correctness
    assert_eq!(original_msgs.len(), extracted_msgs.len(), "Message count mismatch");
    
    for (i, ((orig_poly, orig_blind), (ext_poly, ext_blind))) in 
        original_msgs.iter().zip(extracted_msgs.iter()).enumerate() {
        
        // Check polynomial coefficients
        assert_eq!(orig_poly.coeffs(), ext_poly.coeffs(), 
                  "Polynomial coefficients mismatch for msg[{}]", i);
        
        // Check blind values
        assert_eq!(*orig_blind, *ext_blind, 
                  "Blind value mismatch for msg[{}]", i);
        
        // Check polynomial evaluations at test points
        for test_val in [ExtF::ZERO, ExtF::ONE, ExtF::from_u64(42)] {
            assert_eq!(orig_poly.eval(test_val), ext_poly.eval(test_val),
                      "Polynomial evaluation mismatch for msg[{}] at {:?}", i, test_val);
        }
    }
    
    // Verify cursor consumed all bytes
    assert_eq!(cursor.position() as usize, transcript.len(), 
              "Cursor should have consumed entire transcript");
}

#[test]
fn test_empty_sumcheck_msgs_serialization() {
    // Test edge case: empty message list
    let empty_msgs: Vec<(Polynomial<ExtF>, ExtF)> = vec![];
    
    let mut transcript = Vec::new();
    serialize_sumcheck_msgs(&mut transcript, &empty_msgs);
    
    let mut cursor = Cursor::new(&transcript[..]);
    let extracted_msgs = extract_msgs_ccs(&mut cursor, 0);
    
    assert!(extracted_msgs.is_empty(), "Empty messages should remain empty");
    assert_eq!(transcript.len(), 1, "Empty messages should serialize to 1 byte (length=0)");
}

#[test]
fn test_single_sumcheck_msg_serialization() {
    // Test single message (common case for small CCS instances)
    let poly = Polynomial::new(vec![ExtF::from_u64(1), ExtF::from_u64(2), ExtF::from_u64(3)]);
    let blind = ExtF::from_u64(999);
    let msgs = vec![(poly.clone(), blind)];
    
    let mut transcript = Vec::new();
    serialize_sumcheck_msgs(&mut transcript, &msgs);
    
    let mut cursor = Cursor::new(&transcript[..]);
    let extracted_msgs = extract_msgs_ccs(&mut cursor, 2);
    
    assert_eq!(extracted_msgs.len(), 1, "Should have exactly one message");
    assert_eq!(extracted_msgs[0].0.coeffs(), poly.coeffs(), "Polynomial should match");
    assert_eq!(extracted_msgs[0].1, blind, "Blind should match");
}

#[test]
fn test_fibonacci_expected_msg_count() {
    // Test that our expected message count calculation is correct for Fibonacci CCS
    use neo_arithmetize::fibonacci_ccs;
    
    // For Fibonacci length=3: 1 constraint, witness_size=3
    // Expected ell = max(log2(1), log2(3)) = max(0, 2) = 2
    let ccs = fibonacci_ccs(3);
    let l_constraints = (ccs.num_constraints as f64).log2().ceil() as usize;
    let l_witness = (ccs.witness_size as f64).log2().ceil() as usize;
    let ell = l_constraints.max(l_witness);
    
    assert_eq!(ell, 2, "Fibonacci length=3 should have ell=2");
    
    // For larger Fibonacci instances
    let ccs_large = fibonacci_ccs(1024);
    let l_constraints_large = (ccs_large.num_constraints as f64).log2().ceil() as usize;
    let l_witness_large = (ccs_large.witness_size as f64).log2().ceil() as usize;
    let ell_large = l_constraints_large.max(l_witness_large);
    
    assert_eq!(ell_large, 10, "Fibonacci length=1024 should have ell=10");
}
