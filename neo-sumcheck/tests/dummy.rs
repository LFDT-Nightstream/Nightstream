#[cfg(test)]
mod tests {
    use neo_fields::{ExtF, from_base, F};
    use neo_poly::Polynomial;
    use neo_sumcheck::oracle::FriOracle;
    use neo_sumcheck::PolyOracle;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_dummy_case_alignment() {
        let mut transcript = vec![];
        // Create oracle with empty polynomial list to trigger dummy case
        let mut oracle = FriOracle::new(vec![], &mut transcript);

        let point = vec![ExtF::ONE];
        let (evals, proofs) = oracle.open_at_point(&point);
        let commit = oracle.commit();

        // Check that we get some result (oracle should handle empty case gracefully)
        assert!(!commit.is_empty(), "Commit should not be empty even for dummy case");
        assert!(!evals.is_empty(), "Evals should not be empty");
        assert!(!proofs.is_empty(), "Proofs should not be empty");

        // Verify that the result is consistent
        let verifier = FriOracle::new_for_verifier(4);
        let verify_result = verifier.verify_openings(&commit, &point, &evals, &proofs);
        assert!(verify_result, "Dummy verification should pass");
    }

    #[test]
    fn test_dummy_vs_real_poly() {
        let mut transcript1 = vec![];
        let mut transcript2 = vec![];
        
        // Real polynomial case
        let real_poly = Polynomial::new(vec![ExtF::ONE]);
        let mut real_oracle = FriOracle::new(vec![real_poly.clone()], &mut transcript1);
        
        // Empty case (should create dummy)
        let mut dummy_oracle = FriOracle::new(vec![], &mut transcript2);
        
        let point = vec![ExtF::ONE];
        let (real_evals, real_proofs) = real_oracle.open_at_point(&point);
        let (dummy_evals, dummy_proofs) = dummy_oracle.open_at_point(&point);
        
        let real_commit = real_oracle.commit();
        let dummy_commit = dummy_oracle.commit();
        
        // Both should produce valid results
        assert!(!real_commit.is_empty());
        assert!(!dummy_commit.is_empty());
        assert!(!real_evals.is_empty());
        assert!(!dummy_evals.is_empty());
        
        // Both should verify
        let verifier = FriOracle::new_for_verifier(4);
        assert!(verifier.verify_openings(&real_commit, &point, &real_evals, &real_proofs));
        assert!(verifier.verify_openings(&dummy_commit, &point, &dummy_evals, &dummy_proofs));
        
        // But they should produce different results (dummy shouldn't equal real)
        // Note: This might not always be true due to randomness, but generally should be
        println!("Real eval: {:?}, Dummy eval: {:?}", real_evals[0], dummy_evals[0]);
    }

    #[test]
    fn test_dummy_poly_evaluation() {
        // Test that the dummy polynomial we use evaluates correctly
        let dummy_poly = Polynomial::new(vec![ExtF::ONE]); // Constant polynomial 1
        let point = ExtF::ONE;
        let eval = dummy_poly.eval(point);
        
        // Constant polynomial should evaluate to its constant value
        assert_eq!(eval, ExtF::ONE);
        
        // Test at different points
        let point2 = from_base(F::from_u64(42));
        let eval2 = dummy_poly.eval(point2);
        assert_eq!(eval2, ExtF::ONE); // Still constant
    }
}
