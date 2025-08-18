#[cfg(test)]
mod tests {
    use neo_fields::ExtF;
    use neo_poly::Polynomial;
    use neo_sumcheck::oracle::FriOracle;
    use neo_sumcheck::PolyOracle;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_blinding_consistency() {
        let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(2)]);
        let mut transcript = vec![];
        let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);

        let point = vec![ExtF::ONE];
        let (evals, proofs) = oracle.open_at_point(&point);
        let commit = oracle.commit();

        // Check eval includes blind
        let expected_blinded = poly.eval(point[0]) + oracle.blinds[0];
        assert_eq!(evals[0], expected_blinded, "Eval should include blind");

        // Verify subtracts blind correctly
        let verifier = FriOracle::new_for_verifier(4); // Small domain for test
        let unblinded_eval = evals[0] - oracle.blinds[0];
        let verify_result = verifier.verify_openings(&commit, &point, &[unblinded_eval], &proofs);
        assert!(verify_result, "Verification should pass after subtracting blind");

        // Mismatch case: tamper blind subtraction
        let bad_unblinded = evals[0] - (oracle.blinds[0] + ExtF::ONE);
        let bad_verify = verifier.verify_openings(&commit, &point, &[bad_unblinded], &proofs);
        assert!(!bad_verify, "Verification should fail on blind mismatch");
    }

    #[test]
    fn test_blinding_in_dummy_case() {
        let mut transcript = vec![];
        let mut oracle = FriOracle::new(vec![], &mut transcript); // Empty -> dummy

        let point = vec![ExtF::ONE];
        let (evals, proofs) = oracle.open_at_point(&point);
        let commit = oracle.commit();

        // Dummy should use non-zero poly + blind
        // Note: codewords field is private, but we can test that the oracle works
        assert!(!commit.is_empty(), "Dummy commit should be generated");
        let dummy_poly = Polynomial::new(vec![ExtF::ONE]); // Match your dummy
        let expected_blinded = dummy_poly.eval(point[0]) + oracle.blinds[0];
        assert_eq!(evals[0], expected_blinded, "Dummy eval should include blind");

        // Verify aligns after subtracting blind
        let verifier = FriOracle::new_for_verifier(4);
        let unblinded = evals[0] - oracle.blinds[0];
        let verify_result = verifier.verify_openings(&commit, &point, &[unblinded], &proofs);
        assert!(verify_result, "Dummy verification should pass");
    }
}
