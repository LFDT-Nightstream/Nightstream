//! Property tests for S-action K-vector linearity (DISABLED - see s_action_production_tests.rs)
//!
//! Tests the critical property: S(a*y + b*z) == a*S(y) + b*S(z) for K-vectors
//! This ensures the S-action properly extends from Fq-linearity to K-linearity
//!
//! NOTE: These property tests are disabled in favor of production-realistic tests
//! that use concrete values matching actual ME protocol usage patterns.

#[cfg(test)]
mod unit_tests {
    use neo_math::ring::cf_inv;
    use neo_math::{Fq, SAction, D, K};
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_simple_k_vector_linearity() {
        // Simple test with known values
        let mut coeffs = [Fq::ZERO; D];
        coeffs[0] = Fq::from_u64(2); // S-action multiplies by 2
        let s = SAction::from_ring(cf_inv(coeffs));

        let y = vec![K::new_complex(Fq::ONE, Fq::ZERO), K::new_complex(Fq::ZERO, Fq::ONE)];
        let z = vec![
            K::new_complex(Fq::from_u64(3), Fq::ZERO),
            K::new_complex(Fq::ZERO, Fq::from_u64(4)),
        ];

        let a = K::new_complex(Fq::from_u64(5), Fq::ZERO);
        let b = K::new_complex(Fq::from_u64(7), Fq::ZERO);

        // Compute a*y + b*z
        let mut ay_plus_bz = Vec::new();
        for i in 0..y.len() {
            ay_plus_bz.push(a * y[i] + b * z[i]);
        }

        // Apply S-action
        let s_combination = s.apply_k_vec(&ay_plus_bz).expect("S-action should work");

        // Apply S-action separately
        let s_y = s.apply_k_vec(&y).expect("S-action should work");
        let s_z = s.apply_k_vec(&z).expect("S-action should work");
        let mut a_sy_plus_b_sz = Vec::new();
        for i in 0..y.len() {
            a_sy_plus_b_sz.push(a * s_y[i] + b * s_z[i]);
        }

        // Should be equal
        assert_eq!(s_combination.len(), a_sy_plus_b_sz.len());
        for (lhs, rhs) in s_combination.iter().zip(a_sy_plus_b_sz.iter()) {
            assert_eq!(*lhs, *rhs);
        }
    }
}
