use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::F;
use neo_modint::{Coeff, ModInt};
use p3_field::PrimeCharacteristicRing;

#[test]
fn commit_noise_and_witness_within_configured_bounds_and_ring_modulus() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    // Simple non-zero z to exercise all paths
    let z: Vec<F> = vec![F::ONE; params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);

    let mut transcript = b"bounds".to_vec();
    let (_c, e, blinded_w, _r) = comm.commit(&w, &mut transcript).unwrap();

    // All coefficients are modulo the ring modulus
    for ei in &e {
        for &c in ei.coeffs() {
            assert!(c.as_canonical_u64() < <ModInt as Coeff>::modulus());
        }
    }
    for wi in &blinded_w {
        for &c in wi.coeffs() {
            assert!(c.as_canonical_u64() < <ModInt as Coeff>::modulus());
        }
    }

    // And norms meet configured bounds
    for ei in e { 
        assert!(ei.norm_inf() <= params.e_bound, "Noise bound exceeded: {} > {}", ei.norm_inf(), params.e_bound); 
    }
    let w_bound = params.norm_bound + params.beta;
    for wi in blinded_w { 
        assert!(wi.norm_inf() <= w_bound, "Witness bound exceeded: {} > {}", wi.norm_inf(), w_bound); 
    }
}

#[test]
fn modulus_consistency_across_operations() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);
    
    // Verify that TOY_PARAMS.q matches ModInt::Q
    assert_eq!(params.q, <ModInt as Coeff>::modulus(), 
        "Parameter q ({}) must match ModInt modulus ({})", 
        params.q, <ModInt as Coeff>::modulus());
    
    // Verify that all matrix elements are within the ring modulus
    for row in comm.public_matrix() {
        for elem in row {
            for &coeff in elem.coeffs() {
                assert!(coeff.as_canonical_u64() < <ModInt as Coeff>::modulus(),
                    "Matrix coefficient {} exceeds ring modulus {}", 
                    coeff.as_canonical_u64(), <ModInt as Coeff>::modulus());
            }
        }
    }
}
