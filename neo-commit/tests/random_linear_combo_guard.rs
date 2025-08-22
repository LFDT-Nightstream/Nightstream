use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::F;
use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;
use p3_field::PrimeCharacteristicRing;

#[test]
#[should_panic(expected = "random_linear_combo: rho outside Z_q representative range")]
fn random_linear_combo_rejects_rho_out_of_ring_range() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let zero = RingElement::from_scalar(ModInt::from_u64(0), params.n);
    let c1 = vec![zero.clone(); params.k];
    let c2 = vec![zero; params.k];

    // rho = ModInt::Q + 1 fits in F, but is outside Z_q representative range
    let rho = F::from_u64(<ModInt as Coeff>::modulus() + 1);
    let _ = comm.random_linear_combo(&c1, &c2, rho);
}

#[test]
fn random_linear_combo_accepts_rho_in_range() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let zero = RingElement::from_scalar(ModInt::from_u64(0), params.n);
    let one = RingElement::from_scalar(ModInt::from_u64(1), params.n);
    let c1 = vec![zero.clone(); params.k];
    let c2 = vec![one; params.k];

    // rho = ModInt::Q - 1 is the largest valid value
    let rho = F::from_u64(<ModInt as Coeff>::modulus() - 1);
    let result = comm.random_linear_combo(&c1, &c2, rho);
    
    // Should succeed without panic
    assert_eq!(result.len(), params.k);
}

#[test]
fn random_linear_combo_accepts_zero_rho() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let zero = RingElement::from_scalar(ModInt::from_u64(0), params.n);
    let one = RingElement::from_scalar(ModInt::from_u64(1), params.n);
    let c1 = vec![one.clone(); params.k];
    let c2 = vec![zero.clone(); params.k];

    let rho = F::ZERO;
    let result = comm.random_linear_combo(&c1, &c2, rho);
    
    // Should return c1 + c2 * 0 = c1
    assert_eq!(result.len(), params.k);
    for (res, expected) in result.iter().zip(&c1) {
        assert_eq!(res.coeffs(), expected.coeffs());
    }
}

#[test]
fn random_linear_combo_accepts_max_valid_rho() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let zero = RingElement::from_scalar(ModInt::from_u64(0), params.n);
    let c1 = vec![zero.clone(); params.k];
    let c2 = vec![zero; params.k];

    // Test the boundary: ModInt::Q - 1 should be accepted
    let max_valid_rho = F::from_u64(<ModInt as Coeff>::modulus() - 1);
    let result = comm.random_linear_combo(&c1, &c2, max_valid_rho);
    
    // Should succeed without panic
    assert_eq!(result.len(), params.k);
}
