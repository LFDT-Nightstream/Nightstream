use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;

#[test]
fn test_norm_inf_constant_time() {
    let re = RingElement::from_coeffs(
        vec![
            ModInt::from_u64(1),
            ModInt::from_u64(<ModInt as Coeff>::modulus() - 1),
        ],
        2,
    );
    assert_eq!(re.norm_inf(), 1);
}
