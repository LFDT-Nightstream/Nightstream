use neo_fields::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
fn test_w_non_residue() {
    // Test that W = 7 is a quadratic non-residue in the Goldilocks field
    // A quadratic non-residue x satisfies x^((p-1)/2) ≡ -1 (mod p)
    let exp = (F::ORDER_U64 - 1) / 2;
    let w = F::from_u64(7);
    let legendre = w.exp_u64(exp);
    assert_eq!(legendre, F::NEG_ONE, "W=7 should be a quadratic non-residue");
}

#[test]
fn test_neg_one_is_residue() {
    // Test that -1 is a quadratic residue in the Goldilocks field (this is why it was problematic)
    let exp = (F::ORDER_U64 - 1) / 2;
    let neg_one = F::NEG_ONE;
    let legendre = neg_one.exp_u64(exp);
    assert_eq!(legendre, F::ONE, "-1 should be a quadratic residue (problematic for extension)");
}
