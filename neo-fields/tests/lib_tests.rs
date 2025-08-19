use neo_fields::*;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use rand::Rng;

#[test]
fn test_quadratic_non_residue() {
    // Verify that 7 is a non-residue in Goldilocks field (p = 2^64 - 2^32 + 1)
    // This ensures our extension x^2 - 7 is irreducible and forms a proper field
    let seven = F::from_u64(7);
    let p_minus_1_div_2 = (F::ORDER_U64 - 1) / 2;
    
    // Compute 7^((p-1)/2) mod p. If result is -1, then 7 is a non-residue
    let mut result = F::ONE;
    let mut base = seven;
    let mut exp = p_minus_1_div_2;
    
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    
    // Should be -1 (which is p-1 in the field)
    assert_eq!(result, F::from_u64(F::ORDER_U64 - 1));
    println!("✓ Verified: 7 is a quadratic non-residue in Goldilocks field");
}

#[test]
fn test_inverse_roundtrip() {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let mut val = F::from_u64(rng.random());
        if val == F::ZERO {
            val = F::ONE;
        }
        let inv = inverse(val);
        assert_eq!(val * inv, F::ONE);
    }
}

#[test]
fn test_extf_ops() {
    let mut rng = rand::rng();
    let x = random_extf(&mut rng);
    let y = random_extf(&mut rng);
    let z = x * y;
    assert_eq!(z * y.inverse(), x);
}

#[test]
fn test_extf_traits() {
    let x = from_base(F::from_u64(2));
    let y = from_base(F::from_u64(3));
    assert_eq!(x + y, from_base(F::from_u64(5)));
    assert_eq!(x.inverse() * x, ExtF::ONE);
}

#[test]
fn test_extension_field_structure() {
    // Test that our extension field has the correct structure
    let _a = from_base(F::from_u64(2));
    let b = from_base_pair(0, 1); // This represents the primitive element α where α² = 7
    
    // Verify α² = 7 in our extension
    let alpha_squared = b * b;
    let expected = from_base(F::from_u64(7));
    assert_eq!(alpha_squared, expected);
}
