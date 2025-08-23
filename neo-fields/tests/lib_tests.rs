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
        let inv = val.inverse();
        assert_eq!(val * inv, F::ONE);
    }
}

#[test]
fn test_extf_ops() {
    let x = random_extf();
    let y = random_extf();
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
    let b = ExtF::new_imag(F::ONE); // This represents the primitive element α where α² = 7
    
    // Verify α² = 7 in our extension
    let alpha_squared = b * b;
    let expected = from_base(F::from_u64(7));
    assert_eq!(alpha_squared, expected);
}

#[test]
fn test_field_operations() {
    let a = F::from_u64(2);
    let b = F::from_u64(3);
    
    assert_eq!(a + b, F::from_u64(5));
    assert_eq!(a * b, F::from_u64(6));
    
    let ext_a = embed_base_to_ext(a);
    let ext_b = embed_base_to_ext(b);
    
    assert_eq!(ext_a + ext_b, embed_base_to_ext(F::from_u64(5)));
    assert_eq!(ext_a * ext_b, embed_base_to_ext(F::from_u64(6)));
}

#[test]
fn test_extension_field_operations() {
    let a = ExtF::new_complex(F::from_u64(2), F::from_u64(3));
    let b = ExtF::new_complex(F::from_u64(4), F::from_u64(5));
    
    let sum = a + b;
    let expected_sum = ExtF::new_complex(F::from_u64(6), F::from_u64(8));
    assert_eq!(sum, expected_sum);
    
    // Test norm computation
    assert!(a.abs_norm() > 0);
    assert!(a.abs_norm() <= MAX_BLIND_NORM);
}

#[test]
fn test_projection() {
    let base = F::from_u64(42);
    let ext = embed_base_to_ext(base);
    
    assert_eq!(project_ext_to_base(ext), Some(base));
    
    let non_base = ExtF::new_complex(F::from_u64(1), F::from_u64(2));
    assert_eq!(project_ext_to_base(non_base), None);
}
