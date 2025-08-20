use neo_fields::{embed_base_to_ext, project_ext_to_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_embed_project_roundtrip() {
    let f = F::from_u64(42);
    let e = embed_base_to_ext(f);
    assert_eq!(project_ext_to_base(e), Some(f));

    // Non-projectable (imag != 0)
    let complex = ExtF::new_complex(f, F::ONE);
    assert_eq!(project_ext_to_base(complex), None);
}

#[test]
fn test_project_large_real_value() {
    // Test with a large real value that should still be projectable
    // since MAX_BLIND_NORM is now F::ORDER_U64 / 2, most values should be projectable
    let large_real = ExtF::new_complex(F::from_u64(1000000), F::ZERO);
    assert_eq!(project_ext_to_base(large_real), Some(F::from_u64(1000000)));
}
