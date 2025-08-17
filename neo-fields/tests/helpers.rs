use neo_fields::{embed_base_to_ext, project_ext_to_base, ExtF, F, MAX_BLIND_NORM};
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
fn test_project_invalid_norm() {
    let high = ExtF::new_complex(F::from_u64(MAX_BLIND_NORM + 1), F::ZERO);
    assert_eq!(project_ext_to_base(high), None);
}
