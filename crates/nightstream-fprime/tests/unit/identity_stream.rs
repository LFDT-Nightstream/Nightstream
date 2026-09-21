use super::*;
use serde_json::json;

#[test]
fn streamed_identity_matches_the_original_numeric_array_preimage() {
    let value = json!([0, [7, u64::MAX], []]);
    let mut words = IDENTITY_DOMAIN.to_vec();
    words.extend([
        1,
        3,
        0,
        0, // outer array
        0,
        0,
        0,
        0, // zero
        1,
        2,
        0,
        0, // nested array
        0,
        7,
        0,
        0, // seven
        0,
        0xffff_ffff,
        0xffff_ffff,
        0, // both limbs of u64::MAX
        1,
        0,
        0,
        0, // empty array
    ]);
    let fields: Vec<_> = words.into_iter().map(Goldilocks::from_u64).collect();
    assert_eq!(
        relation_identifier(&value).unwrap(),
        poseidon2::poseidon2_hash(&fields).map(|value| value.as_canonical_u64())
    );
}

#[test]
fn streamed_identity_rejects_non_natural_atoms_and_non_array_nodes() {
    for value in [json!(-1), json!(0.5), json!(null), json!(true), json!("7"), json!({})] {
        assert!(relation_identifier(&json!([0, value])).is_err());
    }
}
