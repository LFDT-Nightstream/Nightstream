//! The circuit key digest against Lean
//! (`Lifecycle.BoundedDepth.CircuitKeyDigest.digest`, SPEC.md section 4.4).
//! Lean emits the digests of both circuits for three identities.

use p3_field::PrimeField64;
use serde_json::Value;

fn words(value: &Value) -> Vec<u64> {
    let words = value.as_array().expect("words").iter();
    words.map(|word| word.as_u64().expect("u64")).collect()
}

#[test]
fn circuit_key_is_the_lean_digest() {
    let file: Value = serde_json::from_str(include_str!(
        "../../../../formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json"
    ))
    .expect("Lean encoding file");
    assert_eq!(file[0].as_u64(), Some(1), "Lean encoding schema");
    let cases = file[1].as_array().expect("circuit keys");
    assert_eq!(cases.len(), 3);
    for case in cases {
        let identity: [u64; 4] = words(&case[0]).try_into().expect("an identity");
        for circuit in [0, 1] {
            let key = super::super::circuit_key(identity, circuit as u64);
            assert_eq!(key.map(|word| word.as_canonical_u64()).to_vec(), words(&case[1 + circuit]));
        }
    }
}
