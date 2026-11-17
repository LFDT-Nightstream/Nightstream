//! Lean `Lifecycle.BoundedDepth.CircuitKeyDigest.digest` (SPEC.md section
//! 4.4), `Export.BoundedDepth.CompressionSponge.ccsDomain` (SPEC.md
//! section 5.4) and `Lifecycle.BoundedDepth.slotDigest` (SPEC.md Algorithm 10).

use p3_field::PrimeField64;
use serde_json::Value;

fn words(value: &Value) -> Vec<u64> {
    let words = value.as_array().expect("words").iter();
    words.map(|word| word.as_u64().expect("u64")).collect()
}

fn lean() -> Value {
    let file: Value = serde_json::from_str(include_str!(
        "../../../../formal/nightstream-fprime/artifacts/nightstream-fprime-encoding-parity-v1.json"
    ))
    .expect("Lean encoding file");
    assert_eq!(file[0].as_u64(), Some(1), "Lean encoding schema");
    file
}

#[test]
fn the_compression_domain_is_the_lean_domain() {
    let domain = super::super::BoundedDepth::compression_profile(&[]).domain;
    assert_eq!(
        domain.map(|word| word.as_canonical_u64()).to_vec(),
        words(&lean()[3][0][0])
    );
}

#[test]
fn circuit_key_is_the_lean_digest() {
    let file = lean();
    let cases = file[1].as_array().expect("circuit keys");
    assert_eq!(cases.len(), 3);
    for case in cases {
        let identity: [u64; 4] = words(&case[0]).try_into().expect("an identity");
        for circuit in [0, 1] {
            let key = super::super::circuit_key(identity, circuit as u64);
            assert_eq!(
                key.map(|word| word.as_canonical_u64()).to_vec(),
                words(&case[1 + circuit])
            );
        }
    }
}

/// SPEC.md Algorithm 10, step 6; Lean `Step.defaultSlot`, `slotDigest`.
#[test]
fn the_default_slot_has_the_lean_digest() {
    let prover = super::prover(2);
    let slot = super::super::words::slot(
        &prover
            .default_slot(1)
            .expect("default slot")
            .accumulator
            .claim,
        0,
    );
    assert_eq!(slot, nightstream_fprime::Slot::default_slot());
    assert_eq!(
        slot.digest().map(|word| word.as_canonical_u64()).to_vec(),
        words(&lean()[5])
    );
}
