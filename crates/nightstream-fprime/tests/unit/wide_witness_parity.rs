use super::*;
use crate::witness::{
    execute_witness_batch, validate_witness_batch, validate_witness_batch_order, validate_witness_coverage,
    RawWitnessBatch,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

#[test]
fn lean_wide_hints_match_native_execution() {
    let fixture: Value = serde_json::from_str(include_str!("../fixtures/pi-rlc-wide-witness-v1.json"))
        .expect("Lean wide witness fixture");
    assert_eq!(fixture[0], 1);
    let inputs = fixture[1].as_u64().unwrap() as usize;
    let private = fixture[2].as_u64().unwrap() as usize;
    assert_eq!((inputs, private), (4, 2021));
    let total = inputs + private;
    let raw: Vec<RawWitnessBatch> = serde_json::from_value(fixture[3].clone()).unwrap();
    let batches = raw
        .into_iter()
        .map(|batch| validate_witness_batch(batch, inputs, total, total, total))
        .collect::<Result<Vec<_>, _>>()
        .expect("causal witness batches");
    validate_witness_batch_order(&batches).unwrap();
    validate_witness_coverage(
        inputs,
        private,
        batches
            .iter()
            .map(|batch| (batch.start, batch.end()))
            .collect(),
    )
    .expect("all helper and retained fields have one producer");
    let cases = fixture[5].as_array().unwrap();
    assert_eq!(cases.len(), 11);
    for (index, case) in cases.iter().enumerate() {
        let mut values = vec![Goldilocks::ZERO; total];
        for (column, value) in case[0].as_array().unwrap().iter().enumerate() {
            let value = value.as_u64().unwrap();
            assert!(value < GOLDILOCKS_MODULUS);
            values[column] = Goldilocks::from_u64(value);
        }
        for batch in &batches {
            execute_witness_batch(batch, &mut values);
        }
        let expected: Vec<u64> = serde_json::from_value(case[1].clone()).unwrap();
        assert_eq!(
            values[inputs..]
                .iter()
                .map(PrimeField64::as_canonical_u64)
                .collect::<Vec<_>>(),
            expected,
            "case {index}: exact helper and retained values"
        );
        let mut coordinates = Vec::new();
        for block in fixture[4].as_array().unwrap() {
            let kind = SlotKind::decode(&block[0]).unwrap();
            for source in block[1].as_array().unwrap() {
                let source = source.as_u64().unwrap() as usize;
                assert!(
                    (1408..2025).contains(&source),
                    "temporary helper must not have a retained slot"
                );
                encode_slot(kind, values[source].as_canonical_u64(), &mut coordinates).unwrap();
            }
        }
        let expected: Vec<u64> = serde_json::from_value(case[2].clone()).unwrap();
        assert_eq!(coordinates.len(), 937);
        let actual = coordinates
            .iter()
            .map(|value| match value {
                -1 => GOLDILOCKS_MODULUS - 1,
                0 => 0,
                1 => 1,
                _ => panic!("coordinate is not strictly below b=2"),
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "case {index}: retained coordinate mapping");
    }
    println!(
        "wide witness parity: {} cases, {} batches, {private} private values, 937 retained coordinates",
        cases.len(),
        batches.len()
    );
}
