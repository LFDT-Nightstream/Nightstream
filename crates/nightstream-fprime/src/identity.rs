//! Poseidon2 identity binding for the canonical Lean-emitted package.

use neo_ccs::crypto::poseidon2_goldilocks as poseidon2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use crate::package::PackageError;
const IDENTITY_DOMAIN: [u64; 29] = [
    78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47, 70, 80, 114, 105, 109, 101, 47, 112, 97, 99, 107, 97, 103,
    101, 47, 118, 50,
];

pub(super) fn relation_identifier(package: &Value) -> Result<[u64; 4], PackageError> {
    let mut input = IDENTITY_DOMAIN.map(Goldilocks::from_u64).to_vec();
    append_value_preimage(package, &mut input)?;
    Ok(poseidon2::poseidon2_hash(&input).map(|value| value.as_canonical_u64()))
}

fn append_identity_node(input: &mut Vec<Goldilocks>, tag: u64, value: u64) {
    input.push(Goldilocks::from_u64(tag));
    input.push(Goldilocks::from_u64(value & 0xffff_ffff));
    input.push(Goldilocks::from_u64(value >> 32));
    input.push(Goldilocks::ZERO);
}

fn append_value_preimage(value: &Value, input: &mut Vec<Goldilocks>) -> Result<(), PackageError> {
    match value {
        Value::Number(number) => {
            let value = number
                .as_u64()
                .ok_or(PackageError::Invalid("non-natural package atom"))?;
            append_identity_node(input, 0, value);
        }
        Value::Array(values) => {
            let length = u64::try_from(values.len()).map_err(|_| PackageError::Invalid("array length"))?;
            append_identity_node(input, 1, length);
            for child in values {
                append_value_preimage(child, input)?;
            }
        }
        _ => return Err(PackageError::Invalid("nonnumeric package value")),
    }
    Ok(())
}
