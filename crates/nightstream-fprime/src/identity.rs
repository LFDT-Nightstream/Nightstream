//! Poseidon2 identity binding for the canonical Lean-emitted package.

use neo_ccs::crypto::poseidon2;
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use serde_json::Value;

use crate::package::PackageError;

fn update_words(input: &mut poseidon2::Poseidon2Hasher, words: &[u64]) {
    let mut fields = [Fq::ZERO; poseidon2::RATE];
    for chunk in words.chunks(poseidon2::RATE) {
        for (field, &word) in fields.iter_mut().zip(chunk) {
            *field = Fq::from_u64(word);
        }
        input.update(&fields[..chunk.len()]);
    }
}

struct CanonicalSink<'a> {
    emit: &'a mut dyn FnMut(&[u64]) -> Result<(), PackageError>,
}

impl CanonicalSink<'_> {
    fn node(&mut self, tag: u64, value: u64) -> Result<(), PackageError> {
        (self.emit)(&[tag, value & 0xffff_ffff, value >> 32, 0])
    }

    fn number(&mut self, value: u64) -> Result<(), PackageError> {
        self.node(0, value)
    }

    fn array(&mut self, length: usize) -> Result<(), PackageError> {
        self.node(
            1,
            u64::try_from(length).map_err(|_| PackageError::Malformed("array length"))?,
        )
    }

    fn value(&mut self, value: &Value) -> Result<(), PackageError> {
        match value {
            Value::Number(number) => self.number(
                number
                    .as_u64()
                    .ok_or(PackageError::Malformed("non-natural package atom"))?,
            ),
            Value::Array(values) => {
                self.array(values.len())?;
                for child in values {
                    self.value(child)?;
                }
                Ok(())
            }
            _ => Err(PackageError::Malformed("nonnumeric package value")),
        }
    }
}

pub(crate) fn append_value_preimage(value: &Value, input: &mut poseidon2::Poseidon2Hasher) -> Result<(), PackageError> {
    CanonicalSink {
        emit: &mut |words| {
            update_words(input, words);
            Ok(())
        },
    }
    .value(value)
}
