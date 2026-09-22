//! The prepared fixed envelope contains only arrays and unsigned field/index words.
//! Count nodes before constructing them; the ordinary JSON recursion check stays enabled.

use std::{fmt, io::Read};

use serde::de::{DeserializeSeed, Error, SeqAccess, Visitor};
use serde_json::Value;

use crate::PackageError;

const EXCEEDED: &str = "prepared fixed envelope exceeds compiler node bound";

struct ValueSeed<'a> {
    remaining: &'a mut usize,
}

impl<'de> DeserializeSeed<'de> for ValueSeed<'_> {
    type Value = Value;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        *self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or_else(|| D::Error::custom(EXCEEDED))?;
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for ValueSeed<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an unsigned integer or an array of prepared values")
    }

    fn visit_u64<E: Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut input: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = input.next_element_seed(ValueSeed {
            remaining: &mut *self.remaining,
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }
}

pub(super) fn decode(input: impl Read, node_limit: usize) -> Result<Value, PackageError> {
    let mut remaining = node_limit;
    let mut decoder = serde_json::Deserializer::from_reader(input);
    let value = ValueSeed {
        remaining: &mut remaining,
    }
    .deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(value)
}

pub(super) fn validate(value: &Value, node_limit: usize) -> Result<(), PackageError> {
    let mut remaining = node_limit;
    let mut stack = vec![std::slice::from_ref(value).iter()];
    while let Some(values) = stack.last_mut() {
        let Some(value) = values.next() else {
            stack.pop();
            continue;
        };
        remaining = remaining
            .checked_sub(1)
            .ok_or(PackageError::Invalid(EXCEEDED))?;
        match value {
            Value::Array(values) => stack.push(values.iter()),
            Value::Number(value) if value.as_u64().is_some() => {}
            _ => {
                return Err(PackageError::Invalid(
                    "prepared fixed envelope requires arrays and u64 values",
                ))
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/prepared_value.rs"]
mod tests;
