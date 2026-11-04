//! Small checked readers shared by the package decoders.
//!
//! Each reader returns `PackageError` with the caller's location text. They
//! check only shape and range; the decoders own the meaning of each value.

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::PackageError;

pub(crate) fn array<'a>(value: &'a Value, location: &'static str) -> Result<&'a [Value], PackageError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or(PackageError::Malformed(location))
}

pub(crate) fn exact_array<'a>(
    value: &'a Value,
    length: usize,
    location: &'static str,
) -> Result<&'a [Value], PackageError> {
    let values = array(value, location)?;
    if values.len() != length {
        return Err(PackageError::Malformed(location));
    }
    Ok(values)
}

pub(crate) fn usize_atom(value: &Value, location: &'static str) -> Result<usize, PackageError> {
    word_to_usize(value.as_u64().ok_or(PackageError::Malformed(location))?, location)
}

pub(crate) fn word_to_usize(value: u64, location: &'static str) -> Result<usize, PackageError> {
    usize::try_from(value).map_err(|_| PackageError::Malformed(location))
}

pub(crate) fn canonical_field(value: u64, location: &'static str) -> Result<(), PackageError> {
    if value >= Goldilocks::ORDER_U64 {
        Err(PackageError::NonCanonicalField { location, value })
    } else {
        Ok(())
    }
}
