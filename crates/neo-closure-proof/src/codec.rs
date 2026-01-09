//! Payload (de)serialization helpers for opaque closure-proof backends.
//!
//! These are used for `ClosureProofV1::OpaqueBytes` payloads after the envelope header has been
//! validated. The goal is to keep decoding rules consistent across backends and apply basic
//! input-size limits to reduce accidental DoS exposure.

#![forbid(unsafe_code)]

use bincode::Options;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::ClosureProofError;

fn bincode_opts() -> impl Options {
    // Match `bincode::{serialize, deserialize}` encoding (fixint, LE) but *reject* trailing bytes.
    // The limit is redundant with the envelope check but helps guard against accidental misuse.
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(crate::opaque::MAX_CLOSURE_PAYLOAD_BYTES as u64)
}

pub fn serialize_payload<T: Serialize>(value: &T) -> Result<Vec<u8>, ClosureProofError> {
    bincode_opts()
        .serialize(value)
        .map_err(|_| ClosureProofError::InvalidOpaqueProofEncoding)
}

pub fn deserialize_payload<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ClosureProofError> {
    bincode_opts()
        .deserialize(bytes)
        .map_err(|_| ClosureProofError::InvalidOpaqueProofEncoding)
}

