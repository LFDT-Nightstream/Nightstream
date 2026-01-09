//! Opaque closure-proof envelope format.
//!
//! `ClosureProofV1::OpaqueBytes` carries a self-describing byte blob so we can evolve the
//! closure backend without changing the outer proof container type.

#![forbid(unsafe_code)]

use crate::ClosureProofError;

pub const BACKEND_ID_WHIR_P3_PLACEHOLDER_V1: u32 = 1;
pub const BACKEND_ID_EXPLICIT_OBLIGATION_CLOSURE_V1: u32 = 2;
/// WHIR-backed proof that (batched) Ajtai openings are correct (dev milestone; not full closure yet).
pub const BACKEND_ID_WHIR_P3_AJTAI_OPENING_ONLY_V1: u32 = 3;
/// WHIR-backed proof that (batched) Ajtai openings are correct AND Z projects to X (dev milestone; not full closure yet).
pub const BACKEND_ID_WHIR_P3_AJTAI_OPENING_PLUS_X_V1: u32 = 4;
/// WHIR-backed proof of full obligation closure (Ajtai opening + bounds + ME consistency).
pub const BACKEND_ID_WHIR_P3_FULL_CLOSURE_V1: u32 = 5;

const MAGIC: [u8; 4] = *b"NCLP";
const ENVELOPE_VERSION_V1: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 4 + 4;

/// Hard cap on opaque payload sizes accepted by the closure verifier.
///
/// This is a defensive limit against obvious DoS vectors; production closure proofs are expected
/// to be far smaller (low 100s KB).
pub const MAX_CLOSURE_PAYLOAD_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

pub fn encode_envelope(backend_id: u32, payload: &[u8]) -> Vec<u8> {
    let payload_len: u32 = payload
        .len()
        .try_into()
        .expect("encode_envelope: payload too large");
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&ENVELOPE_VERSION_V1.to_le_bytes());
    out.extend_from_slice(&backend_id.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn decode_envelope(bytes: &[u8]) -> Result<(u32, &[u8]), ClosureProofError> {
    if bytes.len() < HEADER_LEN {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    if bytes[0..4] != MAGIC {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != ENVELOPE_VERSION_V1 {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    let backend_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if payload_len > MAX_CLOSURE_PAYLOAD_BYTES {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    if bytes.len() != HEADER_LEN + payload_len {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    Ok((backend_id, &bytes[HEADER_LEN..]))
}
