//! Opaque closure-proof envelope format.
//!
//! `ClosureProofV1::OpaqueBytes` carries a self-describing byte blob so we can evolve the
//! closure backend without changing the outer proof container type.

#![forbid(unsafe_code)]

use crate::ClosureProofError;

/// Backend IDs are part of the proof encoding contract.
///
/// Keep this list lean: only IDs that are supported in production should be routable by the
/// verifier. Unknown IDs are rejected as invalid encodings.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendIdV1 {
    /// Dev-only WHIR backend that serializes obligations in the payload.
    /// Not the intended production privacy/size profile.
    WhirP3FullClosureV1 = 5,
    /// Production-target WHIR backend with obligations kept private (no payload obligations).
    WhirP3PrivateFullClosureV1 = 6,
}

impl BackendIdV1 {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

impl TryFrom<u32> for BackendIdV1 {
    type Error = ClosureProofError;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        if v == Self::WhirP3PrivateFullClosureV1.as_u32() {
            return Ok(Self::WhirP3PrivateFullClosureV1);
        }
        if v == Self::WhirP3FullClosureV1.as_u32() {
            return Ok(Self::WhirP3FullClosureV1);
        }
        Err(ClosureProofError::InvalidOpaqueProofEncoding)
    }
}

const MAGIC: [u8; 4] = *b"NCLP";
const ENVELOPE_VERSION_V1: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 4 + 4;

/// Hard cap on opaque payload sizes accepted by the closure verifier.
///
/// This is a defensive limit against obvious DoS vectors; production closure proofs are expected
/// to be far smaller (low 100s KB).
pub const MAX_CLOSURE_PAYLOAD_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

pub fn encode_envelope(backend_id: u32, payload: &[u8]) -> Result<Vec<u8>, ClosureProofError> {
    if payload.len() > MAX_CLOSURE_PAYLOAD_BYTES {
        return Err(ClosureProofError::WhirP3(format!(
            "encode_envelope: payload too large ({} > {})",
            payload.len(),
            MAX_CLOSURE_PAYLOAD_BYTES
        )));
    }
    let payload_len: u32 = payload
        .len()
        .try_into()
        .map_err(|_| ClosureProofError::WhirP3("encode_envelope: payload too large".into()))?;
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&ENVELOPE_VERSION_V1.to_le_bytes());
    out.extend_from_slice(&backend_id.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

pub fn decode_envelope(bytes: &[u8]) -> Result<(u32, &[u8]), ClosureProofError> {
    if bytes.len() < HEADER_LEN {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    if bytes[0..4] != MAGIC {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    let version = u32::from_le_bytes(
        bytes[4..8]
            .try_into()
            .map_err(|_| ClosureProofError::InvalidOpaqueProofEncoding)?,
    );
    if version != ENVELOPE_VERSION_V1 {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    let backend_id = u32::from_le_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| ClosureProofError::InvalidOpaqueProofEncoding)?,
    );
    let payload_len = u32::from_le_bytes(
        bytes[12..16]
            .try_into()
            .map_err(|_| ClosureProofError::InvalidOpaqueProofEncoding)?,
    ) as usize;
    if payload_len > MAX_CLOSURE_PAYLOAD_BYTES {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    if bytes.len() != HEADER_LEN + payload_len {
        return Err(ClosureProofError::InvalidOpaqueProofEncoding);
    }
    Ok((backend_id, &bytes[HEADER_LEN..]))
}
