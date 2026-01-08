//! neo-closure-proof: Phase-2 obligation closure proof types (skeleton).
//!
//! This crate intentionally starts as a **plumbing + serialization** layer:
//! - define a stable closure-proof statement format,
//! - define a proof container format,
//! - provide a tiny "test-only" placeholder proof to wire Phase-1 + Phase-2 together.
//!
//! A real closure-proof backend (transparent / PQ-friendly) is a separate task.

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

/// Closure-proof statement version.
pub const CLOSURE_STATEMENT_V1: u32 = 1;

/// Public statement for obligation closure (Phase 2).
///
/// This binds the closure proof to the same run context as the Phase-1 Spartan statement, and to
/// the exact obligations implied by that statement.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClosureStatementV1 {
    pub version: u32,
    pub context_digest: [u8; 32],
    pub pp_id_digest: [u8; 32],
    pub obligations_digest: [u8; 32],
}

impl ClosureStatementV1 {
    pub fn new(context_digest: [u8; 32], pp_id_digest: [u8; 32], obligations_digest: [u8; 32]) -> Self {
        Self {
            version: CLOSURE_STATEMENT_V1,
            context_digest,
            pp_id_digest,
            obligations_digest,
        }
    }
}

/// Closure proof container (Phase 2).
///
/// WARNING: `TestOnlyDigest` is not a proof of closure. It is only meant to make it easy to plumb
/// `BridgeProofV2` end-to-end while the real closure-proof backend is under construction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClosureProofV1 {
    /// Placeholder "proof" for wiring tests only.
    TestOnlyDigest { digest: [u8; 32] },
    /// Opaque proof bytes for a real backend (FRI/STARK, etc).
    OpaqueBytes { proof_bytes: Vec<u8> },
}

#[derive(Debug, thiserror::Error)]
pub enum ClosureProofError {
    #[error("unsupported closure statement version: {0}")]
    UnsupportedStatementVersion(u32),
    #[error("closure proof backend not implemented")]
    BackendNotImplemented,
    #[error("test-only closure proof mismatch")]
    TestOnlyDigestMismatch,
}

/// Produce a placeholder closure proof bound to `stmt`.
pub fn prove_test_only_v1(stmt: &ClosureStatementV1) -> ClosureProofV1 {
    let mut h = blake3::Hasher::new();
    h.update(b"neo/closure-proof/test-only/v1");
    h.update(&stmt.context_digest);
    h.update(&stmt.pp_id_digest);
    h.update(&stmt.obligations_digest);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(h.finalize().as_bytes());
    ClosureProofV1::TestOnlyDigest { digest }
}

/// Verify a closure proof against its statement.
///
/// Note: only the `TestOnlyDigest` placeholder is supported today.
pub fn verify_closure_v1(stmt: &ClosureStatementV1, proof: &ClosureProofV1) -> Result<(), ClosureProofError> {
    if stmt.version != CLOSURE_STATEMENT_V1 {
        return Err(ClosureProofError::UnsupportedStatementVersion(stmt.version));
    }

    match proof {
        ClosureProofV1::TestOnlyDigest { digest } => {
            let expected = match prove_test_only_v1(stmt) {
                ClosureProofV1::TestOnlyDigest { digest } => digest,
                _ => unreachable!("prove_test_only_v1 must return TestOnlyDigest"),
            };
            if *digest != expected {
                return Err(ClosureProofError::TestOnlyDigestMismatch);
            }
            Ok(())
        }
        ClosureProofV1::OpaqueBytes { .. } => Err(ClosureProofError::BackendNotImplemented),
    }
}

