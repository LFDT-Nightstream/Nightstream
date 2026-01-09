//! neo-closure-proof: Phase-2 obligation closure proof container + backends.
//!
//! This crate defines:
//! - a stable closure-proof statement format (`ClosureStatementV1`),
//! - a proof container (`ClosureProofV1`), and
//! - multiple backends, including:
//!   - an explicit (non-succinct) oracle backend for correctness tests, and
//!   - WHIR-based backends (Plonky3) that prove increasingly strong closure predicates.
//!
//! NOTE: The WHIR full-closure backend is currently a dev milestone (not production-sized yet).
//! NOTE: WHIR backends that serialize obligations in the payload are gated behind the dev-only
//! feature `whir-p3-obligations-public`.

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

mod codec;
mod bounded;
mod contract;
mod encoded;
mod explicit_backend;
mod opaque;

#[cfg(feature = "whir-p3-backend")]
mod whir_p3_backend;

pub use neo_fold::bridge_digests::{compute_accumulator_digest_v2, compute_obligations_digest_v1};

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
/// WARNING: `TestOnlyDigest` is not a proof of closure. It is only meant for wiring tests and
/// should never be used as a production backend.
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
    #[error("invalid opaque closure proof encoding")]
    InvalidOpaqueProofEncoding,
    #[error("missing verification context for backend")]
    MissingVerificationContext,
    #[error("whir-p3 backend error: {0}")]
    WhirP3(String),
    #[error("explicit closure backend error: {0}")]
    Explicit(String),
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

/// Verify a closure proof against its statement (no extra context).
///
/// Backends that need extra context (e.g. `NeoParams`, CCS structure) must be verified via
/// [`verify_closure_v1_with_context`].
pub fn verify_closure_v1(stmt: &ClosureStatementV1, proof: &ClosureProofV1) -> Result<(), ClosureProofError> {
    verify_closure_v1_with_context(stmt, proof, None, None)
}

/// Verify a closure proof against its statement, providing optional context.
pub fn verify_closure_v1_with_context(
    stmt: &ClosureStatementV1,
    proof: &ClosureProofV1,
    params: Option<&neo_params::NeoParams>,
    ccs: Option<&neo_ccs::CcsStructure<neo_math::F>>,
) -> Result<(), ClosureProofError> {
    verify_closure_v1_with_context_and_bus(stmt, proof, params, ccs, None)
}

/// Verify a closure proof against its statement, providing optional context and bus layout.
pub fn verify_closure_v1_with_context_and_bus(
    stmt: &ClosureStatementV1,
    proof: &ClosureProofV1,
    params: Option<&neo_params::NeoParams>,
    ccs: Option<&neo_ccs::CcsStructure<neo_math::F>>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(), ClosureProofError> {
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
        ClosureProofV1::OpaqueBytes { proof_bytes } => {
            let (backend_id, payload) = opaque::decode_envelope(proof_bytes)?;
            match backend_id {
                opaque::BACKEND_ID_EXPLICIT_OBLIGATION_CLOSURE_V1 => {
                    let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
                    let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
                    explicit_backend::verify_explicit_obligation_closure_payload_v1(stmt, payload, params, ccs, bus)
                }
                opaque::BACKEND_ID_WHIR_P3_PLACEHOLDER_V1 => {
                    #[cfg(feature = "whir-p3-backend")]
                    {
                        whir_p3_backend::verify_whir_p3_placeholder_payload_v1(stmt, payload)
                    }
                    #[cfg(not(feature = "whir-p3-backend"))]
                    {
                        let _ = payload;
                        Err(ClosureProofError::BackendNotImplemented)
                    }
                }
                opaque::BACKEND_ID_WHIR_P3_AJTAI_OPENING_ONLY_V1 => {
                    #[cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
                    {
                        let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
                        let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
                        whir_p3_backend::verify_whir_p3_ajtai_opening_only_payload_v1(stmt, payload, params, ccs)
                    }
                    #[cfg(not(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public")))]
                    {
                        let _ = payload;
                        Err(ClosureProofError::BackendNotImplemented)
                    }
                }
                opaque::BACKEND_ID_WHIR_P3_AJTAI_OPENING_PLUS_X_V1 => {
                    #[cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
                    {
                        let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
                        let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
                        whir_p3_backend::verify_whir_p3_ajtai_opening_plus_x_payload_v1(stmt, payload, params, ccs)
                    }
                    #[cfg(not(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public")))]
                    {
                        let _ = payload;
                        Err(ClosureProofError::BackendNotImplemented)
                    }
                }
                opaque::BACKEND_ID_WHIR_P3_FULL_CLOSURE_V1 => {
                    #[cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
                    {
                        let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
                        let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
                        whir_p3_backend::verify_whir_p3_full_closure_payload_v1(stmt, payload, params, ccs, bus)
                    }
                    #[cfg(not(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public")))]
                    {
                        let _ = payload;
                        Err(ClosureProofError::BackendNotImplemented)
                    }
                }
                _ => Err(ClosureProofError::InvalidOpaqueProofEncoding),
            }
        }
    }
}

/// Produce a WHIR-based placeholder `OpaqueBytes` proof bound to `stmt`.
///
/// This is **not** the full obligation-closure proof yet; it is meant to prove out the chosen
/// backend (WHIR) and the statement-binding plumbing end-to-end.
#[cfg(feature = "whir-p3-backend")]
pub fn prove_whir_p3_placeholder_v1(stmt: &ClosureStatementV1) -> ClosureProofV1 {
    ClosureProofV1::OpaqueBytes {
        proof_bytes: whir_p3_backend::prove_whir_p3_placeholder_bytes_v1(stmt),
    }
}

/// Produce a WHIR-based *opening-only* closure proof.
///
/// This backend currently proves only that Ajtai commitments open correctly (batched).
/// It does not yet prove full obligation closure (bounds + ME consistency).
#[cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
pub fn prove_whir_p3_ajtai_opening_only_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
) -> Result<ClosureProofV1, ClosureProofError> {
    let proof_bytes = whir_p3_backend::prove_whir_p3_ajtai_opening_only_bytes_v1(
        stmt, params, ccs, obligations, main_wits, val_wits,
    )?;
    Ok(ClosureProofV1::OpaqueBytes { proof_bytes })
}

/// Produce a WHIR-based *opening + X-projection* closure proof.
///
/// This backend currently proves:
/// - Ajtai commitment openings are correct (batched), and
/// - the witness matrices `Z` project to the public `X` matrices for each obligation.
///
/// It still does not prove full obligation closure (bounds + ME consistency).
#[cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
pub fn prove_whir_p3_ajtai_opening_plus_x_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
) -> Result<ClosureProofV1, ClosureProofError> {
    let proof_bytes = whir_p3_backend::prove_whir_p3_ajtai_opening_plus_x_bytes_v1(
        stmt, params, ccs, obligations, main_wits, val_wits,
    )?;
    Ok(ClosureProofV1::OpaqueBytes { proof_bytes })
}

/// Produce a WHIR-based **full** obligation-closure proof (dev milestone).
///
/// This backend proves:
/// - Ajtai commitment openings are correct (batched),
/// - the witness matrices `Z` project to the public `X`,
/// - ME consistency for core (and bus openings when provided),
/// - and a probabilistic Ajtai digit-range check for `Z` (boundedness).
#[cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]
pub fn prove_whir_p3_full_closure_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<ClosureProofV1, ClosureProofError> {
    let proof_bytes = whir_p3_backend::prove_whir_p3_full_closure_bytes_v1(
        stmt, params, ccs, obligations, main_wits, val_wits, bus,
    )?;
    Ok(ClosureProofV1::OpaqueBytes { proof_bytes })
}

/// Produce an explicit (non-succinct) obligation-closure proof for tests/oracles.
///
/// This encodes the obligations and their witnesses into `OpaqueBytes` and verification recomputes:
/// - the obligations digest binding, and
/// - the full `ReferenceFinalizer` checks (Ajtai opening + bounds + ME y/y_scalars).
///
/// This is **not** production-sized; it is a correctness harness that locks the closure contract.
pub fn prove_explicit_obligation_closure_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    obligations: &neo_fold::shard::ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
) -> Result<ClosureProofV1, ClosureProofError> {
    let proof_bytes = explicit_backend::prove_explicit_obligation_closure_bytes_v1(
        stmt,
        params,
        obligations,
        main_wits,
        val_wits,
    )?;
    Ok(ClosureProofV1::OpaqueBytes { proof_bytes })
}
