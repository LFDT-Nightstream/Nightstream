//! neo-closure-proof: Phase-2 obligation closure proof container + backends.
//!
//! This crate defines:
//! - a stable closure-proof statement format (`ClosureStatementV1`),
//! - a proof container (`ClosureProofV1`), and
//! - backends, including:
//!   - a WHIR-based (Plonky3) backend that proves full obligation closure.
//!
//! NOTE: The obligations-private backend is not implemented yet.

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

mod bounded;
mod codec;
mod contract;
mod digest_binding;
mod encoded;
mod opaque;
mod whir_p3_backend;
mod whir_p3_private_backend;

pub use neo_fold::bridge_digests::compute_obligations_digest_v2;
pub use neo_fold::bridge_digests::{compute_accumulator_digest_v2, compute_obligations_digest_v1};
pub use digest_binding::{
    decode_obligations_digest_binding_shape_v1, prove_obligations_digest_binding_proof_v1,
    verify_obligations_digest_binding_proof_v1, DigestBindingShapeV1,
};

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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClosureProofV1 {
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
    #[error("spartan2 digest-binding error: {0}")]
    Spartan2(String),
    #[error("whir-p3 backend error: {0}")]
    WhirP3(String),
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
        ClosureProofV1::OpaqueBytes { proof_bytes } => {
            let (backend_id_u32, payload) = opaque::decode_envelope(proof_bytes)?;
            let backend_id = opaque::BackendIdV1::try_from(backend_id_u32)?;

            match backend_id {
                opaque::BackendIdV1::WhirP3FullClosureV1 => {
                    let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
                    let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
                    whir_p3_backend::verify_whir_p3_full_closure_payload_v1(stmt, payload, params, ccs, bus)
                }
                opaque::BackendIdV1::WhirP3PrivateFullClosureV1 => {
                    let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
                    let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
                    whir_p3_private_backend::verify_whir_p3_private_full_closure_payload_v1(stmt, payload, params, ccs, bus)
                }
            }
        }
    }
}

/// Production-oriented closure verification entrypoint.
///
/// This rejects the current dev WHIR backend (backend id `5`) and only routes to the
/// obligations-private backend (backend id `6`).
pub fn verify_closure_v1_production_with_context_and_bus(
    stmt: &ClosureStatementV1,
    proof: &ClosureProofV1,
    params: Option<&neo_params::NeoParams>,
    ccs: Option<&neo_ccs::CcsStructure<neo_math::F>>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(), ClosureProofError> {
    if stmt.version != CLOSURE_STATEMENT_V1 {
        return Err(ClosureProofError::UnsupportedStatementVersion(stmt.version));
    }

    let ClosureProofV1::OpaqueBytes { proof_bytes } = proof;
    let (backend_id_u32, payload) = opaque::decode_envelope(proof_bytes)?;
    let backend_id = opaque::BackendIdV1::try_from(backend_id_u32)?;

    match backend_id {
        opaque::BackendIdV1::WhirP3FullClosureV1 => Err(ClosureProofError::BackendNotImplemented),
        opaque::BackendIdV1::WhirP3PrivateFullClosureV1 => {
            let params = params.ok_or(ClosureProofError::MissingVerificationContext)?;
            let ccs = ccs.ok_or(ClosureProofError::MissingVerificationContext)?;
            whir_p3_private_backend::verify_whir_p3_private_full_closure_payload_v1(stmt, payload, params, ccs, bus)
        }
    }
}

/// Produce a WHIR-based **full** obligation-closure proof (dev milestone).
///
/// This backend proves:
/// - Ajtai commitment openings are correct (batched),
/// - the witness matrices `Z` project to the public `X`,
/// - ME consistency for core (and bus openings when provided),
/// - and a probabilistic Ajtai digit-range check for `Z` (boundedness).
pub fn prove_whir_p3_full_closure_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<ClosureProofV1, ClosureProofError> {
    let proof_bytes =
        whir_p3_backend::prove_whir_p3_full_closure_bytes_v1(stmt, params, ccs, obligations, main_wits, val_wits, bus)?;
    Ok(ClosureProofV1::OpaqueBytes { proof_bytes })
}

/// Produce a WHIR-based **obligations-private** obligation-closure proof (production target).
///
/// This backend proves the same closure predicate as [`prove_whir_p3_full_closure_v1`], but does
/// not serialize obligations in the payload. Instead, it includes a digest-binding proof that
/// binds private obligations to `stmt.obligations_digest`.
pub fn prove_whir_p3_private_full_closure_v1(
    stmt: &ClosureStatementV1,
    params: &neo_params::NeoParams,
    ccs: &neo_ccs::CcsStructure<neo_math::F>,
    obligations: &neo_fold::shard::ShardObligations<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    main_wits: &[neo_ccs::Mat<neo_math::F>],
    val_wits: &[neo_ccs::Mat<neo_math::F>],
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<ClosureProofV1, ClosureProofError> {
    let proof_bytes = whir_p3_private_backend::prove_whir_p3_private_full_closure_bytes_v1(
        stmt, params, ccs, obligations, main_wits, val_wits, bus,
    )?;
    Ok(ClosureProofV1::OpaqueBytes { proof_bytes })
}
