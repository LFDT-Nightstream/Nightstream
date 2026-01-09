//! Explicit obligation-closure backend (non-succinct).
//!
//! This backend is meant for:
//! - end-to-end correctness tests,
//! - a reference oracle while the succinct backend is being built.
//!
//! It encodes:
//! - the full obligations (ME instances), and
//! - their witness matrices `Z`,
//! and verification runs `neo_fold::finalize::ReferenceFinalizer`.

#![forbid(unsafe_code)]

use std::sync::Arc;

use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::Mat;
use neo_fold::finalize::{ObligationFinalizer, ReferenceFinalizer};
use neo_fold::shard::ShardObligations;
use neo_math::{F as NeoF, K as NeoK};
use neo_params::NeoParams;

use crate::bounded::BoundedVec;
use crate::encoded::{EncodedMatF, EncodedObligations};
use crate::codec::{deserialize_payload, serialize_payload};
use crate::contract;
use crate::opaque;
use crate::{ClosureProofError, ClosureStatementV1};

const MAX_EXPLICIT_WIT_MATS: usize = 16_384;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ExplicitObligationClosurePayloadV1 {
    obligations: EncodedObligations,
    main_wits: BoundedVec<EncodedMatF, MAX_EXPLICIT_WIT_MATS>,
    val_wits: BoundedVec<EncodedMatF, MAX_EXPLICIT_WIT_MATS>,
}

pub fn prove_explicit_obligation_closure_bytes_v1(
    stmt: &ClosureStatementV1,
    params: &NeoParams,
    obligations: &ShardObligations<Cmt, NeoF, NeoK>,
    main_wits: &[Mat<NeoF>],
    val_wits: &[Mat<NeoF>],
) -> Result<Vec<u8>, ClosureProofError> {
    if obligations.main.len() != main_wits.len() || obligations.val.len() != val_wits.len() {
        return Err(ClosureProofError::Explicit(
            "witness count mismatch for explicit closure proof".into(),
        ));
    }

    let expected_digest = contract::expected_obligations_digest(params, obligations, stmt.pp_id_digest);
    if expected_digest != stmt.obligations_digest {
        return Err(ClosureProofError::Explicit(
            "explicit closure proof obligations_digest does not match statement".into(),
        ));
    }

    let payload = ExplicitObligationClosurePayloadV1 {
        obligations: EncodedObligations::encode(obligations),
        main_wits: main_wits
            .iter()
            .map(EncodedMatF::encode)
            .collect::<Vec<_>>()
            .into(),
        val_wits: val_wits
            .iter()
            .map(EncodedMatF::encode)
            .collect::<Vec<_>>()
            .into(),
    };
    let payload_bytes = serialize_payload(&payload)?;
    Ok(opaque::encode_envelope(
        opaque::BACKEND_ID_EXPLICIT_OBLIGATION_CLOSURE_V1,
        &payload_bytes,
    ))
}

pub fn verify_explicit_obligation_closure_payload_v1(
    stmt: &ClosureStatementV1,
    payload_bytes: &[u8],
    params: &NeoParams,
    ccs: &neo_ccs::CcsStructure<NeoF>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<(), ClosureProofError> {
    // Enforce that the loaded seeded PP matches the statement's pp_id_digest.
    let d = neo_math::D;
    let m = ccs.m;
    contract::require_global_pp_matches_statement(stmt.pp_id_digest, params, d, m)
        .map_err(ClosureProofError::Explicit)?;

    let payload: ExplicitObligationClosurePayloadV1 = deserialize_payload(payload_bytes)?;

    let obligations = payload
        .obligations
        .decode()
        .ok_or(ClosureProofError::InvalidOpaqueProofEncoding)?;
    let main_wits = payload
        .main_wits
        .iter()
        .map(|m| m.decode().ok_or(ClosureProofError::InvalidOpaqueProofEncoding))
        .collect::<Result<Vec<_>, _>>()?;
    let val_wits = payload
        .val_wits
        .iter()
        .map(|m| m.decode().ok_or(ClosureProofError::InvalidOpaqueProofEncoding))
        .collect::<Result<Vec<_>, _>>()?;

    let expected_digest = contract::expected_obligations_digest(params, &obligations, stmt.pp_id_digest);
    if expected_digest != stmt.obligations_digest {
        return Err(ClosureProofError::Explicit(
            "obligations_digest mismatch (proof not bound to Phase-1 obligations)".into(),
        ));
    }

    // Verify closure predicate via the reference oracle.
    let l = AjtaiSModule::from_global_for_dims(neo_math::D, ccs.m).map_err(|e| {
        ClosureProofError::Explicit(format!("AjtaiSModule missing for dims (d={}, m={}): {e:?}", neo_math::D, ccs.m))
    })?;
    let mut fin = ReferenceFinalizer::new_with_bus(params.clone(), Arc::new(ccs.clone()), l, main_wits, val_wits, bus.cloned())
        .map_err(|e| ClosureProofError::Explicit(format!("ReferenceFinalizer::new failed: {e:?}")))?;
    fin.finalize(&obligations)
        .map_err(|e| ClosureProofError::Explicit(format!("finalize failed: {e:?}")))?;
    Ok(())
}
