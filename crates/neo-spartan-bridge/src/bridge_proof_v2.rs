//! BridgeProofV2: a single shareable blob containing:
//! - the Phase-1 Spartan proof (verifier-equivalent up to obligations), and
//! - a Phase-2 closure proof (obligation finalization semantics).

use crate::api::{verify_fold_run, SpartanProof};
use crate::digests::compute_context_digest_v1;
use crate::error::{Result, SpartanBridgeError};
use crate::statement::SpartanShardStatement;
use neo_ajtai::Commitment;
use neo_ccs::CcsStructure;
use neo_closure_proof::{ClosureProofV1, ClosureStatementV1};
use neo_memory::witness::StepInstanceBundle;
use neo_params::NeoParams;
use neo_math::{F as NeoF, K as NeoK};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeProofV2 {
    pub spartan: SpartanProof,
    pub closure_stmt: ClosureStatementV1,
    pub closure: ClosureProofV1,
}

pub fn compute_closure_statement_v1(stmt: &SpartanShardStatement) -> ClosureStatementV1 {
    ClosureStatementV1::new(
        compute_context_digest_v1(stmt),
        stmt.pp_id_digest,
        stmt.obligations_digest,
    )
}

impl BridgeProofV2 {
    pub fn new(spartan: SpartanProof, closure: ClosureProofV1) -> Self {
        let closure_stmt = compute_closure_statement_v1(&spartan.statement);
        Self {
            spartan,
            closure_stmt,
            closure,
        }
    }

    pub fn expected_closure_statement(&self) -> ClosureStatementV1 {
        compute_closure_statement_v1(&self.spartan.statement)
    }
}

/// Verify a `BridgeProofV2`.
///
/// This verifies:
/// 1) the Phase-1 Spartan proof against the verifier context (same as `verify_fold_run`), and
/// 2) the Phase-2 closure proof against `ClosureStatementV1` deterministically derived from the
///    Spartan public statement.
pub fn verify_bridge_proof_v2(
    vk_spartan: &crate::api::SpartanVerifierKey,
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    vm_digest: &[u8; 32],
    steps_public: &[StepInstanceBundle<Commitment, NeoF, NeoK>],
    output_binding: Option<&neo_fold::output_binding::OutputBindingConfig>,
    step_linking: &[(usize, usize)],
    proof: &BridgeProofV2,
) -> Result<bool> {
    let ok = verify_fold_run(
        vk_spartan,
        params,
        ccs,
        vm_digest,
        steps_public,
        output_binding,
        step_linking,
        &proof.spartan,
    )?;
    if !ok {
        return Ok(false);
    }

    let expected = compute_closure_statement_v1(&proof.spartan.statement);
    if proof.closure_stmt != expected {
        return Err(SpartanBridgeError::VerificationError(
            "BridgeProofV2 closure statement mismatch".into(),
        ));
    }

    let bus = neo_fold::memory_sidecar::cpu_bus::try_infer_cpu_bus_layout_for_step_instances(ccs, steps_public)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("BusLayout: {e:?}")))?;
    neo_closure_proof::verify_closure_v1_with_context_and_bus(
        &proof.closure_stmt,
        &proof.closure,
        Some(params),
        Some(ccs),
        bus.as_ref(),
    )
        .map_err(|e| {
        SpartanBridgeError::VerificationError(format!("closure proof verification failed: {e}"))
    })?;

    Ok(true)
}
