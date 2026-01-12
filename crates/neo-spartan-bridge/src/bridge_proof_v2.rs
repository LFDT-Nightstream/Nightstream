//! BridgeProofV2: a single shareable blob containing:
//! - the Phase-1 Spartan proof (verifier-equivalent up to obligations), and
//! - a Phase-2 closure proof (obligation finalization semantics).

use crate::api::{prove_fold_run, verify_fold_run, verify_fold_run_statement_only, SpartanProof};
use crate::circuit::FoldRunWitness;
use crate::digests::compute_context_digest_v1;
use crate::error::{Result, SpartanBridgeError};
use crate::statement::SpartanShardStatement;
use bincode::Options;
use neo_ajtai::Commitment;
use neo_ccs::CcsStructure;
use neo_ccs::Mat;
use neo_closure_proof::{ClosureProofV1, ClosureStatementV1};
use neo_math::{F as NeoF, K as NeoK};
use neo_memory::witness::StepInstanceBundle;
use neo_params::NeoParams;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeProofV2 {
    pub spartan: SpartanProof,
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
        Self { spartan, closure }
    }

    pub fn closure_statement(&self) -> ClosureStatementV1 {
        compute_closure_statement_v1(&self.spartan.statement)
    }
}

const MAX_BRIDGE_PROOF_V2_BYTES: u64 = 128 * 1024 * 1024; // 128 MiB

fn bridge_proof_bincode_opts() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_BRIDGE_PROOF_V2_BYTES)
}

/// Serialize a `BridgeProofV2` into a single blob, with fixed options and explicit size bounds.
pub fn serialize_bridge_proof_v2(proof: &BridgeProofV2) -> Result<Vec<u8>> {
    bridge_proof_bincode_opts()
        .serialize(proof)
        .map_err(|e| SpartanBridgeError::ProvingError(format!("BridgeProofV2 serialize failed: {e}")))
}

/// Deserialize a `BridgeProofV2` blob with explicit size bounds and trailing-bytes rejection.
pub fn deserialize_bridge_proof_v2(bytes: &[u8]) -> Result<BridgeProofV2> {
    if bytes.len() > MAX_BRIDGE_PROOF_V2_BYTES as usize {
        return Err(SpartanBridgeError::VerificationError(format!(
            "BridgeProofV2 too large: len={} exceeds MAX_BRIDGE_PROOF_V2_BYTES={MAX_BRIDGE_PROOF_V2_BYTES}",
            bytes.len()
        )));
    }
    bridge_proof_bincode_opts()
        .deserialize(bytes)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("BridgeProofV2 deserialize failed: {e}")))
}

/// Prove a `BridgeProofV2` using the WHIR full-closure backend.
pub fn prove_bridge_proof_v2_whir_p3_full_closure(
    pk_spartan: &crate::api::SpartanProverKey,
    params: &NeoParams,
    ccs: &CcsStructure<NeoF>,
    witness: FoldRunWitness,
    main_wits: &[Mat<NeoF>],
    val_wits: &[Mat<NeoF>],
) -> Result<BridgeProofV2> {
    let obligations = witness
        .fold_run
        .compute_fold_outputs(&witness.initial_accumulator)
        .obligations;
    let steps_public = witness.steps_public.clone();
    let spartan = prove_fold_run(pk_spartan, params, ccs, witness)?;
    let closure_stmt = compute_closure_statement_v1(&spartan.statement);
    let bus = neo_fold::memory_sidecar::cpu_bus::try_infer_cpu_bus_layout_for_step_instances(ccs, &steps_public)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("BusLayout: {e:?}")))?;
    let closure = neo_closure_proof::prove_whir_p3_full_closure_v1(
        &closure_stmt,
        params,
        ccs,
        &obligations,
        main_wits,
        val_wits,
        bus.as_ref(),
    )
    .map_err(|e| SpartanBridgeError::ProvingError(format!("closure prove (whir full-closure) failed: {e}")))?;
    Ok(BridgeProofV2::new(spartan, closure))
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

    let closure_stmt = compute_closure_statement_v1(&proof.spartan.statement);

    let bus = neo_fold::memory_sidecar::cpu_bus::try_infer_cpu_bus_layout_for_step_instances(ccs, steps_public)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("BusLayout: {e:?}")))?;
    neo_closure_proof::verify_closure_v1_with_context_and_bus(
        &closure_stmt,
        &proof.closure,
        Some(params),
        Some(ccs),
        bus.as_ref(),
    )
    .map_err(|e| SpartanBridgeError::VerificationError(format!("closure proof verification failed: {e}")))?;

    Ok(true)
}

/// Verify a `BridgeProofV2` under the production closure-verifier policy.
///
/// This:
/// 1) verifies the Phase-1 Spartan proof, then
/// 2) derives `ClosureStatementV1` from the Spartan statement, and
/// 3) verifies the Phase-2 closure proof using `neo_closure_proof`'s production verifier entrypoint.
///
/// NOTE: This is expected to fail closed until the obligations-private closure backend is
/// production-audit-ready.
pub fn verify_bridge_proof_v2_production(
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

    let closure_stmt = compute_closure_statement_v1(&proof.spartan.statement);

    let bus = neo_fold::memory_sidecar::cpu_bus::try_infer_cpu_bus_layout_for_step_instances(ccs, steps_public)
        .map_err(|e| SpartanBridgeError::InvalidInput(format!("BusLayout: {e:?}")))?;
    neo_closure_proof::verify_closure_v1_production_with_context_and_bus(
        &closure_stmt,
        &proof.closure,
        Some(params),
        Some(ccs),
        bus.as_ref(),
    )
    .map_err(|e| SpartanBridgeError::VerificationError(format!("closure proof verification failed: {e}")))?;

    Ok(true)
}

/// Verify a `BridgeProofV2` using only an expected Spartan statement.
///
/// This is the fully-compressed verifier entrypoint: the caller provides the pinned Spartan VK
/// and the exact Phase-1 statement it expects, and supplies any optional context required by the
/// chosen closure backend.
pub fn verify_bridge_proof_v2_statement_only(
    vk_spartan: &crate::api::SpartanVerifierKey,
    expected_statement: &SpartanShardStatement,
    proof: &BridgeProofV2,
    params: Option<&NeoParams>,
    ccs: Option<&CcsStructure<NeoF>>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<bool> {
    let ok = verify_fold_run_statement_only(vk_spartan, expected_statement, &proof.spartan)?;
    if !ok {
        return Ok(false);
    }

    let closure_stmt = compute_closure_statement_v1(expected_statement);

    neo_closure_proof::verify_closure_v1_with_context_and_bus(&closure_stmt, &proof.closure, params, ccs, bus)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("closure proof verification failed: {e}")))?;

    Ok(true)
}

/// Verify a `BridgeProofV2` statement-only under the production closure-verifier policy.
///
/// This is the fully-compressed verifier entrypoint intended for production callers. It verifies
/// the Phase-1 proof against the expected statement, then verifies Phase 2 via the production
/// closure verifier.
pub fn verify_bridge_proof_v2_statement_only_production(
    vk_spartan: &crate::api::SpartanVerifierKey,
    expected_statement: &SpartanShardStatement,
    proof: &BridgeProofV2,
    params: Option<&NeoParams>,
    ccs: Option<&CcsStructure<NeoF>>,
    bus: Option<&neo_memory::cpu::BusLayout>,
) -> Result<bool> {
    let ok = verify_fold_run_statement_only(vk_spartan, expected_statement, &proof.spartan)?;
    if !ok {
        return Ok(false);
    }

    let closure_stmt = compute_closure_statement_v1(expected_statement);

    neo_closure_proof::verify_closure_v1_production_with_context_and_bus(&closure_stmt, &proof.closure, params, ccs, bus)
        .map_err(|e| SpartanBridgeError::VerificationError(format!("closure proof verification failed: {e}")))?;

    Ok(true)
}
