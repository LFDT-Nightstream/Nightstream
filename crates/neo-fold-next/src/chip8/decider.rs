//! Owns CHIP-8 adapters from the current construction/audit seam into generic decider targets.
//! Does not own final CHIP-8 proof acceptance; the current target still depends on replay artifacts.

use crate::chip8::final_relation::{
    audit_check_final_statement_with_output, final_proof_component_digests, folded_statement_digest, recursive_seed,
};
use crate::chip8::kernel::SimpleKernelError;
use crate::chip8::proof::{statement_digest, Chip8FinalProof, Chip8Statement};
use crate::decider::spartan2::{
    build_spartan2_decider_relation, prove_spartan2_decider, setup_spartan2_decider,
    validate_spartan2_decider_relation_surface, verify_spartan2_decider, Spartan2DeciderProof,
    Spartan2DeciderProverKey, Spartan2DeciderRelation, Spartan2DeciderTarget, Spartan2DeciderVerifierKey,
};
use crate::finalize::digest32_as_fields;

pub type Chip8DeciderRelation = Spartan2DeciderRelation;

pub fn validate_chip8_decider_relation_surface(relation: &Chip8DeciderRelation) -> Result<(), SimpleKernelError> {
    validate_spartan2_decider_relation_surface(relation).map_err(|err| SimpleKernelError::BridgeFailed(err.to_string()))
}

pub fn build_chip8_decider_relation(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<Chip8DeciderRelation, SimpleKernelError> {
    if statement.folded.digest != folded_statement_digest(&statement.folded) {
        return Err(SimpleKernelError::BridgeFailed(
            "folded statement digest mismatch".into(),
        ));
    }
    if statement.digest != statement_digest(statement) {
        return Err(SimpleKernelError::BridgeFailed("statement digest mismatch".into()));
    }
    audit_check_final_statement_with_output(&statement.public, &statement.folded, proof)?;
    let component_digests = final_proof_component_digests(proof);

    build_spartan2_decider_relation(
        statement.digest,
        statement.folded.digest,
        proof.proof_digest,
        digest32_as_fields(recursive_seed()),
        digest32_as_fields(statement.folded.final_accumulator.terminal_handle.0),
        statement.folded.fold_schedule,
        statement.folded.semantic_step_count,
        proof.chunk_summaries.clone(),
        vec![component_digests.kernel_export_digest],
        component_digests.chunk_transition_digests,
    )
    .map_err(|err| SimpleKernelError::BridgeFailed(err.to_string()))
}

/// Audit helper: this rebuilds the current decider target from replay artifacts.
pub fn audit_check_chip8_decider_relation(
    relation: &Chip8DeciderRelation,
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<(), SimpleKernelError> {
    let expected = build_chip8_decider_relation(statement, proof)?;
    if relation != &expected {
        return Err(SimpleKernelError::BridgeFailed(
            "CHIP-8 decider relation does not match the carried final proof seam".into(),
        ));
    }
    Ok(())
}

pub fn build_chip8_spartan2_decider_target(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<Spartan2DeciderTarget, SimpleKernelError> {
    let relation = build_chip8_decider_relation(statement, proof)?;
    Ok(relation.target())
}

pub fn setup_chip8_spartan2_decider(
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<(Spartan2DeciderProverKey, Spartan2DeciderVerifierKey), SimpleKernelError> {
    let target = build_chip8_spartan2_decider_target(statement, proof)?;
    setup_spartan2_decider(&target.shape()).map_err(|err| SimpleKernelError::BridgeFailed(err.to_string()))
}

pub fn prove_chip8_spartan2_decider(
    pk: &Spartan2DeciderProverKey,
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
) -> Result<Spartan2DeciderProof, SimpleKernelError> {
    let target = build_chip8_spartan2_decider_target(statement, proof)?;
    prove_spartan2_decider(pk, &target).map_err(|err| SimpleKernelError::BridgeFailed(err.to_string()))
}

/// Audit helper: this proves the generic target binding, not standalone CHIP-8 final acceptance.
pub fn audit_check_chip8_spartan2_decider(
    vk: &Spartan2DeciderVerifierKey,
    statement: &Chip8Statement,
    proof: &Chip8FinalProof,
    decider_proof: &Spartan2DeciderProof,
) -> Result<(), SimpleKernelError> {
    let target = build_chip8_spartan2_decider_target(statement, proof)?;
    verify_spartan2_decider(vk, &target, decider_proof).map_err(|err| SimpleKernelError::BridgeFailed(err.to_string()))
}
