//! Owns the RV64IM decider relation surface projected from the verified final seam.
//!
//! This module owns only the surviving direct-Spartan target surface:
//! `Rv64imFinalStatement + Rv64imFinalBuildProof -> Spartan2DeciderRelation`.
//! It does not own any relation-rebuilding artifact/witness adapter above the
//! final seam.

use crate::decider::spartan2::{
    build_spartan2_decider_relation, validate_spartan2_decider_relation_surface, Spartan2DeciderRelation,
};
use crate::finalize::{digest32_as_fields, FixedShapeChunkSummary};
use crate::rv64im::chunk_fold_step::rv64im_chunk_fold_seed;
use crate::rv64im::final_relation::{
    final_proof_component_digests, final_proof_digest_from_component_digests, Rv64imFinalBuildProof,
    Rv64imFinalProofComponentDigests, Rv64imFinalStatement,
};
use crate::rv64im::SimpleKernelError;

pub type Rv64imDeciderRelation = Spartan2DeciderRelation;

pub fn validate_rv64im_decider_relation_surface(relation: &Rv64imDeciderRelation) -> Result<(), SimpleKernelError> {
    validate_spartan2_decider_relation_surface(relation).map_err(|err| SimpleKernelError::Bridge(err.to_string()))
}

pub fn build_rv64im_decider_relation_from_final_surface(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imDeciderRelation, SimpleKernelError> {
    let component_digests = final_proof_component_digests(proof);
    build_rv64im_main_relation_backend_relation_from_main_surface(statement, &proof.chunk_summaries, &component_digests)
}

pub(crate) fn build_rv64im_main_relation_backend_relation_from_main_surface(
    statement: &Rv64imFinalStatement,
    chunk_summaries: &[FixedShapeChunkSummary],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Spartan2DeciderRelation, SimpleKernelError> {
    if statement.folded.chunk_count as usize != chunk_summaries.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk count does not match final proof chunk summaries".into(),
        ));
    }
    if statement.folded.chunk_count as usize != component_digests.chunk_transition_digests.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk count does not match final proof replay witness".into(),
        ));
    }

    build_spartan2_decider_relation(
        statement.digest,
        statement.folded.digest,
        final_proof_digest_from_component_digests(&statement.folded, chunk_summaries, component_digests),
        digest32_as_fields(rv64im_chunk_fold_seed()),
        digest32_as_fields(statement.folded.final_accumulator.terminal_handle.0),
        statement.folded.fold_schedule,
        statement.folded.semantic_step_count,
        chunk_summaries.to_vec(),
        vec![component_digests.kernel_export_proof_digest],
        component_digests.chunk_transition_digests.clone(),
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))
}
