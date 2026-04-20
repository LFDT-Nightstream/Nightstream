//! Owns the legacy audit-only RV64IM shell decider relation projected from the
//! final seam.
//!
//! The live Goal 3 verifier no longer consumes this surface. It remains only
//! for audit/debug callers that still want the old fixed-shape shell relation
//! projected from an authoritative final seam.

use serde::{Deserialize, Serialize};

use crate::decider::spartan2::{
    build_spartan2_decider_relation, validate_spartan2_decider_relation_surface, Spartan2DeciderRelation,
    Spartan2DeciderTarget,
};
use crate::finalize::{digest32_as_fields, FixedShapeChunkSummary};
use crate::rv64im::chunk_fold_step::{rv64im_chunk_fold_seed, Rv64imChunkFoldCarry};
use crate::rv64im::final_relation::{
    final_proof_component_digests, final_proof_digest_from_component_digests,
    rv64im_chunk_fold_carry_recursive_accumulator_digest, Rv64imFinalBuildProof, Rv64imFinalProofComponentDigests,
    Rv64imFinalStatement,
};
use crate::rv64im::main_recursion::{build_rv64im_main_recursion_verifier_key_fs, Rv64imEncodedPublicInput};
use crate::rv64im::recursion_spartan::build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs;
use crate::rv64im::SimpleKernelError;
use neo_transcript::{Poseidon2Transcript, Transcript};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imLegacyShellDeciderRelation {
    pub public_statement_digest: [u8; 32],
    pub relation_digest: [u8; 32],
    pub final_proof_digest: [u8; 32],
    pub initial_handle_digest: [neo_math::F; crate::finalize::FIXED_SHAPE_DIGEST_FIELD_LEN],
    pub terminal_handle_digest: [neo_math::F; crate::finalize::FIXED_SHAPE_DIGEST_FIELD_LEN],
    pub fold_schedule: crate::proof::FoldSchedule,
    pub semantic_step_count: u64,
    pub chunk_summaries: Vec<FixedShapeChunkSummary>,
    pub base_component_digests: Vec<[u8; 32]>,
    pub chunk_transition_bindings: Vec<crate::decider::spartan2::Spartan2ChunkTransitionBinding>,
    pub x_last: Rv64imEncodedPublicInput,
    pub folded_accumulator_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Rv64imLegacyShellDeciderRelation {
    fn shell_relation(&self) -> Spartan2DeciderRelation {
        Spartan2DeciderRelation {
            public_statement_digest: self.public_statement_digest,
            relation_digest: self.relation_digest,
            final_proof_digest: self.final_proof_digest,
            initial_handle_digest: self.initial_handle_digest,
            terminal_handle_digest: self.terminal_handle_digest,
            fold_schedule: self.fold_schedule,
            semantic_step_count: self.semantic_step_count,
            chunk_summaries: self.chunk_summaries.clone(),
            base_component_digests: self.base_component_digests.clone(),
            chunk_transition_bindings: self.chunk_transition_bindings.clone(),
            digest: self.shell_digest(),
        }
    }

    fn shell_digest(&self) -> [u8; 32] {
        let relation = Spartan2DeciderRelation {
            public_statement_digest: self.public_statement_digest,
            relation_digest: self.relation_digest,
            final_proof_digest: self.final_proof_digest,
            initial_handle_digest: self.initial_handle_digest,
            terminal_handle_digest: self.terminal_handle_digest,
            fold_schedule: self.fold_schedule,
            semantic_step_count: self.semantic_step_count,
            chunk_summaries: self.chunk_summaries.clone(),
            base_component_digests: self.base_component_digests.clone(),
            chunk_transition_bindings: self.chunk_transition_bindings.clone(),
            digest: [0; 32],
        };
        relation
            .target()
            .relation()
            .expect("RV64IM decider shell relation construction must be valid")
            .digest
    }

    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/terminal_decider/relation");
        tr.append_message(
            b"neo.fold.next/rv64im/terminal_decider/relation/shell_digest",
            &self.shell_digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/terminal_decider/relation/x_last",
            &self.x_last.bytes(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/terminal_decider/relation/folded_accumulator_digest",
            &self.folded_accumulator_digest,
        );
        tr.digest32()
    }

    pub fn target(&self) -> Spartan2DeciderTarget {
        self.shell_relation().target()
    }
}

pub fn validate_rv64im_legacy_shell_decider_relation_surface(
    relation: &Rv64imLegacyShellDeciderRelation,
) -> Result<(), SimpleKernelError> {
    validate_spartan2_decider_relation_surface(&relation.shell_relation())
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if relation.digest != relation.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM terminal decider relation digest mismatch".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_legacy_shell_decider_relation_from_final_surface(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imLegacyShellDeciderRelation, SimpleKernelError> {
    let component_digests = final_proof_component_digests(proof);
    build_rv64im_main_relation_backend_relation_from_main_surface(statement, &proof.chunk_summaries, &component_digests)
}

pub(crate) fn build_rv64im_main_relation_backend_relation_from_main_surface(
    statement: &Rv64imFinalStatement,
    chunk_summaries: &[FixedShapeChunkSummary],
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Rv64imLegacyShellDeciderRelation, SimpleKernelError> {
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

    let shell = build_spartan2_decider_relation(
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
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;

    let vk_fs = build_rv64im_main_recursion_verifier_key_fs()?;
    let x_last = build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
        &vk_fs,
        statement.folded.chunk_count,
        &statement.folded.final_accumulator,
    )?;
    let folded_accumulator_digest = rv64im_chunk_fold_carry_recursive_accumulator_digest(&Rv64imChunkFoldCarry {
        main: crate::proof::Carry {
            claims: statement.folded.final_accumulator.final_main_claims.clone(),
            witnesses: Vec::new(),
        },
        terminal_handle: statement.folded.final_accumulator.terminal_handle,
    });

    let mut relation = Rv64imLegacyShellDeciderRelation {
        public_statement_digest: shell.public_statement_digest,
        relation_digest: shell.relation_digest,
        final_proof_digest: shell.final_proof_digest,
        initial_handle_digest: shell.initial_handle_digest,
        terminal_handle_digest: shell.terminal_handle_digest,
        fold_schedule: shell.fold_schedule,
        semantic_step_count: shell.semantic_step_count,
        chunk_summaries: shell.chunk_summaries,
        base_component_digests: shell.base_component_digests,
        chunk_transition_bindings: shell.chunk_transition_bindings,
        x_last,
        folded_accumulator_digest,
        digest: [0; 32],
    };
    relation.digest = relation.expected_digest();
    Ok(relation)
}
