use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{F, K};
use neo_reductions::error::PiCcsError;

use super::digest::chunk_relation_digest;
use super::types::{
    CcsTransitionState, ChunkComputation, ChunkRelationArtifacts, ChunkRelationResult, ChunkReplayWitness,
};
use crate::proof::{Carry, ChunkProof, ChunkResult, PiDecArtifact, PiRlcArtifact, PublicChunk};

impl CcsTransitionState {
    pub(super) fn into_relation_result(self) -> Result<ChunkRelationResult, PiCcsError> {
        Ok(chunk_relation_result_from_transition(self))
    }
}

impl ChunkComputation {
    pub(crate) fn into_chunk_result(self, chunk: &crate::proof::ChunkInput) -> ChunkResult {
        chunk_result_from_transition(self.transition, chunk.public(), self.ccs_proof)
    }

    pub(crate) fn into_chunk_result_with_public_chunk(self, public_chunk: PublicChunk) -> ChunkResult {
        chunk_result_from_transition(self.transition, public_chunk, self.ccs_proof)
    }
}

fn chunk_result_from_transition(
    transition: CcsTransitionState,
    public_chunk: PublicChunk,
    ccs_proof: neo_reductions::api::PiCcsProof,
) -> ChunkResult {
    let CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
        ..
    } = transition;
    let relation_digest = chunk_relation_digest(&ccs_outputs, &parent, &children);
    ChunkResult {
        proof: ChunkProof {
            chunk: public_chunk,
            relation_digest,
            ccs_outputs,
            ccs_proof,
            rlc: PiRlcArtifact { parent },
            dec: PiDecArtifact {
                children: children.clone(),
            },
        },
        next_main: Carry {
            claims: children,
            witnesses: z_split,
        },
    }
}

pub(super) fn chunk_relation_result_from_transition(transition: CcsTransitionState) -> ChunkRelationResult {
    let CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
        ..
    } = transition;
    let relation_digest = chunk_relation_digest(&ccs_outputs, &parent, &children);
    ChunkRelationResult {
        next_main: Carry {
            claims: children,
            witnesses: z_split,
        },
        artifacts: ChunkRelationArtifacts { relation_digest },
    }
}

pub(super) fn chunk_relation_result_from_parts(
    ccs_outputs: &[CeClaim<Commitment, F, K>],
    parent: CeClaim<Commitment, F, K>,
    children: Vec<CeClaim<Commitment, F, K>>,
    z_split: Vec<Mat<F>>,
) -> ChunkRelationResult {
    let relation_digest = chunk_relation_digest(ccs_outputs, &parent, &children);
    ChunkRelationResult {
        next_main: Carry {
            claims: children,
            witnesses: z_split,
        },
        artifacts: ChunkRelationArtifacts { relation_digest },
    }
}

pub(super) fn chunk_replay_witness_and_result_from_parts(
    transition: CcsTransitionState,
    ccs_replay_proof: neo_reductions::optimized_engine::PiCcsReplayProofWitness,
) -> (ChunkReplayWitness, ChunkRelationResult) {
    let CcsTransitionState {
        ccs_outputs,
        parent,
        children,
        z_split,
        ..
    } = transition;
    let relation_digest = chunk_relation_digest(&ccs_outputs, &parent, &children);
    (
        ChunkReplayWitness {
            ccs_outputs,
            ccs_replay_proof,
        },
        ChunkRelationResult {
            next_main: Carry {
                claims: children,
                witnesses: z_split,
            },
            artifacts: ChunkRelationArtifacts { relation_digest },
        },
    )
}
