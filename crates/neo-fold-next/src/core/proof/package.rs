//! Owns full-session and packaged proof data shapes.

use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use serde::{Deserialize, Serialize};

use super::{ChunkProof, FoldSchedule, PublicChunk};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunProof {
    pub fold_schedule: FoldSchedule,
    pub chunks: Vec<ChunkProof>,
    pub final_main_claims: Vec<CeClaim<Commitment, F, K>>,
}

impl RunProof {
    pub fn public_step_count(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.chunk.steps.len())
            .sum()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicStatement {
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub chunks: Vec<PublicChunk>,
    pub final_main_claims: Vec<CeClaim<Commitment, F, K>>,
    pub digest: [u8; 32],
}

impl PublicStatement {
    pub fn public_step_count(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.steps.len()).sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalProof {
    pub session: RunProof,
    pub proof_digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackagedProof {
    pub statement: PublicStatement,
    pub proof: FinalProof,
}
