//! Owns one-chunk proof artifacts produced by `Pi_CCS -> Pi_RLC -> Pi_DEC`.

use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{F, K};
use neo_reductions::api::PiCcsProof;
use serde::{Deserialize, Serialize};

use super::{Carry, PublicChunk};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiRlcArtifact {
    pub parent: CeClaim<Commitment, F, K>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiDecArtifact {
    pub children: Vec<CeClaim<Commitment, F, K>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkProof {
    pub chunk: PublicChunk,
    pub relation_digest: [u8; 32],
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub ccs_proof: PiCcsProof,
    pub rlc: PiRlcArtifact,
    pub dec: PiDecArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkResult {
    pub proof: ChunkProof,
    pub next_main: Carry,
}
