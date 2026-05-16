//! Owns native SuperNeo IVC carrier data shapes.

use neo_math::F;

use crate::chunk_folding::ChunkReplayWitness;
use crate::proof::{Carry, ChunkInput, ChunkProvePerf, RunProvePerf};

#[derive(Clone, Debug, PartialEq)]
pub struct SuperNeoIvcTranscriptSnapshot {
    pub state: [F; neo_params::poseidon2_goldilocks::WIDTH],
    pub absorbed: usize,
}

#[derive(Clone, Debug)]
pub struct SuperNeoIvcState {
    pub chunk_count: u64,
    pub step_count: u64,
    pub carry: Carry,
    pub transcript: SuperNeoIvcTranscriptSnapshot,
}

#[derive(Clone, Debug)]
pub struct SuperNeoIvcStepRelation {
    pub chunk_index: u64,
    pub chunk: ChunkInput,
    pub state_in: SuperNeoIvcState,
    pub state_out: SuperNeoIvcState,
    pub replay_witness: ChunkReplayWitness,
    pub fold_digest: [u8; 32],
    pub chunk_relation_digest: [u8; 32],
    pub perf: ChunkProvePerf,
}

#[derive(Clone, Debug)]
pub struct SuperNeoIvcBuild {
    pub relations: Vec<SuperNeoIvcStepRelation>,
    pub final_state: SuperNeoIvcState,
    pub cache_build_ms: f64,
    pub total_ms: f64,
}

impl SuperNeoIvcBuild {
    pub fn prove_perf(&self) -> RunProvePerf {
        RunProvePerf {
            chunks: self
                .relations
                .iter()
                .map(|relation| relation.perf)
                .collect(),
            total_ms: self.total_ms,
        }
    }
}
