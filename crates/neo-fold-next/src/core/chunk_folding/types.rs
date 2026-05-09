use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_reductions::optimized_engine::{PiCcsReplayProofWitness, PiCcsReplayTerminalState};
use serde::{Deserialize, Serialize};

use crate::proof::Carry;

#[derive(Clone, Copy)]
pub struct CommitmentMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Commitment]) -> Commitment + Clone + Copy,
    MB: Fn(&[Commitment], u32) -> Commitment + Clone + Copy,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

#[derive(Clone, Debug)]
pub struct ChunkRelationArtifacts {
    pub relation_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct ChunkRelationResult {
    pub next_main: Carry,
    pub artifacts: ChunkRelationArtifacts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkReplayWitness {
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub ccs_replay_proof: PiCcsReplayProofWitness,
}

pub(crate) struct ChunkReplayTrace {
    pub ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub ccs_replay_proof: PiCcsReplayProofWitness,
    pub ccs_post_transcript_state: [F; neo_params::poseidon2_goldilocks::WIDTH],
    pub ccs_post_transcript_absorbed: usize,
    pub terminal_state: PiCcsReplayTerminalState,
    pub parent: CeClaim<Commitment, F, K>,
    pub children: Vec<CeClaim<Commitment, F, K>>,
    pub z_split: Vec<Mat<F>>,
}

pub(super) struct ChunkPreparedInputs {
    pub(super) start_index: usize,
    pub(super) fresh_step_count: usize,
    pub(super) fresh_claims: Vec<CcsClaim<Commitment, F>>,
    pub(super) fresh_witnesses: Vec<CcsWitness<F>>,
    pub(super) public_chunk_digest: [F; 4],
    pub(super) prepare_inputs_ms: f64,
}

pub(super) struct BorrowedChunkPreparedInputs<'a> {
    pub(super) start_index: usize,
    pub(super) fresh_step_count: usize,
    pub(super) fresh_claims: &'a [CcsClaim<Commitment, F>],
    pub(super) fresh_witnesses: &'a [CcsWitness<F>],
    pub(super) public_chunk_digest: [F; 4],
    pub(super) prepare_inputs_ms: f64,
}

pub(super) struct CcsTransitionState {
    pub(super) ccs_outputs: Vec<CeClaim<Commitment, F, K>>,
    pub(super) parent: CeClaim<Commitment, F, K>,
    pub(super) children: Vec<CeClaim<Commitment, F, K>>,
    pub(super) z_split: Vec<Mat<F>>,
}

pub(super) struct ChunkTransitionCore {
    pub(super) parent: CeClaim<Commitment, F, K>,
    pub(super) children: Vec<CeClaim<Commitment, F, K>>,
    pub(super) z_split: Vec<Mat<F>>,
}

pub(crate) struct ChunkComputation {
    pub(super) transition: CcsTransitionState,
    pub(super) ccs_proof: neo_reductions::api::PiCcsProof,
}
