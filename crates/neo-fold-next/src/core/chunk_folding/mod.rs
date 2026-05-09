//! Owns one-chunk SuperNeo folding for the shared core path.
//!
//! Ownership:
//! - prepares fresh chunk claims plus the incoming accumulator carry
//! - runs or verifies `Π_CCS`
//! - applies `Π_RLC -> Π_DEC` to derive the next carry
//! - packages replay witnesses, proof artifacts, and audit digests
//! - does not own VM-specific wrappers, Spartan compression, or run orchestration

mod digest;
mod pi_ccs;
mod prepare;
mod prove;
mod replay;
mod result;
mod trace;
mod transition;
mod types;
mod verify;

pub(crate) use digest::{chunk_relation_digest, claim_digests};
pub use pi_ccs::build_inert_chunk_replay_proof_witness;
pub(crate) use prove::{compute_chunk_relation_for_prover_chunk_with_perf, compute_chunk_relation_with_perf};
pub(crate) use replay::{
    compute_chunk_replay_witness_and_relation_with_instance_digest_and_me_input_handle_and_perf,
    compute_chunk_replay_witness_and_relation_with_instance_digest_and_perf,
};
pub use replay::{replay_chunk_relation, replay_chunk_relation_with_perf};
pub(crate) use trace::trace_chunk_relation_with_witness_and_instance_digest_and_me_input_handle;
pub(crate) use types::ChunkReplayTrace;
pub use types::{ChunkRelationArtifacts, ChunkRelationResult, ChunkReplayWitness, CommitmentMixers};
pub use verify::verify_chunk_relation_with_witness;
pub(crate) use verify::{
    verify_chunk_relation_with_witness_and_instance_digest_and_me_input_handle_with_perf,
    verify_chunk_relation_with_witness_and_instance_digest_with_perf,
};
