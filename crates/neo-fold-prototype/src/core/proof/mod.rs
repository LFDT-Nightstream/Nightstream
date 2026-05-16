//! Owns the generic session proof boundary.
//!
//! This folder contains data shapes only. The `session`, `chunk_folding`, and
//! `finalize` modules own the protocol work that consumes these shapes.

mod chunk;
mod package;
mod perf;
mod schedule;
mod step;

pub use chunk::{ChunkProof, ChunkResult, PiDecArtifact, PiRlcArtifact};
pub use package::{FinalProof, PackagedProof, PublicStatement, RunProof};
pub use perf::{ChunkProvePerf, ChunkVerifyPerf, RunProvePerf, RunVerifyPerf};
pub use schedule::FoldSchedule;
pub(crate) use step::{partition_prover_step_inputs, ProverChunkInput};
pub use step::{partition_public_steps, partition_step_inputs, Carry, ChunkInput, PublicChunk, PublicStep, StepInput};
