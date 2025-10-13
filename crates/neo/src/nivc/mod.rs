//! NIVC (Non-Uniform IVC) support à la HyperNova
//!
//! This module provides a pragmatic NIVC driver on top of the existing IVC folding
//! implementation. It allows selecting one of multiple step CCS relations per step
//! and folds only that lane's running instance, achieving an "à‑la‑carte" cost profile.
//!
//! ## Design highlights
//!
//! - Keeps a per‑type ("lane") running ME instance and witness.
//! - Maintains a global y (compact state) shared across lanes.
//! - Binds the selected lane index into the step public input (and thus the FS transcript).
//! - Reuses `prove_ivc_step_chained`/`verify_ivc_step` for per‑step proving and verification.
//!
//! NOTE: For production‑grade scalability, consider switching the "lanes state" to a
//! Merkle tree and proving a single leaf update in‑circuit. This initial driver does not
//! add in‑circuit constraints for unchanged lanes; it preserves à‑la‑carte cost by
//! only folding the chosen lane each step.
//!
//! ## Module Structure
//!
//! - `api/` - Public API surface (types, state, program, errors)
//! - `pipeline/` - Core operations (prover, verifier, finalizer)
//! - `internal/` - Internal utilities (digest, binding)

// Re-export the public API surface
pub mod api;
pub mod pipeline;
mod internal;

// Top-level re-exports for convenience
pub use api::{
    NivcError,
    NivcProgram, NivcStepSpec,
    NivcState, NivcAccumulators, LaneRunningState,
    NivcStepProof, NivcChainProof,
    LaneId, StepIdx, Result,
};

pub use pipeline::{
    prove_step,
    verify_chain,
    finalize as finalize_nivc_chain,
    finalize_with_options as finalize_nivc_chain_with_options,
    NivcFinalizeOptions,
};

// For backwards compatibility, provide impl on NivcState
impl NivcState {
    /// Execute one NIVC step (forwards to pipeline::prover::step)
    pub fn step(
        &mut self,
        lane_idx: usize,
        step_io: &[crate::F],
        step_witness: &[crate::F],
    ) -> anyhow::Result<NivcStepProof> {
        pipeline::prove_step(self, lane_idx, step_io, step_witness)
    }
}

// For backwards compatibility with original verify_nivc_chain function
pub fn verify_nivc_chain(
    program: &NivcProgram,
    params: &crate::NeoParams,
    chain: &NivcChainProof,
    initial_y: &[crate::F],
) -> anyhow::Result<bool> {
    pipeline::verify_chain(program, params, chain, initial_y)
}

