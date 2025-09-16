//! Chain-style step/verify API over Neo IVC
//!
//! This module provides a minimal ergonomic wrapper around the production IVC
//! pipeline so examples can look like:
//!
//!   - `step(state, io, witness) -> State`
//!   - `verify(state, io) -> bool`
//!
//! Internally it uses the secure `IvcBatchBuilder` with linked-witness EV and
//! finalizes to a single proof on `verify`.

use crate::{F, NeoParams};
use crate::ivc::{Accumulator, EmissionPolicy, IvcBatchBuilder, StepBindingSpec};
use neo_ccs::CcsStructure;

/// Minimal state wrapper for the simple API
pub struct State {
    params: NeoParams,
    binding: StepBindingSpec,
    builder: IvcBatchBuilder,
}

impl State {
    /// Initialize a new State for a given step CCS and initial y-state.
    ///
    /// - `y0` is the initial compact y (the running state exposed by IVC folding)
    /// - `binding` must be a trusted binding specification for the step circuit
    pub fn new(
        params: NeoParams,
        step_ccs: CcsStructure<F>,
        y0: Vec<F>,
        binding: StepBindingSpec,
    ) -> anyhow::Result<Self> {
        let acc = Accumulator {
            c_z_digest: [0u8; 32],
            c_coords: vec![],
            y_compact: y0,
            step: 0,
        };

        let builder = IvcBatchBuilder::new_with_bindings(
            params.clone(),
            step_ccs.clone(),
            acc,
            EmissionPolicy::Never, // accumulate and prove on verify()
            binding.clone(),
        )?;

        Ok(Self { params, binding, builder })
    }
}

/// Advance one step of the IVC chain.
///
/// - `io` are per-step public inputs (can be empty)
/// - `witness` is the step circuit witness
///
/// Note: This collects steps into an internal batch (no proof yet).
pub fn step(mut state: State, io: &[F], witness: &[F]) -> State {
    // Extract y_step using binding offsets (trusted circuit spec)
    let y_step: Vec<F> = state
        .binding
        .y_step_offsets
        .iter()
        .map(|&idx| witness[idx])
        .collect();

    // Append the step to the batch
    // - Bind step public input (io)
    // - Use linked witness EV with secure rho derivation handled internally
    let _ = state
        .builder
        .append_step(witness, Some(io), &y_step)
        .expect("failed to append IVC step");

    state
}

/// Finalize the current batch into a proof and verify it.
///
/// Returns true if verification succeeded or there was nothing to verify.
pub fn verify(mut state: State, _io: &[F]) -> bool {
    // Extract pending batch (if any)
    let Some(batch) = state.builder.finalize() else {
        // No steps pending; treat as vacuously true
        return true;
    };

    // Prove the batch and verify against the exact CCS + public input
    // Keep copies of CCS and public input for verification after proving
    let ccs = batch.ccs.clone();
    let public_input = batch.public_input.clone();

    match crate::ivc::prove_batch_data(&state.params, batch) {
        Ok(proof) => match crate::verify(&ccs, &public_input, &proof) {
            Ok(ok) => ok,
            Err(_) => false,
        },
        Err(_) => false,
    }
}
