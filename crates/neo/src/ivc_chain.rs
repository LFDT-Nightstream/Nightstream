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

/// Finalize the current batch into a proof and return it along with verification parameters.
///
/// This separates proof generation from verification for clearer API design.
/// Returns `Ok(Some((proof, batch_ccs, batch_public_input)))` if there were pending steps, `Ok(None)` if batch was empty.
pub fn finalize_and_prove(mut state: State) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
    // Extract pending batch (if any)
    let Some(batch) = state.builder.finalize()? else {
        // No steps pending
        return Ok(None);
    };

    // Keep copies of CCS and public input for verification
    let batch_ccs = batch.ccs.clone();
    let batch_public_input = batch.public_input.clone();

    // Generate the proof from accumulated batch data
    let proof = crate::ivc::prove_batch_data(&state.params, batch)?;
    Ok(Some((proof, batch_ccs, batch_public_input)))
}
