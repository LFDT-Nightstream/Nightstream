//! Owns the WASM semantic kernel proof boundary above Stage 1/2/3.

mod openings;
mod types;

use super::builder::WasmTraceBuilder;
use super::relation::{prove_wasm_relation, verify_wasm_relation};
use super::step_build::WasmStepBuild;
use openings::{build_kernel_opening_summary, verify_kernel_opening_summary};

pub use types::{
    WasmKernelError, WasmKernelOpeningSummary, WasmKernelOutput, WasmKernelPreparedStepSummary, WasmKernelProof,
    WasmKernelProverInput, WasmKernelPublicInput, WasmKernelRelationOpeningSummary, WasmKernelSelectedRowRef,
    WasmKernelVerifierInput,
};

pub fn prove_simple_kernel(
    input: &WasmKernelProverInput<'_>,
) -> Result<(WasmKernelOutput, WasmKernelProof), WasmKernelError> {
    let prepared_steps = build_prepared(input.trace)?;

    let relation = prove_wasm_relation(
        input.trace,
        &input.public.initial_locals,
        &input.pc_rom,
        &input.pc_edge_kinds,
        &input.function_entries,
        &input.public.transcript_seed,
    )
    .map_err(WasmKernelError::Relation)?;
    if relation.boundary_rows.len() != prepared_steps.len() {
        return Err(WasmKernelError::Bridge(format!(
            "wasm relation exported {} boundary rows for {} prepared steps",
            relation.boundary_rows.len(),
            prepared_steps.len()
        )));
    }

    let mut proof = WasmKernelProof {
        relation,
        opening_summary: empty_opening_summary(),
    };
    let opening_summary = build_kernel_opening_summary(&proof, &prepared_steps);
    proof.opening_summary = opening_summary.clone();
    let output = WasmKernelOutput {
        prepared_steps,
        opening_summary,
    };

    Ok((output, proof))
}

pub fn verify_simple_kernel(
    input: &WasmKernelVerifierInput<'_>,
    proof: &WasmKernelProof,
) -> Result<WasmKernelOutput, WasmKernelError> {
    let prepared_steps = build_prepared(input.trace)?;

    verify_wasm_relation(
        input.trace,
        &input.public.initial_locals,
        &input.pc_rom,
        &input.pc_edge_kinds,
        &input.function_entries,
        &input.public.transcript_seed,
        &proof.relation,
    )
    .map_err(WasmKernelError::Relation)?;

    if proof.relation.boundary_rows.len() != prepared_steps.len() {
        return Err(WasmKernelError::Bridge(format!(
            "wasm relation exported {} boundary rows for {} prepared steps",
            proof.relation.boundary_rows.len(),
            prepared_steps.len()
        )));
    }

    verify_kernel_opening_summary(&proof.opening_summary, proof, &prepared_steps).map_err(WasmKernelError::Bridge)?;

    Ok(WasmKernelOutput {
        prepared_steps,
        opening_summary: proof.opening_summary.clone(),
    })
}

fn build_prepared(trace: &[crate::ir::WasmStepTrace]) -> Result<Vec<WasmStepBuild>, WasmKernelError> {
    let vm = crate::ccs::WasmVmSpec::default();
    let builder = WasmTraceBuilder::new();
    builder
        .build_steps(&vm, trace)
        .map_err(|err| WasmKernelError::InvalidWitness(err.to_string()))
}

fn empty_opening_summary() -> WasmKernelOpeningSummary {
    WasmKernelOpeningSummary {
        relation: WasmKernelRelationOpeningSummary {
            lookup_rows_digest: [0u8; 32],
            memory_events_digest: [0u8; 32],
            boundary_rows_digest: [0u8; 32],
            lookup_row_count: 0,
            memory_event_count: 0,
            boundary_row_count: 0,
            final_stack_slot_count: 0,
            final_local_slot_count: 0,
            first_lookup_row: None,
            last_lookup_row: None,
            first_memory_event: None,
            last_memory_event: None,
            digest: [0u8; 32],
        },
        prepared_steps: WasmKernelPreparedStepSummary {
            steps_digest: [0u8; 32],
            step_count: 0,
            first_step: None,
            last_step: None,
            digest: [0u8; 32],
        },
        digest: [0u8; 32],
    }
}
