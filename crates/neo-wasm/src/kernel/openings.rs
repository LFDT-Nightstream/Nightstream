//! Owns compact selected-row and kernel-opening summary surfaces for the WASM kernel.

use neo_transcript::{Poseidon2Transcript, Transcript};

use super::types::{
    WasmKernelOpeningSummary, WasmKernelPreparedStepSummary, WasmKernelProof, WasmKernelRelationOpeningSummary,
    WasmKernelSelectedRowRef,
};
use crate::relation::{WasmBoundaryRow, WasmLookupRow, WasmMemoryEvent};
use crate::step_build::WasmStepBuild;

pub fn build_kernel_opening_summary(
    proof: &WasmKernelProof,
    prepared_steps: &[WasmStepBuild],
) -> WasmKernelOpeningSummary {
    let relation = build_relation_summary(proof);
    let prepared_steps = build_prepared_step_summary(prepared_steps);

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_summary");
    tr.append_message(b"wasm/kernel_opening_summary/relation", &relation.digest);
    tr.append_message(b"wasm/kernel_opening_summary/prepared_steps", &prepared_steps.digest);

    WasmKernelOpeningSummary {
        relation,
        prepared_steps,
        digest: tr.digest32(),
    }
}

pub fn verify_kernel_opening_summary(
    expected: &WasmKernelOpeningSummary,
    proof: &WasmKernelProof,
    prepared_steps: &[WasmStepBuild],
) -> Result<(), String> {
    let recomputed = build_kernel_opening_summary(proof, prepared_steps);
    if &recomputed != expected {
        return Err("wasm kernel opening summary mismatch".into());
    }
    Ok(())
}

fn build_relation_summary(proof: &WasmKernelProof) -> WasmKernelRelationOpeningSummary {
    let lookup_rows_digest = digest_lookup_rows(&proof.relation.lookup_rows);
    let memory_events_digest = digest_memory_events(&proof.relation.memory_events);
    let boundary_rows_digest = digest_boundary_rows(&proof.relation.boundary_rows);
    let first_lookup_row = proof
        .relation
        .lookup_rows
        .first()
        .map(|row| lookup_row_ref(0, row));
    let last_lookup_row = proof
        .relation
        .lookup_rows
        .last()
        .map(|row| lookup_row_ref(proof.relation.lookup_rows.len() as u64 - 1, row));
    let first_memory_event = proof
        .relation
        .memory_events
        .first()
        .map(|event| memory_event_ref(0, event));
    let last_memory_event = proof
        .relation
        .memory_events
        .last()
        .map(|event| memory_event_ref(proof.relation.memory_events.len() as u64 - 1, event));

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_relation");
    tr.append_message(b"wasm/kernel_opening_relation/lookup_rows_digest", &lookup_rows_digest);
    tr.append_message(
        b"wasm/kernel_opening_relation/memory_events_digest",
        &memory_events_digest,
    );
    tr.append_message(
        b"wasm/kernel_opening_relation/boundary_rows_digest",
        &boundary_rows_digest,
    );
    tr.append_u64s(
        b"wasm/kernel_opening_relation/counts",
        &[
            proof.relation.lookup_rows.len() as u64,
            proof.relation.memory_events.len() as u64,
            proof.relation.boundary_rows.len() as u64,
            proof.relation.final_stack_slots.len() as u64,
            proof.relation.final_local_slots.len() as u64,
        ],
    );
    append_optional_ref(
        &mut tr,
        b"wasm/kernel_opening_relation/first_lookup_row",
        first_lookup_row.as_ref(),
    );
    append_optional_ref(
        &mut tr,
        b"wasm/kernel_opening_relation/last_lookup_row",
        last_lookup_row.as_ref(),
    );
    append_optional_ref(
        &mut tr,
        b"wasm/kernel_opening_relation/first_memory_event",
        first_memory_event.as_ref(),
    );
    append_optional_ref(
        &mut tr,
        b"wasm/kernel_opening_relation/last_memory_event",
        last_memory_event.as_ref(),
    );

    WasmKernelRelationOpeningSummary {
        lookup_rows_digest,
        memory_events_digest,
        boundary_rows_digest,
        lookup_row_count: proof.relation.lookup_rows.len() as u64,
        memory_event_count: proof.relation.memory_events.len() as u64,
        boundary_row_count: proof.relation.boundary_rows.len() as u64,
        final_stack_slot_count: proof.relation.final_stack_slots.len() as u64,
        final_local_slot_count: proof.relation.final_local_slots.len() as u64,
        first_lookup_row,
        last_lookup_row,
        first_memory_event,
        last_memory_event,
        digest: tr.digest32(),
    }
}

fn build_prepared_step_summary(prepared_steps: &[WasmStepBuild]) -> WasmKernelPreparedStepSummary {
    let steps_digest = digest_prepared_steps(prepared_steps);
    let first_step = prepared_steps
        .first()
        .map(|step| prepared_step_ref(0, step));
    let last_step = prepared_steps
        .last()
        .map(|step| prepared_step_ref(prepared_steps.len() as u64 - 1, step));

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_prepared_steps");
    tr.append_message(b"wasm/kernel_opening_prepared_steps/steps_digest", &steps_digest);
    tr.append_u64s(
        b"wasm/kernel_opening_prepared_steps/counts",
        &[prepared_steps.len() as u64],
    );
    append_optional_ref(
        &mut tr,
        b"wasm/kernel_opening_prepared_steps/first_step",
        first_step.as_ref(),
    );
    append_optional_ref(
        &mut tr,
        b"wasm/kernel_opening_prepared_steps/last_step",
        last_step.as_ref(),
    );

    WasmKernelPreparedStepSummary {
        steps_digest,
        step_count: prepared_steps.len() as u64,
        first_step,
        last_step,
        digest: tr.digest32(),
    }
}

fn digest_lookup_rows(rows: &[WasmLookupRow]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_lookup_rows");
    tr.append_u64s(b"wasm/kernel_opening_lookup_rows/count", &[rows.len() as u64]);
    for row in rows {
        let mut fields = vec![
            row.trace_index as u64,
            row.cycle,
            row.pc_before,
            row.shout_opcode as u64,
            u64::from(row.shout_id),
            u64::from(row.arity.width()),
            row.inputs.len() as u64,
        ];
        fields.extend(row.inputs.iter().map(|value| u64::from(*value)));
        fields.push(row.outputs.len() as u64);
        fields.extend(row.outputs.iter().map(|value| u64::from(*value)));
        tr.append_u64s(b"wasm/kernel_opening_lookup_rows/row", &fields);
    }
    tr.digest32()
}

fn digest_memory_events(events: &[WasmMemoryEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_memory_events");
    tr.append_u64s(b"wasm/kernel_opening_memory_events/count", &[events.len() as u64]);
    for event in events {
        tr.append_u64s(
            b"wasm/kernel_opening_memory_events/event",
            &[
                event.family as u64,
                event.kind as u64,
                event.trace_index.map(|v| v as u64).unwrap_or(u64::MAX),
                event.cycle,
                event.addr0,
                event.addr1,
                event.value0,
                event.value1,
            ],
        );
    }
    tr.digest32()
}

fn digest_boundary_rows(rows: &[WasmBoundaryRow]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_boundary_rows");
    tr.append_u64s(b"wasm/kernel_opening_boundary_rows/count", &[rows.len() as u64]);
    for row in rows {
        tr.append_u64s(
            b"wasm/kernel_opening_boundary_rows/row",
            &[
                row.trace_index as u64,
                row.cycle,
                row.pc_before,
                row.pc_after,
                row.sp_before,
                row.sp_after,
                row.memory_pages_before.map(u64::from).unwrap_or(u64::MAX),
                row.memory_pages_after.map(u64::from).unwrap_or(u64::MAX),
                u64::from(row.halted),
            ],
        );
    }
    tr.digest32()
}

fn digest_prepared_steps(steps: &[WasmStepBuild]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_prepared_step_rows");
    tr.append_u64s(b"wasm/kernel_opening_prepared_step_rows/count", &[steps.len() as u64]);
    for step in steps {
        tr.append_message(b"wasm/kernel_opening_prepared_step_rows/label", step.label.as_bytes());
        tr.append_u64s(
            b"wasm/kernel_opening_prepared_step_rows/meta",
            &[step.assignment.len() as u64],
        );
        tr.append_fields(b"wasm/kernel_opening_prepared_step_rows/assignment", &step.assignment);
    }
    tr.digest32()
}

fn lookup_row_ref(logical_index: u64, row: &WasmLookupRow) -> WasmKernelSelectedRowRef {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_lookup_row_ref_value");
    let mut fields = vec![
        row.trace_index as u64,
        row.cycle,
        row.pc_before,
        row.shout_opcode as u64,
        u64::from(row.shout_id),
        u64::from(row.arity.width()),
        row.inputs.len() as u64,
    ];
    fields.extend(row.inputs.iter().map(|value| u64::from(*value)));
    fields.push(row.outputs.len() as u64);
    fields.extend(row.outputs.iter().map(|value| u64::from(*value)));
    tr.append_u64s(b"wasm/kernel_opening_lookup_row_ref_value/row", &fields);
    selected_ref(b"wasm/kernel_opening_lookup_row_ref", logical_index, tr.digest32())
}

fn memory_event_ref(logical_index: u64, event: &WasmMemoryEvent) -> WasmKernelSelectedRowRef {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_memory_event_ref_value");
    tr.append_u64s(
        b"wasm/kernel_opening_memory_event_ref_value/event",
        &[
            event.family as u64,
            event.kind as u64,
            event.trace_index.map(|v| v as u64).unwrap_or(u64::MAX),
            event.cycle,
            event.addr0,
            event.addr1,
            event.value0,
            event.value1,
        ],
    );
    selected_ref(b"wasm/kernel_opening_memory_event_ref", logical_index, tr.digest32())
}

fn prepared_step_ref(logical_index: u64, step: &WasmStepBuild) -> WasmKernelSelectedRowRef {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_opening_prepared_step_ref_value");
    tr.append_message(
        b"wasm/kernel_opening_prepared_step_ref_value/label",
        step.label.as_bytes(),
    );
    tr.append_u64s(
        b"wasm/kernel_opening_prepared_step_ref_value/meta",
        &[step.assignment.len() as u64],
    );
    tr.append_fields(
        b"wasm/kernel_opening_prepared_step_ref_value/assignment",
        &step.assignment,
    );
    selected_ref(b"wasm/kernel_opening_prepared_step_ref", logical_index, tr.digest32())
}

fn selected_ref(label: &'static [u8], logical_index: u64, value_digest: [u8; 32]) -> WasmKernelSelectedRowRef {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/wasm/kernel_selected_row_ref");
    tr.append_message(b"wasm/kernel_selected_row_ref/label", label);
    tr.append_u64s(b"wasm/kernel_selected_row_ref/index", &[logical_index]);
    tr.append_message(b"wasm/kernel_selected_row_ref/value_digest", &value_digest);
    WasmKernelSelectedRowRef {
        logical_index,
        value_digest,
        digest: tr.digest32(),
    }
}

fn append_optional_ref(
    tr: &mut Poseidon2Transcript,
    label: &'static [u8],
    reference: Option<&WasmKernelSelectedRowRef>,
) {
    tr.append_u64s(label, &[u64::from(reference.is_some())]);
    if let Some(reference) = reference {
        tr.append_u64s(label, &[reference.logical_index]);
        tr.append_message(label, &reference.value_digest);
        tr.append_message(label, &reference.digest);
    }
}
