//! Owns deterministic Stage-3 continuity and bridge semantics for accepted-proof verification.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::isa::Rv32Opcode;
use crate::rv32im::kernel::RootExecutionBundle;
use crate::rv32im::lower::Rv32ExpandedRow;

use super::ContinuityEvent;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3SemanticsProof {
    pub continuity_digest: [u8; 32],
    pub root_semantic_rows_digest: [u8; 32],
    pub row_chunk_routes_digest: [u8; 32],
    pub prepared_step_bindings_digest: [u8; 32],
    pub stage2_temporal_digest: [u8; 32],
    pub initial_pc: u64,
    pub final_pc: u64,
    pub real_row_count: u64,
    pub first_real_step_index: u64,
    pub last_real_step_index: u64,
    pub digest: [u8; 32],
}

impl Stage3SemanticsProof {
    pub(crate) fn new(
        continuity_digest: [u8; 32],
        root_execution: &RootExecutionBundle,
        stage2_temporal_digest: [u8; 32],
        initial_pc: u64,
        final_pc: u64,
        continuity: &[ContinuityEvent],
    ) -> Self {
        let proof = Self {
            continuity_digest,
            root_semantic_rows_digest: root_execution.semantic_rows_digest,
            row_chunk_routes_digest: root_execution.row_chunk_routes_digest,
            prepared_step_bindings_digest: root_execution.prepared_step_bindings.digest,
            stage2_temporal_digest,
            initial_pc,
            final_pc,
            real_row_count: continuity.len() as u64,
            first_real_step_index: continuity
                .first()
                .map(|event| event.step_index as u64)
                .unwrap_or(0),
            last_real_step_index: continuity
                .last()
                .map(|event| event.step_index as u64)
                .unwrap_or(0),
            digest: [0; 32],
        };
        Self {
            digest: proof.expected_digest(),
            ..proof
        }
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage3_semantics_proof");
        tr.append_message(
            b"rv32im/stage3_semantics_proof/continuity_digest",
            &self.continuity_digest,
        );
        tr.append_message(
            b"rv32im/stage3_semantics_proof/root_semantic_rows_digest",
            &self.root_semantic_rows_digest,
        );
        tr.append_message(
            b"rv32im/stage3_semantics_proof/row_chunk_routes_digest",
            &self.row_chunk_routes_digest,
        );
        tr.append_message(
            b"rv32im/stage3_semantics_proof/prepared_step_bindings_digest",
            &self.prepared_step_bindings_digest,
        );
        tr.append_message(
            b"rv32im/stage3_semantics_proof/stage2_temporal_digest",
            &self.stage2_temporal_digest,
        );
        tr.append_u64s(
            b"rv32im/stage3_semantics_proof/meta",
            &[
                self.initial_pc,
                self.final_pc,
                self.real_row_count,
                self.first_real_step_index,
                self.last_real_step_index,
            ],
        );
        tr.digest32()
    }
}

pub fn verify_stage3_semantics(
    continuity: &[ContinuityEvent],
    rows: &[Rv32ExpandedRow],
    root_execution: &RootExecutionBundle,
    initial_pc: u64,
    final_pc: u64,
) -> Result<(), String> {
    if continuity.is_empty() {
        return Err("stage3 continuity cannot be empty".into());
    }
    if root_execution.semantic_rows.len() != rows.len() || root_execution.row_chunk_routes.len() != rows.len() {
        return Err("stage3 root execution bridge length mismatch".into());
    }
    for (logical_index, route) in root_execution.row_chunk_routes.iter().enumerate() {
        if route.logical_index != logical_index as u64 {
            return Err(format!(
                "stage3 row-to-chunk route lost logical ordering at semantic row {}",
                logical_index
            ));
        }
    }

    let real_rows = rows.iter().filter(|row| row.is_real).collect::<Vec<_>>();
    if real_rows.len() != continuity.len() {
        return Err("stage3 continuity count does not match the real-row count".into());
    }

    let first = real_rows[0];
    if u64::from(first.pc) != initial_pc {
        return Err(format!(
            "stage3 start-boundary mismatch: expected initial pc 0x{initial_pc:016x}, got 0x{:016x}",
            first.pc
        ));
    }

    for (index, (event, row)) in continuity.iter().zip(real_rows.iter()).enumerate() {
        let successor_pc = real_rows.get(index + 1).map(|next| next.pc);
        let final_step = index + 1 == real_rows.len();
        let continuity_holds = successor_pc.is_none_or(|pc| row.next_pc == pc);
        if event.step_index != row.step_index
            || event.pc != row.pc
            || event.next_pc != row.next_pc
            || event.successor_pc != successor_pc
            || event.final_step != final_step
            || event.continuity_holds != continuity_holds
        {
            return Err(format!(
                "stage3 semantic bridge mismatch at real row step {}",
                row.step_index
            ));
        }
    }

    let last = real_rows
        .last()
        .ok_or_else(|| "stage3 continuity is missing the final real row".to_string())?;
    let last_event = continuity
        .last()
        .ok_or_else(|| "stage3 continuity is missing the final bridge event".to_string())?;
    if !last.is_commit_row || !last.halted || last.trace_opcode != Some(Rv32Opcode::Ecall) {
        return Err("stage3 final-boundary is not a terminating sequence-final ECALL row".into());
    }
    if !last_event.final_step || last_event.successor_pc.is_some() {
        return Err("stage3 final continuity event is malformed".into());
    }
    if u64::from(last.next_pc) != final_pc {
        return Err(format!(
            "stage3 final-boundary mismatch: expected final pc 0x{final_pc:016x}, got 0x{:016x}",
            last.next_pc
        ));
    }
    Ok(())
}
