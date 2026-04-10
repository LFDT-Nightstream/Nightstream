//! Owns the published Nightstream statement boundary and proof-binding digests.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::finalize::FixedShapeChunkSummary;
use crate::proof::FoldSchedule;

pub mod chip8;
pub mod rv64im;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NightstreamStatement {
    pub public_io_digest: [u8; 32],
    pub verifier_context_digest: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub semantic_step_count: u64,
    pub chunk_summaries: Vec<FixedShapeChunkSummary>,
    pub linkage_root: [u8; 32],
    pub proof_binding_root: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NightstreamProofBindingInputs {
    pub main_decider_proof_digest: [u8; 32],
    pub main_residual_proof_digest: [u8; 32],
    pub side_bridge_artifact_digest: [u8; 32],
    pub linkage_artifact_digest: [u8; 32],
}

impl NightstreamStatement {
    pub fn core_digest(&self) -> [u8; 32] {
        nightstream_statement_core_digest(self)
    }

    pub fn digest(&self) -> [u8; 32] {
        nightstream_statement_digest(self)
    }
}

pub fn nightstream_statement_core_digest(statement: &NightstreamStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/statement_core");
    tr.append_message(b"neo.fold.next/nightstream/statement_core/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/statement_core/public_io_digest",
        &statement.public_io_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/statement_core/verifier_context_digest",
        &statement.verifier_context_digest,
    );
    tr.append_u64s(
        b"neo.fold.next/nightstream/statement_core/fold_schedule",
        &statement.fold_schedule.meta_words(),
    );
    tr.append_u64s(
        b"neo.fold.next/nightstream/statement_core/meta",
        &[statement.semantic_step_count, statement.chunk_summaries.len() as u64],
    );
    for summary in &statement.chunk_summaries {
        tr.append_message(
            b"neo.fold.next/nightstream/statement_core/chunk_summary",
            &summary.digest(),
        );
    }
    tr.append_message(
        b"neo.fold.next/nightstream/statement_core/linkage_root",
        &statement.linkage_root,
    );
    tr.digest32()
}

pub fn nightstream_proof_binding_root(
    statement_core_digest: [u8; 32],
    inputs: &NightstreamProofBindingInputs,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/proof_binding_root");
    tr.append_message(b"neo.fold.next/nightstream/proof_binding_root/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/statement_core_digest",
        &statement_core_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/main_decider_proof_digest",
        &inputs.main_decider_proof_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/main_residual_proof_digest",
        &inputs.main_residual_proof_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/side_bridge_artifact_digest",
        &inputs.side_bridge_artifact_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/linkage_artifact_digest",
        &inputs.linkage_artifact_digest,
    );
    tr.digest32()
}

pub fn nightstream_statement_digest(statement: &NightstreamStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/statement");
    tr.append_message(b"neo.fold.next/nightstream/statement/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/statement/statement_core_digest",
        &nightstream_statement_core_digest(statement),
    );
    tr.append_message(
        b"neo.fold.next/nightstream/statement/proof_binding_root",
        &statement.proof_binding_root,
    );
    tr.digest32()
}
