//! Owns the published Nightstream statement boundary and proof-binding digests.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::proof::FoldSchedule;

pub mod chip8;
pub mod rv32im;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NightstreamStatement {
    pub public_io_digest: [u8; 32],
    pub verifier_context_digest: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub semantic_step_count: u64,
    pub proof_binding_root: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NightstreamProofBindingInputs {
    pub main_proof_digest: [u8; 32],
    pub side_proof_digest: [u8; 32],
    pub public_statement_digest: [u8; 32],
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
    tr.append_message(b"neo.fold.next/nightstream/statement_core/version", b"v2");
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
        &[statement.semantic_step_count],
    );
    tr.digest32()
}

pub fn nightstream_proof_binding_root(
    statement_core_digest: [u8; 32],
    inputs: &NightstreamProofBindingInputs,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/proof_binding_root");
    tr.append_message(b"neo.fold.next/nightstream/proof_binding_root/version", b"v3");
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/statement_core_digest",
        &statement_core_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/main_proof_digest",
        &inputs.main_proof_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/side_proof_digest",
        &inputs.side_proof_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/proof_binding_root/public_statement_digest",
        &inputs.public_statement_digest,
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
