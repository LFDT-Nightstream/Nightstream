//! Owns the exact Stage 1 row-binding and helper-result summaries for the RV32IM parity slice.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::kernel::{
    family_word, opcode_word, trace_virtual_opcode_word, Stage1ArtifactSurface, Stage1PackagedOpeningProof,
};
use crate::rv32im::lower::{Rv32ExpandedRow, Rv32TraceVirtualOpcode};
use crate::rv32im::tables::Rv32FamilyTag;

use crate::rv32im::isa::Rv32Opcode;

use super::semantic_inputs::{build_sem_inputs, sem_inputs_digest, SemIn};
use super::semantics::{build_stage1_semantics_proof_from_digests, Stage1SemanticsProof};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1RowBinding {
    pub trace_index: usize,
    pub step_index: usize,
    pub sequence_index: usize,
    pub fetch_pc: u32,
    pub fetched_word: u32,
    pub opcode: Rv32Opcode,
    pub trace_opcode: Option<Rv32Opcode>,
    pub trace_virtual_opcode: Option<Rv32TraceVirtualOpcode>,
    pub family: Rv32FamilyTag,
    pub next_pc: u32,
    pub alu_result: u32,
    pub effective_addr: Option<u32>,
    pub writes_rd: bool,
    pub rd: u8,
    pub rd_after: u32,
    pub is_first_in_sequence: bool,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_effect_row: bool,
    pub is_commit_row: bool,
    pub is_real: bool,
    pub preserves_x0: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1Summary {
    pub rows: Vec<Stage1RowBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BytecodeShoutProof {
    pub rows_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AluShoutProof {
    pub sem_inputs_digest: [u8; 32],
    pub effect_trace_index: u64,
    pub commit_trace_index: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BranchShoutProof {
    pub sem_inputs_digest: [u8; 32],
    pub first_trace_index: u64,
    pub last_trace_index: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1AddressCorrectnessProof {
    pub row_count: u64,
    pub effect_row_count: u64,
    pub commit_row_count: u64,
    pub real_row_count: u64,
    pub preserves_x0_count: u64,
    pub rows_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1LinkageProof {
    pub rows_digest: [u8; 32],
    pub sem_inputs_digest: [u8; 32],
    pub mix: u64,
    pub first_trace_index: u64,
    pub effect_trace_index: u64,
    pub commit_trace_index: u64,
    pub last_trace_index: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage1ProofBundle {
    pub sem_inputs: Vec<SemIn>,
    pub row_bindings: Vec<Stage1RowBinding>,
    pub bytecode: BytecodeShoutProof,
    pub alu: AluShoutProof,
    pub branch: BranchShoutProof,
    pub semantics: Stage1SemanticsProof,
    pub address_correctness: Stage1AddressCorrectnessProof,
    pub linkage: Stage1LinkageProof,
    pub selected_opening: Stage1PackagedOpeningProof,
    pub digest: [u8; 32],
}

pub(crate) fn stage1_row_words(row: &Stage1RowBinding) -> [u64; 23] {
    [
        row.trace_index as u64,
        row.step_index as u64,
        row.sequence_index as u64,
        u64::from(row.fetch_pc),
        row.fetched_word as u64,
        opcode_word(row.opcode),
        row.trace_opcode.map(opcode_word).unwrap_or(0),
        row.trace_virtual_opcode
            .map(trace_virtual_opcode_word)
            .unwrap_or(0),
        row.trace_opcode.is_some() as u64,
        row.trace_virtual_opcode.is_some() as u64,
        family_word(row.family),
        u64::from(row.next_pc),
        u64::from(row.alu_result),
        u64::from(row.effective_addr.unwrap_or(0)),
        row.writes_rd as u64,
        row.rd as u64,
        u64::from(row.rd_after),
        row.is_first_in_sequence as u64,
        row.virtual_sequence_remaining.unwrap_or(u16::MAX) as u64,
        row.is_effect_row as u64,
        row.is_commit_row as u64,
        row.is_real as u64,
        row.preserves_x0 as u64,
    ]
}

pub fn stage1_row_word_width() -> usize {
    stage1_row_words(&Stage1RowBinding {
        trace_index: 0,
        step_index: 0,
        sequence_index: 0,
        fetch_pc: 0,
        fetched_word: 0,
        opcode: Rv32Opcode::Addi,
        trace_opcode: None,
        trace_virtual_opcode: None,
        family: Rv32FamilyTag::NativeAlu,
        next_pc: 0,
        alu_result: 0,
        effective_addr: None,
        writes_rd: false,
        rd: 0,
        rd_after: 0,
        is_first_in_sequence: false,
        virtual_sequence_remaining: None,
        is_effect_row: false,
        is_commit_row: false,
        is_real: false,
        preserves_x0: true,
    })
    .len()
}

pub(crate) fn stage1_row_digest(row: &Stage1RowBinding) -> [u8; 32] {
    let words = stage1_row_words(row);
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_selected_row");
    tr.append_u64s_iter(b"stage1/row", words.len(), words.into_iter());
    tr.digest32()
}

pub(crate) fn stage1_row_binding_from_row(row: &Rv32ExpandedRow) -> Stage1RowBinding {
    Stage1RowBinding {
        trace_index: row.trace_index,
        step_index: row.step_index,
        sequence_index: row.sequence_index,
        fetch_pc: row.pc,
        fetched_word: row.word,
        opcode: row.opcode,
        trace_opcode: row.trace_opcode,
        trace_virtual_opcode: row.trace_virtual_opcode,
        family: row.family,
        next_pc: row.next_pc,
        alu_result: row.alu_result,
        effective_addr: row.effective_addr,
        writes_rd: row.writes_rd,
        rd: row.rd,
        rd_after: row.rd_after,
        is_first_in_sequence: row.is_first_in_sequence,
        virtual_sequence_remaining: row.virtual_sequence_remaining,
        is_effect_row: row.is_effect_row,
        is_commit_row: row.is_commit_row,
        is_real: row.is_real,
        preserves_x0: row.rd == 0 || !row.writes_rd,
    }
}

pub fn build_stage1_summary(rows: &[Rv32ExpandedRow]) -> Stage1Summary {
    Stage1Summary {
        rows: rows.iter().map(stage1_row_binding_from_row).collect(),
    }
}

impl BytecodeShoutProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_bytecode_shout_proof");
        tr.append_message(b"rv32im/stage1_bytecode_shout_proof/rows_digest", &self.rows_digest);
        tr.append_message(
            b"rv32im/stage1_bytecode_shout_proof/packaged_digest",
            &self.packaged_digest,
        );
        tr.digest32()
    }
}

impl AluShoutProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_alu_shout_proof");
        tr.append_message(
            b"rv32im/stage1_alu_shout_proof/sem_inputs_digest",
            &self.sem_inputs_digest,
        );
        tr.append_u64s(
            b"rv32im/stage1_alu_shout_proof/meta",
            &[self.effect_trace_index, self.commit_trace_index],
        );
        tr.digest32()
    }
}

impl BranchShoutProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_branch_shout_proof");
        tr.append_message(
            b"rv32im/stage1_branch_shout_proof/sem_inputs_digest",
            &self.sem_inputs_digest,
        );
        tr.append_u64s(
            b"rv32im/stage1_branch_shout_proof/meta",
            &[self.first_trace_index, self.last_trace_index],
        );
        tr.digest32()
    }
}

impl Stage1AddressCorrectnessProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_address_correctness_proof");
        tr.append_message(
            b"rv32im/stage1_address_correctness_proof/rows_digest",
            &self.rows_digest,
        );
        tr.append_u64s(
            b"rv32im/stage1_address_correctness_proof/meta",
            &[
                self.row_count,
                self.effect_row_count,
                self.commit_row_count,
                self.real_row_count,
                self.preserves_x0_count,
            ],
        );
        tr.digest32()
    }
}

impl Stage1LinkageProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_linkage_proof");
        tr.append_message(b"rv32im/stage1_linkage_proof/rows_digest", &self.rows_digest);
        tr.append_message(
            b"rv32im/stage1_linkage_proof/sem_inputs_digest",
            &self.sem_inputs_digest,
        );
        tr.append_u64s(
            b"rv32im/stage1_linkage_proof/meta",
            &[
                self.mix,
                self.first_trace_index,
                self.effect_trace_index,
                self.commit_trace_index,
                self.last_trace_index,
            ],
        );
        tr.digest32()
    }
}

impl Stage1ProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_proof_bundle");
        tr.append_message(b"rv32im/stage1_proof_bundle/bytecode", &self.bytecode.digest);
        tr.append_message(b"rv32im/stage1_proof_bundle/alu", &self.alu.digest);
        tr.append_message(b"rv32im/stage1_proof_bundle/branch", &self.branch.digest);
        tr.append_message(b"rv32im/stage1_proof_bundle/semantics", &self.semantics.digest);
        tr.append_message(
            b"rv32im/stage1_proof_bundle/address_correctness",
            &self.address_correctness.digest,
        );
        tr.append_message(b"rv32im/stage1_proof_bundle/linkage", &self.linkage.digest);
        tr.append_message(
            b"rv32im/stage1_proof_bundle/selected_opening",
            &self.selected_opening.digest,
        );
        tr.append_u64s(
            b"rv32im/stage1_proof_bundle/meta",
            &[self.sem_inputs.len() as u64, self.row_bindings.len() as u64],
        );
        tr.digest32()
    }
}

pub fn build_stage1_proof_bundle(
    rows: &[Rv32ExpandedRow],
    summary: &Stage1Summary,
    artifact: &Stage1ArtifactSurface,
    selected_opening: &Stage1PackagedOpeningProof,
) -> Stage1ProofBundle {
    let sem_inputs = build_sem_inputs(rows);
    let sem_inputs_digest = sem_inputs_digest(&sem_inputs);
    let first_trace_index = summary
        .rows
        .first()
        .map(|row| row.trace_index as u64)
        .unwrap_or(0);
    let effect_trace_index = summary
        .rows
        .iter()
        .find(|row| row.is_effect_row)
        .map(|row| row.trace_index as u64)
        .unwrap_or(0);
    let commit_trace_index = summary
        .rows
        .iter()
        .find(|row| row.is_commit_row)
        .map(|row| row.trace_index as u64)
        .unwrap_or(0);
    let last_trace_index = summary
        .rows
        .last()
        .map(|row| row.trace_index as u64)
        .unwrap_or(0);

    let bytecode = BytecodeShoutProof {
        rows_digest: artifact.rows.rows_digest,
        packaged_digest: selected_opening.digest,
        digest: [0; 32],
    };
    let bytecode = BytecodeShoutProof {
        digest: bytecode.expected_digest(),
        ..bytecode
    };
    let alu = AluShoutProof {
        sem_inputs_digest,
        effect_trace_index,
        commit_trace_index,
        digest: [0; 32],
    };
    let alu = AluShoutProof {
        digest: alu.expected_digest(),
        ..alu
    };
    let branch = BranchShoutProof {
        sem_inputs_digest,
        first_trace_index,
        last_trace_index,
        digest: [0; 32],
    };
    let branch = BranchShoutProof {
        digest: branch.expected_digest(),
        ..branch
    };
    let address_correctness = Stage1AddressCorrectnessProof {
        row_count: artifact.claim.row_count as u64,
        effect_row_count: artifact.claim.effect_row_count as u64,
        commit_row_count: artifact.claim.commit_row_count as u64,
        real_row_count: artifact.claim.real_row_count as u64,
        preserves_x0_count: artifact.claim.preserves_x0_count as u64,
        rows_digest: artifact.rows.rows_digest,
        digest: [0; 32],
    };
    let address_correctness = Stage1AddressCorrectnessProof {
        digest: address_correctness.expected_digest(),
        ..address_correctness
    };
    let semantics =
        build_stage1_semantics_proof_from_digests(sem_inputs_digest, artifact.rows.rows_digest, &sem_inputs);
    let linkage = Stage1LinkageProof {
        rows_digest: artifact.rows.rows_digest,
        sem_inputs_digest,
        mix: artifact.claim.mix,
        first_trace_index,
        effect_trace_index,
        commit_trace_index,
        last_trace_index,
        digest: [0; 32],
    };
    let linkage = Stage1LinkageProof {
        digest: linkage.expected_digest(),
        ..linkage
    };
    let bundle = Stage1ProofBundle {
        sem_inputs,
        row_bindings: summary.rows.clone(),
        bytecode,
        alu,
        branch,
        semantics,
        address_correctness,
        linkage,
        selected_opening: selected_opening.clone(),
        digest: [0; 32],
    };
    Stage1ProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}
