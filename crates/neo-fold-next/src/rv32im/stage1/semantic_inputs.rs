//! Owns the canonical Stage 1 semantic-input surface derived from lowered RV32IM rows.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::isa::Rv32Opcode;
use crate::rv32im::kernel::{family_word, opcode_word, trace_virtual_opcode_word};
use crate::rv32im::lower::{Rv32ExpandedRow, Rv32TraceVirtualOpcode};
use crate::rv32im::tables::Rv32FamilyTag;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SemIn {
    pub trace_index: usize,
    pub step_index: usize,
    pub sequence_index: usize,
    pub pc: u32,
    pub opcode: Rv32Opcode,
    pub trace_opcode: Option<Rv32Opcode>,
    pub trace_virtual_opcode: Option<Rv32TraceVirtualOpcode>,
    pub family: Rv32FamilyTag,
    pub arch_rs1: u8,
    pub arch_rs1_value: u32,
    pub arch_rs2: u8,
    pub arch_rs2_value: u32,
    pub arch_rd: u8,
    pub arch_rd_before: u32,
    pub arch_imm: i32,
    pub rs1: u8,
    pub rs1_value: u32,
    pub rs2: u8,
    pub rs2_value: u32,
    pub rd: u8,
    pub rd_before: u32,
    pub rd_after: u32,
    pub imm: i32,
    pub effective_addr: Option<u32>,
    pub memory_before: Option<u32>,
    pub memory_after: Option<u32>,
    pub mem_width_bytes: Option<u8>,
    pub mem_unsigned: Option<bool>,
    pub writes_rd: bool,
    pub writes_ram: bool,
    pub is_first_in_sequence: bool,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_effect_row: bool,
    pub is_commit_row: bool,
    pub is_real: bool,
}

fn memory_payload_shape(opcode: Rv32Opcode) -> (Option<u8>, Option<bool>) {
    match opcode {
        Rv32Opcode::Lb => (Some(1), Some(false)),
        Rv32Opcode::Lbu => (Some(1), Some(true)),
        Rv32Opcode::Lh => (Some(2), Some(false)),
        Rv32Opcode::Lhu => (Some(2), Some(true)),
        Rv32Opcode::Lw => (Some(4), Some(false)),
        Rv32Opcode::Sb => (Some(1), None),
        Rv32Opcode::Sh => (Some(2), None),
        Rv32Opcode::Sw => (Some(4), None),
        _ => (None, None),
    }
}

pub fn sem_in_from_row(row: &Rv32ExpandedRow) -> SemIn {
    let (mem_width_bytes, mem_unsigned) = memory_payload_shape(row.opcode);
    SemIn {
        trace_index: row.trace_index,
        step_index: row.step_index,
        sequence_index: row.sequence_index,
        pc: row.pc,
        opcode: row.opcode,
        trace_opcode: row.trace_opcode,
        trace_virtual_opcode: row.trace_virtual_opcode,
        family: row.family,
        arch_rs1: row.arch_rs1,
        arch_rs1_value: row.arch_rs1_value,
        arch_rs2: row.arch_rs2,
        arch_rs2_value: row.arch_rs2_value,
        arch_rd: row.arch_rd,
        arch_rd_before: row.arch_rd_before,
        arch_imm: row.arch_imm,
        rs1: row.rs1,
        rs1_value: row.rs1_value,
        rs2: row.rs2,
        rs2_value: row.rs2_value,
        rd: row.rd,
        rd_before: row.rd_before,
        rd_after: row.rd_after,
        imm: row.imm,
        effective_addr: row.effective_addr,
        memory_before: row.memory_before,
        memory_after: row.memory_after,
        mem_width_bytes,
        mem_unsigned,
        writes_rd: row.writes_rd,
        writes_ram: row.writes_ram,
        is_first_in_sequence: row.is_first_in_sequence,
        virtual_sequence_remaining: row.virtual_sequence_remaining,
        is_effect_row: row.is_effect_row,
        is_commit_row: row.is_commit_row,
        is_real: row.is_real,
    }
}

pub fn build_sem_inputs(rows: &[Rv32ExpandedRow]) -> Vec<SemIn> {
    rows.iter().map(sem_in_from_row).collect()
}

pub(crate) fn sem_in_words(input: &SemIn) -> [u64; 39] {
    [
        input.trace_index as u64,
        input.step_index as u64,
        input.sequence_index as u64,
        u64::from(input.pc),
        opcode_word(input.opcode),
        input.trace_opcode.map(opcode_word).unwrap_or(0),
        input
            .trace_virtual_opcode
            .map(trace_virtual_opcode_word)
            .unwrap_or(0),
        input.trace_opcode.is_some() as u64,
        input.trace_virtual_opcode.is_some() as u64,
        family_word(input.family),
        input.arch_rs1 as u64,
        u64::from(input.arch_rs1_value),
        input.arch_rs2 as u64,
        u64::from(input.arch_rs2_value),
        input.arch_rd as u64,
        u64::from(input.arch_rd_before),
        u64::from(input.arch_imm as u32),
        input.rs1 as u64,
        u64::from(input.rs1_value),
        input.rs2 as u64,
        u64::from(input.rs2_value),
        input.rd as u64,
        u64::from(input.rd_before),
        u64::from(input.rd_after),
        u64::from(input.imm as u32),
        u64::from(input.effective_addr.unwrap_or(0)),
        input.effective_addr.is_some() as u64,
        u64::from(input.memory_before.unwrap_or(0)),
        input.memory_before.is_some() as u64,
        u64::from(input.memory_after.unwrap_or(0)),
        input.memory_after.is_some() as u64,
        input.mem_width_bytes.unwrap_or(0) as u64,
        input.mem_width_bytes.is_some() as u64,
        input.mem_unsigned.unwrap_or(false) as u64,
        input.mem_unsigned.is_some() as u64,
        input.writes_rd as u64,
        input.writes_ram as u64,
        input.is_first_in_sequence as u64,
        input.virtual_sequence_remaining.unwrap_or(u16::MAX) as u64,
    ]
}

pub fn sem_in_digest(input: &SemIn) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_sem_in");
    tr.append_u64s_iter(
        b"rv32im/stage1_sem_in/words",
        sem_in_words(input).len(),
        sem_in_words(input),
    );
    tr.append_u64s(
        b"rv32im/stage1_sem_in/flags",
        &[
            input.is_effect_row as u64,
            input.is_commit_row as u64,
            input.is_real as u64,
        ],
    );
    tr.digest32()
}

pub fn sem_inputs_digest(inputs: &[SemIn]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage1_sem_inputs");
    tr.append_u64s(b"rv32im/stage1_sem_inputs/len", &[inputs.len() as u64]);
    for input in inputs {
        tr.append_message(b"rv32im/stage1_sem_inputs/entry", &sem_in_digest(input));
    }
    tr.digest32()
}
