//! Owns the public RV32IM expanded-row surface and ordinary-step lowering.

use serde::{Deserialize, Serialize};

use super::execute::ExecutedStep;
use super::isa::Rv32Opcode;
use super::tables::Rv32FamilyTag;
use super::trace_expand::lower_inline_rows;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Rv32TraceVirtualOpcode {
    Movsign,
    Advice,
    ChangeDivisor,
    AssertValidDiv0,
    AssertMulNoOverflow,
    AssertLte,
    AssertValidUnsignedRemainder,
    AssertSignedDivIdentity,
    AssertSignedRemainderBounds,
    Move,
    SignExtendWord,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Rv32TraceOpcode {
    Real(Rv32Opcode),
    Virtual(Rv32TraceVirtualOpcode),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32ExpandedRow {
    pub trace_index: usize,
    pub step_index: usize,
    pub sequence_index: usize,
    pub pc: u32,
    pub next_pc: u32,
    pub word: u32,
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
    pub alu_result: u32,
    pub effective_addr: Option<u32>,
    pub memory_before: Option<u32>,
    pub memory_after: Option<u32>,
    pub writes_rd: bool,
    pub writes_ram: bool,
    pub halted: bool,
    pub is_first_in_sequence: bool,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_effect_row: bool,
    pub is_commit_row: bool,
    pub is_real: bool,
}

fn writes_rd_for_opcode(opcode: Rv32Opcode, rd: u8) -> bool {
    matches!(
        opcode,
        Rv32Opcode::Addi
            | Rv32Opcode::Add
            | Rv32Opcode::Sub
            | Rv32Opcode::Andi
            | Rv32Opcode::And
            | Rv32Opcode::Ori
            | Rv32Opcode::Or
            | Rv32Opcode::Xori
            | Rv32Opcode::Xor
            | Rv32Opcode::Slti
            | Rv32Opcode::Slt
            | Rv32Opcode::Sltiu
            | Rv32Opcode::Sltu
            | Rv32Opcode::Slli
            | Rv32Opcode::Sll
            | Rv32Opcode::Srli
            | Rv32Opcode::Srl
            | Rv32Opcode::Srai
            | Rv32Opcode::Sra
            | Rv32Opcode::Lui
            | Rv32Opcode::Auipc
            | Rv32Opcode::Mul
            | Rv32Opcode::Mulh
            | Rv32Opcode::Mulhsu
            | Rv32Opcode::Mulhu
            | Rv32Opcode::Div
            | Rv32Opcode::Divu
            | Rv32Opcode::Rem
            | Rv32Opcode::Remu
            | Rv32Opcode::Lb
            | Rv32Opcode::Lbu
            | Rv32Opcode::Lh
            | Rv32Opcode::Lhu
            | Rv32Opcode::Lw
            | Rv32Opcode::Jal
            | Rv32Opcode::Jalr
    ) && rd != 0
}

fn ordinary_row(step: &ExecutedStep, trace_index: usize) -> Rv32ExpandedRow {
    Rv32ExpandedRow {
        trace_index,
        step_index: step.step_index,
        sequence_index: 0,
        pc: step.prev.pc,
        next_pc: step.next.pc,
        word: step.word,
        opcode: step.decoded.opcode,
        trace_opcode: Some(step.decoded.opcode),
        trace_virtual_opcode: None,
        family: step.family,
        arch_rs1: step.decoded.rs1,
        arch_rs1_value: step.rs1_value,
        arch_rs2: step.decoded.rs2,
        arch_rs2_value: step.rs2_value,
        arch_rd: step.decoded.rd,
        arch_rd_before: step.rd_before,
        arch_imm: step.decoded.imm,
        rs1: step.decoded.rs1,
        rs1_value: step.rs1_value,
        rs2: step.decoded.rs2,
        rs2_value: step.rs2_value,
        rd: step.decoded.rd,
        rd_before: step.rd_before,
        rd_after: step.next.read_reg(step.decoded.rd),
        imm: step.decoded.imm,
        alu_result: step.alu_result,
        effective_addr: step.effective_addr,
        memory_before: step.memory_before,
        memory_after: step.memory_after,
        writes_rd: writes_rd_for_opcode(step.decoded.opcode, step.decoded.rd),
        writes_ram: matches!(step.decoded.opcode, Rv32Opcode::Sb | Rv32Opcode::Sh | Rv32Opcode::Sw),
        halted: step.next.halted,
        is_first_in_sequence: true,
        virtual_sequence_remaining: None,
        is_effect_row: true,
        is_commit_row: true,
        is_real: true,
    }
}

pub fn lower_step(step: &ExecutedStep, trace_index_start: usize) -> Vec<Rv32ExpandedRow> {
    lower_inline_rows(step, trace_index_start).unwrap_or_else(|| vec![ordinary_row(step, trace_index_start)])
}
