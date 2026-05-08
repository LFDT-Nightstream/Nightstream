//! Owns trace-time inline helper expansion for RV32IM opcodes.
//!
//! RV32IM currently keeps architectural rows as the canonical execution rows.
//! The type surface is retained because Stage 1 packages use the same row
//! schema for real and virtual rows, but no legacy word-op helper lowering is
//! emitted.

use super::execute::ExecutedStep;
use super::isa::Rv32Opcode;
use super::lower::{Rv32ExpandedRow, Rv32TraceOpcode};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TraceInstructionSpec {
    pub(crate) opcode: Rv32TraceOpcode,
    pub(crate) rd: u8,
    pub(crate) rs1: u8,
    pub(crate) rs2: u8,
    pub(crate) imm: i32,
    pub(crate) hint: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InlineTracePlan {
    pub(crate) steps: Vec<TraceInstructionSpec>,
    pub(crate) effect_index: usize,
}

pub(crate) fn canonical_trace_plan(
    _opcode: Rv32Opcode,
    _rs1_value: u32,
    _rs2_value: u32,
    _rs1: u8,
    _rs2: u8,
    _rd: u8,
) -> Option<InlineTracePlan> {
    None
}

pub(crate) fn lower_inline_rows(_step: &ExecutedStep, _trace_index_start: usize) -> Option<Vec<Rv32ExpandedRow>> {
    None
}
