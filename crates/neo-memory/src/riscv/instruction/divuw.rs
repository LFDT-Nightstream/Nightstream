use crate::riscv::instruction::{
    rv64_word_helper_decomposition, DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator,
};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Divuw;

impl InstructionDescriptor for Divuw {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Divuw)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Divuw, rs1, rs2, xlen)
}

/// RV64 DIVUW decomposition:
///
/// 1. `v_out <- VDIVUW(rs1, rs2)` computes the raw low-word unsigned quotient bits.
/// 2. `v_sign <- VMOVSIGNW(v_out)` extracts the sign-fill mask.
/// 3. `v_out <- compose(v_out, v_sign)` locally reconstructs the RV64 result.
/// 4. the final non-virtual row commits `rd <- v_out`.
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    rv64_word_helper_decomposition(rd, rs1, rs2, RiscvOpcode::VirtualDivuWord, alloc)
}
