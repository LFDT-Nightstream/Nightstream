use crate::riscv::instruction::{
    rv64_word_helper_decomposition, DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator,
};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Mulw;

impl InstructionDescriptor for Mulw {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Mulw)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Mulw, rs1, rs2, xlen)
}

/// Jolt-style MULW decomposition sequence.
///
/// We keep the semantics on existing helper lookup tables:
/// 1. `v_out <- VMULW(rs1, rs2)` computes the raw low-word product bits.
/// 2. `v_sign <- VMOVSIGNW(v_out)` extracts the sign-fill mask.
/// 3. `v_out <- compose(v_out, v_sign)` locally reconstructs the RV64 result.
/// 4. the final non-virtual row commits `rd <- v_out`.
///
/// This keeps the virtual rows aligned with existing Shout-owned helper tables
/// instead of introducing a native virtual `Mulw` lookup row.
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    rv64_word_helper_decomposition(rd, rs1, rs2, RiscvOpcode::VirtualMulWord, alloc)
}
