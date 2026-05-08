//! Owns RV32IM parity-slice opcode family tags and lowering metadata.

use serde::{Deserialize, Serialize};

use super::isa::Rv32Opcode;

pub const RV32IM_VERTICAL_SLICE_FIXTURE_ID: &str = "vertical_add_sw_lw_ecall_v1";
pub const RV32IM_NATIVE_ALU_FOCUS_FIXTURE_ID: &str = "native_add_chain_x0_ecall_v1";
pub const RV32IM_ALIGNED_MEMORY_FOCUS_FIXTURE_ID: &str = "aligned_negative_offset_roundtrip_v1";
pub const RV32IM_CONTROL_FLOW_FOCUS_FIXTURE_ID: &str = "control_flow_ecall_only_v1";
pub const RV32IM_CONTROL_FLOW_JAL_FIXTURE_ID: &str = "control_flow_jal_skip_ecall_v1";
pub const RV32IM_CONTROL_FLOW_JALR_FIXTURE_ID: &str = "control_flow_jalr_skip_ecall_v1";
pub const RV32IM_CONTROL_FLOW_BEQ_FIXTURE_ID: &str = "control_flow_beq_taken_skip_ecall_v1";
pub const RV32IM_CONTROL_FLOW_BNE_FIXTURE_ID: &str = "control_flow_bne_taken_skip_ecall_v1";
pub const RV32IM_NATIVE_LOGIC_COMPARE_FIXTURE_ID: &str = "native_logic_compare_chain_ecall_v1";
pub const RV32IM_NATIVE_SHIFT_FIXTURE_ID: &str = "native_shift_chain_ecall_v1";
pub const RV32IM_NATIVE_RV32_WRAP_FIXTURE_ID: &str = "native_rv32_arith_wrap_chain_ecall_v1";
pub const RV32IM_NATIVE_RV32_SHIFT_MASK_FIXTURE_ID: &str = "native_rv32_shift_mask_chain_ecall_v1";
pub const RV32IM_NATIVE_UPPER_FIXTURE_ID: &str = "native_sub_lui_auipc_fence_ecall_v1";
pub const RV32IM_NARROW_MEMORY_LOAD_FIXTURE_ID: &str = "narrow_memory_load_extract_extend_ecall_v1";
pub const RV32IM_NARROW_MEMORY_STORE_FIXTURE_ID: &str = "narrow_memory_store_blend_ecall_v1";
pub const RV32IM_MULTIPLY_LOW_FIXTURE_ID: &str = "multiply_low_mul_ecall_v1";
pub const RV32IM_MULTIPLY_HIGH_FIXTURE_ID: &str = "multiply_high_mulh_mulhu_mulhsu_ecall_v1";
pub const RV32IM_UNSIGNED_DIVREM_FIXTURE_ID: &str = "unsigned_divrem_chain_ecall_v1";
pub const RV32IM_SIGNED_DIVREM_FIXTURE_ID: &str = "signed_divrem_chain_ecall_v1";
pub const RV32IM_CONTROL_FLOW_BLT_FIXTURE_ID: &str = "control_flow_blt_taken_skip_ecall_v1";
pub const RV32IM_CONTROL_FLOW_BGE_FIXTURE_ID: &str = "control_flow_bge_taken_skip_ecall_v1";
pub const RV32IM_CONTROL_FLOW_BLTU_FIXTURE_ID: &str = "control_flow_bltu_taken_skip_ecall_v1";
pub const RV32IM_CONTROL_FLOW_BGEU_FIXTURE_ID: &str = "control_flow_bgeu_taken_skip_ecall_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Rv32FamilyTag {
    NativeAlu,
    AlignedMemory,
    NarrowMemory,
    Multiply,
    UnsignedDivRem,
    SignedDivRem,
    ControlFlow,
}

pub fn opcode_family(opcode: Rv32Opcode) -> Rv32FamilyTag {
    match opcode {
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
        | Rv32Opcode::Fence => Rv32FamilyTag::NativeAlu,
        Rv32Opcode::Lb | Rv32Opcode::Lbu | Rv32Opcode::Lh | Rv32Opcode::Lhu | Rv32Opcode::Sb | Rv32Opcode::Sh => {
            Rv32FamilyTag::NarrowMemory
        }
        Rv32Opcode::Lw | Rv32Opcode::Sw => Rv32FamilyTag::AlignedMemory,
        Rv32Opcode::Mul | Rv32Opcode::Mulh | Rv32Opcode::Mulhsu | Rv32Opcode::Mulhu => Rv32FamilyTag::Multiply,
        Rv32Opcode::Divu | Rv32Opcode::Remu => Rv32FamilyTag::UnsignedDivRem,
        Rv32Opcode::Div | Rv32Opcode::Rem => Rv32FamilyTag::SignedDivRem,
        Rv32Opcode::Jal
        | Rv32Opcode::Jalr
        | Rv32Opcode::Beq
        | Rv32Opcode::Bne
        | Rv32Opcode::Blt
        | Rv32Opcode::Bge
        | Rv32Opcode::Bltu
        | Rv32Opcode::Bgeu
        | Rv32Opcode::Ecall => Rv32FamilyTag::ControlFlow,
    }
}
