//! Program-building and machine-level RV32IM API.

pub use super::super::builder::{build_program, Rv32ProgramBuild};
pub use super::super::isa::{
    decode_instruction, encode_add, encode_addi, encode_and, encode_andi, encode_auipc, encode_beq, encode_bge,
    encode_bgeu, encode_blt, encode_bltu, encode_bne, encode_div, encode_divu, encode_ecall, encode_fence, encode_jal,
    encode_jalr, encode_lb, encode_lbu, encode_lh, encode_lhu, encode_lui, encode_lw, encode_mul, encode_mulh,
    encode_mulhsu, encode_mulhu, encode_or, encode_ori, encode_rem, encode_remu, encode_sb, encode_sh, encode_sll,
    encode_slli, encode_slt, encode_slti, encode_sltiu, encode_sltu, encode_sra, encode_srai, encode_srl, encode_srli,
    encode_sub, encode_sw, encode_xor, encode_xori, MemoryWord, Rv32BuildError, Rv32DecodedInstruction, Rv32Opcode,
    Rv32Program, Rv32State,
};
pub use super::super::kernel::{
    aligned_memory_focus_manifest, build_aligned_memory_focus_parity_case, build_all_parity_cases,
    build_control_flow_beq_parity_case, build_control_flow_bge_parity_case, build_control_flow_bgeu_parity_case,
    build_control_flow_blt_parity_case, build_control_flow_bltu_parity_case, build_control_flow_bne_parity_case,
    build_control_flow_focus_parity_case, build_control_flow_jal_parity_case, build_control_flow_jalr_parity_case,
    build_multiply_high_parity_case, build_multiply_low_parity_case, build_narrow_memory_load_parity_case,
    build_narrow_memory_store_parity_case, build_native_alu_focus_parity_case, build_native_logic_compare_parity_case,
    build_native_rv32_shift_mask_parity_case, build_native_rv32_wrap_parity_case, build_native_shift_parity_case,
    build_native_upper_parity_case, build_parity_case_from_source, build_signed_divrem_parity_case,
    build_unsigned_divrem_parity_case, build_vertical_slice_parity_case, control_flow_beq_manifest,
    control_flow_bge_manifest, control_flow_bgeu_manifest, control_flow_blt_manifest, control_flow_bltu_manifest,
    control_flow_bne_manifest, control_flow_focus_manifest, control_flow_jal_manifest, control_flow_jalr_manifest,
    multiply_high_manifest, multiply_low_manifest, narrow_memory_load_manifest, narrow_memory_store_manifest,
    native_alu_focus_manifest, native_logic_compare_manifest, native_rv32_shift_mask_manifest,
    native_rv32_wrap_manifest, native_shift_manifest, native_upper_manifest, parity_source_cases,
    signed_divrem_manifest, unsigned_divrem_manifest, vertical_slice_manifest, Rv32imParityCaseManifest,
    Rv32imParityDerivedCase, Rv32imParitySourceCase,
};
pub use super::super::lower::{Rv32ExpandedRow, Rv32TraceOpcode, Rv32TraceVirtualOpcode};
