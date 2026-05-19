//! Owns the sharded source/derived parity artifacts for the RV32IM parity corpus.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::builder::build_program;
use crate::rv32im::ccs::{
    RV32IM_PARITY_CASE_NAME_LABEL, RV32IM_PARITY_EXECUTION_DIGEST_LABEL, RV32IM_PARITY_FINAL_STATE_DIGEST_LABEL,
    RV32IM_PARITY_INITIAL_MEMORY_LABEL, RV32IM_PARITY_INITIAL_REGS_LABEL, RV32IM_PARITY_KERNEL_FINAL_MIX_LABEL,
    RV32IM_PARITY_PROGRAM_WORDS_LABEL, RV32IM_PARITY_ROOT0_DIGEST_LABEL, RV32IM_PARITY_STAGE1_DIGEST_LABEL,
    RV32IM_PARITY_STAGE1_MIX_LABEL, RV32IM_PARITY_STAGE2_DIGEST_LABEL, RV32IM_PARITY_STAGE2_RAM_MIX_LABEL,
    RV32IM_PARITY_STAGE2_REG_MIX_LABEL, RV32IM_PARITY_STAGE3_CONTINUITY_MIX_LABEL, RV32IM_PARITY_STAGE3_DIGEST_LABEL,
    RV32IM_PARITY_TRANSCRIPT_APP_LABEL, RV32IM_PARITY_TRANSCRIPT_SEED_LABEL,
};
use crate::rv32im::isa::{
    encode_add, encode_addi, encode_and, encode_andi, encode_auipc, encode_beq, encode_bge, encode_bgeu, encode_blt,
    encode_bltu, encode_bne, encode_div, encode_divu, encode_ecall, encode_fence, encode_jal, encode_jalr, encode_lb,
    encode_lbu, encode_lh, encode_lhu, encode_lui, encode_lw, encode_mul, encode_mulh, encode_mulhsu, encode_mulhu,
    encode_or, encode_ori, encode_rem, encode_remu, encode_sb, encode_sh, encode_sll, encode_slli, encode_slt,
    encode_slti, encode_sltiu, encode_sltu, encode_sra, encode_srai, encode_srl, encode_srli, encode_sub, encode_sw,
    encode_xor, encode_xori, MemoryWord, Rv32BuildError, Rv32Opcode, Rv32Program, Rv32State,
};
use crate::rv32im::layout::{
    RV32IM_PARITY_LOWERING_VERSION_ID, RV32IM_PARITY_PROTOCOL_VERSION_ID, RV32_REGISTER_COUNT,
};
use crate::rv32im::lower::Rv32ExpandedRow;
use crate::rv32im::stage1::{build_stage1_summary, Stage1Summary};
use crate::rv32im::stage2::{build_stage2_summary, Stage2Summary};
use crate::rv32im::stage3::{build_stage3_summary, Stage3Summary};
use crate::rv32im::tables::{
    Rv32FamilyTag, RV32IM_ALIGNED_MEMORY_FOCUS_FIXTURE_ID, RV32IM_CONTROL_FLOW_BEQ_FIXTURE_ID,
    RV32IM_CONTROL_FLOW_BGEU_FIXTURE_ID, RV32IM_CONTROL_FLOW_BGE_FIXTURE_ID, RV32IM_CONTROL_FLOW_BLTU_FIXTURE_ID,
    RV32IM_CONTROL_FLOW_BLT_FIXTURE_ID, RV32IM_CONTROL_FLOW_BNE_FIXTURE_ID, RV32IM_CONTROL_FLOW_FOCUS_FIXTURE_ID,
    RV32IM_CONTROL_FLOW_JALR_FIXTURE_ID, RV32IM_CONTROL_FLOW_JAL_FIXTURE_ID, RV32IM_MULTIPLY_HIGH_FIXTURE_ID,
    RV32IM_MULTIPLY_LOW_FIXTURE_ID, RV32IM_NARROW_MEMORY_LOAD_FIXTURE_ID, RV32IM_NARROW_MEMORY_STORE_FIXTURE_ID,
    RV32IM_NATIVE_ALU_FOCUS_FIXTURE_ID, RV32IM_NATIVE_LOGIC_COMPARE_FIXTURE_ID,
    RV32IM_NATIVE_RV32_SHIFT_MASK_FIXTURE_ID, RV32IM_NATIVE_RV32_WRAP_FIXTURE_ID, RV32IM_NATIVE_SHIFT_FIXTURE_ID,
    RV32IM_NATIVE_UPPER_FIXTURE_ID, RV32IM_SIGNED_DIVREM_FIXTURE_ID, RV32IM_UNSIGNED_DIVREM_FIXTURE_ID,
    RV32IM_VERTICAL_SLICE_FIXTURE_ID,
};

use super::transcript::{LoggingTranscript, TranscriptRecord};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imParityCaseManifest {
    pub name: String,
    pub fixture_id: String,
    pub protocol_version_id: u64,
    pub lowering_version_id: u64,
    pub family_tags: Vec<Rv32FamilyTag>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imParitySourceCase {
    pub manifest: Rv32imParityCaseManifest,
    pub start_pc: u32,
    pub program_words: Vec<u32>,
    pub initial_registers: [u32; RV32_REGISTER_COUNT],
    pub initial_memory: Vec<MemoryWord>,
    pub transcript_seed: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imKernelSummary {
    pub root0_digest: [u8; 32],
    pub stage1_digest: [u8; 32],
    pub stage2_digest: [u8; 32],
    pub stage3_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub final_state_digest: [u8; 32],
    pub stage1_mix: u64,
    pub stage2_reg_mix: u64,
    pub stage2_ram_mix: u64,
    pub stage3_continuity_mix: u64,
    pub kernel_final_mix: u64,
    pub transcript_final_digest: [u8; 32],
    pub final_pc: u32,
    pub final_registers: [u32; RV32_REGISTER_COUNT],
    pub final_memory: Vec<MemoryWord>,
    pub halted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imParityDerivedCase {
    pub manifest: Rv32imParityCaseManifest,
    pub execution_rows: Vec<Rv32ExpandedRow>,
    pub stage1: Stage1Summary,
    pub stage2: Stage2Summary,
    pub stage3: Stage3Summary,
    pub transcript: TranscriptRecord,
    pub kernel: Rv32imKernelSummary,
}

fn make_manifest(name: &str, fixture_id: &str, family_tags: Vec<Rv32FamilyTag>) -> Rv32imParityCaseManifest {
    Rv32imParityCaseManifest {
        name: name.into(),
        fixture_id: fixture_id.into(),
        protocol_version_id: RV32IM_PARITY_PROTOCOL_VERSION_ID,
        lowering_version_id: RV32IM_PARITY_LOWERING_VERSION_ID,
        family_tags,
    }
}

pub fn vertical_slice_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "vertical_add_sw_lw_ecall",
        RV32IM_VERTICAL_SLICE_FIXTURE_ID,
        vec![
            Rv32FamilyTag::NativeAlu,
            Rv32FamilyTag::AlignedMemory,
            Rv32FamilyTag::ControlFlow,
        ],
    )
}

pub fn native_alu_focus_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "native_add_chain_x0_ecall",
        RV32IM_NATIVE_ALU_FOCUS_FIXTURE_ID,
        vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
    )
}

pub fn native_logic_compare_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "native_logic_compare_chain_ecall",
        RV32IM_NATIVE_LOGIC_COMPARE_FIXTURE_ID,
        vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
    )
}

pub fn native_shift_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "native_shift_chain_ecall",
        RV32IM_NATIVE_SHIFT_FIXTURE_ID,
        vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
    )
}

pub fn native_rv32_wrap_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "native_rv32_wrap_chain_ecall",
        RV32IM_NATIVE_RV32_WRAP_FIXTURE_ID,
        vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
    )
}

pub fn native_rv32_shift_mask_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "native_rv32_shift_mask_chain_ecall",
        RV32IM_NATIVE_RV32_SHIFT_MASK_FIXTURE_ID,
        vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
    )
}

pub fn native_upper_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "native_sub_lui_auipc_fence_ecall",
        RV32IM_NATIVE_UPPER_FIXTURE_ID,
        vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
    )
}

pub fn narrow_memory_load_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "narrow_memory_load_extract_extend_ecall",
        RV32IM_NARROW_MEMORY_LOAD_FIXTURE_ID,
        vec![Rv32FamilyTag::NarrowMemory, Rv32FamilyTag::ControlFlow],
    )
}

pub fn narrow_memory_store_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "narrow_memory_store_blend_ecall",
        RV32IM_NARROW_MEMORY_STORE_FIXTURE_ID,
        vec![Rv32FamilyTag::NarrowMemory, Rv32FamilyTag::ControlFlow],
    )
}

pub fn multiply_low_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "multiply_low_mul_ecall",
        RV32IM_MULTIPLY_LOW_FIXTURE_ID,
        vec![Rv32FamilyTag::Multiply, Rv32FamilyTag::ControlFlow],
    )
}

pub fn multiply_high_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "multiply_high_mulh_mulhu_mulhsu_ecall",
        RV32IM_MULTIPLY_HIGH_FIXTURE_ID,
        vec![Rv32FamilyTag::Multiply, Rv32FamilyTag::ControlFlow],
    )
}

pub fn unsigned_divrem_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "unsigned_divrem_chain_ecall",
        RV32IM_UNSIGNED_DIVREM_FIXTURE_ID,
        vec![Rv32FamilyTag::UnsignedDivRem, Rv32FamilyTag::ControlFlow],
    )
}

pub fn signed_divrem_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "signed_divrem_chain_ecall",
        RV32IM_SIGNED_DIVREM_FIXTURE_ID,
        vec![Rv32FamilyTag::SignedDivRem, Rv32FamilyTag::ControlFlow],
    )
}

pub fn aligned_memory_focus_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "aligned_negative_offset_roundtrip",
        RV32IM_ALIGNED_MEMORY_FOCUS_FIXTURE_ID,
        vec![
            Rv32FamilyTag::NativeAlu,
            Rv32FamilyTag::AlignedMemory,
            Rv32FamilyTag::ControlFlow,
        ],
    )
}

pub fn control_flow_focus_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_ecall_only",
        RV32IM_CONTROL_FLOW_FOCUS_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow],
    )
}

pub fn control_flow_jal_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_jal_skip_ecall",
        RV32IM_CONTROL_FLOW_JAL_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow],
    )
}

pub fn control_flow_jalr_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_jalr_skip_ecall",
        RV32IM_CONTROL_FLOW_JALR_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow],
    )
}

pub fn control_flow_beq_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_beq_taken_skip_ecall",
        RV32IM_CONTROL_FLOW_BEQ_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow],
    )
}

pub fn control_flow_bne_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_bne_taken_skip_ecall",
        RV32IM_CONTROL_FLOW_BNE_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow],
    )
}

pub fn control_flow_blt_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_blt_taken_skip_ecall",
        RV32IM_CONTROL_FLOW_BLT_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow, Rv32FamilyTag::NativeAlu],
    )
}

pub fn control_flow_bge_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_bge_taken_skip_ecall",
        RV32IM_CONTROL_FLOW_BGE_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow, Rv32FamilyTag::NativeAlu],
    )
}

pub fn control_flow_bltu_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_bltu_taken_skip_ecall",
        RV32IM_CONTROL_FLOW_BLTU_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow, Rv32FamilyTag::NativeAlu],
    )
}

pub fn control_flow_bgeu_manifest() -> Rv32imParityCaseManifest {
    make_manifest(
        "control_flow_bgeu_taken_skip_ecall",
        RV32IM_CONTROL_FLOW_BGEU_FIXTURE_ID,
        vec![Rv32FamilyTag::ControlFlow, Rv32FamilyTag::NativeAlu],
    )
}

fn vertical_slice_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[10] = 0x1000;
    let program_words = vec![
        encode_addi(1, 0, 5),
        encode_add(2, 1, 1),
        encode_sw(2, 10, 0),
        encode_lw(3, 10, 0),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: vertical_slice_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: registers,
        initial_memory: vec![MemoryWord { addr: 0x1000, value: 0 }],
        transcript_seed: b"rv32im-vertical-slice-v1".to_vec(),
    }
}

fn native_alu_focus_source_case() -> Rv32imParitySourceCase {
    let program_words = vec![
        encode_addi(1, 0, 7),
        encode_addi(2, 1, 9),
        encode_add(3, 2, 1),
        encode_addi(0, 3, 5),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: native_alu_focus_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-native-alu-focus-v1".to_vec(),
    }
}

fn native_logic_compare_source_case() -> Rv32imParitySourceCase {
    let program_words = vec![
        encode_addi(1, 0, 5),
        encode_addi(2, 0, 3),
        encode_and(3, 1, 2),
        encode_andi(4, 1, 6),
        encode_or(5, 1, 2),
        encode_ori(6, 2, 8),
        encode_xor(7, 1, 2),
        encode_xori(8, 1, 7),
        encode_slt(9, 2, 1),
        encode_slti(10, 2, 4),
        encode_sltu(11, 2, 1),
        encode_sltiu(12, 1, 4),
        encode_fence(),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: native_logic_compare_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-native-logic-compare-v1".to_vec(),
    }
}

fn native_shift_source_case() -> Rv32imParitySourceCase {
    let program_words = vec![
        encode_addi(1, 0, 1),
        encode_slli(2, 1, 4),
        encode_addi(3, 0, -16),
        encode_srli(4, 2, 2),
        encode_srai(5, 3, 2),
        encode_addi(6, 0, 3),
        encode_sll(7, 1, 6),
        encode_srl(8, 2, 6),
        encode_sra(9, 3, 6),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: native_shift_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-native-shift-v1".to_vec(),
    }
}

fn native_rv32_wrap_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[3] = 0x7fff_ffff;
    registers[4] = 2;
    registers[5] = 0;
    registers[6] = 1;
    Rv32imParitySourceCase {
        manifest: native_rv32_wrap_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, -1),
            encode_addi(2, 1, 2),
            encode_add(7, 3, 4),
            encode_sub(8, 5, 6),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-native-word-arith-v1".to_vec(),
    }
}

fn native_rv32_shift_mask_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = 1;
    registers[2] = 0x8000_0000;
    registers[6] = 40;
    Rv32imParitySourceCase {
        manifest: native_rv32_shift_mask_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_slli(3, 1, 31),
            encode_srli(4, 2, 4),
            encode_srai(5, 2, 4),
            encode_sll(7, 1, 6),
            encode_srl(8, 2, 6),
            encode_sra(9, 2, 6),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-native-word-shift-v1".to_vec(),
    }
}

fn native_upper_source_case() -> Rv32imParitySourceCase {
    let program_words = vec![
        encode_addi(1, 0, 9),
        encode_addi(2, 0, 4),
        encode_sub(3, 1, 2),
        encode_lui(4, 0x1234_5000),
        encode_auipc(5, 0x0000_2000),
        encode_fence(),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: native_upper_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-native-upper-v1".to_vec(),
    }
}

fn narrow_memory_load_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[10] = 0x3000;
    Rv32imParitySourceCase {
        manifest: narrow_memory_load_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_lb(1, 10, 0),
            encode_lbu(2, 10, 1),
            encode_lh(3, 10, 0),
            encode_lhu(4, 10, 2),
            encode_lw(5, 10, 0),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![MemoryWord {
            addr: 0x3000,
            value: 0x807f_80ff,
        }],
        transcript_seed: b"rv32im-narrow-memory-load-v1".to_vec(),
    }
}

fn narrow_memory_store_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = u32::MAX;
    registers[2] = 0x0123;
    registers[3] = 0x1234_5067;
    registers[10] = 0x4000;
    Rv32imParitySourceCase {
        manifest: narrow_memory_store_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_sb(1, 10, 1),
            encode_sh(2, 10, 2),
            encode_sw(3, 10, 4),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![MemoryWord {
            addr: 0x4000,
            value: 0x4433_2211,
        }],
        transcript_seed: b"rv32im-narrow-memory-store-v1".to_vec(),
    }
}

fn multiply_low_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = 3;
    registers[2] = 5;
    registers[3] = u32::MAX;
    registers[4] = 5;
    Rv32imParitySourceCase {
        manifest: multiply_low_manifest(),
        start_pc: 0,
        program_words: vec![encode_mul(5, 1, 2), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-multiply-low-v1".to_vec(),
    }
}

fn multiply_high_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    // Keep this parity case within the live DEC budget while still producing
    // non-zero high-word results for the Mulh/Mulhu/Mulhsu lowering path.
    registers[1] = 1 << 30;
    registers[2] = 1 << 29;
    registers[3] = 1 << 31;
    registers[4] = 1 << 28;
    registers[5] = 1 << 30;
    registers[6] = 1 << 27;
    Rv32imParitySourceCase {
        manifest: multiply_high_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_mulh(7, 1, 2),
            encode_mulhu(8, 3, 4),
            encode_mulhsu(9, 5, 6),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-multiply-high-v1".to_vec(),
    }
}

fn unsigned_divrem_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = 20;
    registers[2] = 6;
    registers[3] = u32::MAX;
    registers[4] = 3;
    registers[9] = 9;
    registers[13] = 0x8000_0001;
    Rv32imParitySourceCase {
        manifest: unsigned_divrem_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_divu(5, 1, 2),
            encode_remu(6, 1, 2),
            encode_divu(11, 9, 10),
            encode_remu(12, 9, 10),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-unsigned-divrem-v1".to_vec(),
    }
}

fn signed_divrem_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = (-20i32) as u32;
    registers[2] = 6;
    registers[3] = i32::MIN as u32;
    registers[4] = (-1i32) as u32;
    registers[9] = (-9i32) as u32;
    registers[10] = 4;
    registers[13] = 7;
    registers[17] = 0x8000_0001;
    Rv32imParitySourceCase {
        manifest: signed_divrem_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_div(5, 1, 2),
            encode_rem(6, 1, 2),
            encode_div(7, 3, 4),
            encode_rem(8, 3, 4),
            encode_div(15, 13, 14),
            encode_rem(16, 13, 14),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-signed-divrem-v1".to_vec(),
    }
}

fn aligned_memory_focus_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[10] = 0x2008;
    let program_words = vec![
        encode_addi(1, 0, 42),
        encode_sw(1, 10, -8),
        encode_lw(2, 10, -8),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: aligned_memory_focus_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: registers,
        initial_memory: vec![
            MemoryWord {
                addr: 0x2000,
                value: 13,
            },
            MemoryWord {
                addr: 0x2008,
                value: 99,
            },
        ],
        transcript_seed: b"rv32im-aligned-memory-focus-v1".to_vec(),
    }
}

fn control_flow_focus_source_case() -> Rv32imParitySourceCase {
    Rv32imParitySourceCase {
        manifest: control_flow_focus_manifest(),
        start_pc: 0,
        program_words: vec![encode_ecall()],
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-focus-v1".to_vec(),
    }
}

fn control_flow_jal_source_case() -> Rv32imParitySourceCase {
    Rv32imParitySourceCase {
        manifest: control_flow_jal_manifest(),
        start_pc: 0,
        program_words: vec![encode_jal(1, 8), encode_ecall(), encode_ecall()],
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-jal-v1".to_vec(),
    }
}

fn control_flow_jalr_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[5] = 8;
    Rv32imParitySourceCase {
        manifest: control_flow_jalr_manifest(),
        start_pc: 0,
        program_words: vec![encode_jalr(1, 5, 0), encode_ecall(), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-jalr-v1".to_vec(),
    }
}

fn control_flow_beq_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = 11;
    registers[2] = 11;
    Rv32imParitySourceCase {
        manifest: control_flow_beq_manifest(),
        start_pc: 0,
        program_words: vec![encode_beq(1, 2, 8), encode_ecall(), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-beq-v1".to_vec(),
    }
}

fn control_flow_bne_source_case() -> Rv32imParitySourceCase {
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[1] = 11;
    registers[2] = 12;
    Rv32imParitySourceCase {
        manifest: control_flow_bne_manifest(),
        start_pc: 0,
        program_words: vec![encode_bne(1, 2, 8), encode_ecall(), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-bne-v1".to_vec(),
    }
}

fn control_flow_blt_source_case() -> Rv32imParitySourceCase {
    Rv32imParitySourceCase {
        manifest: control_flow_blt_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, -1),
            encode_addi(2, 0, 1),
            encode_blt(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-blt-v1".to_vec(),
    }
}

fn control_flow_bge_source_case() -> Rv32imParitySourceCase {
    Rv32imParitySourceCase {
        manifest: control_flow_bge_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, 1),
            encode_addi(2, 0, -1),
            encode_bge(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-bge-v1".to_vec(),
    }
}

fn control_flow_bltu_source_case() -> Rv32imParitySourceCase {
    Rv32imParitySourceCase {
        manifest: control_flow_bltu_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, 1),
            encode_addi(2, 0, 2),
            encode_bltu(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-bltu-v1".to_vec(),
    }
}

fn control_flow_bgeu_source_case() -> Rv32imParitySourceCase {
    Rv32imParitySourceCase {
        manifest: control_flow_bgeu_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, 2),
            encode_addi(2, 0, 1),
            encode_bgeu(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u32; RV32_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv32im-control-flow-bgeu-v1".to_vec(),
    }
}

pub fn parity_source_cases() -> Vec<Rv32imParitySourceCase> {
    vec![
        vertical_slice_source_case(),
        native_alu_focus_source_case(),
        native_logic_compare_source_case(),
        native_shift_source_case(),
        native_rv32_wrap_source_case(),
        native_rv32_shift_mask_source_case(),
        native_upper_source_case(),
        narrow_memory_load_source_case(),
        narrow_memory_store_source_case(),
        multiply_low_source_case(),
        multiply_high_source_case(),
        unsigned_divrem_source_case(),
        signed_divrem_source_case(),
        aligned_memory_focus_source_case(),
        control_flow_focus_source_case(),
        control_flow_jal_source_case(),
        control_flow_jalr_source_case(),
        control_flow_beq_source_case(),
        control_flow_bne_source_case(),
        control_flow_blt_source_case(),
        control_flow_bge_source_case(),
        control_flow_bltu_source_case(),
        control_flow_bgeu_source_case(),
    ]
}

pub(crate) fn opcode_word(opcode: Rv32Opcode) -> u64 {
    match opcode {
        Rv32Opcode::Addi => 0,
        Rv32Opcode::Add => 1,
        Rv32Opcode::Sub => 2,
        Rv32Opcode::Andi => 3,
        Rv32Opcode::And => 4,
        Rv32Opcode::Ori => 5,
        Rv32Opcode::Or => 6,
        Rv32Opcode::Xori => 7,
        Rv32Opcode::Xor => 8,
        Rv32Opcode::Slti => 9,
        Rv32Opcode::Slt => 10,
        Rv32Opcode::Sltiu => 11,
        Rv32Opcode::Sltu => 12,
        Rv32Opcode::Slli => 13,
        Rv32Opcode::Sll => 14,
        Rv32Opcode::Srli => 15,
        Rv32Opcode::Srl => 16,
        Rv32Opcode::Srai => 17,
        Rv32Opcode::Sra => 18,
        Rv32Opcode::Lui => 19,
        Rv32Opcode::Auipc => 20,
        Rv32Opcode::Fence => 21,
        Rv32Opcode::Ecall => 22,
        Rv32Opcode::Jal => 23,
        Rv32Opcode::Jalr => 24,
        Rv32Opcode::Beq => 25,
        Rv32Opcode::Bne => 26,
        Rv32Opcode::Blt => 27,
        Rv32Opcode::Bge => 28,
        Rv32Opcode::Bltu => 29,
        Rv32Opcode::Bgeu => 30,
        Rv32Opcode::Lb => 31,
        Rv32Opcode::Lbu => 32,
        Rv32Opcode::Lh => 33,
        Rv32Opcode::Lhu => 34,
        Rv32Opcode::Lw => 35,
        Rv32Opcode::Sb => 36,
        Rv32Opcode::Sh => 37,
        Rv32Opcode::Sw => 38,
        Rv32Opcode::Mul => 39,
        Rv32Opcode::Mulh => 40,
        Rv32Opcode::Mulhsu => 41,
        Rv32Opcode::Mulhu => 42,
        Rv32Opcode::Div => 43,
        Rv32Opcode::Divu => 44,
        Rv32Opcode::Rem => 45,
        Rv32Opcode::Remu => 46,
    }
}

pub(crate) fn family_word(family: Rv32FamilyTag) -> u64 {
    match family {
        Rv32FamilyTag::NativeAlu => 0,
        Rv32FamilyTag::AlignedMemory => 1,
        Rv32FamilyTag::ControlFlow => 2,
        Rv32FamilyTag::NarrowMemory => 3,
        Rv32FamilyTag::Multiply => 4,
        Rv32FamilyTag::UnsignedDivRem => 5,
        Rv32FamilyTag::SignedDivRem => 6,
    }
}

pub(crate) fn register_read_role_word(role: crate::rv32im::stage2::RegisterReadRole) -> u64 {
    match role {
        crate::rv32im::stage2::RegisterReadRole::Rs1 => 0,
        crate::rv32im::stage2::RegisterReadRole::Rs2 => 1,
    }
}

pub(crate) fn ram_access_kind_word(kind: crate::rv32im::stage2::RamAccessKind) -> u64 {
    match kind {
        crate::rv32im::stage2::RamAccessKind::Read => 0,
        crate::rv32im::stage2::RamAccessKind::Write => 1,
    }
}

pub(crate) fn trace_virtual_opcode_word(opcode: crate::rv32im::lower::Rv32TraceVirtualOpcode) -> u64 {
    match opcode {
        crate::rv32im::lower::Rv32TraceVirtualOpcode::Movsign => 0,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::Advice => 1,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::ChangeDivisor => 2,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::AssertValidDiv0 => 3,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::AssertMulNoOverflow => 4,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::AssertLte => 5,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::AssertValidUnsignedRemainder => 6,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::AssertSignedDivIdentity => 7,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::AssertSignedRemainderBounds => 8,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::Move => 9,
        crate::rv32im::lower::Rv32TraceVirtualOpcode::SignExtendWord => 10,
    }
}

fn append_u64_matrix_digest(app_label: &'static [u8], sections: &[(&'static [u8], Vec<u64>)]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(app_label);
    for (label, words) in sections {
        tr.append_u64s(label, words);
    }
    tr.digest32()
}

fn flatten_memory_words(words: &[MemoryWord]) -> Vec<u64> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for word in words {
        out.push(u64::from(word.addr));
        out.push(u64::from(word.value));
    }
    out
}

fn flatten_row(row: &Rv32ExpandedRow) -> Vec<u64> {
    let mut out = vec![
        row.trace_index as u64,
        row.step_index as u64,
        row.sequence_index as u64,
        u64::from(row.pc),
        u64::from(row.next_pc),
        row.word as u64,
        opcode_word(row.opcode),
        row.trace_opcode.map(opcode_word).unwrap_or(0),
        row.trace_virtual_opcode
            .map(trace_virtual_opcode_word)
            .unwrap_or(0),
        row.trace_opcode.is_some() as u64,
        row.trace_virtual_opcode.is_some() as u64,
        family_word(row.family),
        row.rs1 as u64,
        u64::from(row.rs1_value),
        row.rs2 as u64,
        u64::from(row.rs2_value),
        row.rd as u64,
        u64::from(row.rd_before),
        u64::from(row.rd_after),
        row.imm as u64,
        u64::from(row.alu_result),
        row.writes_rd as u64,
        row.writes_ram as u64,
        row.halted as u64,
        row.is_first_in_sequence as u64,
        row.virtual_sequence_remaining.unwrap_or(u16::MAX) as u64,
        row.is_effect_row as u64,
        row.is_commit_row as u64,
        row.is_real as u64,
    ];
    out.push(u64::from(row.effective_addr.unwrap_or(0)));
    out.push(u64::from(row.memory_before.unwrap_or(0)));
    out.push(u64::from(row.memory_after.unwrap_or(0)));
    out
}

pub(super) fn flatten_stage1(stage1: &Stage1Summary) -> Vec<u64> {
    let mut out = Vec::new();
    for row in &stage1.rows {
        out.extend([
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
        ]);
    }
    out
}

pub(super) fn flatten_stage2(stage2: &Stage2Summary) -> Vec<u64> {
    let mut out = Vec::new();
    out.push(stage2.register_reads.len() as u64);
    for event in &stage2.register_reads {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            register_read_role_word(event.role),
            event.reg as u64,
            u64::from(event.value),
        ]);
    }
    out.push(stage2.register_writes.len() as u64);
    for event in &stage2.register_writes {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            event.reg as u64,
            u64::from(event.previous),
            u64::from(event.next),
        ]);
    }
    out.push(stage2.ram_events.len() as u64);
    for event in &stage2.ram_events {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            ram_access_kind_word(event.kind),
            u64::from(event.addr),
            u64::from(event.previous),
            u64::from(event.next),
        ]);
    }
    out.push(stage2.twist_links.len() as u64);
    for event in &stage2.twist_links {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            family_word(event.family),
            u64::from(event.routed_write_value.unwrap_or(0)),
            u64::from(event.routed_memory_before.unwrap_or(0)),
            u64::from(event.routed_memory_after.unwrap_or(0)),
        ]);
    }
    out
}

pub(super) fn flatten_stage3(stage3: &Stage3Summary) -> Vec<u64> {
    let mut out = vec![stage3.halted as u64, stage3.continuity.len() as u64];
    for event in &stage3.continuity {
        out.extend([
            event.step_index as u64,
            u64::from(event.pc),
            u64::from(event.next_pc),
            u64::from(event.successor_pc.unwrap_or(0)),
            event.final_step as u64,
            event.continuity_holds as u64,
        ]);
    }
    out
}

fn flatten_registers(values: &[u32; RV32_REGISTER_COUNT]) -> Vec<u64> {
    values.iter().copied().map(u64::from).collect()
}

pub(crate) fn rv32im_source_case_digest(source: &Rv32imParitySourceCase) -> [u8; 32] {
    append_u64_matrix_digest(
        b"neo.fold.next/rv32im/source_digest_v1",
        &[
            (
                b"source/protocol",
                vec![source.manifest.protocol_version_id, source.manifest.lowering_version_id],
            ),
            (
                b"source/program",
                source
                    .program_words
                    .iter()
                    .map(|word| *word as u64)
                    .collect(),
            ),
            (b"source/regs", flatten_registers(&source.initial_registers)),
            (b"source/memory", flatten_memory_words(&source.initial_memory)),
            (
                b"source/seed",
                source
                    .transcript_seed
                    .iter()
                    .map(|byte| *byte as u64)
                    .collect(),
            ),
        ],
    )
}

pub(crate) fn digest_rows(rows: &[Rv32ExpandedRow]) -> [u8; 32] {
    let mut sections = Vec::with_capacity(rows.len());
    for row in rows {
        sections.push((b"execution/row".as_slice(), flatten_row(row)));
    }
    append_u64_matrix_digest(b"neo.fold.next/rv32im/execution_digest_v1", &sections)
}

fn digest_final_state(final_state: &Rv32State) -> [u8; 32] {
    append_u64_matrix_digest(
        b"neo.fold.next/rv32im/final_state_digest_v1",
        &[
            (b"final/pc", vec![u64::from(final_state.pc)]),
            (b"final/halted", vec![final_state.halted as u64]),
            (b"final/registers", flatten_registers(&final_state.regs)),
            (b"final/memory", flatten_memory_words(&final_state.memory_words())),
        ],
    )
}

pub(super) fn build_kernel_transcript_and_summary_from_parts(
    source: &Rv32imParitySourceCase,
    rows: &[Rv32ExpandedRow],
    stage1: &Stage1Summary,
    stage2: &Stage2Summary,
    stage3: &Stage3Summary,
    final_state: &Rv32State,
) -> (TranscriptRecord, Rv32imKernelSummary) {
    let root0_digest = rv32im_source_case_digest(source);
    let stage1_digest = append_u64_matrix_digest(
        b"neo.fold.next/rv32im/stage1_digest_v1",
        &[(b"stage1/rows", flatten_stage1(stage1))],
    );
    let stage2_digest = append_u64_matrix_digest(
        b"neo.fold.next/rv32im/stage2_digest_v1",
        &[(b"stage2/summary", flatten_stage2(stage2))],
    );
    let stage3_digest = append_u64_matrix_digest(
        b"neo.fold.next/rv32im/stage3_digest_v1",
        &[(b"stage3/summary", flatten_stage3(stage3))],
    );
    let execution_digest = digest_rows(rows);
    let final_state_digest = digest_final_state(final_state);

    let mut transcript = LoggingTranscript::new(RV32IM_PARITY_TRANSCRIPT_APP_LABEL);
    transcript.append_message(RV32IM_PARITY_TRANSCRIPT_SEED_LABEL, &source.transcript_seed);
    transcript.append_message(RV32IM_PARITY_CASE_NAME_LABEL, source.manifest.name.as_bytes());
    transcript.append_u64s(
        RV32IM_PARITY_PROGRAM_WORDS_LABEL,
        &source
            .program_words
            .iter()
            .map(|word| *word as u64)
            .collect::<Vec<_>>(),
    );
    transcript.append_u64s(
        RV32IM_PARITY_INITIAL_REGS_LABEL,
        &flatten_registers(&source.initial_registers),
    );
    transcript.append_u64s(
        RV32IM_PARITY_INITIAL_MEMORY_LABEL,
        &flatten_memory_words(&source.initial_memory),
    );
    transcript.append_message(RV32IM_PARITY_ROOT0_DIGEST_LABEL, &root0_digest);
    let stage1_mix = transcript.challenge_field(RV32IM_PARITY_STAGE1_MIX_LABEL);
    transcript.append_message(RV32IM_PARITY_STAGE1_DIGEST_LABEL, &stage1_digest);
    let stage2_reg_mix = transcript.challenge_field(RV32IM_PARITY_STAGE2_REG_MIX_LABEL);
    let stage2_ram_mix = transcript.challenge_field(RV32IM_PARITY_STAGE2_RAM_MIX_LABEL);
    transcript.append_message(RV32IM_PARITY_STAGE2_DIGEST_LABEL, &stage2_digest);
    let stage3_continuity_mix = transcript.challenge_field(RV32IM_PARITY_STAGE3_CONTINUITY_MIX_LABEL);
    transcript.append_message(RV32IM_PARITY_STAGE3_DIGEST_LABEL, &stage3_digest);
    transcript.append_message(RV32IM_PARITY_EXECUTION_DIGEST_LABEL, &execution_digest);
    transcript.append_message(RV32IM_PARITY_FINAL_STATE_DIGEST_LABEL, &final_state_digest);
    let kernel_final_mix = transcript.challenge_field(RV32IM_PARITY_KERNEL_FINAL_MIX_LABEL);
    let transcript_final_digest = transcript.digest32();
    let transcript = transcript.finish();

    let kernel = Rv32imKernelSummary {
        root0_digest,
        stage1_digest,
        stage2_digest,
        stage3_digest,
        execution_digest,
        final_state_digest,
        stage1_mix,
        stage2_reg_mix,
        stage2_ram_mix,
        stage3_continuity_mix,
        kernel_final_mix,
        transcript_final_digest,
        final_pc: final_state.pc,
        final_registers: final_state.regs,
        final_memory: final_state.memory_words(),
        halted: final_state.halted,
    };
    (transcript, kernel)
}

pub fn build_parity_case_from_source(
    source: Rv32imParitySourceCase,
    max_steps: usize,
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    let program = Rv32Program::new(source.start_pc, source.program_words.clone());
    let initial_state = Rv32State::new(source.start_pc, source.initial_registers, &source.initial_memory);
    let build = build_program(&program, &initial_state, max_steps)?;

    let stage1 = build_stage1_summary(&build.rows);
    let stage2 = build_stage2_summary(&build.rows);
    let stage3 = build_stage3_summary(&build.rows);

    let (transcript, kernel) = build_kernel_transcript_and_summary_from_parts(
        &source,
        &build.rows,
        &stage1,
        &stage2,
        &stage3,
        &build.final_state,
    );

    Ok((
        source.clone(),
        Rv32imParityDerivedCase {
            manifest: source.manifest.clone(),
            execution_rows: build.rows,
            stage1,
            stage2,
            stage3,
            transcript,
            kernel,
        },
    ))
}

pub fn build_vertical_slice_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        vertical_slice_source_case(),
        vertical_slice_source_case().program_words.len(),
    )
}

pub fn build_native_alu_focus_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        native_alu_focus_source_case(),
        native_alu_focus_source_case().program_words.len(),
    )
}

pub fn build_native_logic_compare_parity_case(
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        native_logic_compare_source_case(),
        native_logic_compare_source_case().program_words.len(),
    )
}

pub fn build_native_shift_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        native_shift_source_case(),
        native_shift_source_case().program_words.len(),
    )
}

pub fn build_native_rv32_wrap_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        native_rv32_wrap_source_case(),
        native_rv32_wrap_source_case().program_words.len(),
    )
}

pub fn build_native_rv32_shift_mask_parity_case(
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        native_rv32_shift_mask_source_case(),
        native_rv32_shift_mask_source_case().program_words.len(),
    )
}

pub fn build_native_upper_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        native_upper_source_case(),
        native_upper_source_case().program_words.len(),
    )
}

pub fn build_narrow_memory_load_parity_case(
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        narrow_memory_load_source_case(),
        narrow_memory_load_source_case().program_words.len(),
    )
}

pub fn build_narrow_memory_store_parity_case(
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        narrow_memory_store_source_case(),
        narrow_memory_store_source_case().program_words.len(),
    )
}

pub fn build_multiply_low_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        multiply_low_source_case(),
        multiply_low_source_case().program_words.len(),
    )
}

pub fn build_multiply_high_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        multiply_high_source_case(),
        multiply_high_source_case().program_words.len(),
    )
}

pub fn build_unsigned_divrem_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        unsigned_divrem_source_case(),
        unsigned_divrem_source_case().program_words.len(),
    )
}

pub fn build_signed_divrem_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        signed_divrem_source_case(),
        signed_divrem_source_case().program_words.len(),
    )
}

pub fn build_aligned_memory_focus_parity_case(
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        aligned_memory_focus_source_case(),
        aligned_memory_focus_source_case().program_words.len(),
    )
}

pub fn build_control_flow_focus_parity_case(
) -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError> {
    build_parity_case_from_source(
        control_flow_focus_source_case(),
        control_flow_focus_source_case().program_words.len(),
    )
}

pub fn build_control_flow_jal_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_jal_source_case(),
        control_flow_jal_source_case().program_words.len(),
    )
}

pub fn build_control_flow_jalr_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_jalr_source_case(),
        control_flow_jalr_source_case().program_words.len(),
    )
}

pub fn build_control_flow_beq_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_beq_source_case(),
        control_flow_beq_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bne_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_bne_source_case(),
        control_flow_bne_source_case().program_words.len(),
    )
}

pub fn build_control_flow_blt_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_blt_source_case(),
        control_flow_blt_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bge_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_bge_source_case(),
        control_flow_bge_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bltu_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_bltu_source_case(),
        control_flow_bltu_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bgeu_parity_case() -> Result<(Rv32imParitySourceCase, Rv32imParityDerivedCase), Rv32BuildError>
{
    build_parity_case_from_source(
        control_flow_bgeu_source_case(),
        control_flow_bgeu_source_case().program_words.len(),
    )
}

pub fn build_all_parity_cases() -> Result<Vec<(Rv32imParitySourceCase, Rv32imParityDerivedCase)>, Rv32BuildError> {
    parity_source_cases()
        .into_iter()
        .map(|source| {
            let max_steps = source.program_words.len();
            build_parity_case_from_source(source, max_steps)
        })
        .collect()
}
