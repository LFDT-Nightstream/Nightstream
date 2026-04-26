//! Owns the sharded source/derived parity artifacts for the RV64IM parity corpus.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv64im::builder::build_program;
use crate::rv64im::ccs::{
    RV64IM_PARITY_CASE_NAME_LABEL, RV64IM_PARITY_EXECUTION_DIGEST_LABEL, RV64IM_PARITY_FINAL_STATE_DIGEST_LABEL,
    RV64IM_PARITY_INITIAL_MEMORY_LABEL, RV64IM_PARITY_INITIAL_REGS_LABEL, RV64IM_PARITY_KERNEL_FINAL_MIX_LABEL,
    RV64IM_PARITY_PROGRAM_WORDS_LABEL, RV64IM_PARITY_ROOT0_DIGEST_LABEL, RV64IM_PARITY_STAGE1_DIGEST_LABEL,
    RV64IM_PARITY_STAGE1_MIX_LABEL, RV64IM_PARITY_STAGE2_DIGEST_LABEL, RV64IM_PARITY_STAGE2_RAM_MIX_LABEL,
    RV64IM_PARITY_STAGE2_REG_MIX_LABEL, RV64IM_PARITY_STAGE3_CONTINUITY_MIX_LABEL, RV64IM_PARITY_STAGE3_DIGEST_LABEL,
    RV64IM_PARITY_TRANSCRIPT_APP_LABEL, RV64IM_PARITY_TRANSCRIPT_SEED_LABEL,
};
use crate::rv64im::isa::{
    encode_add, encode_addi, encode_addiw, encode_addw, encode_and, encode_andi, encode_auipc, encode_beq, encode_bge,
    encode_bgeu, encode_blt, encode_bltu, encode_bne, encode_div, encode_divu, encode_divuw, encode_divw, encode_ecall,
    encode_fence, encode_jal, encode_jalr, encode_lb, encode_lbu, encode_ld, encode_lh, encode_lhu, encode_lui,
    encode_lw, encode_lwu, encode_mul, encode_mulh, encode_mulhsu, encode_mulhu, encode_mulw, encode_or, encode_ori,
    encode_rem, encode_remu, encode_remuw, encode_remw, encode_sb, encode_sd, encode_sh, encode_sll, encode_slli,
    encode_slliw, encode_sllw, encode_slt, encode_slti, encode_sltiu, encode_sltu, encode_sra, encode_srai,
    encode_sraiw, encode_sraw, encode_srl, encode_srli, encode_srliw, encode_srlw, encode_sub, encode_subw, encode_sw,
    encode_xor, encode_xori, MemoryWord, Rv64BuildError, Rv64Opcode, Rv64Program, Rv64State,
};
use crate::rv64im::layout::{
    RV64IM_PARITY_LOWERING_VERSION_ID, RV64IM_PARITY_PROTOCOL_VERSION_ID, RV64_REGISTER_COUNT,
};
use crate::rv64im::lower::Rv64ExpandedRow;
use crate::rv64im::stage1::{build_stage1_summary, Stage1Summary};
use crate::rv64im::stage2::{build_stage2_summary, Stage2Summary};
use crate::rv64im::stage3::{build_stage3_summary, Stage3Summary};
use crate::rv64im::tables::{
    Rv64FamilyTag, RV64IM_ALIGNED_MEMORY_FOCUS_FIXTURE_ID, RV64IM_CONTROL_FLOW_BEQ_FIXTURE_ID,
    RV64IM_CONTROL_FLOW_BGEU_FIXTURE_ID, RV64IM_CONTROL_FLOW_BGE_FIXTURE_ID, RV64IM_CONTROL_FLOW_BLTU_FIXTURE_ID,
    RV64IM_CONTROL_FLOW_BLT_FIXTURE_ID, RV64IM_CONTROL_FLOW_BNE_FIXTURE_ID, RV64IM_CONTROL_FLOW_FOCUS_FIXTURE_ID,
    RV64IM_CONTROL_FLOW_JALR_FIXTURE_ID, RV64IM_CONTROL_FLOW_JAL_FIXTURE_ID, RV64IM_MULTIPLY_HIGH_FIXTURE_ID,
    RV64IM_MULTIPLY_LOW_FIXTURE_ID, RV64IM_NARROW_MEMORY_LOAD_FIXTURE_ID, RV64IM_NARROW_MEMORY_STORE_FIXTURE_ID,
    RV64IM_NATIVE_ALU_FOCUS_FIXTURE_ID, RV64IM_NATIVE_LOGIC_COMPARE_FIXTURE_ID, RV64IM_NATIVE_SHIFT_FIXTURE_ID,
    RV64IM_NATIVE_UPPER_FIXTURE_ID, RV64IM_NATIVE_WORD_ARITH_FIXTURE_ID, RV64IM_NATIVE_WORD_SHIFT_FIXTURE_ID,
    RV64IM_SIGNED_DIVREM_FIXTURE_ID, RV64IM_UNSIGNED_DIVREM_FIXTURE_ID, RV64IM_VERTICAL_SLICE_FIXTURE_ID,
};

use super::transcript::{LoggingTranscript, TranscriptRecord};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imParityCaseManifest {
    pub name: String,
    pub fixture_id: String,
    pub protocol_version_id: u64,
    pub lowering_version_id: u64,
    pub family_tags: Vec<Rv64FamilyTag>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imParitySourceCase {
    pub manifest: Rv64imParityCaseManifest,
    pub start_pc: u64,
    pub program_words: Vec<u32>,
    pub initial_registers: [u64; RV64_REGISTER_COUNT],
    pub initial_memory: Vec<MemoryWord>,
    pub transcript_seed: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelSummary {
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
    pub final_pc: u64,
    pub final_registers: [u64; RV64_REGISTER_COUNT],
    pub final_memory: Vec<MemoryWord>,
    pub halted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imParityDerivedCase {
    pub manifest: Rv64imParityCaseManifest,
    pub execution_rows: Vec<Rv64ExpandedRow>,
    pub stage1: Stage1Summary,
    pub stage2: Stage2Summary,
    pub stage3: Stage3Summary,
    pub transcript: TranscriptRecord,
    pub kernel: Rv64imKernelSummary,
}

fn make_manifest(name: &str, fixture_id: &str, family_tags: Vec<Rv64FamilyTag>) -> Rv64imParityCaseManifest {
    Rv64imParityCaseManifest {
        name: name.into(),
        fixture_id: fixture_id.into(),
        protocol_version_id: RV64IM_PARITY_PROTOCOL_VERSION_ID,
        lowering_version_id: RV64IM_PARITY_LOWERING_VERSION_ID,
        family_tags,
    }
}

pub fn vertical_slice_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "vertical_add_sd_ld_ecall",
        RV64IM_VERTICAL_SLICE_FIXTURE_ID,
        vec![
            Rv64FamilyTag::NativeAlu,
            Rv64FamilyTag::AlignedMemory,
            Rv64FamilyTag::ControlFlow,
        ],
    )
}

pub fn native_alu_focus_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "native_add_chain_x0_ecall",
        RV64IM_NATIVE_ALU_FOCUS_FIXTURE_ID,
        vec![Rv64FamilyTag::NativeAlu, Rv64FamilyTag::ControlFlow],
    )
}

pub fn native_logic_compare_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "native_logic_compare_chain_ecall",
        RV64IM_NATIVE_LOGIC_COMPARE_FIXTURE_ID,
        vec![Rv64FamilyTag::NativeAlu, Rv64FamilyTag::ControlFlow],
    )
}

pub fn native_shift_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "native_shift_chain_ecall",
        RV64IM_NATIVE_SHIFT_FIXTURE_ID,
        vec![Rv64FamilyTag::NativeAlu, Rv64FamilyTag::ControlFlow],
    )
}

pub fn native_word_arith_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "native_word_arith_chain_ecall",
        RV64IM_NATIVE_WORD_ARITH_FIXTURE_ID,
        vec![Rv64FamilyTag::NativeAlu, Rv64FamilyTag::ControlFlow],
    )
}

pub fn native_word_shift_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "native_word_shift_chain_ecall",
        RV64IM_NATIVE_WORD_SHIFT_FIXTURE_ID,
        vec![Rv64FamilyTag::NativeAlu, Rv64FamilyTag::ControlFlow],
    )
}

pub fn native_upper_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "native_sub_lui_auipc_fence_ecall",
        RV64IM_NATIVE_UPPER_FIXTURE_ID,
        vec![Rv64FamilyTag::NativeAlu, Rv64FamilyTag::ControlFlow],
    )
}

pub fn narrow_memory_load_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "narrow_memory_load_extract_extend_ecall",
        RV64IM_NARROW_MEMORY_LOAD_FIXTURE_ID,
        vec![Rv64FamilyTag::NarrowMemory, Rv64FamilyTag::ControlFlow],
    )
}

pub fn narrow_memory_store_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "narrow_memory_store_blend_ecall",
        RV64IM_NARROW_MEMORY_STORE_FIXTURE_ID,
        vec![Rv64FamilyTag::NarrowMemory, Rv64FamilyTag::ControlFlow],
    )
}

pub fn multiply_low_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "multiply_low_mul_mulw_ecall",
        RV64IM_MULTIPLY_LOW_FIXTURE_ID,
        vec![Rv64FamilyTag::Multiply, Rv64FamilyTag::ControlFlow],
    )
}

pub fn multiply_high_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "multiply_high_mulh_mulhu_mulhsu_ecall",
        RV64IM_MULTIPLY_HIGH_FIXTURE_ID,
        vec![Rv64FamilyTag::Multiply, Rv64FamilyTag::ControlFlow],
    )
}

pub fn unsigned_divrem_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "unsigned_divrem_chain_ecall",
        RV64IM_UNSIGNED_DIVREM_FIXTURE_ID,
        vec![Rv64FamilyTag::UnsignedDivRem, Rv64FamilyTag::ControlFlow],
    )
}

pub fn signed_divrem_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "signed_divrem_chain_ecall",
        RV64IM_SIGNED_DIVREM_FIXTURE_ID,
        vec![Rv64FamilyTag::SignedDivRem, Rv64FamilyTag::ControlFlow],
    )
}

pub fn aligned_memory_focus_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "aligned_negative_offset_roundtrip",
        RV64IM_ALIGNED_MEMORY_FOCUS_FIXTURE_ID,
        vec![
            Rv64FamilyTag::NativeAlu,
            Rv64FamilyTag::AlignedMemory,
            Rv64FamilyTag::ControlFlow,
        ],
    )
}

pub fn control_flow_focus_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_ecall_only",
        RV64IM_CONTROL_FLOW_FOCUS_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow],
    )
}

pub fn control_flow_jal_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_jal_skip_ecall",
        RV64IM_CONTROL_FLOW_JAL_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow],
    )
}

pub fn control_flow_jalr_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_jalr_skip_ecall",
        RV64IM_CONTROL_FLOW_JALR_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow],
    )
}

pub fn control_flow_beq_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_beq_taken_skip_ecall",
        RV64IM_CONTROL_FLOW_BEQ_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow],
    )
}

pub fn control_flow_bne_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_bne_taken_skip_ecall",
        RV64IM_CONTROL_FLOW_BNE_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow],
    )
}

pub fn control_flow_blt_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_blt_taken_skip_ecall",
        RV64IM_CONTROL_FLOW_BLT_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow, Rv64FamilyTag::NativeAlu],
    )
}

pub fn control_flow_bge_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_bge_taken_skip_ecall",
        RV64IM_CONTROL_FLOW_BGE_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow, Rv64FamilyTag::NativeAlu],
    )
}

pub fn control_flow_bltu_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_bltu_taken_skip_ecall",
        RV64IM_CONTROL_FLOW_BLTU_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow, Rv64FamilyTag::NativeAlu],
    )
}

pub fn control_flow_bgeu_manifest() -> Rv64imParityCaseManifest {
    make_manifest(
        "control_flow_bgeu_taken_skip_ecall",
        RV64IM_CONTROL_FLOW_BGEU_FIXTURE_ID,
        vec![Rv64FamilyTag::ControlFlow, Rv64FamilyTag::NativeAlu],
    )
}

fn vertical_slice_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[10] = 0x1000;
    let program_words = vec![
        encode_addi(1, 0, 5),
        encode_add(2, 1, 1),
        encode_sd(2, 10, 0),
        encode_ld(3, 10, 0),
        encode_ecall(),
    ];

    Rv64imParitySourceCase {
        manifest: vertical_slice_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: registers,
        initial_memory: vec![MemoryWord { addr: 0x1000, value: 0 }],
        transcript_seed: b"rv64im-vertical-slice-v1".to_vec(),
    }
}

fn native_alu_focus_source_case() -> Rv64imParitySourceCase {
    let program_words = vec![
        encode_addi(1, 0, 7),
        encode_addi(2, 1, 9),
        encode_add(3, 2, 1),
        encode_addi(0, 3, 5),
        encode_ecall(),
    ];

    Rv64imParitySourceCase {
        manifest: native_alu_focus_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-native-alu-focus-v1".to_vec(),
    }
}

fn native_logic_compare_source_case() -> Rv64imParitySourceCase {
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

    Rv64imParitySourceCase {
        manifest: native_logic_compare_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-native-logic-compare-v1".to_vec(),
    }
}

fn native_shift_source_case() -> Rv64imParitySourceCase {
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

    Rv64imParitySourceCase {
        manifest: native_shift_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-native-shift-v1".to_vec(),
    }
}

fn native_word_arith_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[3] = 0x0000_0000_7fff_ffff;
    registers[4] = 2;
    registers[5] = 0;
    registers[6] = 1;
    Rv64imParitySourceCase {
        manifest: native_word_arith_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addiw(1, 0, -1),
            encode_addiw(2, 1, 2),
            encode_addw(7, 3, 4),
            encode_subw(8, 5, 6),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-native-word-arith-v1".to_vec(),
    }
}

fn native_word_shift_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = 1;
    registers[2] = 0xffff_ffff_8000_0000;
    registers[6] = 40;
    Rv64imParitySourceCase {
        manifest: native_word_shift_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_slliw(3, 1, 31),
            encode_srliw(4, 2, 4),
            encode_sraiw(5, 2, 4),
            encode_sllw(7, 1, 6),
            encode_srlw(8, 2, 6),
            encode_sraw(9, 2, 6),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-native-word-shift-v1".to_vec(),
    }
}

fn native_upper_source_case() -> Rv64imParitySourceCase {
    let program_words = vec![
        encode_addi(1, 0, 9),
        encode_addi(2, 0, 4),
        encode_sub(3, 1, 2),
        encode_lui(4, 0x1234_5000),
        encode_auipc(5, 0x0000_2000),
        encode_fence(),
        encode_ecall(),
    ];

    Rv64imParitySourceCase {
        manifest: native_upper_manifest(),
        start_pc: 0,
        program_words,
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-native-upper-v1".to_vec(),
    }
}

fn narrow_memory_load_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[10] = 0x3000;
    Rv64imParitySourceCase {
        manifest: narrow_memory_load_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_lb(1, 10, 0),
            encode_lbu(2, 10, 1),
            encode_lh(3, 10, 0),
            encode_lhu(4, 10, 2),
            encode_lw(5, 10, 0),
            encode_lwu(6, 10, 4),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![MemoryWord {
            addr: 0x3000,
            value: 0x89ab_cdef_807f_80ff,
        }],
        transcript_seed: b"rv64im-narrow-memory-load-v1".to_vec(),
    }
}

fn narrow_memory_store_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = u64::MAX;
    registers[2] = 0x0123;
    registers[3] = 0x1234_5067;
    registers[10] = 0x4000;
    Rv64imParitySourceCase {
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
            value: 0x8877_6655_4433_2211,
        }],
        transcript_seed: b"rv64im-narrow-memory-store-v1".to_vec(),
    }
}

fn multiply_low_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = 3;
    registers[2] = 5;
    registers[3] = u64::MAX;
    registers[4] = 5;
    Rv64imParitySourceCase {
        manifest: multiply_low_manifest(),
        start_pc: 0,
        program_words: vec![encode_mul(5, 1, 2), encode_mulw(6, 3, 4), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-multiply-low-v1".to_vec(),
    }
}

fn multiply_high_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    // Keep this parity case within the live DEC budget while still producing
    // non-zero high-word results for the Mulh/Mulhu/Mulhsu lowering path.
    registers[1] = 1 << 40;
    registers[2] = 1 << 35;
    registers[3] = 1 << 41;
    registers[4] = 1 << 34;
    registers[5] = 1 << 42;
    registers[6] = 1 << 33;
    Rv64imParitySourceCase {
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
        transcript_seed: b"rv64im-multiply-high-v1".to_vec(),
    }
}

fn unsigned_divrem_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = 20;
    registers[2] = 6;
    registers[3] = u64::MAX;
    registers[4] = 3;
    registers[9] = 9;
    registers[13] = 0xffff_ffff_8000_0001;
    Rv64imParitySourceCase {
        manifest: unsigned_divrem_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_divu(5, 1, 2),
            encode_remu(6, 1, 2),
            encode_divuw(7, 3, 4),
            encode_remuw(8, 3, 4),
            encode_divu(11, 9, 10),
            encode_remu(12, 9, 10),
            encode_divuw(15, 13, 14),
            encode_remuw(16, 13, 14),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-unsigned-divrem-v1".to_vec(),
    }
}

fn signed_divrem_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = (-20i64) as u64;
    registers[2] = 6;
    registers[3] = i64::MIN as u64;
    registers[4] = (-1i64) as u64;
    registers[9] = (-9i64) as u64;
    registers[10] = 4;
    registers[13] = 7;
    registers[17] = 0xffff_ffff_8000_0001;
    Rv64imParitySourceCase {
        manifest: signed_divrem_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_div(5, 1, 2),
            encode_rem(6, 1, 2),
            encode_div(7, 3, 4),
            encode_rem(8, 3, 4),
            encode_divw(11, 9, 10),
            encode_remw(12, 9, 10),
            encode_div(15, 13, 14),
            encode_rem(16, 13, 14),
            encode_divw(19, 17, 18),
            encode_remw(20, 17, 18),
            encode_ecall(),
        ],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-signed-divrem-v1".to_vec(),
    }
}

fn aligned_memory_focus_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[10] = 0x2008;
    let program_words = vec![
        encode_addi(1, 0, 42),
        encode_sd(1, 10, -8),
        encode_ld(2, 10, -8),
        encode_ecall(),
    ];

    Rv64imParitySourceCase {
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
        transcript_seed: b"rv64im-aligned-memory-focus-v1".to_vec(),
    }
}

fn control_flow_focus_source_case() -> Rv64imParitySourceCase {
    Rv64imParitySourceCase {
        manifest: control_flow_focus_manifest(),
        start_pc: 0,
        program_words: vec![encode_ecall()],
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-focus-v1".to_vec(),
    }
}

fn control_flow_jal_source_case() -> Rv64imParitySourceCase {
    Rv64imParitySourceCase {
        manifest: control_flow_jal_manifest(),
        start_pc: 0,
        program_words: vec![encode_jal(1, 8), encode_ecall(), encode_ecall()],
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-jal-v1".to_vec(),
    }
}

fn control_flow_jalr_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[5] = 8;
    Rv64imParitySourceCase {
        manifest: control_flow_jalr_manifest(),
        start_pc: 0,
        program_words: vec![encode_jalr(1, 5, 0), encode_ecall(), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-jalr-v1".to_vec(),
    }
}

fn control_flow_beq_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = 11;
    registers[2] = 11;
    Rv64imParitySourceCase {
        manifest: control_flow_beq_manifest(),
        start_pc: 0,
        program_words: vec![encode_beq(1, 2, 8), encode_ecall(), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-beq-v1".to_vec(),
    }
}

fn control_flow_bne_source_case() -> Rv64imParitySourceCase {
    let mut registers = [0u64; RV64_REGISTER_COUNT];
    registers[1] = 11;
    registers[2] = 12;
    Rv64imParitySourceCase {
        manifest: control_flow_bne_manifest(),
        start_pc: 0,
        program_words: vec![encode_bne(1, 2, 8), encode_ecall(), encode_ecall()],
        initial_registers: registers,
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-bne-v1".to_vec(),
    }
}

fn control_flow_blt_source_case() -> Rv64imParitySourceCase {
    Rv64imParitySourceCase {
        manifest: control_flow_blt_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, -1),
            encode_addi(2, 0, 1),
            encode_blt(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-blt-v1".to_vec(),
    }
}

fn control_flow_bge_source_case() -> Rv64imParitySourceCase {
    Rv64imParitySourceCase {
        manifest: control_flow_bge_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, 1),
            encode_addi(2, 0, -1),
            encode_bge(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-bge-v1".to_vec(),
    }
}

fn control_flow_bltu_source_case() -> Rv64imParitySourceCase {
    Rv64imParitySourceCase {
        manifest: control_flow_bltu_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, 1),
            encode_addi(2, 0, 2),
            encode_bltu(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-bltu-v1".to_vec(),
    }
}

fn control_flow_bgeu_source_case() -> Rv64imParitySourceCase {
    Rv64imParitySourceCase {
        manifest: control_flow_bgeu_manifest(),
        start_pc: 0,
        program_words: vec![
            encode_addi(1, 0, 2),
            encode_addi(2, 0, 1),
            encode_bgeu(1, 2, 8),
            encode_ecall(),
            encode_ecall(),
        ],
        initial_registers: [0u64; RV64_REGISTER_COUNT],
        initial_memory: vec![],
        transcript_seed: b"rv64im-control-flow-bgeu-v1".to_vec(),
    }
}

pub fn parity_source_cases() -> Vec<Rv64imParitySourceCase> {
    vec![
        vertical_slice_source_case(),
        native_alu_focus_source_case(),
        native_logic_compare_source_case(),
        native_shift_source_case(),
        native_word_arith_source_case(),
        native_word_shift_source_case(),
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

pub(crate) fn opcode_word(opcode: Rv64Opcode) -> u64 {
    match opcode {
        Rv64Opcode::Addi => 0,
        Rv64Opcode::Add => 1,
        Rv64Opcode::Sub => 2,
        Rv64Opcode::Andi => 3,
        Rv64Opcode::And => 4,
        Rv64Opcode::Ori => 5,
        Rv64Opcode::Or => 6,
        Rv64Opcode::Xori => 7,
        Rv64Opcode::Xor => 8,
        Rv64Opcode::Slti => 9,
        Rv64Opcode::Slt => 10,
        Rv64Opcode::Sltiu => 11,
        Rv64Opcode::Sltu => 12,
        Rv64Opcode::Slli => 13,
        Rv64Opcode::Sll => 14,
        Rv64Opcode::Srli => 15,
        Rv64Opcode::Srl => 16,
        Rv64Opcode::Srai => 17,
        Rv64Opcode::Sra => 18,
        Rv64Opcode::Lui => 19,
        Rv64Opcode::Auipc => 20,
        Rv64Opcode::Fence => 21,
        Rv64Opcode::Ld => 22,
        Rv64Opcode::Sd => 23,
        Rv64Opcode::Ecall => 24,
        Rv64Opcode::Jal => 25,
        Rv64Opcode::Jalr => 26,
        Rv64Opcode::Beq => 27,
        Rv64Opcode::Bne => 28,
        Rv64Opcode::Blt => 29,
        Rv64Opcode::Bge => 30,
        Rv64Opcode::Bltu => 31,
        Rv64Opcode::Bgeu => 32,
        Rv64Opcode::Lb => 33,
        Rv64Opcode::Lbu => 34,
        Rv64Opcode::Lh => 35,
        Rv64Opcode::Lhu => 36,
        Rv64Opcode::Lw => 37,
        Rv64Opcode::Lwu => 38,
        Rv64Opcode::Sb => 39,
        Rv64Opcode::Sh => 40,
        Rv64Opcode::Sw => 41,
        Rv64Opcode::Mul => 42,
        Rv64Opcode::Mulh => 43,
        Rv64Opcode::Mulhsu => 44,
        Rv64Opcode::Mulhu => 45,
        Rv64Opcode::Mulw => 46,
        Rv64Opcode::Div => 47,
        Rv64Opcode::Divu => 48,
        Rv64Opcode::Rem => 49,
        Rv64Opcode::Remu => 50,
        Rv64Opcode::Divw => 51,
        Rv64Opcode::Divuw => 52,
        Rv64Opcode::Remw => 53,
        Rv64Opcode::Remuw => 54,
        Rv64Opcode::Addiw => 55,
        Rv64Opcode::Addw => 56,
        Rv64Opcode::Subw => 57,
        Rv64Opcode::Slliw => 58,
        Rv64Opcode::Sllw => 59,
        Rv64Opcode::Srliw => 60,
        Rv64Opcode::Srlw => 61,
        Rv64Opcode::Sraiw => 62,
        Rv64Opcode::Sraw => 63,
    }
}

pub(crate) fn family_word(family: Rv64FamilyTag) -> u64 {
    match family {
        Rv64FamilyTag::NativeAlu => 0,
        Rv64FamilyTag::AlignedMemory => 1,
        Rv64FamilyTag::ControlFlow => 2,
        Rv64FamilyTag::NarrowMemory => 3,
        Rv64FamilyTag::Multiply => 4,
        Rv64FamilyTag::UnsignedDivRem => 5,
        Rv64FamilyTag::SignedDivRem => 6,
    }
}

pub(crate) fn register_read_role_word(role: crate::rv64im::stage2::RegisterReadRole) -> u64 {
    match role {
        crate::rv64im::stage2::RegisterReadRole::Rs1 => 0,
        crate::rv64im::stage2::RegisterReadRole::Rs2 => 1,
    }
}

pub(crate) fn ram_access_kind_word(kind: crate::rv64im::stage2::RamAccessKind) -> u64 {
    match kind {
        crate::rv64im::stage2::RamAccessKind::Read => 0,
        crate::rv64im::stage2::RamAccessKind::Write => 1,
    }
}

pub(crate) fn trace_virtual_opcode_word(opcode: crate::rv64im::lower::Rv64TraceVirtualOpcode) -> u64 {
    match opcode {
        crate::rv64im::lower::Rv64TraceVirtualOpcode::Movsign => 0,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::Advice => 1,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::ChangeDivisor => 2,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::AssertValidDiv0 => 3,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::AssertMulNoOverflow => 4,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::AssertLte => 5,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::AssertValidUnsignedRemainder => 6,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::AssertSignedDivIdentity => 7,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::AssertSignedRemainderBounds => 8,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::Move => 9,
        crate::rv64im::lower::Rv64TraceVirtualOpcode::SignExtendWord => 10,
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
        out.push(word.addr);
        out.push(word.value);
    }
    out
}

fn flatten_row(row: &Rv64ExpandedRow) -> Vec<u64> {
    let mut out = vec![
        row.trace_index as u64,
        row.step_index as u64,
        row.sequence_index as u64,
        row.pc,
        row.next_pc,
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
        row.rs1_value,
        row.rs2 as u64,
        row.rs2_value,
        row.rd as u64,
        row.rd_before,
        row.rd_after,
        row.imm as u64,
        row.alu_result,
        row.writes_rd as u64,
        row.writes_ram as u64,
        row.halted as u64,
        row.is_first_in_sequence as u64,
        row.virtual_sequence_remaining.unwrap_or(u16::MAX) as u64,
        row.is_effect_row as u64,
        row.is_commit_row as u64,
        row.is_real as u64,
    ];
    out.push(row.effective_addr.unwrap_or(0));
    out.push(row.memory_before.unwrap_or(0));
    out.push(row.memory_after.unwrap_or(0));
    out
}

pub(super) fn flatten_stage1(stage1: &Stage1Summary) -> Vec<u64> {
    let mut out = Vec::new();
    for row in &stage1.rows {
        out.extend([
            row.trace_index as u64,
            row.step_index as u64,
            row.sequence_index as u64,
            row.fetch_pc,
            row.fetched_word as u64,
            opcode_word(row.opcode),
            row.trace_opcode.map(opcode_word).unwrap_or(0),
            row.trace_virtual_opcode
                .map(trace_virtual_opcode_word)
                .unwrap_or(0),
            row.trace_opcode.is_some() as u64,
            row.trace_virtual_opcode.is_some() as u64,
            family_word(row.family),
            row.next_pc,
            row.alu_result,
            row.effective_addr.unwrap_or(0),
            row.writes_rd as u64,
            row.rd as u64,
            row.rd_after,
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
            event.value,
        ]);
    }
    out.push(stage2.register_writes.len() as u64);
    for event in &stage2.register_writes {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            event.reg as u64,
            event.previous,
            event.next,
        ]);
    }
    out.push(stage2.ram_events.len() as u64);
    for event in &stage2.ram_events {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            ram_access_kind_word(event.kind),
            event.addr,
            event.previous,
            event.next,
        ]);
    }
    out.push(stage2.twist_links.len() as u64);
    for event in &stage2.twist_links {
        out.extend([
            event.trace_index as u64,
            event.step_index as u64,
            family_word(event.family),
            event.routed_write_value.unwrap_or(0),
            event.routed_memory_before.unwrap_or(0),
            event.routed_memory_after.unwrap_or(0),
        ]);
    }
    out
}

pub(super) fn flatten_stage3(stage3: &Stage3Summary) -> Vec<u64> {
    let mut out = vec![stage3.halted as u64, stage3.continuity.len() as u64];
    for event in &stage3.continuity {
        out.extend([
            event.step_index as u64,
            event.pc,
            event.next_pc,
            event.successor_pc.unwrap_or(0),
            event.final_step as u64,
            event.continuity_holds as u64,
        ]);
    }
    out
}

fn flatten_registers(values: &[u64; RV64_REGISTER_COUNT]) -> Vec<u64> {
    values.to_vec()
}

pub(crate) fn rv64im_source_case_digest(source: &Rv64imParitySourceCase) -> [u8; 32] {
    append_u64_matrix_digest(
        b"neo.fold.next/rv64im/source_digest_v1",
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

pub(crate) fn digest_rows(rows: &[Rv64ExpandedRow]) -> [u8; 32] {
    let mut sections = Vec::with_capacity(rows.len());
    for row in rows {
        sections.push((b"execution/row".as_slice(), flatten_row(row)));
    }
    append_u64_matrix_digest(b"neo.fold.next/rv64im/execution_digest_v1", &sections)
}

fn digest_final_state(final_state: &Rv64State) -> [u8; 32] {
    append_u64_matrix_digest(
        b"neo.fold.next/rv64im/final_state_digest_v1",
        &[
            (b"final/pc", vec![final_state.pc]),
            (b"final/halted", vec![final_state.halted as u64]),
            (b"final/registers", flatten_registers(&final_state.regs)),
            (b"final/memory", flatten_memory_words(&final_state.memory_words())),
        ],
    )
}

pub(super) fn build_kernel_transcript_and_summary_from_parts(
    source: &Rv64imParitySourceCase,
    rows: &[Rv64ExpandedRow],
    stage1: &Stage1Summary,
    stage2: &Stage2Summary,
    stage3: &Stage3Summary,
    final_state: &Rv64State,
) -> (TranscriptRecord, Rv64imKernelSummary) {
    let root0_digest = rv64im_source_case_digest(source);
    let stage1_digest = append_u64_matrix_digest(
        b"neo.fold.next/rv64im/stage1_digest_v1",
        &[(b"stage1/rows", flatten_stage1(stage1))],
    );
    let stage2_digest = append_u64_matrix_digest(
        b"neo.fold.next/rv64im/stage2_digest_v1",
        &[(b"stage2/summary", flatten_stage2(stage2))],
    );
    let stage3_digest = append_u64_matrix_digest(
        b"neo.fold.next/rv64im/stage3_digest_v1",
        &[(b"stage3/summary", flatten_stage3(stage3))],
    );
    let execution_digest = digest_rows(rows);
    let final_state_digest = digest_final_state(final_state);

    let mut transcript = LoggingTranscript::new(RV64IM_PARITY_TRANSCRIPT_APP_LABEL);
    transcript.append_message(RV64IM_PARITY_TRANSCRIPT_SEED_LABEL, &source.transcript_seed);
    transcript.append_message(RV64IM_PARITY_CASE_NAME_LABEL, source.manifest.name.as_bytes());
    transcript.append_u64s(
        RV64IM_PARITY_PROGRAM_WORDS_LABEL,
        &source
            .program_words
            .iter()
            .map(|word| *word as u64)
            .collect::<Vec<_>>(),
    );
    transcript.append_u64s(RV64IM_PARITY_INITIAL_REGS_LABEL, &source.initial_registers);
    transcript.append_u64s(
        RV64IM_PARITY_INITIAL_MEMORY_LABEL,
        &flatten_memory_words(&source.initial_memory),
    );
    transcript.append_message(RV64IM_PARITY_ROOT0_DIGEST_LABEL, &root0_digest);
    let stage1_mix = transcript.challenge_field(RV64IM_PARITY_STAGE1_MIX_LABEL);
    transcript.append_message(RV64IM_PARITY_STAGE1_DIGEST_LABEL, &stage1_digest);
    let stage2_reg_mix = transcript.challenge_field(RV64IM_PARITY_STAGE2_REG_MIX_LABEL);
    let stage2_ram_mix = transcript.challenge_field(RV64IM_PARITY_STAGE2_RAM_MIX_LABEL);
    transcript.append_message(RV64IM_PARITY_STAGE2_DIGEST_LABEL, &stage2_digest);
    let stage3_continuity_mix = transcript.challenge_field(RV64IM_PARITY_STAGE3_CONTINUITY_MIX_LABEL);
    transcript.append_message(RV64IM_PARITY_STAGE3_DIGEST_LABEL, &stage3_digest);
    transcript.append_message(RV64IM_PARITY_EXECUTION_DIGEST_LABEL, &execution_digest);
    transcript.append_message(RV64IM_PARITY_FINAL_STATE_DIGEST_LABEL, &final_state_digest);
    let kernel_final_mix = transcript.challenge_field(RV64IM_PARITY_KERNEL_FINAL_MIX_LABEL);
    let transcript_final_digest = transcript.digest32();
    let transcript = transcript.finish();

    let kernel = Rv64imKernelSummary {
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
    source: Rv64imParitySourceCase,
    max_steps: usize,
) -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    let program = Rv64Program::new(source.start_pc, source.program_words.clone());
    let initial_state = Rv64State::new(source.start_pc, source.initial_registers, &source.initial_memory);
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
        Rv64imParityDerivedCase {
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

pub fn build_vertical_slice_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        vertical_slice_source_case(),
        vertical_slice_source_case().program_words.len(),
    )
}

pub fn build_native_alu_focus_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        native_alu_focus_source_case(),
        native_alu_focus_source_case().program_words.len(),
    )
}

pub fn build_native_logic_compare_parity_case(
) -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        native_logic_compare_source_case(),
        native_logic_compare_source_case().program_words.len(),
    )
}

pub fn build_native_shift_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        native_shift_source_case(),
        native_shift_source_case().program_words.len(),
    )
}

pub fn build_native_word_arith_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        native_word_arith_source_case(),
        native_word_arith_source_case().program_words.len(),
    )
}

pub fn build_native_word_shift_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        native_word_shift_source_case(),
        native_word_shift_source_case().program_words.len(),
    )
}

pub fn build_native_upper_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        native_upper_source_case(),
        native_upper_source_case().program_words.len(),
    )
}

pub fn build_narrow_memory_load_parity_case(
) -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        narrow_memory_load_source_case(),
        narrow_memory_load_source_case().program_words.len(),
    )
}

pub fn build_narrow_memory_store_parity_case(
) -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        narrow_memory_store_source_case(),
        narrow_memory_store_source_case().program_words.len(),
    )
}

pub fn build_multiply_low_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        multiply_low_source_case(),
        multiply_low_source_case().program_words.len(),
    )
}

pub fn build_multiply_high_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        multiply_high_source_case(),
        multiply_high_source_case().program_words.len(),
    )
}

pub fn build_unsigned_divrem_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        unsigned_divrem_source_case(),
        unsigned_divrem_source_case().program_words.len(),
    )
}

pub fn build_signed_divrem_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        signed_divrem_source_case(),
        signed_divrem_source_case().program_words.len(),
    )
}

pub fn build_aligned_memory_focus_parity_case(
) -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        aligned_memory_focus_source_case(),
        aligned_memory_focus_source_case().program_words.len(),
    )
}

pub fn build_control_flow_focus_parity_case(
) -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError> {
    build_parity_case_from_source(
        control_flow_focus_source_case(),
        control_flow_focus_source_case().program_words.len(),
    )
}

pub fn build_control_flow_jal_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_jal_source_case(),
        control_flow_jal_source_case().program_words.len(),
    )
}

pub fn build_control_flow_jalr_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_jalr_source_case(),
        control_flow_jalr_source_case().program_words.len(),
    )
}

pub fn build_control_flow_beq_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_beq_source_case(),
        control_flow_beq_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bne_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_bne_source_case(),
        control_flow_bne_source_case().program_words.len(),
    )
}

pub fn build_control_flow_blt_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_blt_source_case(),
        control_flow_blt_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bge_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_bge_source_case(),
        control_flow_bge_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bltu_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_bltu_source_case(),
        control_flow_bltu_source_case().program_words.len(),
    )
}

pub fn build_control_flow_bgeu_parity_case() -> Result<(Rv64imParitySourceCase, Rv64imParityDerivedCase), Rv64BuildError>
{
    build_parity_case_from_source(
        control_flow_bgeu_source_case(),
        control_flow_bgeu_source_case().program_words.len(),
    )
}

pub fn build_all_parity_cases() -> Result<Vec<(Rv64imParitySourceCase, Rv64imParityDerivedCase)>, Rv64BuildError> {
    parity_source_cases()
        .into_iter()
        .map(|source| {
            let max_steps = source.program_words.len();
            build_parity_case_from_source(source, max_steps)
        })
        .collect()
}
