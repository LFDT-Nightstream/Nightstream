//! Owns deterministic RV32IM perf/debug source cases reused across Rust-only benchmarks and Rust↔Lean compatibility checks.

use crate::rv32im::isa::{
    encode_add, encode_addi, encode_and, encode_beq, encode_divu, encode_ecall, encode_lw, encode_mul, encode_remu,
    encode_slli, encode_sw, encode_xor, MemoryWord,
};
use crate::rv32im::kernel::{Rv32imParityCaseManifest, Rv32imParitySourceCase};
use crate::rv32im::layout::{
    RV32IM_PARITY_LOWERING_VERSION_ID, RV32IM_PARITY_PROTOCOL_VERSION_ID, RV32_REGISTER_COUNT,
};
use crate::rv32im::tables::Rv32FamilyTag;

pub const RV32IM_MIXED_OPCODE_PERF_DEFAULT_N: usize = 100;
pub const RV32IM_MIXED_OPCODE_PERF_BLOCK_LEN: usize = 13;

const START_PC: u32 = 0x1000;
const PERF_MEMORY_ADDR: u32 = 0x100;

pub fn mixed_opcode_perf_expected_x1(opcode_count: usize) -> usize {
    opcode_count.div_ceil(RV32IM_MIXED_OPCODE_PERF_BLOCK_LEN)
}

pub fn build_mixed_opcode_perf_source_case(opcode_count: usize) -> Rv32imParitySourceCase {
    let mixed_block = [
        encode_addi(1, 1, 1),
        encode_addi(2, 2, 3),
        encode_add(3, 1, 2),
        encode_slli(4, 3, 1),
        encode_xor(5, 4, 2),
        encode_mul(6, 5, 1),
        encode_divu(7, 6, 1),
        encode_remu(8, 6, 1),
        encode_beq(1, 0, 8),
        encode_sw(7, 0, PERF_MEMORY_ADDR as i16),
        encode_lw(9, 0, PERF_MEMORY_ADDR as i16),
        encode_and(11, 9, 5),
        encode_addi(12, 11, 7),
    ];

    let mut program_words = Vec::with_capacity(opcode_count + 1);
    while program_words.len() < opcode_count {
        for word in mixed_block {
            if program_words.len() == opcode_count {
                break;
            }
            program_words.push(word);
        }
    }
    program_words.push(encode_ecall());

    let mut transcript_seed = b"rv32im-mixed-opcode-perf-snapshot-v1".to_vec();
    transcript_seed.extend_from_slice(&(opcode_count as u64).to_le_bytes());

    Rv32imParitySourceCase {
        manifest: Rv32imParityCaseManifest {
            name: "mixed_opcode_perf_snapshot".into(),
            fixture_id: "mixed_opcode_perf_snapshot_v1".into(),
            protocol_version_id: RV32IM_PARITY_PROTOCOL_VERSION_ID,
            lowering_version_id: RV32IM_PARITY_LOWERING_VERSION_ID,
            family_tags: vec![
                Rv32FamilyTag::NativeAlu,
                Rv32FamilyTag::Multiply,
                Rv32FamilyTag::UnsignedDivRem,
                Rv32FamilyTag::AlignedMemory,
                Rv32FamilyTag::ControlFlow,
            ],
        },
        start_pc: START_PC,
        program_words,
        initial_registers: [0; RV32_REGISTER_COUNT],
        initial_memory: vec![MemoryWord {
            addr: PERF_MEMORY_ADDR,
            value: 0,
        }],
        transcript_seed,
    }
}
