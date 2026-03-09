//! RV64IM Validation Test Suite
//!
//! Validates the supported RV64IM execution surface:
//! - I: Base Integer
//! - M: Multiply/Divide
//!
//! Legacy internal A/C coverage lives in `riscv_legacy_ac_validation.rs`.

use neo_memory::riscv::lookups::*;
use neo_memory::riscv::packed::{build_rv_packed_cols, rv_packed_supported_opcode};
use neo_memory::RiscvProofProfile;
use p3_goldilocks::Goldilocks;
#[path = "common/riscv_exec_helpers.rs"]
mod riscv_exec_helpers;

use riscv_exec_helpers::run_program;

// =============================================================================
// I Extension: Base Integer
// =============================================================================

#[test]
fn test_i_arithmetic() {
    // Test ADD, SUB, ADDI
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 100,
        }, // x1 = 100
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 50,
        }, // x2 = 50
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 150
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // x4 = 50
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[1], 100, "ADDI x1, x0, 100");
    assert_eq!(regs[2], 50, "ADDI x2, x0, 50");
    assert_eq!(regs[3], 150, "ADD x3, x1, x2");
    assert_eq!(regs[4], 50, "SUB x4, x1, x2");
}

#[test]
fn test_i_logical() {
    // Test AND, OR, XOR
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0b1010,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0b1100,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::And,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 0b1000
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // 0b1110
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Xor,
            rd: 5,
            rs1: 1,
            rs2: 2,
        }, // 0b0110
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 0b1000, "AND");
    assert_eq!(regs[4], 0b1110, "OR");
    assert_eq!(regs[5], 0b0110, "XOR");
}

#[test]
fn test_i_shifts() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0b1010,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 0b101000
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Srl,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // 0b10
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 5,
            rs1: 1,
            imm: 3,
        }, // 0b1010000
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 0b101000, "SLL");
    assert_eq!(regs[4], 0b10, "SRL");
    assert_eq!(regs[5], 0b1010000, "SLLI");
}

#[test]
fn test_i_sra_positive() {
    // SRA on positive number - just shifts right
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0b10000,
        }, // 16
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sra,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 4
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 4, "SRA 16 >> 2 = 4");
}

#[test]
fn test_i_srl_vs_sra_positive() {
    // For positive numbers, SRL and SRA should give the same result
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 64,
        }, // 64
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Srl,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 8
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sra,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // 8
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 8, "SRL 64 >> 3 = 8");
    assert_eq!(regs[4], 8, "SRA 64 >> 3 = 8");
    assert_eq!(regs[3], regs[4], "SRL and SRA equal for positive");
}

#[test]
fn test_i_comparisons() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 20,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 1 (10 < 20)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 4,
            rs1: 2,
            rs2: 1,
        }, // 0 (20 < 10)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sltu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        }, // 1
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 1, "SLT 10 < 20");
    assert_eq!(regs[4], 0, "SLT 20 < 10");
    assert_eq!(regs[5], 1, "SLTU");
}

#[test]
fn test_i_branches() {
    // BEQ test: skip one instruction if equal
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // skip next
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 999,
        }, // should skip
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 42,
        }, // should execute
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 0, "BEQ should skip x3 assignment");
    assert_eq!(regs[4], 42, "BEQ should reach x4 assignment");
}

#[test]
fn test_i_jal_jalr() {
    let program = vec![
        RiscvInstruction::Jal { rd: 1, imm: 8 }, // x1 = 4, jump to +8
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 999,
        }, // skip
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 42,
        }, // land here
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[1], 4, "JAL stores return address");
    assert_eq!(regs[2], 0, "JAL skips instruction");
    assert_eq!(regs[3], 42, "JAL jumps to target");
}

#[test]
fn test_i_lui_auipc() {
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 0x12345 }, // x1 = 0x12345000
        RiscvInstruction::Auipc { rd: 2, imm: 0x1 },   // x2 = PC + 0x1000 = 4 + 0x1000
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[1], 0x12345 << 12, "LUI");
    assert_eq!(regs[2], 4 + (0x1 << 12), "AUIPC");
}

#[test]
fn test_i_load_store() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        }, // addr
        // Construct 0xABCD using legal immediates:
        //   LUI  x2, 0xB      => x2 = 0xB000
        //   ADDI x2, x2, -0x433 => x2 = 0xABCD
        RiscvInstruction::Lui { rd: 2, imm: 0xB },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 2,
            imm: -0x433,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        }, // mem[0x100] = 0xABCD
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 3,
            rs1: 1,
            imm: 0,
        }, // x3 = mem[0x100]
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 0xABCD, "LW loads stored value");
}

// =============================================================================
// M Extension: Multiply/Divide
// =============================================================================

#[test]
fn test_m_multiply() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 6,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 42
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 42, "MUL 7 * 6 = 42");
}

#[test]
fn test_m_mulhu_unsigned() {
    // Test high bits of unsigned multiplication.
    // Use values large enough that the 128-bit product has non-zero upper 64 bits.
    // 0x1_0000_0000 * 0x1_0000_0000 = 0x1_0000_0000_0000_0000 (upper 64 bits = 1).
    let program = vec![
        // Build 0x1_0000_0000 in x1:  ADDI x1, x0, 1  then  SLLI x1, x1, 32
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 1,
            rs1: 1,
            imm: 32,
        }, // x1 = 0x1_0000_0000
        // Copy to x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            rs2: 0,
        }, // x2 = 0x1_0000_0000
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = upper 64 bits of (0x1_0000_0000 * 0x1_0000_0000) = 1
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[1], 0x1_0000_0000u64, "x1 operand");
    assert_eq!(regs[2], 0x1_0000_0000u64, "x2 operand");
    assert_eq!(regs[3], 1u64, "MULHU result: upper 64 bits of 2^64 should be 1");
}

#[test]
fn test_m_divide() {
    // Use unsigned division to avoid sign issues
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 6
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: 43,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 6,
            rs1: 5,
            rs2: 2,
        }, // 1
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 6, "DIVU 42 / 7 = 6");
    assert_eq!(regs[4], 0, "REMU 42 % 7 = 0");
    assert_eq!(regs[6], 1, "REMU 43 % 7 = 1");
}

#[test]
fn test_m_divide_by_zero() {
    // RISC-V spec: division by zero returns -1 for DIV, dividend for REM
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 2,
            rs1: 1,
            rs2: 0,
        }, // x0 = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 3,
            rs1: 1,
            rs2: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[2], u64::MAX, "DIV by zero returns -1");
    assert_eq!(regs[3], 42, "REM by zero returns dividend");
}

// =============================================================================
// RV64: Word Operations (W-suffix)
// =============================================================================

#[test]
fn test_rv64_addw_basic() {
    // Simple ADDW test with small positive numbers
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 50,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Addw,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 150
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 150, "ADDW 100 + 50 = 150");
}

#[test]
fn test_rv64_subw_basic() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 30,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Subw,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 70
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 70, "SUBW 100 - 30 = 70");
}

#[test]
fn test_rv64_mulw_basic() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 50,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Mulw,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 5000
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 5000, "MULW 100 * 50 = 5000");
}

#[test]
fn test_rv64_sllw_basic() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0b1010,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Sllw,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // 0b101000
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 0b101000, "SLLW shift left by 2");
}

// =============================================================================
// System Instructions
// =============================================================================

#[test]
fn test_fence_nop() {
    // FENCE should be a no-op in our model
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::Fence { pred: 0xF, succ: 0xF },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[1], 42);
    assert_eq!(regs[2], 100);
}

#[test]
fn test_rv64im_profile_accepts_mulh_and_mulhsu() {
    let profile = RiscvProofProfile::rv64im();
    for op in [RiscvOpcode::Mulh, RiscvOpcode::Mulhsu] {
        let inst = RiscvInstruction::RAlu {
            op,
            rd: 1,
            rs1: 2,
            rs2: 3,
        };
        assert!(
            profile.supports_instruction(&inst),
            "helper-owned RV64 multiply-high op should be part of the current RV64IM proving profile: {op:?}"
        );
    }
}

#[test]
fn test_rv64im_profile_accepts_base_div_rem_ops() {
    let profile = RiscvProofProfile::rv64im();
    for op in [RiscvOpcode::Div, RiscvOpcode::Divu, RiscvOpcode::Rem, RiscvOpcode::Remu] {
        let inst = RiscvInstruction::RAlu {
            op,
            rd: 1,
            rs1: 2,
            rs2: 3,
        };
        assert!(
            profile.supports_instruction(&inst),
            "base RV64 div/rem op should be part of the current RV64IM proving profile: {op:?}"
        );
    }
}

#[test]
fn test_rv64_packed_support_covers_exact_base_m_path() {
    assert!(rv_packed_supported_opcode(RiscvOpcode::Mul, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Mulh, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Mulhu, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Mulhsu, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Div, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Divu, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Rem, 64));
    assert!(rv_packed_supported_opcode(RiscvOpcode::Remu, 64));

    assert!(!rv_packed_supported_opcode(RiscvOpcode::Add, 64));
}

#[test]
fn test_rv64_packed_mul_cols_allow_non_injective_transport_values() {
    let lhs = (-2i64) as u64;
    let rhs = 3u64;
    let val = lhs.wrapping_mul(rhs);

    let cols = build_rv_packed_cols::<Goldilocks>(RiscvOpcode::Mul, lhs, rhs, val, 64)
        .expect("rv64 packed mul cols should use exact field transport for non-injective words");
    assert_eq!(cols.len(), 66);
}

// =============================================================================
// Decode/Encode Roundtrip
// =============================================================================

#[test]
fn test_encode_decode_roundtrip_r_type() {
    let instructions = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 2,
            rs2: 3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 4,
            rs1: 5,
            rs2: 6,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::And,
            rd: 7,
            rs1: 8,
            rs2: 9,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Or,
            rd: 10,
            rs1: 11,
            rs2: 12,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Xor,
            rd: 13,
            rs1: 14,
            rs2: 15,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 16,
            rs1: 17,
            rs2: 18,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Srl,
            rd: 19,
            rs1: 20,
            rs2: 21,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sra,
            rd: 22,
            rs1: 23,
            rs2: 24,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 25,
            rs1: 26,
            rs2: 27,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sltu,
            rd: 28,
            rs1: 29,
            rs2: 30,
        },
    ];

    for instr in instructions {
        let encoded = encode_instruction(&instr);
        let decoded = decode_instruction(encoded).expect("decode failed");
        let re_encoded = encode_instruction(&decoded);
        assert_eq!(encoded, re_encoded, "Roundtrip failed for {:?}", instr);
    }
}

#[test]
fn test_encode_decode_roundtrip_m_extension() {
    let instructions = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 1,
            rs1: 2,
            rs2: 3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 4,
            rs1: 5,
            rs2: 6,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 7,
            rs1: 8,
            rs2: 9,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 10,
            rs1: 11,
            rs2: 12,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 13,
            rs1: 14,
            rs2: 15,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 16,
            rs1: 17,
            rs2: 18,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 19,
            rs1: 20,
            rs2: 21,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 22,
            rs1: 23,
            rs2: 24,
        },
    ];

    for instr in instructions {
        let encoded = encode_instruction(&instr);
        let decoded = decode_instruction(encoded).expect("decode failed");
        let re_encoded = encode_instruction(&decoded);
        assert_eq!(encoded, re_encoded, "M-ext roundtrip failed for {:?}", instr);
    }
}

#[test]
fn test_encode_decode_roundtrip_w_suffix() {
    let instructions = vec![
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Addw,
            rd: 1,
            rs1: 2,
            rs2: 3,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Subw,
            rd: 4,
            rs1: 5,
            rs2: 6,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Sllw,
            rd: 7,
            rs1: 8,
            rs2: 9,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Srlw,
            rd: 10,
            rs1: 11,
            rs2: 12,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Sraw,
            rd: 13,
            rs1: 14,
            rs2: 15,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Mulw,
            rd: 16,
            rs1: 17,
            rs2: 18,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Divw,
            rd: 19,
            rs1: 20,
            rs2: 21,
        },
        RiscvInstruction::RAluw {
            op: RiscvOpcode::Remw,
            rd: 22,
            rs1: 23,
            rs2: 24,
        },
    ];

    for instr in instructions {
        let encoded = encode_instruction(&instr);
        let decoded = decode_instruction(encoded).expect("decode failed");
        let re_encoded = encode_instruction(&decoded);
        assert_eq!(encoded, re_encoded, "W-suffix roundtrip failed for {:?}", instr);
    }
}

// =============================================================================
// Complex Program: Fibonacci
// =============================================================================

#[test]
fn test_fibonacci_rv64() {
    // Compute fib(10) = 55
    let program = vec![
        // x1 = n = 10
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 10,
        },
        // x2 = fib(0) = 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0,
        },
        // x3 = fib(1) = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        // x4 = counter = 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 0,
        },
        // Loop start (offset 16)
        // if counter >= n, branch to end
        RiscvInstruction::Branch {
            cond: BranchCondition::Ge,
            rs1: 4,
            rs2: 1,
            imm: 24,
        },
        // x5 = x2 + x3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 2,
            rs2: 3,
        },
        // x2 = x3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 3,
            rs2: 0,
        },
        // x3 = x5
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 5,
            rs2: 0,
        },
        // counter++
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 4,
            imm: 1,
        },
        // jump back to loop start
        RiscvInstruction::Jal { rd: 0, imm: -20 },
        // x10 = result (for output)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 2,
            rs2: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[10], 55, "fib(10) = 55");
}

// =============================================================================
// Complex Program: GCD (using DIV/REM)
// =============================================================================

#[test]
fn test_gcd_euclidean() {
    // GCD(48, 18) = 6 using Euclidean algorithm with unsigned remainder
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 48,
        }, // a = 48
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 18,
        }, // b = 18
        // Loop: while b != 0
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 2,
            rs2: 0,
            imm: 20,
        }, // if b==0, exit
        // t = a % b (unsigned to avoid overflow issues)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // a = b
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 2,
            rs2: 0,
        },
        // b = t
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 3,
            rs2: 0,
        },
        // loop back
        RiscvInstruction::Jal { rd: 0, imm: -16 },
        // Result in x1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 1,
            rs2: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[10], 6, "GCD(48, 18) = 6");
}
