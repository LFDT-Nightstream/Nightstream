//! Legacy internal validation for unsupported A/C execution surfaces.
//!
//! These tests remain as internal reference coverage while the product contract
//! is explicitly `RV64IM` only.

use neo_memory::riscv::lookups::*;

#[path = "common/riscv_exec_helpers.rs"]
mod riscv_exec_helpers;

use riscv_exec_helpers::run_program;

// =============================================================================
// A Extension: Atomics
// =============================================================================

#[test]
fn test_a_load_reserved_store_conditional() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x200,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::LoadReserved {
            op: RiscvMemOp::LrW,
            rd: 3,
            rs1: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::StoreConditional {
            op: RiscvMemOp::ScW,
            rd: 5,
            rs1: 1,
            rs2: 4,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 6,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[3], 42);
    assert_eq!(regs[5], 0);
    assert_eq!(regs[6], 100);
}

#[test]
fn test_a_amoadd() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x200,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 5,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[4], 10);
    assert_eq!(regs[5], 15);
}

#[test]
fn test_a_amoswap() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x200,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoswapW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 5,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[4], 42);
    assert_eq!(regs[5], 100);
}

#[test]
fn test_a_amoand_amoor() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x200,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0b1111,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 0b1010,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoandW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 5,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let regs = run_program(program, 64);
    assert_eq!(regs[4], 0b1111);
    assert_eq!(regs[5], 0b1010);
}

// =============================================================================
// C Extension: Compressed Instructions
// =============================================================================

#[test]
fn test_c_decode_nop() {
    let instr = decode_compressed_instruction(0x0001).expect("decode failed");
    match instr {
        RiscvInstruction::Nop => {}
        _ => panic!("Expected C.NOP, got {:?}", instr),
    }
}

#[test]
fn test_c_decode_produces_valid_instructions() {
    let nop = decode_compressed_instruction(0x0001).unwrap();
    assert!(matches!(nop, RiscvInstruction::Nop));

    let ebreak = decode_compressed_instruction(0x9002).unwrap();
    assert!(matches!(ebreak, RiscvInstruction::Ebreak));
}

#[test]
fn test_c_compressed_detection() {
    assert_ne!(0x0001u16 & 0b11, 0b11);
    assert_ne!(0x9002u16 & 0b11, 0b11);

    let addi_32bit = encode_instruction(&RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 1,
        rs1: 0,
        imm: 42,
    });
    assert_eq!(addi_32bit & 0b11, 0b11);
}

#[test]
fn test_c_mixed_program() {
    let mut bytes = Vec::new();

    let addi = encode_instruction(&RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 1,
        rs1: 0,
        imm: 42,
    });
    bytes.extend_from_slice(&addi.to_le_bytes());
    bytes.extend_from_slice(&0x0001u16.to_le_bytes());
    let halt = encode_instruction(&RiscvInstruction::Halt);
    bytes.extend_from_slice(&halt.to_le_bytes());

    let program = decode_program(&bytes).expect("decode mixed program");

    assert_eq!(program.len(), 3);
    assert!(matches!(
        program[0],
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 42
        }
    ));
    assert!(matches!(program[1], RiscvInstruction::Nop));
    assert!(matches!(program[2], RiscvInstruction::Halt));
}
