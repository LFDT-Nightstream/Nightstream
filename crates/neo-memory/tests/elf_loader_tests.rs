//! Tests for the ELF binary loader.

use neo_memory::riscv::elf_loader::{load_elf, load_raw_binary, ElfError};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn test_load_raw_binary() {
    let instructions = vec![
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
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let bytes = encode_program(&instructions);

    let loaded = load_raw_binary(&bytes, 0x80000000).unwrap();

    assert_eq!(loaded.entry, 0x80000000);
    assert_eq!(loaded.instructions.len(), 4);
    assert_eq!(loaded.instructions[0].0, 0x80000000);
}

#[test]
fn test_loaded_program_get_instructions() {
    let instructions = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::Halt,
    ];
    let bytes = encode_program(&instructions);

    let loaded = load_raw_binary(&bytes, 0).unwrap();
    let extracted = loaded.get_instructions();

    assert_eq!(extracted.len(), 2);
}

#[test]
fn test_load_raw_binary_at_different_addresses() {
    let instructions = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 100,
        },
        RiscvInstruction::Halt,
    ];
    let bytes = encode_program(&instructions);

    // Test at address 0
    let loaded_at_0 = load_raw_binary(&bytes, 0).unwrap();
    assert_eq!(loaded_at_0.entry, 0);
    assert_eq!(loaded_at_0.instructions[0].0, 0);

    // Test at typical RISC-V entry point
    let loaded_at_8000 = load_raw_binary(&bytes, 0x80000000).unwrap();
    assert_eq!(loaded_at_8000.entry, 0x80000000);
    assert_eq!(loaded_at_8000.instructions[0].0, 0x80000000);
}

#[test]
fn test_loaded_program_code_size() {
    let instructions = vec![
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
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let bytes = encode_program(&instructions);

    let loaded = load_raw_binary(&bytes, 0).unwrap();

    // Each instruction is 4 bytes
    assert_eq!(loaded.code_size(), 4 * 4);
}

#[test]
fn test_elf_invalid_magic() {
    // Need at least 52 bytes to pass the initial length check
    let invalid_data = vec![0x00; 52];
    let result = load_elf(&invalid_data);
    assert!(
        matches!(result, Err(ElfError::InvalidMagic)),
        "Expected InvalidMagic, got {:?}",
        result
    );
}

#[test]
fn test_elf_too_short() {
    let short_data = vec![0x7f, b'E', b'L', b'F'];
    let result = load_elf(&short_data);
    assert!(matches!(result, Err(ElfError::TooShort)));
}

#[test]
fn test_raw_binary_misaligned() {
    // 5 bytes - not aligned to 4
    let misaligned = vec![0x13, 0x00, 0x00, 0x00, 0x00];
    let result = load_raw_binary(&misaligned, 0);
    assert!(matches!(result, Err(ElfError::TooShort)));
}

#[test]
fn test_complex_program_roundtrip() {
    use neo_memory::riscv::lookups::BranchCondition;

    // Fibonacci program
    let instructions = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 2,
            rs2: 0,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 4,
            rs2: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 3,
            imm: -1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 3,
            rs2: 0,
            imm: -16,
        },
        RiscvInstruction::Halt,
    ];

    let bytes = encode_program(&instructions);
    let loaded = load_raw_binary(&bytes, 0x80000000).unwrap();

    assert_eq!(loaded.instructions.len(), 9);

    // Verify the extracted instructions work
    let extracted = loaded.get_instructions();
    assert_eq!(extracted.len(), 9);
}
