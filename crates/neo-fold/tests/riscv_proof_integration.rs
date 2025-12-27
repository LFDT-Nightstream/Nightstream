//! RISC-V Program Execution Tests (No Full Proof Generation)
//!
//! This test file demonstrates RISC-V program execution and tracing
//! using Neo's VmCpu, Twist, and Shout traits.
//!
//! For full proof generation tests, see `riscv_full_proof.rs`.
//!
//! ## Credits
//!
//! The RISC-V CPU implementation is inspired by Jolt (MIT/Apache-2.0 license).
//! <https://github.com/a16z/jolt>

#![allow(non_snake_case)]

use neo_memory::riscv::lookups::{
    BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables,
};
use neo_vm_trace::trace_program;

// ============================================================================
// Test 1: Simple Arithmetic Program
// ============================================================================

/// Test RISC-V execution for a simple arithmetic program.
///
/// Program: Compute 2 + 3 + 5 = 10 using ADD operations.
#[test]
fn riscv_exec_simple_arithmetic() {
    let xlen = 32;

    // Simple program: x1 = 2, x2 = 3, x3 = x1 + x2 (5), x4 = x3 + 5 (10)
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 3,
            imm: 5,
        },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(xlen);
    let shout_tables = RiscvShoutTables::new(xlen);

    // Execute and trace
    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    // Verify final result
    let last_step = trace.steps.last().unwrap();
    assert_eq!(last_step.regs_after[4], 10, "Expected result 10");

    println!("✓ RISC-V Exec: Simple Arithmetic");
    println!("  Steps executed: {}", trace.len());
    println!("  Shout events: {}", trace.total_shout_events());
    println!("  Final result (x4): {}", last_step.regs_after[4]);
}

// ============================================================================
// Test 2: Memory-Intensive Program
// ============================================================================

/// Test RISC-V execution for a program with memory operations.
///
/// Program: Store values to memory, load them back, compute sum.
#[test]
fn riscv_exec_memory_program() {
    let xlen = 32;

    // Program: Store 10 and 20 to memory, load them, compute sum
    let program = vec![
        // x1 = 0x100 (base address)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        },
        // x2 = 10
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 10,
        },
        // x3 = 20
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 20,
        },
        // mem[x1] = x2 (store 10)
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        // mem[x1+4] = x3 (store 20)
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 3,
            imm: 4,
        },
        // x4 = mem[x1] (load 10)
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 4,
            rs1: 1,
            imm: 0,
        },
        // x5 = mem[x1+4] (load 20)
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 5,
            rs1: 1,
            imm: 4,
        },
        // x6 = x4 + x5 (30)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 4,
            rs2: 5,
        },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(xlen);
    let shout_tables = RiscvShoutTables::new(xlen);

    // Execute and trace
    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    assert_eq!(last_step.regs_after[6], 30, "Expected 10 + 20 = 30");

    println!("✓ RISC-V Exec: Memory Program");
    println!("  Steps: {}", trace.len());
    println!("  Twist events: {}", trace.total_twist_events());
    println!("  Result (x6 = x4 + x5): {}", last_step.regs_after[6]);
}

// ============================================================================
// Test 3: Multiplication Program (M Extension)
// ============================================================================

/// Test RISC-V execution for a program using M extension (MUL/DIV).
#[test]
fn riscv_exec_multiplication_program() {
    let xlen = 32;

    // Program: Compute 7 * 13 = 91, then 91 / 7 = 13
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
            imm: 13,
        },
        // x3 = x1 * x2 = 91
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = x3 / x1 = 13
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 4,
            rs1: 3,
            rs2: 1,
        },
        // x5 = x3 % x1 = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 5,
            rs1: 3,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(xlen);
    let shout_tables = RiscvShoutTables::new(xlen);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    assert_eq!(last_step.regs_after[3], 91, "7 * 13 = 91");
    assert_eq!(last_step.regs_after[4], 13, "91 / 7 = 13");
    assert_eq!(last_step.regs_after[5], 0, "91 % 7 = 0");

    println!("✓ RISC-V Exec: M Extension (MUL/DIV)");
    println!("  7 * 13 = {}", last_step.regs_after[3]);
    println!("  91 / 7 = {}", last_step.regs_after[4]);
    println!("  91 % 7 = {}", last_step.regs_after[5]);
}

// ============================================================================
// Test 4: Fibonacci Sequence
// ============================================================================

/// Test RISC-V execution for Fibonacci computation.
#[test]
fn riscv_exec_fibonacci() {
    // Fibonacci program: compute F(5) = 5
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0,
        }, // x1 = 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        }, // x2 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        }, // x3 = 5 (counter)
        // Loop:
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // x4 = x1 + x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 2,
            rs2: 0,
        }, // x1 = x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 4,
            rs2: 0,
        }, // x2 = x4
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 3,
            imm: -1,
        }, // x3--
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 3,
            rs2: 0,
            imm: -16,
        },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let fib_result = last_step.regs_after[2];
    assert_eq!(fib_result, 8, "Fibonacci(6) should be 8");

    println!("✓ RISC-V Exec: Fibonacci");
    println!("  Program steps: {}", trace.len());
    println!("  Fibonacci result: {}", fib_result);
}

// ============================================================================
// Test 5: GCD with Euclidean Algorithm
// ============================================================================

/// Test RISC-V execution for GCD computation using Euclidean algorithm.
///
/// Program: Compute GCD(48, 18) = 6
#[test]
fn riscv_exec_gcd_euclidean() {
    let xlen = 32;

    // GCD using Euclidean algorithm
    let program = vec![
        // 0: x1 = 48 (a)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 48,
        },
        // 4: x2 = 18 (b)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 18,
        },
        // 8: if x2 == 0, goto halt (at 28, so +20 from 8)
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 2,
            rs2: 0,
            imm: 20,
        },
        // 12: x3 = x2 (temp = b)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 2,
            rs2: 0,
        },
        // 16: x2 = x1 % x2 (b = a % b)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 2,
            rs1: 1,
            rs2: 2,
        },
        // 20: x1 = x3 (a = temp)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 3,
            rs2: 0,
        },
        // 24: jump back to 8 (-16 bytes from 24)
        RiscvInstruction::Jal { rd: 0, imm: -16 },
        // 28: halt
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(xlen);
    let shout_tables = RiscvShoutTables::new(xlen);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let gcd_result = last_step.regs_after[1];
    assert_eq!(gcd_result, 6, "GCD(48, 18) should be 6");

    println!("✓ RISC-V Exec: GCD (Euclidean Algorithm)");
    println!("  GCD(48, 18) = {}", gcd_result);
    println!("  Steps: {}", trace.len());
    println!("  Shout events: {}", trace.total_shout_events());
}

// ============================================================================
// Test 6: Factorial
// ============================================================================

/// Test RISC-V execution for factorial computation.
///
/// Program: Compute 5! = 120
#[test]
fn riscv_exec_factorial() {
    let xlen = 32;

    // Factorial program: n! where n=5
    let program = vec![
        // x1 = 5 (n)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        // x2 = 1 (result)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        // Loop (addr 8):
        // x2 = x2 * x1 (result *= n)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 2,
            rs1: 2,
            rs2: 1,
        },
        // x1 = x1 - 1 (n--)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: -1,
        },
        // if x1 != 0, goto -8 (loop)
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 1,
            rs2: 0,
            imm: -8,
        },
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(xlen);
    cpu.load_program(0, program);

    let memory = RiscvMemory::new(xlen);
    let shout_tables = RiscvShoutTables::new(xlen);

    let trace = trace_program(cpu, memory, shout_tables, 100).unwrap();
    assert!(trace.did_halt());

    let last_step = trace.steps.last().unwrap();
    let factorial_result = last_step.regs_after[2];
    assert_eq!(factorial_result, 120, "5! should be 120");

    println!("✓ RISC-V Exec: Factorial (5!)");
    println!("  Program steps: {}", trace.len());
    println!("  Result: 5! = {}", factorial_result);
}
