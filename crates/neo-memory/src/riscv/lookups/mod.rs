//! RISC-V lookup-based execution helpers (RV32-focused proving integration).
//!
//! This module provides:
//! - Instruction decoding/encoding
//! - A traceable CPU that emits Twist (memory) and Shout (ALU/compare lookup) events
//! - Lookup helpers/tables for proving integrations
//!
//! # Proving integration scope (today)
//!
//! The shared-bus RV32 B1 proving path assumes:
//! - `xlen == 32` (RV32)
//! - no compressed (RVC) instructions
//! - 4-byte aligned PC and control-flow targets
//!
//! Note: Shout operand keys are encoded via bit interleaving into a `u64` key
//! (2×32-bit → 64-bit), which is sufficient for RV32. RV64 proving requires a wider key.
//!
//! # Architecture
//!
//! ## Lookup Tables (Shout)
//!
//! ALU operations are proven using Neo's Shout (read-only memory) protocol:
//! - The **index** encodes operands via bit interleaving
//! - The **value** is the operation result
//! - MLEs enable efficient sumcheck verification
//!
//! ## Memory (Twist)
//!
//! Load/store and atomic operations use Neo's Twist (read-write memory) protocol.
//!
//! # Instruction Categories
//!
//! ## Base Integer (I Extension)
//! - **Arithmetic**: ADD, ADDI, SUB
//! - **Logical**: AND, ANDI, OR, ORI, XOR, XORI
//! - **Shifts**: SLL, SLLI, SRL, SRLI, SRA, SRAI
//! - **Compare**: SLT, SLTI, SLTU, SLTIU
//! - **Branches**: BEQ, BNE, BLT, BGE, BLTU, BGEU
//! - **Jumps**: JAL, JALR
//! - **Upper Immediate**: LUI, AUIPC
//! - **Loads**: LB, LBU, LH, LHU, LW, LWU, LD
//! - **Stores**: SB, SH, SW, SD
//!
//! ## RV64 Word Operations
//! - ADDW, SUBW, ADDIW
//! - SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW
//!
//! ## Multiply/Divide (M Extension)
//! - MUL, MULH, MULHU, MULHSU
//! - DIV, DIVU, REM, REMU
//! - MULW, DIVW, DIVUW, REMW, REMUW (RV64)
//!
//! ## Atomics (A Extension)
//! - **Load-Reserved**: LR.W, LR.D
//! - **Store-Conditional**: SC.W, SC.D
//! - **AMO**: AMOSWAP, AMOADD, AMOXOR, AMOAND, AMOOR, AMOMIN, AMOMAX, AMOMINU, AMOMAXU
//!
//! ## Compressed (C Extension)
//! - All quadrant 0, 1, and 2 instructions
//! - Automatic detection of 16-bit vs 32-bit instructions
//!
//! ## System
//! - ECALL, EBREAK
//! - FENCE, FENCE.I
//!
//! # Example
//!
//! ```ignore
//! use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvMemory, RiscvShoutTables};
//! use neo_vm_trace::trace_program;
//!
//! // Load and decode a RISC-V binary
//! let program = decode_program(&binary_bytes)?;
//!
//! // Execute with full tracing
//! let mut cpu = RiscvCpu::new(32); // RV32
//! cpu.load_program(0, program);
//! let memory = RiscvMemory::new(32);
//! let shout = RiscvShoutTables::new(32);
//!
//! let trace = trace_program(cpu, memory, shout, 1000)?;
//! // trace now contains all steps for proving
//! ```

mod alu;
mod bits;
mod cpu;
mod decode;
mod encode;
mod isa;
mod memory;
mod mle;
mod tables;
mod trace;

use neo_vm_trace::TwistId;

/// Canonical Twist instance id for RISC-V data RAM.
pub const RAM_ID: TwistId = TwistId(0);

/// Canonical Twist instance id for the program ROM (B1 instruction fetch).
pub const PROG_ID: TwistId = TwistId(1);

pub use alu::{compute_op, lookup_entry};
pub use bits::{interleave_bits, uninterleave_bits};
pub use cpu::RiscvCpu;
pub use decode::{decode_compressed_instruction, decode_instruction, decode_program, RiscvFormat};
pub use encode::{encode_instruction, encode_program};
pub use isa::{BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};
pub use memory::{RiscvMemory, RiscvMemoryEvent};
pub use mle::{
    evaluate_add_mle, evaluate_and_mle, evaluate_eq_mle, evaluate_neq_mle, evaluate_opcode_mle, evaluate_or_mle,
    evaluate_sll_mle, evaluate_slt_mle, evaluate_sltu_mle, evaluate_sra_mle, evaluate_srl_mle, evaluate_sub_mle,
    evaluate_xor_mle,
};
pub use tables::{RangeCheckTable, RiscvLookupEvent, RiscvLookupTable, RiscvShoutTables};
pub use trace::{
    analyze_trace, build_final_memory_state, build_opcode_lut_table, extract_program_io, trace_to_plain_lut_trace,
    trace_to_plain_lut_traces_by_opcode, trace_to_plain_mem_trace, TraceConversionSummary, TraceToProofConfig,
};
