//! RISC-V RV64IMAC instruction support for Neo's proving system.
//!
//! This module implements a complete **RV64IMAC** RISC-V instruction set, providing:
//! - Instruction decoding (32-bit and 16-bit compressed)
//! - Instruction encoding
//! - CPU execution with tracing
//! - Lookup tables for ALU operations (Shout protocol)
//! - Memory operations (Twist protocol)
//!
//! # Supported RISC-V Extensions
//!
//! | Extension | Description | Status |
//! |-----------|-------------|--------|
//! | **I** | Base Integer (RV64I) | ✅ Full |
//! | **M** | Multiply/Divide | ✅ Full |
//! | **A** | Atomics (LR/SC, AMO) | ✅ Full |
//! | **C** | Compressed (16-bit) | ✅ Full |
//! | **Zbb** | Bitmanip (subset) | ✅ ANDN |
//!
//! This provides feature parity with [Jolt](https://github.com/a16z/jolt).
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
//! // Load and decode a RISC-V binary (supports compressed instructions)
//! let program = decode_program(&binary_bytes)?;
//!
//! // Execute with full tracing
//! let mut cpu = RiscvCpu::new(64); // RV64
//! cpu.load_program(0, &program);
//! let memory = RiscvMemory::new(64);
//! let shout = RiscvShoutTables::new(64);
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
