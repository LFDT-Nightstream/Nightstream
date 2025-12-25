//! Neo Memory: Twist & Shout protocols with RV64IMAC RISC-V support.
//!
//! This crate provides:
//!
//! - **Twist**: Read-write memory argument for proving memory operations
//! - **Shout**: Read-only memory / lookup table argument for proving ALU operations
//! - **RISC-V RV64IMAC**: Complete instruction set implementation
//!
//! # RISC-V Support
//!
//! The [`riscv_lookups`] module implements the full **RV64IMAC** instruction set:
//!
//! | Extension | Description |
//! |-----------|-------------|
//! | **I** | Base Integer (64-bit) |
//! | **M** | Multiply/Divide |
//! | **A** | Atomics (LR/SC, AMO) |
//! | **C** | Compressed (16-bit instructions) |
//!
//! This matches [Jolt's](https://github.com/a16z/jolt) RISC-V support.
//!
//! # Key Modules
//!
//! - [`riscv_lookups`]: RISC-V instruction decoding, encoding, execution, and lookup tables
//! - [`riscv_ccs`]: Constraint system (CCS) for RISC-V instruction verification
//! - [`elf_loader`]: Load ELF and raw RISC-V binaries
//! - [`output_check`]: Output sumcheck for binding program I/O to proofs
//! - [`twist`]: Read-write memory protocol
//! - [`shout`]: Read-only memory / lookup protocol

pub mod addr;
pub mod ajtai;
pub mod bit_ops;
pub mod builder;
pub mod elf_loader;
pub mod encode;
pub mod mem_init;
pub mod mle;
pub mod output_check;
pub mod plain;
pub mod r1cs_adapter;
pub mod riscv_ccs;
pub mod riscv_lookups;
pub mod shout;
pub mod sumcheck_proof;
pub mod ts_common;
pub mod twist;
pub mod twist_oracle;
pub mod witness;

pub use builder::*;
pub use encode::*;
pub use mem_init::*;
pub use mle::*;
pub use output_check::{
    generate_output_sumcheck_proof, generate_output_sumcheck_proof_and_challenges, verify_output_sumcheck,
    verify_output_sumcheck_rounds_get_state, OutputBindingProof, OutputCheckError, OutputSumcheckParams, OutputSumcheckProof,
    OutputSumcheckProver, OutputSumcheckState, ProgramIO,
};
pub use plain::*;
pub use r1cs_adapter::*;
pub use sumcheck_proof::BatchedAddrProof;
pub use witness::*;
