//! Neo Memory: Twist & Shout protocols plus RISC-V helpers.
//!
//! This crate provides:
//!
//! - **Twist**: Read-write memory argument for proving memory operations
//! - **Shout**: Read-only memory / lookup table argument for proving ALU operations
//! - **RISC-V helpers**: instruction decode/encode, tracing CPU, and CCS builders
//!
//! # RISC-V Support
//!
//! The current proving integration is RV32-focused (e.g. the shared-bus RV32 B1 path assumes
//! `xlen == 32`, no compressed instructions, and 4-byte aligned control flow).
//! RV64 proving is not yet supported by the Shout key encoding used in this path.
//!
//! # Key Modules
//!
//! - [`riscv::lookups`]: RISC-V instruction decoding, encoding, execution, and lookup tables
//! - [`riscv::ccs`]: Constraint system (CCS) for RISC-V instruction verification
//! - [`riscv::elf_loader`]: Load ELF and raw RISC-V binaries
//! - [`output_check`]: Output sumcheck for binding program I/O to proofs
//! - [`twist`]: Read-write memory protocol
//! - [`shout`]: Read-only memory / lookup protocol

pub mod addr;
pub mod ajtai;
pub mod bit_ops;
pub mod builder;
pub mod cpu;
pub mod mem_init;
pub mod mle;
pub mod output_check;
pub mod plain;
pub mod riscv;
pub mod shout;
pub mod sparse_time;
pub mod sumcheck_proof;
pub mod ts_common;
pub mod twist;
pub mod twist_oracle;
pub mod witness;

pub use builder::*;
pub use cpu::constraints::*;
pub use cpu::r1cs_adapter::*;
pub use mem_init::*;
pub use mle::*;
pub use output_check::{
    generate_output_sumcheck_proof, generate_output_sumcheck_proof_and_challenges, verify_output_sumcheck,
    verify_output_sumcheck_rounds_get_state, OutputBindingProof, OutputCheckError, OutputSumcheckParams,
    OutputSumcheckProof, OutputSumcheckProver, OutputSumcheckState, ProgramIO,
};
pub use plain::*;
pub use sumcheck_proof::BatchedAddrProof;
pub use witness::*;
