//! RISC-V support for the canonical RV64IM proving path plus retained internal
//! width-specific helpers.
//!
//! The supported product contract is currently:
//! - canonical proving path: real ELF via `Rv64TraceWiring::from_elf(...)`
//! - ISA profile: RV64IM
//! - not supported: compressed instructions (`C`) and atomics (`A`)
//!
//! RV32 code that remains in this module tree is temporary internal
//! migration/reference coverage, not a supported product-facing path.

pub mod ccs;
pub mod decomposition_semantics;
pub mod elf_loader;
pub mod exec_table;
pub mod instruction;
pub mod lookups;
pub mod lowering;
pub mod memory_layout;
pub mod packed;
pub mod profile;
pub mod rom_init;
pub mod shout_oracle;
pub mod sparse_access;
pub mod trace;
