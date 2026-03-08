//! RISC-V support (RV32-focused proving integration).
//!
//! This module groups RISC-V-specific components under `neo_memory::riscv::*`.

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
