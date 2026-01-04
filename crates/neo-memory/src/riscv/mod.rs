//! RISC-V support (RV32-focused proving integration).
//!
//! This module groups RISC-V-specific components under `neo_memory::riscv::*`.

pub mod ccs;
pub mod elf_loader;
pub mod lookups;
pub mod rom_init;
pub mod shard;
pub mod shout_oracle;
