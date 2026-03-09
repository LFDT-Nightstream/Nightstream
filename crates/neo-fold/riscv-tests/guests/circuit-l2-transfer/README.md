# `circuit-l2-transfer` guest

Legacy directory name for the canonical RV64IM note-spend guest source.

Notes:
- The maintained note repro path is the real ELF RV64IM flow in `crates/neo-fold/riscv-tests/test_rv64_note_from_elf.rs`.
- `crates/neo-fold/riscv-tests/support/rv64_guest.rs` builds this guest into an RV64IM ELF for the canonical spend repro.
- The old RV32 compiled-ROM path has been removed.
