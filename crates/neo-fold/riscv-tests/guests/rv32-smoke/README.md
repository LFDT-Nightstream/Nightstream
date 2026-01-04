# `rv32-smoke` guest

This is a tiny RV32IM no-std program used by Neo’s end-to-end RISC-V prove+verify tests.

## Regenerating the committed ROM bytes

From this directory:

```bash
python3 export_rom_rs.py
```

This updates `crates/neo-fold/riscv-tests/binaries/rv32_smoke_rom.rs`.

Prereq: `rustup target add riscv32im-unknown-none-elf`.
