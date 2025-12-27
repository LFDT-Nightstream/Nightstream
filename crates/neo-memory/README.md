# neo-memory

Twist & Shout memory/lookup protocols, shared CPU-bus integration, and RV64IMAC RISC-V helpers.

## What’s In This Crate

- **Twist**: read/write memory argument (Route A)
- **Shout**: read-only lookup argument (Route A)
- **Shared CPU bus integration**: bind CPU semantics to Twist/Shout fields inside the CPU witness
- **RISC-V RV64IMAC**: instruction decode/execute + lookup-table helpers (kept in `riscv_*` modules)
- **Output binding**: output sumcheck utilities for binding program I/O to proofs

## Modules

| Module | Description |
|--------|-------------|
| `twist.rs` | Twist proof metadata + helpers |
| `twist_oracle.rs` | Sum-check oracles for Twist/Shout |
| `shout.rs` | Shout proof metadata + helpers |
| `addr.rs` | Bit-address validation + encoding utilities |
| `cpu/` | Shared-bus layout + CPU binding constraints + R1CS adapter |
| `builder.rs` | Build per-step witness bundles from a `VmTrace` |
| `elf_loader.rs` | Load ELF and raw RISC-V binaries |
| `output_check.rs` | Output sumcheck (bind program I/O) |
| `riscv_lookups/` | RV64IMAC decode/execute + lookup tables |
| `riscv_ccs.rs` | RISC-V CCS helpers |
| `riscv_shout_oracle.rs` | RISC-V-specific Shout oracle helpers |
| `witness.rs` | Witness/instance data structures |

## Shared CPU Bus (Important)

In shared-bus mode, Twist/Shout do **not** have independent commitment namespaces. Their access-row
columns live in the **tail of the CPU witness** `z`, and the fold/sidecar logic consumes only CPU-derived
openings. This prevents CPU↔memory forking at the commitment level.

Security still requires **semantic binding** inside the CPU constraints: flag-gated checks must also
enforce padding-to-zero so inactive bus fields cannot float.

## Tests

```bash
cargo test -p neo-memory --release
```
