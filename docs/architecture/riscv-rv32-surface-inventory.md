# Remaining `Rv32*` Surface Inventory

This note tracks the remaining `Rv32*`-named surfaces after the canonical
product path moved to real-ELF RV64IM proving.

The classification below is intentionally pragmatic:
- **true RV32**: legitimate RV32-only implementation that should not be renamed
  or deleted casually
- **shared but poorly named**: behavior is already shared or reused by RV64, but
  the symbol still carries an `rv32` prefix
- **dead**: old note-path baggage that has already been removed and should not be
  reintroduced
- **publicly misleading**: surface that would incorrectly imply RV32 is still a
  supported peer product path

## True RV32

These remain valid RV32-specific implementations or internal legacy coverage:

- `crates/neo-fold/src/riscv_trace_shard.rs`
- `crates/neo-memory/src/riscv/trace/layout.rs`
- `crates/neo-memory/src/riscv/trace/witness.rs`
- RV32 packed shout/oracle machinery in `neo-memory` / `neo-fold`
- RV32-specific redteam, perf, and integration suites that still validate shared
  invariants

## Shared But Poorly Named

These helpers are already reused by both widths and should continue migrating
toward width-generic names:

- `rv32_decode_lookup_*`
- `rv32_trace_lookup_*`

Primary current reuse sites include:

- `crates/neo-fold/src/rv64_trace_shard.rs`
- `crates/neo-fold/src/rv64_trace_shard/helpers.rs`
- `crates/neo-memory/src/builder.rs`
- `crates/neo-fold/src/memory_sidecar/cpu_bus.rs`

## Dead

These old RV32 note-path surfaces have already been removed and should stay gone:

- the compiled-ROM note test flow
- generated note/transfer ROM blobs
- the ROM export script
- the old split-tiling note fixture

## Publicly Misleading

These are the surfaces that should not define the supported product contract:

- public proof-profile wording that implied RV32 remained a supported peer path
- documentation that implied the deleted RV32 compiled-ROM note flow still
  existed

The supported product contract is now:

- canonical path: real ELF via `Rv64TraceWiring::from_elf(...)`
- ISA: RV64IM
- not supported: `A`, `C`
- retained RV32 code is temporary internal migration/reference coverage only
