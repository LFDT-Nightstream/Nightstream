# RV64IM Program-ROM Bridge

## Purpose

This note explains the current blocker to supporting arbitrary `RV64IM` ELFs in
the proving path, what the analogous mechanism in Jolt looks like, and the
concrete bridge Nightstream should add.

The goal is:

- execute arbitrary real `RV64IM` ELF layout at runtime
- prove instruction fetches soundly
- stop assuming one executable region / one contiguous program blob

This note is about the **proof-side fetch model**, not about arithmetic opcode
support.

## Current Problem

The canonical RV64IM runner already uses real ELF input:

- [rv64_trace_shard.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/src/rv64_trace_shard.rs)

Runtime is already more general than the proof path:

- the CPU can load instructions sparsely by absolute address
- the ELF loader already preserves multiple `PT_LOAD` segments

But the proving path still assumes:

1. there is exactly **one** executable `PT_LOAD` segment
2. program ROM can be represented as **one contiguous executable region**
3. fetch proving can use guest `pc_before` directly as the program-ROM address

That assumption is enforced in the canonical RV64IM path in:

- [rv64_trace_shard.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/src/rv64_trace_shard.rs)

The practical consequence is:

- valid ELF parsing is not enough
- valid sparse execution is not enough
- proving still fails or is rejected for multi-executable-segment layouts

## What "single executable-region assumption" means

Today the proof effectively assumes:

> every fetched instruction address must lie in one single executable segment
> range that can be modeled as one contiguous program ROM region

That is **not**:

- an ISA rule
- an ELF rule
- a compiler rule

It is a current proving-model shortcut.

## Why This Blocks Arbitrary RV64IM ELF Support

A valid ELF may have code in multiple executable segments, for example:

- segment A at `0x1000..0x1fff`
- segment B at `0x4000..0x4fff`

Execution can still be valid:

- control flow can jump between those regions
- the emulator can fetch from both

But the proof currently wants:

- one executable segment
- one program-ROM base
- one contiguous program-ROM blob

So the current branch supports:

- the canonical real-ELF RV64IM note path

but not yet:

- arbitrary valid RV64IM ELF layouts

## How Jolt Does The Analogous Thing

Jolt does **not** use one custom adapter per ELF.

It separates:

1. **runtime addresses**
2. **proof-domain addresses**

### Runtime side

Jolt loads ELF sections into memory at their declared addresses:

- [external/jolt/tracer/src/emulator/mod.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/external/jolt/tracer/src/emulator/mod.rs)

So execution uses the real ELF layout.

### Proof side

Jolt remaps code into a compact bytecode / virtual-PC domain:

- [external/jolt/jolt-core/src/zkvm/bytecode/mod.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/external/jolt/jolt-core/src/zkvm/bytecode/mod.rs)

The key structure is:

- `BytecodePCMapper`

This maps:

- real instruction address

to:

- compact proof-domain PC index

Jolt also remaps RAM addresses into a compact logical RAM domain:

- [external/jolt/jolt-core/src/zkvm/ram/mod.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/external/jolt/jolt-core/src/zkvm/ram/mod.rs)
- [external/jolt/common/src/jolt_device.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/external/jolt/common/src/jolt_device.rs)

### Architectural lesson

The important pattern is:

- execution uses real sparse addresses
- proof uses compact logical domains

For Nightstream, the analogous missing piece is:

- a compact logical **program-ROM / fetch** domain

We already did the analogous thing for RV64 RAM.

## What Nightstream Should Do

We should add a **segmented executable-address remap** for the proving path.

That means:

1. keep guest `pc_before` as the architectural fetch address
2. collect all executable `PT_LOAD` segments
3. assign each executable segment a compact logical range in a proof-domain ROM
4. prove a mapping:
   - `logical_prog_addr = φ(pc_before)`
5. use `logical_prog_addr` for:
   - program ROM layout/init
   - fetch proving
   - decode lookup injection
   - any program-bus checks

This is the program-ROM analogue of the RV64 RAM bridge.

## Recommended Design

### New concept

Add an executable layout/remap object, analogous in spirit to the RAM remap:

- executable segment guest ranges
- logical proof ranges
- deterministic remap function `φ`

### Required property

For every fetched instruction:

- CPU semantics prove the guest fetch address `pc_before`
- the ROM bridge proves `logical_prog_addr = φ(pc_before)`
- the fetch/opening/program-ROM checks are performed against `logical_prog_addr`

### Why this is sound

This closes the gap between:

- guest execution semantics
- proof-domain ROM checks

without forcing all executable code to lie in one contiguous guest-address
region.

## What Not To Do

### Not the real fix

Do **not** solve this by:

- adding one custom adapter per ELF
- hand-special-casing note guests
- pretending the parser is the main problem

### Temporary stopgap only

A weaker stopgap is:

- stitch executable segments into one large zero-filled contiguous ROM span

That can broaden support slightly, but it has real drawbacks:

- large holes inflate ROM size
- large holes inflate proving size
- it still keeps the wrong proving abstraction

This can be acceptable as a short-term bridge, but it is not the right final
design for arbitrary RV64IM ELF support.

## Files Likely In Scope For The Real Fix

### Primary files

- [crates/neo-fold/src/rv64_trace_shard.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/src/rv64_trace_shard.rs)
- [crates/neo-fold/src/rv64_trace_shard/helpers.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/src/rv64_trace_shard/helpers.rs)
- [crates/neo-memory/src/riscv/elf_loader.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-memory/src/riscv/elf_loader.rs)
- [crates/neo-memory/src/riscv/rom_init.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-memory/src/riscv/rom_init.rs)
- [crates/neo-memory/src/riscv/exec_table.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-memory/src/riscv/exec_table.rs)

### Secondary files

- [crates/neo-memory/src/builder.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-memory/src/builder.rs)
- [crates/neo-fold/src/memory_sidecar/cpu_bus.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/src/memory_sidecar/cpu_bus.rs)
- [crates/neo-fold/src/memory_sidecar/memory/transcript_and_common.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/src/memory_sidecar/memory/transcript_and_common.rs)

## Tests That Should Fail Today

These are the right driver tests for the real fix.

### 1. Multiple executable segments

Add a test that builds or loads an RV64IM ELF with:

- two executable `PT_LOAD` segments
- entrypoint in segment A
- a control-flow transfer into segment B

Expected **today**:

- `Rv64TraceWiring::from_elf(...).prepare()` rejects the ELF

Expected **after the bridge**:

- prepare succeeds
- simulate succeeds
- prove/verify succeeds

### 2. Large executable gap

Add a test with:

- executable segment A
- executable segment B far away in guest address space

Expected **today**:

- rejection because the proof path still wants one executable region

Expected **after the bridge**:

- succeeds without forcing a giant sparse contiguous ROM region

### 3. Entry segment not equal to all code

Add a test where:

- entrypoint is in one executable segment
- not all subsequent code is in that same segment

Expected **today**:

- rejection or later failure from the single-segment proof assumption

Expected **after the bridge**:

- succeeds because code fetches are remapped across the full executable set

### 4. Canonical note repros remain green

The current real-ELF note repros must continue to pass:

- [test_rv64_note_from_elf.rs](/Users/nicolasarqueros/.codex/worktrees/7ac0/halo3/crates/neo-fold/riscv-tests/test_rv64_note_from_elf.rs)

This proves the generalized ROM bridge did not regress the maintained path.

## Suggested Test File

Add a new focused integration suite, for example:

- `crates/neo-fold/riscv-tests/test_rv64_multi_segment_from_elf.rs`

Use:

- `FoldingMode::Optimized`

and make sure the test programs are true RV64IM:

- no `A`
- no `C`

## Summary

The current blocker to arbitrary RV64IM ELF support is not mainly:

- parsing
- tracing
- arithmetic opcode support

It is:

- the proof-side instruction-fetch model still assuming one contiguous
  executable region

Jolt’s relevant architectural lesson is:

- use real addresses for execution
- use compact logical domains for proof

Nightstream already does that for RAM.

To support arbitrary RV64IM ELF layouts, Nightstream needs the same idea for:

- program ROM
- instruction fetch
- decode/program-bus proving
