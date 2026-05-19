# Rv64IMNarrowMemoryOpcodeSemantics Spec

## Purpose

- **What it is**: The exact theorem-facing opcode owner for RV64IM narrow loads and narrow stores.
- **What it is not**: It is not the helper-arithmetic owner, not the aligned-word RAM authentication owner, and not the trace/kernel theorem owner.
- **Protocol role**: It sits above `Rv64IMNarrowMemoryLoweringSemantics` and exposes exact decoded-row opcode consequences for `LB`, `LBU`, `LH`, `LHU`, `LW`, `LWU`, `SB`, `SH`, and `SW`.

## Central Objects

- `NarrowMemoryWidths(MemWidth)` fixes the theorem-facing width constants used by the architectural opcode surface.
- `NarrowMemoryOpcode` enumerates the architectural narrow-memory opcodes.
- `NarrowMemoryOpcodeBound(widths, row, opcode)` is the exact decoded-row contract for that opcode.

## Required Theorem Surface

The module must expose:

- `flags_of_narrowMemoryOpcodeBound`
- `rowClassFlags_of_narrowMemoryOpcodeBound`
- `flags_of_narrowMemoryOpcodeSemantics`
- `classFlags_of_narrowMemoryOpcodeSemantics`
- `x0WritePreserved_of_narrowMemoryOpcodeSemantics`
- `sequenceCorrect_of_narrowMemoryOpcodeSemantics`
- `sequenceDeterministic_of_narrowMemoryOpcodeSemantics`

## Exact Opcode Contracts

For each narrow-memory opcode, the theorem-facing decoded-row contract fixes:

- load versus store role,
- `usesRs2`,
- `writesAluToRd = 0`,
- `memWidth`,
- `memUnsigned`,
- and the fact that the row is not control-flow, multiply, div/rem, or W-width arithmetic.

These contracts are:

- `LB`, `LBU`: byte loads
- `LH`, `LHU`: halfword loads
- `LW`, `LWU`: word loads
- `SB`: byte store
- `SH`: halfword store
- `SW`: word store

`rd = x0` is handled by the separate theorem
`x0WritePreserved_of_narrowMemoryOpcodeSemantics`, because the kernel spec
makes narrow-load write suppression depend on `rd`, not only on opcode.

## Proof Obligations

- The owner must not weaken the exact family lowering theorem to a generic “memory row” statement.
- The width/unsigned/load/store contract must remain theorem-visible at the opcode owner.
- The preserved committed-sequence correctness and determinism theorems must still be available at the opcode owner.
- The owner must remain consistent with the aligned-word RAM model and the helper arithmetic fixed by `riscv-kernel.md`.
- The owner must preserve the kernel-spec `x0` sink/write-suppression rule.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - memory-support virtual instructions
  - narrow-memory helper arithmetic
  - `LB / LBU / LH / LHU / LW / LWU`
  - `SB / SH / SW`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/NarrowMemoryOpcodeSemantics.lean` | Exact narrow-memory opcode owner |
| `Nightstream/Rv64IM/Execution/NarrowMemoryOpcodeSemanticsInterface.lean` | Theorem-facing re-export surface |
