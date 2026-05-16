# Rv64IMNarrowMemorySemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for RV64IM narrow loads and narrow stores lowered through aligned 64-bit RAM words.
- **What it is not**: It is not the Stage-1 helper-arithmetic owner and it does not replace Stage-2 aligned-word RAM authentication.
- **Protocol role**: It turns the canonical `narrowMemory` opcode-class proof into exact lowered-sequence semantic closure for the narrow memory family.

## Covered Architectural Opcodes

- Loads: `LB`, `LBU`, `LH`, `LHU`, `LW`, `LWU`
- Stores: `SB`, `SH`, `SW`

Native aligned 64-bit `LD` / `SD` rows are out of scope for this owner. They
belong to the dedicated exact aligned-memory opcode owner above
`Rv64IMStepComposition`.

## Central Object

`NarrowMemoryExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = narrowMemory`,
- `ExecutionCorrect` for those rows,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- preserved-state and touched-state consequences for the lowered sequences.

The fixed committed-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`.

## Required Constructors

The module must expose:

- `narrowMemoryExecutionFacts_of_opcodeClassFacts`
- `narrowMemoryExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_narrowMemoryExecutionFacts`
- `adjacentStates_of_narrowMemoryExecutionFacts`
- `preparedStep_matches_row_of_narrowMemoryExecutionFacts`
- `successor_matches_rows_of_narrowMemoryExecutionFacts`
- `row_has_opcodeClass_at_index_of_narrowMemoryExecutionFacts`

## Proof Obligations

- Exact indexed row/step/successor consequences remain theorem-visible at the family owner.
- This owner does not re-own `align_down_8`, `byte_offset_8`, `extract_extend`, or `blend`; those remain with `Rv64IMNarrowMemoryHelpers`.
- This owner does not re-own the exact `extractExtend` / `blend` to
  `ALU_RESULT` bridge; that is packaged by the dedicated exact narrow-memory
  helper-result owner above exact family closure.
- This owner does not re-own authenticated aligned RAM load/store payload facts; those are packaged by the dedicated exact narrow-memory payload owner above exact family closure.
- `LB` / `LBU` / `LH` / `LHU` / `LW` / `LWU` and `SB` / `SH` / `SW` still use the exact committed row lists fixed by `riscv-kernel.md`, not prover-chosen alternatives.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - memory-support virtual instructions
  - narrow-memory helper arithmetic
  - `LB / LBU / LH / LHU / LW / LWU`
  - `SB / SH / SW`
- **Associated Jolt sources**:
  - `./external/jolt/book/src/how/architecture/ram.md`
  - `./external/jolt/book/src/how/architecture/emulation.md`
  - `./external/jolt/tracer/src/instruction/lwu.rs`
  - `./external/jolt/tracer/src/instruction/sh.rs`
  - `./external/jolt/tracer/src/instruction/sb.rs`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/NarrowMemorySemantics.lean` | Exact narrow-memory semantic owner |
| `Nightstream/Rv64IM/Execution/NarrowMemorySemanticsInterface.lean` | Theorem-facing re-export surface |
