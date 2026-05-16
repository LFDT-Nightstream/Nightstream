# Rv64IMWordShiftSemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for RV64 word-width arithmetic and shift semantics.
- **What it is not**: It is not the Stage-1 shift-slot owner and it does not replace the generic opcode-class semantic bundle.
- **Protocol role**: It turns the canonical `wordShift` opcode-class proof into exact architectural semantics for the W-variants and their committed lowerings.

## Covered Architectural Opcodes

- `ADDW`, `ADDIW`
- `SUBW`
- `SLLW`, `SLLIW`
- `SRLW`, `SRLIW`
- `SRAW`, `SRAIW`

## Central Object

`WordShiftExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = wordShift`,
- `ExecutionCorrect` for those rows,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- exact row-wise agreement that every row in the package belongs to the
  `wordShift` opcode class.

The fixed committed-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`.
The exact opcode binding and the corrected `SRAW` / `SRAIW` specialization live
in the dedicated opcode owner above this module.

## Exact Semantic Targets

This module owns class-level execution facts only. The exact word-width
arithmetic and shift targets are discharged one layer above this module:

- `Rv64IMWordShiftLoweringSemantics` preserves the committed-sequence theorem
  package for the word/shift family.
- `Rv64IMWordShiftOpcodeSemantics` specializes that preserved sequence package
  to the exact word-shift opcode carried by `StepComposition`.
- `Rv64IMWordShiftWordArithmetic` then turns that exact opcode-bound surface
  into exact word-result and authenticated writeback equalities.
- The corrected `SRAW` / `SRAIW` semantics therefore remain theorem-facing at
  exact word-result granularity rather than only at raw class-packaging
  granularity.

## Required Constructors

The module must expose:

- `wordShiftExecutionFacts_of_opcodeClassFacts`
- `wordShiftExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_wordShiftExecutionFacts`
- `adjacentStates_of_wordShiftExecutionFacts`
- `preparedStep_matches_row_of_wordShiftExecutionFacts`
- `successor_matches_rows_of_wordShiftExecutionFacts`
- `row_has_opcodeClass_at_index_of_wordShiftExecutionFacts`

## Proof Obligations

- Exact indexed row/step/successor consequences remain theorem-visible at the
  class owner.
- The class owner may not weaken the family to a broader opcode class.
- Exact opcode binding, low-5-bit shift discipline, and corrected `SRAW` /
  `SRAIW` specialization belong to the dedicated lowering/opcode owners above
  this module.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `ADDW / ADDIW`
  - `SUBW`
  - `SLLW / SLLIW`
  - `SRLW / SRLIW`
  - `SRAW / SRAIW`
- **Associated Jolt sources**:
  - `./external/jolt/tracer/src/instruction/sraw.rs`
  - `./external/jolt/tracer/src/instruction/sraiw.rs`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/WordShiftSemantics.lean` | Exact word/shift semantic owner |
| `Nightstream/Rv64IM/Execution/WordShiftSemanticsInterface.lean` | Theorem-facing re-export surface |
