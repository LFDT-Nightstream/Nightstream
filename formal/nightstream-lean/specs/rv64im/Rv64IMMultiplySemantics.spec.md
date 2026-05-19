# Rv64IMMultiplySemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for the RV64IM multiply family.
- **What it is not**: It is not the Stage-1 byte-level multiplication-slot owner and it does not replace the generic opcode-class semantic bundle.
- **Protocol role**: It turns the canonical `multiply` opcode-class proof into exact architectural semantics for low-half, high-half, and word-width multiply instructions.

## Covered Architectural Opcodes

- `MUL`
- `MULH`
- `MULHU`
- `MULHSU`
- `MULW`

## Central Object

`MultiplyExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = multiply`,
- `ExecutionCorrect` for those rows,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- exact row-wise agreement that every row in the package belongs to the
  `multiply` opcode class.

The fixed committed-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`.
Exact opcode binding and opcode-specific decoded-row consequences live in the
dedicated opcode owner above this module.

## Exact Semantic Targets

This module owns class-level execution facts only. The exact multiply opcode
contracts are discharged one layer above this module:

- `Rv64IMMultiplyLoweringSemantics` preserves the committed-sequence theorem
  package for the multiply family.
- `Rv64IMMultiplyOpcodeSemantics` specializes that preserved sequence package
  to the exact multiply opcode carried by `StepComposition`.
- Exact `MULW` W-width distinction and decoded-row opcode binding therefore
  remain theorem-facing, but at opcode granularity rather than at raw
  class-packaging granularity.

## Required Constructors

The module must expose:

- `multiplyExecutionFacts_of_opcodeClassFacts`
- `multiplyExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_multiplyExecutionFacts`
- `adjacentStates_of_multiplyExecutionFacts`
- `preparedStep_matches_row_of_multiplyExecutionFacts`
- `successor_matches_rows_of_multiplyExecutionFacts`
- `row_has_opcodeClass_at_index_of_multiplyExecutionFacts`

## Proof Obligations

- Exact indexed row/step/successor consequences remain theorem-visible at the family owner.
- The family owner may not weaken multiply closure to a broader opcode class.
- Exact opcode binding, decoded-row flag consequences, and `MULW` W-width distinction belong to the dedicated lowering/opcode owners above this module.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `MUL`
  - `MULH`
  - `MULHU`
  - `MULHSU`
  - `MULW`
- **Associated Jolt sources**:
  - `./external/jolt/tracer/src/instruction/mul.rs`
  - `./external/jolt/tracer/src/instruction/mulh.rs`
  - `./external/jolt/tracer/src/instruction/mulhsu.rs`
  - `./external/jolt/tracer/src/instruction/mulhu.rs`
  - `./external/jolt/tracer/src/instruction/mulw.rs`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/MultiplySemantics.lean` | Exact multiply-family semantic owner |
| `Nightstream/Rv64IM/Execution/MultiplySemanticsInterface.lean` | Theorem-facing re-export surface |

## Downstream Owners

- `Nightstream/Rv64IM/Execution/MultiplyLoweringSemantics.lean`
- `Nightstream/Rv64IM/Execution/MultiplyOpcodeSemantics.lean`
