# Rv64IMNativeAluSemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for the native one-row RV64IM arithmetic, logic, comparison, upper-immediate, and trivial system family.
- **What it is not**: It is not the Stage-1 execution-row binding owner, not the committed-sequence owner for multi-row lowerings, and not the kernel theorem owner.
- **Protocol role**: It turns the canonical `nativeAlu` opcode-class proof into exact one-row architectural semantics for the native non-memory, non-multiply, non-div/rem opcodes.

## Covered Architectural Opcodes

This family covers the architectural instructions whose committed lowered sequence is one native row:

- `ADD`, `ADDI`
- `SUB`
- `AND`, `ANDI`
- `OR`, `ORI`
- `XOR`, `XORI`
- `SLT`, `SLTI`
- `SLTU`, `SLTIU`
- `LUI`
- `AUIPC`
- `FENCE`
- `ECALL`

## Central Object

`NativeAluExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = nativeAlu`,
- `ExecutionCorrect` for that native row,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- exact row-wise agreement that every row in the package belongs to the
  `nativeAlu` opcode class.

The fixed committed-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`,
rather than by this local family-semantic owner. Exact opcode binding and
opcode-specific decoded-row consequences live in the dedicated opcode owner
above this module.

## Exact Semantic Targets

This module owns class-level execution facts only. The exact native-ALU opcode
contracts are discharged one layer above this module:

- `Rv64IMNativeAluLoweringSemantics` preserves the committed-sequence theorem
  package for the native one-row family.
- `Rv64IMNativeAluOpcodeSemantics` specializes that preserved sequence package
  to the exact native opcode carried by `StepComposition`.
- Exact `ECALL` termination therefore remains theorem-facing, but at opcode
  granularity rather than at raw class-packaging granularity.

## Required Constructors

The module must expose:

- `nativeAluExecutionFacts_of_opcodeClassFacts`
- `nativeAluExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_nativeAluExecutionFacts`
- `adjacentStates_of_nativeAluExecutionFacts`
- `preparedStep_matches_row_of_nativeAluExecutionFacts`
- `successor_matches_rows_of_nativeAluExecutionFacts`
- `row_has_opcodeClass_at_index_of_nativeAluExecutionFacts`

so consumers can recover the exact native-ALU semantic bundle from either:

- one exact `nativeAlu` opcode-class semantic package, or
- one `StepCompositionProofPackage`.

## Proof Obligations

- Exact indexed row/step/successor consequences remain theorem-visible at the family owner, rather than being recoverable only by unpacking `classFacts`.
- The family owner may not weaken native ALU closure to a broader opcode class.
- Exact opcode binding, decoded-row flag consequences, and `ECALL` termination belong to the dedicated lowering/opcode owners above this module.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - primitive virtual-instruction catalog
  - per-instruction virtual sequences
  - final-boundary / termination claim

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/NativeAluSemantics.lean` | Exact native-ALU semantic owner |
| `Nightstream/Rv64IM/Execution/NativeAluSemanticsInterface.lean` | Theorem-facing re-export surface |

## Downstream Owners

- `Nightstream/Rv64IM/Execution/NativeAluLoweringSemantics.lean`
- `Nightstream/Rv64IM/Execution/NativeAluOpcodeSemantics.lean`
