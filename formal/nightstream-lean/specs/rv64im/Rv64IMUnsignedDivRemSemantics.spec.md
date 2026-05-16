# Rv64IMUnsignedDivRemSemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for unsigned RV64IM division and remainder semantics.
- **What it is not**: It is not the base unsigned DIV/REM soundness package and it does not own signed divisor-adjustment or signed remainder reconstruction.
- **Protocol role**: It turns the canonical `unsignedDivRem` opcode-class proof and `UnsignedDivRemSoundnessProofPackage` into exact architectural semantics for the unsigned DIV/REM family.

## Covered Architectural Opcodes

- `DIVU`
- `REMU`
- `DIVUW`
- `REMUW`

## Central Object

`UnsignedDivRemExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = unsignedDivRem`,
- the underlying `UnsignedDivRemSoundnessProofPackage`,
- `ExecutionCorrect` for those rows,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- the theorem-visible `MULU_NO_OVERFLOW` support and its execution-level witness,
- the theorem-visible `UnsignedDivRemSpec`,
- the theorem-level unsigned determinism consequence.

The fixed advice-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`.

## Exact Semantic Targets

For `DIVU` and `REMU`, the accepted quotient/remainder pair `(Q, R)` must
satisfy `UnsignedDivRemSpec(RS1, Q, RS2, R)`. The final architectural result is:

- `DIVU`: `out = Q`
- `REMU`: `out = R`

For the W-variants:

- first truncate to the low 32-bit operands,
- interpret those truncated operands as unsigned 32-bit values,
- prove the same unsigned quotient/remainder target on those 32-bit values,
- sign-extend the low 32 bits of the final quotient/remainder to 64 bits.

## Required Constructors

The module must expose:

- `unsignedDivRemExecutionFacts_of_opcodeClassFacts`
- `unsignedDivRemExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_unsignedDivRemExecutionFacts`
- `adjacentStates_of_unsignedDivRemExecutionFacts`
- `preparedStep_matches_row_of_unsignedDivRemExecutionFacts`
- `successor_matches_rows_of_unsignedDivRemExecutionFacts`
- `row_has_opcodeClass_at_index_of_unsignedDivRemExecutionFacts`
- `mulUNoOverflowBound_of_unsignedDivRemExecutionFacts`
- `mulUNoOverflow_of_unsignedDivRemExecutionFacts`
- `unsignedDivRemSpec_of_unsignedDivRemExecutionFacts`
- `unsignedDivRemDeterministic_of_unsignedDivRemExecutionFacts`

## Proof Obligations

- The execution-level owner factors through `Rv64IMUnsignedDivRemSoundness`, rather than restating a weaker unsigned target.
- Exact indexed row/step/successor consequences remain theorem-visible at the family owner.
- The `MULU_NO_OVERFLOW` guard remains theorem-visible at the execution level.
- The accepted unsigned quotient/remainder pair is deterministic.
- `DIVUW` and `REMUW` inherit the same unsigned theorem target on truncated 32-bit operands.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `DIVU`
  - `REMU`
  - `DIVW / DIVUW / REMW / REMUW`
  - `MULU_NO_OVERFLOW`
- **Associated Jolt sources**:
  - `./external/jolt/tracer/src/instruction/divu.rs`
  - `./external/jolt/tracer/src/instruction/remu.rs`
  - `./external/jolt/tracer/src/instruction/divuw.rs`
  - `./external/jolt/tracer/src/instruction/remuw.rs`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/UnsignedDivRemSemantics.lean` | Exact unsigned DIV/REM semantic owner |
| `Nightstream/Rv64IM/Execution/UnsignedDivRemSemanticsInterface.lean` | Theorem-facing re-export surface |
