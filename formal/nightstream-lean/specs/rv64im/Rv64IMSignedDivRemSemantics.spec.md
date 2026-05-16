# Rv64IMSignedDivRemSemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for signed RV64IM division and remainder semantics.
- **What it is not**: It is not the advice-generation algorithm and it does not replace the lower-level signed DIV/REM soundness package.
- **Protocol role**: It turns the canonical `signedDivRem` opcode-class proof and `SignedDivRemProofPackage` into exact architectural semantics for signed DIV/REM lowerings.

## Covered Architectural Opcodes

- `DIV`
- `REM`
- `DIVW`
- `REMW`

## Central Object

`SignedDivRemExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = signedDivRem`,
- the underlying `SignedDivRemProofPackage`,
- `ExecutionCorrect` for those rows,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- the signed divisor-adjustment and dividend-sign remainder consequences as execution-level facts.

The fixed advice-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`.

## Exact Semantic Targets

For `DIV` and `REM`, the accepted quotient/remainder pair `(Q, R_signed)` must
satisfy `SignedDivRemSpec(RS1, Q, RS2, R_signed)`. The final architectural
result is:

- `DIV`: `out = Q`
- `REM`: `out = R_signed`

For the W-variants:

- first truncate to the low 32-bit operands,
- sign-extend those truncated operands to 64-bit semantic inputs,
- prove the same signed quotient/remainder target on those sign-extended word values,
- sign-extend the low 32 bits of the final quotient/remainder to 64 bits.

The execution-level contract must preserve:

- `ChangeDivisorCorrect`,
- `RemainderFromDividendSign`,
- the final `SignedDivRemSpec`.

## Required Constructors

The module must expose:

- `signedDivRemExecutionFacts_of_opcodeClassFacts`
- `signedDivRemExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_signedDivRemExecutionFacts`
- `adjacentStates_of_signedDivRemExecutionFacts`
- `preparedStep_matches_row_of_signedDivRemExecutionFacts`
- `successor_matches_rows_of_signedDivRemExecutionFacts`
- `row_has_opcodeClass_at_index_of_signedDivRemExecutionFacts`
- `changeDivisorCorrect_of_signedDivRemExecutionFacts`
- `remainderFromDividendSign_of_signedDivRemExecutionFacts`
- `signedDivRemSpec_of_signedDivRemExecutionFacts`

## Proof Obligations

- The execution-level owner factors through `Rv64IMSignedDivRemSoundness`, rather than weakening to row-local helper facts.
- Exact indexed row/step/successor consequences remain theorem-visible at the family owner.
- The unique signed-overflow divisor adjustment remains theorem-visible.
- Signed remainder reconstruction uses the dividend sign.
- `DIVW` and `REMW` inherit the same signed theorem target on sign-extended 32-bit operands.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `DIV`
  - `REM`
  - `DIVW / DIVUW / REMW / REMUW`
  - `CHANGE_DIVISOR`
  - `SIGNED_DIVREM_SPEC`
- **Associated Jolt sources**:
  - `./external/jolt/tracer/src/instruction/div.rs`
  - `./external/jolt/tracer/src/instruction/rem.rs`
  - `./external/jolt/tracer/src/instruction/divw.rs`
  - `./external/jolt/tracer/src/instruction/remw.rs`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/SignedDivRemSemantics.lean` | Exact signed DIV/REM semantic owner |
| `Nightstream/Rv64IM/Execution/SignedDivRemSemanticsInterface.lean` | Theorem-facing re-export surface |
