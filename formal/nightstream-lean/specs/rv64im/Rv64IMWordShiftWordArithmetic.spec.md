# Rv64IMWordShiftWordArithmetic Spec

## Purpose

- **What it is**: The exact execution-level owner for word-level arithmetic
  equalities in the RV64IM word/shift family.
- **What it is not**: It is not the Stage-1 shift-slot owner, and it does not
  replace the exact opcode or lowering owners beneath it.
- **Protocol role**: It turns exact opcode-bound word/shift evidence into exact
  Stage-1 `aluResult` word equalities and exact authenticated non-`x0`
  writeback-word equalities.

## Covered Architectural Opcodes

- `ADDW`, `ADDIW`
- `SUBW`
- `SLLW`, `SLLIW`
- `SRLW`, `SRLIW`
- `SRAW`, `SRAIW`

## Exact Semantic Targets

For each covered opcode, the module owns two theorem shapes:

- exact Stage-1 equality:
  `executionRow.results.aluResult = WordShiftWordResult(..., opcode)`
- exact authenticated non-`x0` equality:
  `limbPairToWord(registerTwist.wvReg) = WordShiftWordResult(..., opcode)`

The corrected `SRAW` / `SRAIW` behavior therefore becomes theorem-facing at the
word-result layer, not only at the decode/flag layer.

## Required Constructors

The module must expose opcode-specialized theorems for:

- `addw`, `addiw`
- `subw`
- `sllw`, `slliw`
- `srlw`, `srliw`
- `sraw`, `sraiw`

Each opcode must have:

- one exact `aluResult` theorem
- one exact authenticated non-`x0` writeback theorem

## Proof Obligations

- The module must factor through `Rv64IMWordShiftOpcodeSemantics` and the
  explicit word/limb representation bridge already owned by `StepComposition`.
- The module may not introduce a new evaluator disconnected from
  `StepComposition.wordShiftWordOps` / `wordShiftEncodedOps`.
- Authenticated writeback theorems must require `rd ≠ x0`.
- Immediate forms must use the decoded immediate exactly; register forms must
  use the authenticated `rvRs2` value exactly.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `ADDW / ADDIW`
  - `SUBW`
  - `SLLW / SLLIW`
  - `SRLW / SRLIW`
  - `SRAW / SRAIW`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/WordShiftWordArithmetic.lean` | Exact execution-level word/shift word arithmetic |
| `Nightstream/Rv64IM/Execution/WordShiftWordArithmeticInterface.lean` | Theorem-facing re-export surface |
