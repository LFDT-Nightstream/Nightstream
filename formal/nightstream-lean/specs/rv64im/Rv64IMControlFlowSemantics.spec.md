# Rv64IMControlFlowSemantics Spec

## Purpose

- **What it is**: The exact execution-level owner for RV64IM jumps and branches.
- **What it is not**: It is not the Stage-3 continuity owner and it does not replace the Stage-1 control-flow row binding.
- **Protocol role**: It turns the canonical `controlFlow` opcode-class proof into exact architectural `PC_NEXT`, link-register, branch-predicate, and taken-target-alignment facts.

## Covered Architectural Opcodes

- `JAL`
- `JALR`
- `BEQ`, `BNE`
- `BLT`, `BGE`
- `BLTU`, `BGEU`

## Central Object

`ControlFlowExecutionFacts(proof)` packages:

- exact class equality `proof.opcodeClass = controlFlow`,
- `ExecutionCorrect` for the control-flow row,
- exact indexed frame/row equality,
- exact adjacent-frame state equality,
- exact indexed prepared-step/row equality,
- exact indexed successor/adjacent-row equality,
- exact branch predicate semantics by opcode,
- exact `PC_NEXT` routing semantics,
- exact link-register semantics for `JAL` and `JALR`,
- the authenticated taken-target alignment discharge.

The fixed committed-sequence proof package for this family is owned by
`Rv64IMStepComposition` and preserved by `Rv64IMExactOpcodeFamilySemantics`.

## Exact Semantic Targets

For jumps:

- `JAL`: `ALU_OUT = PC + 4` and `PC_NEXT = PC + IMM`
- `JALR`: `ALU_OUT = PC + 4` and `PC_NEXT = (RS1 + IMM) & ~1`

For branches:

- `BEQ`: `taken ↔ RS1 = RS2`
- `BNE`: `taken ↔ RS1 ≠ RS2`
- `BLT`: `taken ↔ RS1 <_s RS2`
- `BGE`: `taken ↔ RS1 ≥_s RS2`
- `BLTU`: `taken ↔ RS1 <_u RS2`
- `BGEU`: `taken ↔ RS1 ≥_u RS2`

and for every branch row:

- `PC_NEXT = PC + IMM` if `taken = 1`,
- `PC_NEXT = PC + 4` if `taken = 0`.

## Required Constructors

The module must expose:

- `controlFlowExecutionFacts_of_opcodeClassFacts`
- `controlFlowExecutionFacts_of_stepComposition`
- `frame_row_eq_at_index_of_controlFlowExecutionFacts`
- `adjacentStates_of_controlFlowExecutionFacts`
- `preparedStep_matches_row_of_controlFlowExecutionFacts`
- `successor_matches_rows_of_controlFlowExecutionFacts`
- `row_has_opcodeClass_at_index_of_controlFlowExecutionFacts`
- `takenTargetAlignment_of_controlFlowExecutionFacts`

## Proof Obligations

- Exact indexed row/step/successor consequences remain theorem-visible at the family owner.
- The taken-target alignment discharge remains theorem-visible and is not hidden inside Stage-1 row-local proofs.
- `JALR` semantics include both the `& ~1` clearing rule and the resulting 4-byte alignment obligation required by the kernel.
- Branch rows do not perform an architectural destination-register write.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - normative address model
  - `JAL`
  - `JALR`
  - `BEQ` / `BNE` / `BLT` / `BGE` / `BLTU` / `BGEU`
  - taken-target alignment

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/ControlFlowSemantics.lean` | Exact control-flow semantic owner |
| `Nightstream/Rv64IM/Execution/ControlFlowSemanticsInterface.lean` | Theorem-facing re-export surface |
