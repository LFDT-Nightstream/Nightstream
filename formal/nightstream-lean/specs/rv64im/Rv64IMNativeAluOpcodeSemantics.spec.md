# Rv64IMNativeAluOpcodeSemantics Spec

## Purpose

- **What it is**: The exact theorem-facing opcode owner for the RV64IM native one-row ALU, comparison, upper-immediate, and trivial system family.
- **What it is not**: It is not the class-level native-ALU semantic owner, not the Stage-1 decode owner, and not the trace/kernel lift owner.
- **Protocol role**: It sits above `Rv64IMNativeAluLoweringSemantics` and closes the exact decoded-row opcode gap for `ADD`, `ADDI`, `SUB`, `AND`, `ANDI`, `OR`, `ORI`, `XOR`, `XORI`, `SLT`, `SLTI`, `SLTU`, `SLTIU`, `LUI`, `AUIPC`, `FENCE`, and `ECALL`.

## Central Objects

- `NativeAluOpcode` enumerates the architectural opcodes in this family.
- `NativeAluAluOps(AluOp)` fixes the theorem-facing ALU operation tags used by the decoded-row contract.
- `NativeAluOpcodeBound(ops, row, opcode)` is the exact decoded-row contract for that opcode.

## Required Theorem Surface

The module must expose:

- `opcodeBound_of_nativeAluOpcodeSemantics`
- `flags_of_nativeAluOpcodeSemantics`
- `registerOperands_of_nativeAluOpcodeSemantics`
- `nonX0WriteFacts_of_nativeAluOpcodeSemantics`
- `activeWrite_of_nativeAluOpcodeSemantics`
- `passiveWrite_of_nativeAluOpcodeSemantics`
- `authenticatedWriteback_of_activeNativeAluOpcodeSemantics`
- `routedWriteback_of_activeNativeAluOpcodeSemantics`
- `authenticatedRoutedWriteback_of_activeNativeAluOpcodeSemantics`
- `encodedAluOut_of_activeNativeAluOpcodeSemantics`
- `encodedAluResult_of_activeNativeAluOpcodeSemantics`
- `authenticatedEncodedAluOut_of_activeNativeAluOpcodeSemantics`
- `authenticatedEncodedAluResult_of_activeNativeAluOpcodeSemantics`
- one exact theorem-facing specialization for each native-ALU opcode listed above
- exact opcode-specialized authenticated writeback consequences for each
  native-ALU opcode with `writesArchitecturalRd = true`
- exact opcode-specialized passive no-write consequences for `FENCE` and
  `ECALL`
- `x0WritePreserved_of_nativeAluOpcodeSemantics`
- `ecall_terminates_of_nativeAluOpcodeSemantics`
- `sequenceCorrect_of_nativeAluOpcodeSemantics`
- `sequenceDeterministic_of_nativeAluOpcodeSemantics`

## Exact Opcode Contracts

For each native-ALU opcode, the theorem-facing decoded-row contract fixes:

- the row is not `jal`, `jalr`, branch, load, store, W-width arithmetic, multiply, div, or rem,
- `usesRs2`,
- `writesMemToRd = false`,
- and the exact `aluOp` chosen from `NativeAluAluOps`.

The exact opcode owner therefore distinguishes:

- binary versus immediate arithmetic/logic/comparison opcodes through `usesRs2`,
- `LUI` versus `AUIPC` versus `FENCE` versus `ECALL` through the exact `aluOp`,
- `ECALL` through a theorem-facing terminating-row consequence,
- and the kernel-spec `rd = x0` sink rule through the separate theorem
  `x0WritePreserved_of_nativeAluOpcodeSemantics`.

Above that decoded-row contract, the exact opcode owner also exposes:

- authenticated register operands from the concrete Twist instantiation,
- the non-`x0` active/passive architectural write split determined jointly by
  opcode and `rd`,
- and the authenticated writeback equality
  `wvReg = rdNext` when the opcode writes an architectural destination and
  `rd ≠ x0`.
- and, through the exact row-local routing boundary, the routed writeback
  equalities `rdNext = aluWritebackValue` and `wvReg = aluWritebackValue` under
  the same active non-`x0` conditions.
- and, through the explicit Stage-1/Stage-2 representation bridge, the encoded
  equalities
  `wordToLimbPair(executionRow.lane.aluOut) = aluWritebackValue`,
  `wordToLimbPair(executionRow.results.aluResult) = aluWritebackValue`,
  and hence
  `wvReg = wordToLimbPair(executionRow.lane.aluOut)` and
  `wvReg = wordToLimbPair(executionRow.results.aluResult)`
  under the same active non-`x0` conditions.

Those consequences must remain visible both generically and as exact opcode
specializations:

- `ADD`, `ADDI`, `SUB`, `AND`, `ANDI`, `OR`, `ORI`, `XOR`, `XORI`, `SLT`,
  `SLTI`, `SLTU`, `SLTIU`, `LUI`, and `AUIPC` each re-export exact active
  authenticated writeback under `rd ≠ x0`,
- `FENCE` and `ECALL` each re-export the exact passive non-`x0` no-write
  contract,
- and `ECALL` separately re-exports termination.

## Proof Obligations

- The opcode owner may not weaken native ALU closure back to a class-only or family-only statement.
- The exact decoded-row contract must remain theorem-visible at opcode granularity.
- `ECALL` termination must remain tied to the exact native opcode, not to a looser family predicate.
- The owner must preserve the kernel-spec convention that `rd = x0` forces
  `PreservesRd = 1`, `WritesAluToRd = 0`, and `WritesMemToRd = 0`.
- The owner must not infer authenticated writeback from opcode alone; it must
  use the Stage-1/Stage-2 register-write activation bridge exposed by
  `Rv64IMStepComposition`.
- The preserved committed-sequence correctness and determinism consequences must remain visible at the opcode owner.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - primitive virtual-instruction catalog
  - per-instruction virtual sequences
  - `ADD / ADDI / SUB`
  - `AND / ANDI / OR / ORI / XOR / XORI`
  - `SLT / SLTI / SLTU / SLTIU`
  - `LUI`
  - `AUIPC`
  - `FENCE`
  - `ECALL`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/NativeAluOpcodeSemantics.lean` | Exact native-ALU opcode owner |
| `Nightstream/Rv64IM/Execution/NativeAluOpcodeSemanticsInterface.lean` | Theorem-facing re-export surface |
