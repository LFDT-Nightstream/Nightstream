# Rv64IMMultiplyOpcodeSemantics Spec

## Purpose

- **What it is**: The exact theorem-facing opcode owner for the RV64IM multiply family.
- **What it is not**: It is not the class-level multiply semantic owner, not the Stage-1 multiplication-slot owner, and not the trace/kernel lift owner.
- **Protocol role**: It sits above `Rv64IMMultiplyLoweringSemantics` and closes the exact decoded-row opcode gap for `MUL`, `MULH`, `MULHU`, `MULHSU`, and `MULW`.

## Central Objects

- `MultiplyOpcode` enumerates the architectural multiply opcodes.
- `MultiplyAluOps(AluOp)` fixes the theorem-facing ALU-operation tags used by the decoded-row contract.
- `MultiplyOpcodeBound(ops, row, opcode)` is the exact decoded-row contract for that opcode.

## Required Theorem Surface

The module must expose:

- `opcodeBound_of_multiplyOpcodeSemantics`
- `flags_of_multiplyOpcodeSemantics`
- `registerOperands_of_multiplyOpcodeSemantics`
- `activeWrite_of_multiplyOpcodeSemantics`
- `authenticatedWriteback_of_activeMultiplyOpcodeSemantics`
- one exact theorem-facing specialization for each multiply opcode listed above
- exact opcode-specialized authenticated writeback consequences for `MUL`,
  `MULH`, `MULHU`, `MULHSU`, and `MULW`
- `routedWriteback_of_activeMultiplyOpcodeSemantics`
- `authenticatedRoutedWriteback_of_activeMultiplyOpcodeSemantics`
- `encodedAluOut_of_activeMultiplyOpcodeSemantics`
- `encodedAluResult_of_activeMultiplyOpcodeSemantics`
- `authenticatedEncodedAluOut_of_activeMultiplyOpcodeSemantics`
- `authenticatedEncodedAluResult_of_activeMultiplyOpcodeSemantics`
- `x0WritePreserved_of_multiplyOpcodeSemantics`
- `sequenceCorrect_of_multiplyOpcodeSemantics`
- `sequenceDeterministic_of_multiplyOpcodeSemantics`

## Exact Opcode Contracts

For each multiply opcode, the theorem-facing decoded-row contract fixes:

- the row is not control-flow, memory, or div/rem,
- `isMul = true`,
- `usesRs2 = true`,
- `writesMemToRd = false`,
- `isWOp`,
- and the exact `aluOp` chosen from `MultiplyAluOps`.

The exact opcode owner therefore distinguishes:

- low-half `MUL`,
- high-half `MULH`, `MULHU`, `MULHSU`,
- and `MULW` through `isWOp = true` while sharing the low-multiply ALU tag.

`rd = x0` is handled by the separate theorem
`x0WritePreserved_of_multiplyOpcodeSemantics`, because the kernel spec makes
write suppression depend on `rd`, not only on opcode.

Above that decoded-row contract, the exact opcode owner also exposes:

- authenticated register operands from the concrete Twist instantiation,
- the non-`x0` active architectural write consequence,
- the authenticated writeback equality
  `wvReg = rdNext` when `rd ≠ x0`,
- the exact routed writeback equalities `rdNext = aluWritebackValue` and
  `wvReg = aluWritebackValue` when `rd ≠ x0`,
- the explicit Stage-1/Stage-2 representation equalities
  `wordToLimbPair(executionRow.lane.aluOut) = aluWritebackValue`,
  `wordToLimbPair(executionRow.results.aluResult) = aluWritebackValue`,
  and therefore
  `wvReg = wordToLimbPair(executionRow.lane.aluOut)` and
  `wvReg = wordToLimbPair(executionRow.results.aluResult)` when `rd ≠ x0`,
- and those authenticated writeback consequences specialized at each exact
  multiply opcode.

## Proof Obligations

- The opcode owner may not weaken multiply closure back to a class-only or family-only statement.
- The exact decoded-row contract must remain theorem-visible at opcode granularity.
- `MULW` must remain theorem-visible as the W-width multiply opcode rather than being collapsed into the generic low-half multiply case.
- The owner must preserve the kernel-spec convention that `rd = x0` forces
  `PreservesRd = 1`, `WritesAluToRd = 0`, and `WritesMemToRd = 0`.
- The owner must not infer authenticated writeback from opcode alone; it must
  use the Stage-1/Stage-2 register-write activation bridge exposed by
  `Rv64IMStepComposition`.
- The preserved committed-sequence correctness and determinism consequences must remain visible at the opcode owner.

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
  - `./external/jolt/tracer/src/instruction/mulhu.rs`
  - `./external/jolt/tracer/src/instruction/mulhsu.rs`
  - `./external/jolt/tracer/src/instruction/mulw.rs`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Execution/MultiplyOpcodeSemantics.lean` | Exact multiply opcode owner |
| `Nightstream/Rv64IM/Execution/MultiplyOpcodeSemanticsInterface.lean` | Theorem-facing re-export surface |
