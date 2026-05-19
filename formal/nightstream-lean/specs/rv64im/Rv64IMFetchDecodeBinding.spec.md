# Rv64IMFetchDecodeBinding Spec

## Purpose

- **What it is**: The theorem-facing Stage-1 decode and handoff contract for
  one authenticated expanded-bytecode row.
- **What it is not**: It is not the bytecode read-check itself and it does not
  own ALU, branch, register, or RAM authentication.
- **Protocol role**: It fixes the exact decoded row object that Stage 1 exports
  into the main lane and `C_decode_handoff`, including non-final-row write
  hygiene and x0-preservation.

## Target Formulas

Define the authenticated decode/handoff object:

$$
\mathrm{DecodeHandoff}
:=
(
\mathrm{usesRs2},
\mathrm{isLoad},
\mathrm{isStore},
\mathrm{memWidth},
\mathrm{memUnsigned},
\mathrm{isFirstInSequence},
\mathrm{isLastInSequence}
).
$$

Define the theorem-facing decoded row object `DecodedStage1Row` carrying at
least:

- `valid`,
- `instructionWordArch`,
- `unexpandedPc`,
- `virtualOpcode`,
- sequence-boundary flags,
- register selectors `rd`, `rs1`, `rs2`,
- decoded immediate `imm`,
- lane-write and control-flow flags,
- memory flags and width metadata,
- execution tags such as `aluOp`, `branchOp`, `divremKind`, `isWOp`,
  `isMul`, `isDiv`, and `isRem`.

Define the exact handoff projection:

$$
\mathrm{toDecodeHandoff}(row)
$$

by copying the Stage-2-consumed fields from the authenticated decoded row.

Define:

$$
\mathrm{DecodeHandoffBound}(row, handoff)
\iff
handoff = \mathrm{toDecodeHandoff}(row).
$$

Define the architectural x0 preservation rule:

$$
\mathrm{X0WritePreserved}(x0, row)
$$

meaning:

$$
row.rd = x0
\Longrightarrow
row.preservesRd = 1
\land
row.writesAluToRd = 0
\land
row.writesMemToRd = 0.
$$

Define the non-final-row write hygiene rule:

$$
\mathrm{NonFinalRdTargetBound}(\mathrm{isArchitectural}, row)
$$

meaning:

$$
row.isLastInSequence = 0
\Longrightarrow
(row.writesAluToRd = 1 \lor row.writesMemToRd = 1)
\Longrightarrow
\neg \mathrm{isArchitectural}(row.rd).
$$

Define the full Stage-1 fetch/decode boundary:

$$
\mathrm{FetchDecodeBound}
(
\mathrm{bytecodeTable},
\mathrm{expandedPc},
x0,
\mathrm{isArchitectural},
row
)
$$

meaning:

- `bytecodeTable(expandedPc) = some row`,
- `row.valid = 1`,
- `DecodeHandoffBound(row, toDecodeHandoff(row))`,
- `X0WritePreserved(x0, row)`,
- `NonFinalRdTargetBound(isArchitectural, row)`.

The decoded row also exports:

$$
\mathrm{advanceArchPc}(row) := row.isLastInSequence,
$$

which is the Stage-1 contribution to the root-lane `AdvanceArchPc` control.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - expanded bytecode row channel
  - decode-handoff surface
  - committed sequence-boundary flags
  - x0 preservation
  - non-final-row virtual-register-only write rule

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage1/FetchDecodeBinding.lean` | Exact Stage-1 decoded-row and handoff theorem surface |
| `Nightstream/Rv64IM/Stage1/FetchDecodeBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Handoff | `DecodeHandoff` | structure | Definitional | Fixes the exact Stage-2 handoff tuple exported by Stage 1 |
| Decoded row | `DecodedStage1Row` | structure | Definitional | Carries the exact authenticated expanded-bytecode row metadata used by Stage 1 |
| Projection | `DecodedStage1Row.toDecodeHandoff` | def | Definitional | Projects the Stage-2 handoff fields from the decoded row |
| Projection | `DecodedStage1Row.advanceArchPc` | def | Definitional | Fixes `AdvanceArchPc` to the authenticated sequence-final flag |
| Boundary | `DecodeHandoffBound` | def | Definitional | Handoff commitment equals the decoded-row projection exactly |
| Boundary | `X0WritePreserved` | def | Definitional | x0-targeted rows cannot actually write x0 |
| Boundary | `NonFinalRdTargetBound` | def | Definitional | Non-final writing rows cannot target architectural registers |
| Boundary | `FetchDecodeBound` | def | Definitional | Fetch plus decode plus handoff plus write-hygiene package for one authenticated row |
| Theorem | `decodeHandoffBound_refl` | theorem | Theorem-Target | The canonical row-projected handoff satisfies the handoff boundary |
| Theorem | `advanceArchPc_eq_isLastInSequence` | theorem | Theorem-Target | Stage-1 `AdvanceArchPc` is exactly the authenticated `isLastInSequence` flag |
| Theorem | `fetchDecodeBound_bytecodeRow` | theorem | Theorem-Target | The Stage-1 boundary contains one concrete bytecode row at the fetched expanded address |
| Theorem | `fetchDecodeBound_valid` | theorem | Theorem-Target | Every accepted fetched row is active |
| Theorem | `fetchDecodeBound_handoff` | theorem | Theorem-Target | The committed handoff tuple is the decoded-row projection |
| Theorem | `fetchDecodeBound_x0Preserved` | theorem | Theorem-Target | x0 preservation is part of the accepted Stage-1 boundary |
| Theorem | `fetchDecodeBound_nonFinalRdTarget` | theorem | Theorem-Target | Non-final-row virtual-register-only write hygiene is part of the accepted Stage-1 boundary |

## Proof Obligations

- The Stage-1 accepted row is the committed expanded-bytecode row, not a
  prover-chosen internal decomposition.
- The handoff tuple exported to Stage 2 is determined exactly by that
  authenticated row.
- x0-preservation and non-final-row architectural-write exclusion are part of
  the accepted decode boundary, not implementation intuition.
- `AdvanceArchPc` is derived from authenticated sequence-boundary metadata.

## Out of Scope

- bytecode Shout read-checking
- ALU / branch slot authentication
- register / RAM history
- final ISA-equivalence of the enclosing committed sequence
