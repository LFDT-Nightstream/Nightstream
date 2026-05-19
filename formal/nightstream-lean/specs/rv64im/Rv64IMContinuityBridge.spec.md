# Rv64IMContinuityBridge Spec

## Purpose

- **What it is**: The theorem-facing Stage-3 owner for RV64IM active-prefix PC continuity.
- **What it is not**: It is not Stage-2 state closure and it does not define final-boundary semantics by itself.
- **Protocol role**: It fixes the active-prefix continuity rows, the pair mask, and the refinement from row-local continuity evidence to the generic `PcAdjacentBridge`.

## Target Formulas

Define the exact active-prefix pair mask:

$$
\mathrm{PairMaskN}(N, j) := [j + 1 < N].
$$

Define one row-local continuity witness:

$$
\mathrm{ContinuityRow}
:=
(\mathrm{rowIndex}, \mathrm{pairMask}, \mathrm{pcNext}, \mathrm{shiftedPc}).
$$

Define:

$$
\mathrm{ContinuityRowBound}(N, row)
$$

meaning:

- `pairMask = PairMaskN(N, rowIndex)`,
- if `pairMask = 1`, then `shiftedPc = pcNext`.

Define:

$$
\mathrm{ContinuityRowsBound}(rows, postPc, prePc, N)
$$

meaning every active adjacent pair `j < N - 1` is witnessed by one row in
`rows` whose `pcNext = postPc(j)` and `shiftedPc = prePc(j+1)` and which
satisfies `ContinuityRowBound`.

The key refinement target is:

$$
\mathrm{ContinuityRowsBound}(rows, postPc, prePc, N)
\Longrightarrow
\mathrm{PcAdjacentBridge}(postPc, prePc, N).
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - Continuity support relation
  - ContinuityCheck
  - Stage-3 semantic PC bridge

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage3/ContinuityBridge.lean` | Stage-3 continuity owner |
| `Nightstream/Rv64IM/Stage3/ContinuityBridgeInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Mask | `PairMaskN` | def | Definitional | Fixes the exact active-prefix predecessor mask |
| Row | `ContinuityRow` | structure | Definitional | Packages one row-local Stage-3 continuity witness |
| Boundary | `ContinuityRowBound` | def | Definitional | One continuity row matches the active-prefix mask and PC equality rule |
| Boundary | `ContinuityRowsBound` | def | Definitional | The row set covers every active adjacent pair |
| Bridge | `RowProjectionBinding` | structure | Definitional | Packages the row-export binding carried alongside continuity |
| Package | `Stage3ProofPackage` | structure | Definitional | Packages continuity rows, row-export bindings, and the continuity theorem input |
| Theorem | `pcAdjacentBridge_of_continuityRowsBound` | theorem | Theorem-Target | Row-local continuity evidence implies the generic PC-adjacent bridge |
| Constructor | `pcAdjacentBridgeProofPackage_of_stage3` | def | Definitional | Builds the generic PC-adjacent bridge package from Stage-3 continuity evidence |

## Out of Scope

- register / RAM temporal closure
- final halted-execution claim
- opening provenance chain
