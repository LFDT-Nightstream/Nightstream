# Rv64IMStage3Refinement Spec

## Purpose

- **What it is**: The theorem-facing owner that packages Stage-3 continuity together with the final halted-execution claim.
- **What it is not**: It is not the Stage-1 or Stage-2 semantic owner.
- **Protocol role**: It refines Stage-3 evidence into the two generic semantic surfaces consumed downstream: `PcAdjacentBridgeProofPackage` and `FinalBoundaryClaimProofPackage`.

## Target Formulas

Define:

$$
\mathrm{Stage3RefinementPackage}
$$

carrying:

- one `Stage3ProofPackage`,
- one `FinalBoundaryClaimProofPackage`.

The exported refinement targets are:

$$
\mathrm{pcAdjacentBridgeProofPackageOfStage3Refinement}(pkg),
$$

and

$$
\mathrm{fullHaltedExecutionClaimOfStage3Refinement}(pkg).
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - Stage-3 semantic PC bridge
  - full halted execution claim

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage3/Stage3Refinement.lean` | Stage-3 refinement owner |
| `Nightstream/Rv64IM/Stage3/Stage3RefinementInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Package | `Stage3RefinementPackage` | structure | Definitional | Packages the Stage-3 continuity owner together with the final-boundary owner |
| Constructor | `pcAdjacentBridgeProofPackage_of_stage3Refinement` | def | Definitional | Exports the generic PC-adjacent bridge package |
| Constructor | `fullHaltedExecutionClaim_of_stage3Refinement` | def | Definitional | Exports the full halted-execution claim |

## Out of Scope

- Stage-1 decode or execution ownership
- Stage-2 register or RAM history ownership
