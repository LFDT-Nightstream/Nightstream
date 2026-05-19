# Rv64IMTwistTemporalInstantiation Spec

## Purpose

- **What it is**: The theorem-facing owner that turns the concrete Stage-2 history bundles into the generic adjacent-state closure surface.
- **What it is not**: It is not the PC bridge and it does not define final-boundary semantics.
- **Protocol role**: It packages register-history and RAM-history owners into the generic `Stage2TemporalContext` used by strong kernel soundness.

## Target Formulas

Define the concrete Stage-2 history bundle package:

$$
\mathrm{Stage2HistoryBundles}
$$

carrying one register-history bundle and one RAM-history bundle.

Define the context constructor:

$$
\mathrm{stage2TemporalContextOfHistoryBundles}(history, rowLinks)
$$

which packages:

- the register timeline from the register-history bundle,
- the RAM timeline from the RAM-history bundle,
- the shared row-link evidence.

Define the package constructor:

$$
\mathrm{stage2TemporalClosureProofPackageOfHistoryBundles}
$$

which lifts the concrete history bundles plus an `AdjacentStateClosed` proof
into the generic `Stage2TemporalClosureProofPackage`.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - Stage-2 temporal closure object
  - adjacent-state linking theorem obligation

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage2/TwistTemporalInstantiation.lean` | Stage-2 temporal-instantiation owner |
| `Nightstream/Rv64IM/Stage2/TwistTemporalInstantiationInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Bundles | `Stage2HistoryBundles` | structure | Definitional | Packages the register and RAM Stage-2 history owners together |
| Context | `stage2TemporalContextOfHistoryBundles` | def | Definitional | Builds the generic Stage-2 temporal context from the concrete history owners |
| Package | `stage2TemporalClosureProofPackage_of_historyBundles` | def | Definitional | Lifts concrete Stage-2 history owners into the generic adjacent-state closure package |
| Theorem | `adjacentStateClosed_of_stage2TemporalClosureProofPackage` | theorem | Theorem-Target | Recovers the exact adjacent-state closure obligation from the package |

## Out of Scope

- PC adjacency
- final halted-execution claim
