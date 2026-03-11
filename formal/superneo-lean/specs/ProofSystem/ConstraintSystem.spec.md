# ConstraintSystem — Barrel Re-export

## Purpose

- **What it is**: Barrel module that re-exports `SuperNeo.ProofSystem.ConstraintSystem.CCS` (Definition 11 structures, Definition 12/13 relation objects, relaxed CE carrier, and Definition 14 parameter constructors).
- **Key property**: Importing `SuperNeo.ProofSystem.ConstraintSystem` provides the Section 7.1 proof-system relation layer, including coherent theorem-instance packages (`Section71Objects`, `Section71Instance`), without importing the submodule directly.
- **Protocol role**: Facade for proof-system consumers that need CCS/CE relation objects (Section 7.1).

## Target Formulas

- Re-export only; no new formulas. All formulas are those of `CCS.spec.md`.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449–455.
- Definition 12 (Norm-bounded CCS), lines 457–459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461–465.
- Definition 14 (Global Reduction Parameters), lines 467–475.

## Module Mapping

| Lean module | Paper section |
|-------------|---------------|
| `SuperNeo.ProofSystem.ConstraintSystem` | Barrel; re-exports CCS (Definitions 12–13) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Barrel | re-exports from CCS | `CommitmentMap`, `InputProjector`, `CCSStructure`, `GlobalParams`, `Section71Objects`, `Section71Instance`, `CCS`, `CE`, `CERelaxed`, `CERelaxed.ofCE` | Definitional |

## Proof Obligations and Closure Plan

- Re-export only; no additional formulas or proof obligations beyond `CCS.spec.md`.

## Assumption Ledger

This barrel introduces no additional boundary assumptions.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.ProofSystem.ConstraintSystem.CCS`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`, `SuperNeo.FoldingProtocol`: use CCS types for folding reductions.
  - `SuperNeo.ProofSystem.ConstraintSystemInterface`: imports this module for interface boundary.

## Implementation Plan

Keep barrel minimal; no new definitions.

## Quality Expectations

Barrel stays a single import line; spec documents re-export scope.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.

## Out of Scope

- New definitions or theorems; barrel is aggregation-only.
