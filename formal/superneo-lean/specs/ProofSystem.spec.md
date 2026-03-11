# ProofSystem.spec.md

## Purpose

- **What it is**: Top-level barrel re-export that aggregates all `ProofSystem.*` sub-modules into a single import.
- **Key property**: `import SuperNeo.ProofSystem` transitively provides Types, Negligible, Security, Lattice, LatticeReductions, ConstraintSystem, SumCheck, Folding, and Protocol.
- **Protocol role**: Convenience namespace for downstream modules that need the full proof-system layer (e.g., `ProtocolTheorem`).

## Target Formulas

- `import SuperNeo.ProofSystem` = `import SuperNeo.ProofSystem.Protocol` (transitive re-export).

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 6 (Security Model), lines 404-447: interactive reduction definitions.
- Section 7 (Folding Protocol), lines 449-596: protocol composition context.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| (barrel) | re-exports from sub-modules | Definitional |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Barrel | `barrelContract` | Importing `SuperNeo.ProofSystem` exposes the curated proof-system stack | Definitional |

## Proof Obligations

- No local proof obligations. All obligations live in sub-modules.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.ProofSystem.Protocol` (transitive: all sub-modules).
- **Consumers**:
  - `SuperNeo.ProtocolTheorem`: imports `ProofSystem` for Lattice, Security, SumCheck surfaces.
  - `SuperNeo.ProofSystem.Protocol`: uses `ProofSystem` sub-module surfaces.

## Quality Expectations

- Barrel file stays minimal (under 30 lines).
- No logic duplication — all substance lives in sub-modules.

## Acceptance Criteria

- `lake build` succeeds.
- Interface stays thin.

## Out of Scope

- Any proof logic — this is purely a re-export barrel.
