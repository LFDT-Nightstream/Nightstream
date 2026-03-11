# Folding — Barrel Re-export

## Purpose

- **What it is**: Barrel module that re-exports `SuperNeo.ProofSystem.Folding.PiCCS`, `PiRLC`, and `PiDEC` (the three folding reduction wrappers).
- **Key property**: Importing `SuperNeo.ProofSystem.Folding` provides the compatibility assumption wrappers together with the paper-facing realized/provider entrypoints, the generic proof-system `Section71Instance` plus compact-specialization entrypoints, the packaged theorem-native `ProtocolSection71Setup` entrypoints, the fully paper-faithful `ProtocolSection71TheoremInstance` entrypoints, the single-object `ProtocolSection71Context` entrypoints, the explicit protocol-side Definition-14 `ProtocolSection71Data` entrypoints, and the explicit protocol-side Section 7.5 `ProtocolTargetData` entrypoints for Π_CCS, Π_RLC, and Π_DEC without importing submodules directly.
- **Protocol role**: Facade for proof-system consumers that need the three folding reductions (Section 7.2–7.5).

## Target Formulas

- Re-export only; no new formulas. All formulas are those of `PiCCS.spec.md`, `PiRLC.spec.md`, and `PiDEC.spec.md`.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.2 (Folding scheme for CCS via interactive reductions), lines 468–480.
- Section 7.3 (Π_CCS), lines 481–548; Section 7.4 (Π_RLC), lines 549–583; Section 7.5 (Π_DEC), lines 585–593.

## Module Mapping

| Lean module | Paper section |
|-------------|---------------|
| `SuperNeo.ProofSystem.Folding` | Barrel; re-exports PiCCS (Lemma 3), PiRLC (Lemma 4), PiDEC (Theorem 7) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Barrel | re-exports from PiCCS, PiRLC, PiDEC | Folding reduction wrappers | Definitional |

## Proof Obligations and Closure Plan

None (barrel file).

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.ProofSystem.Folding.PiCCS`, `PiRLC`, `PiDEC`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Protocol`, `SuperNeo.FoldingProtocol`: use folding reductions for composition.
  - `SuperNeo.ProofSystem.FoldingInterface`: imports this module for interface boundary.

## Implementation Plan

Keep barrel minimal; no new definitions.

## Quality Expectations

Barrel stays thin; spec documents re-export scope.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.

## Out of Scope

- New definitions or theorems; barrel is aggregation-only.
