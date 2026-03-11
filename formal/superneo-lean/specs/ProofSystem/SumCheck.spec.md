# SumCheck — Barrel Re-export

## Purpose

- **What it is**: Barrel module that re-exports `SuperNeo.ProofSystem.SumCheck.General` and `SuperNeo.ProofSystem.SumCheck.PrefixSoundnessEndpoint` (Instance, Transcript, Accepted, ClaimTrue, SoundnessErrorBoundary, `TheoremPackage`, constructive package constructors, Lund/Schwartz–Zippel bound surface, soundness/completeness surfaces, extraction theorems, and the faithful prefix-dependent aligned positive-round endpoint).
- **Key property**: Importing `SuperNeo.ProofSystem.SumCheck` provides the structured SumCheck API without importing the submodule directly.
- **Protocol role**: Facade for proof-system consumers that need sum-check types and theorem package (Section 7.3 Π_CCS, Section 7.4 Π_RLC).

## Target Formulas

- Re-export only; no new formulas. All formulas are those of `General.spec.md` and `SingleRound.spec.md`.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 6 (The sum-check protocol), lines 352–355.
- Section 7.3 (Π_CCS), lines 481–548; Section 7.4 (Π_RLC), lines 549–583.

## Module Mapping

| Lean module | Paper section |
|-------------|---------------|
| `SuperNeo.ProofSystem.SumCheck` | Barrel; re-exports General plus the prefix-dependent aligned positive-round endpoint (Definition 6, Sections 7.3–7.4) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Barrel | re-exports from General and PrefixSoundnessEndpoint | SumCheck API (`TheoremPackage`, `soundness`, `completeness`, `lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix`) | Definitional |

## Proof Obligations and Closure Plan

None (barrel file).

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.ProofSystem.SumCheck.General`, `SuperNeo.ProofSystem.SumCheck.PrefixSoundnessEndpoint`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`, `SuperNeo.FoldingProtocol`: use sum-check for folding reductions.
  - `SuperNeo.ProofSystem.SumCheckInterface`: imports this module for interface boundary.

## Implementation Plan

Keep barrel minimal; no new definitions.

## Quality Expectations

Barrel stays thin; spec documents re-export scope.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.

## Out of Scope

- New definitions or theorems; barrel is aggregation-only.
