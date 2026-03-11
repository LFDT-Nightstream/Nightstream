# Regression.spec.md

## Purpose

- **What it is**: Golden-vector parity entrypoint that imports pre-computed Goldilocks field vectors for regression testing.
- **Key property**: Module compiles iff golden vectors remain consistent with the current field implementation.
- **Protocol role**: Serves as a compile-time regression gate ensuring Goldilocks parameter stability across refactors.

## Target Formulas

- Compilation success → golden vectors type-check against `GoldilocksGolden` = regression parity holds.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Appendix B.2 (Goldilocks parameters), lines 709-727: golden vectors validate Goldilocks field constants.

## Module Mapping

No exported declarations — this module is a parity entrypoint.

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| (none) | — | No exported declarations — this module is a parity entrypoint. | Definitional |

## Proof Obligations and Closure Plan

- No proof obligations. Module exists solely as a compilation-gated regression check.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.Golden.GoldilocksGolden` (imports golden vector definitions).
- **Consumers**:
  - `Main.lean`: depends on `Regression` transitively for full-project compilation gate.

## Implementation Plan

- No implementation work needed. File is stable.

## Quality Expectations

- File compiles without error.
- Spec documents the regression role clearly.

## Acceptance Criteria

- `lake build` succeeds.
- No exported symbols to curate.

## Out of Scope

- Runtime regression tests (this is compile-time only).
