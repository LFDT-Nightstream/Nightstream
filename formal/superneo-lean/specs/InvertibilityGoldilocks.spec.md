# InvertibilityGoldilocks

## Purpose

- **What it is**: The concrete Goldilocks instantiation of Theorem 8 for the SuperNeo ring.
- **Key property**: The paper floor `goldilocksPaperBInv = 383` satisfies the standalone low-norm invertibility theorem in `R_q`, and the narrower active `paperCarrier`-difference boundary follows as a corollary.
- **Protocol role**: Supplies the constructive Goldilocks invertibility theorem consumed by the active protocol route and exposes the stronger standalone low-norm theorem surface at the paper floor.

## Target Formulas

- `lowNormInvertibilityAssumption goldilocksPaperBInv`
- `lowNormInvertibilityAssumption 5`
- `paperCarrierDiffInvertibilityAssumption`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Theorem 8 (Low-norm invertibility), Section 5/6, lines 375-378.
- Appendix B.2 concrete floor `b_inv = 383`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/InvertibilityGoldilocks.lean` | Theorem 8 + Appendix B.2 concrete Goldilocks specialization |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|---|---|---|---|
| Theorems | `lowNormInvertibilityAssumption_paperBInv_goldilocks` | Concrete Goldilocks proof of the standalone Theorem-8 boundary at `B = goldilocksPaperBInv` | Theorem-Target |
| Theorems | `lowNormInvertibilityAssumption_five_goldilocks` | Corollary specialization at the narrower threshold `B = 5` | Theorem-Target |
| Theorems | `paperCarrierDiffInvertibilityAssumption_goldilocks` | Concrete Goldilocks proof of active nonzero `paperCarrier`-difference invertibility | Theorem-Target |

## Assumption Ledger

- No extra theorem-facing boundary is introduced in this module beyond the ambient Goldilocks parameters already fixed in `Parameters.lean` and `InvertibilityAxioms.lean`.

## Dependency and Consumer Map

- **Dependencies**:
  - `SuperNeo/InvertibilityAxioms.lean`
  - `SuperNeo/PolynomialBridge.lean`
  - `SuperNeo/Parameters.lean`
- **Consumers**:
  - `SuperNeo/ProtocolTarget.lean`
  - `SuperNeo/ProtocolTheorem.lean`

## Quality Expectations

- The constructive proof must discharge the actual paper floor `goldilocksPaperBInv = 383`, not just the narrower protocol threshold `5`.
- The active `paperCarrier`-difference theorem must be a theorem corollary of the standalone low-norm result, not a separate ad hoc boundary.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.

## Out of Scope

- Carrier-parametric or field-parametric invertibility libraries beyond the concrete Goldilocks theorem used by SuperNeo.
