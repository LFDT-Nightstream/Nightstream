# Dimensions

## Purpose

- **What it is**: Concrete cyclotomic dimension constants `η = 81` and ring degree `d = 54` for the Goldilocks instantiation, plus the field-to-ring dimension map `nF(nR) = d · nR`.
- **Key property**: `nF(nR) = 54 · nR` and `nF` is monotone: `a ≤ b → nF a ≤ nF b`.
- **Protocol role**: Every module that converts between ring-vector length `nR` and field-vector length `nF` uses `nF_def`. Embedding (Definition 7) and bar-lift (Definition 8) rely on `d = 54` for block sizes. Protocol relations (Section 7.1) use dimension helpers for CCS shape constraints.
- **Scope**: Goldilocks cyclotomic `Φ_{81}(X)` of degree 54 only.

## Target Formulas

- `η = 81`, `d = 54`
- `nF(nR) = d · nR`
- `nFIn(nRIn) = d · nRIn`
- `a ≤ b → nF a ≤ nF b` (monotonicity)
- `0 < nR → 0 < nF nR` (positivity transfer)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 1 (Fields, Rings, and Dimensions), Section 4, lines 275-282: `d` = degree of cyclotomic `Φ`, `nF = d · nR`.
- Appendix B.2 (Goldilocks parameters), lines 709-727: concrete `d = 54`, `η = 81`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Dimensions.lean` | Definition 1 (dimensions) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Constants | `eta`, `d` | def | Definitional | `η = 81`, `d = 54` |
| Dimension maps | `nF`, `nFIn` | def | Definitional | `nF nR = d * nR` |
| Concrete values | `eta_eq_81`, `d_eq_54` | theorem | Theorem-Target | Exact equalities |
| Positivity | `eta_pos`, `d_pos` | theorem | Theorem-Target | `0 < η`, `0 < d` |
| Positivity transfer | `nF_pos_of_pos` | theorem | Theorem-Target | `0 < nR → 0 < nF nR` |
| Algebra | `nF_add`, `nF_mul` | theorem | Theorem-Target | `nF(a+b) = nF a + nF b`, `nF(a*b) = nF a * b` |
| Monotonicity | `nF_mono`, `nFIn_mono` | theorem | Theorem-Target | `a ≤ b → nF a ≤ nF b` |

## Proof Obligations and Closure Plan

All obligations closed.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

Upstream dependencies: none.

Downstream consumers:
- `SuperNeo/Ring.lean`: uses `d` for `D := d` and `hasRingDegreeShape`.
- `SuperNeo/Embedding.lean`: uses `d` for block size in `embedElem`/`unembedElem`.
- `SuperNeo/BarLift.lean`: uses `d` for chunk block sizes.
- `SuperNeo/Parameters.lean`: uses `eta`, `d` for sanity checks.

## Implementation Plan

No further work required; module is proof-complete.

## Quality Expectations

Dimension theorems must compose algebraically (e.g., `nF_add` enables `nF (a + b)` rewriting without unfolding).

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `nF_add` and `nF_mul` are used by downstream modules without ad-hoc `simp` lemmas.

## Out of Scope

- Other cyclotomic choices (Mersenne-61, Almost-Goldilocks).
- Runtime dimension parametricity (constants are fixed at `η = 81`, `d = 54`).
