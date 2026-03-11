# Parameters

## Purpose

- **What it is**: Concrete Appendix B.2 parameter constants for the Goldilocks instantiation: decomposition base `b = 2`, depth `k = 14`, norm bound `B = b^k = 16384`, security parameter `κ = 18`, max instances `K_max = 61`, matrix count `T = 216`, and extension degree `extDegreeK = 2`.
- **Key property**: `B = b^k = 16384 < q/2` and `b < q/2` (required for balanced decomposition and norm bounds).
- **Protocol role**: `B` and `b` feed Π_DEC (Section 7.5) decomposition bounds. `κ` sets the security level for invertibility (Theorem 8). `K_max` and `T` constrain the folding scheme's global parameters (Definition 14). Challenge coefficients satisfy `cCoeffMin = -2 ≤ c_i ≤ 2 = cCoeffMax`.

## Target Formulas

- `b = 2`, `k = 14`, `B = b^k = 16384`
- `B < q/2` (= `16384 < 9223372034707292160`)
- `κ = 18`, `K_max = 61`, `T = 216`
- `extDegreeK = 2`
- `cCoeffMin = -2`, `cCoeffMax = 2`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Appendix B.2 (Goldilocks: `2^64 - 2^32 + 1`), lines 709-727: all concrete parameter values.
- Definition 1, Section 4, lines 275-282: norm bounds `b, B = b^k < q/2`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Parameters.lean` | Appendix B.2 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Decomposition | `b`, `k`, `B` | def | Definitional | `b = 2`, `k = 14`, `B = b^k` |
| Security | `kappa` | def | Definitional | `κ = 18` |
| Folding bounds | `Kmax`, `T` | def | Definitional | `K_max = 61`, `T = 216` |
| Extension | `extDegreeK` | def | Definitional | `extDegreeK = 2` |
| Challenge | `cCoeffMin`, `cCoeffMax` | def | Definitional | `-2` and `2` |
| Modulus link | `modulus_def` | theorem | Theorem-Target | `modulus = q` |
| Concrete equalities | `b_eq_2`, `k_eq_14`, `B_eq_16384`, etc. | theorem | Theorem-Target | Exact values |
| Positivity | `b_pos`, `k_pos`, `B_pos`, `kappa_pos`, etc. | theorem | Theorem-Target | `0 < x` for each |
| Bound | `b_lt_modulus_half` | theorem | Theorem-Target | `b < q/2` |
| Bound | `B_lt_modulus` | theorem | Theorem-Target | `B < q` |
| Sanity | `concreteParameters` | theorem | Theorem-Target | Conjunction of all sanity checks |

## Proof Obligations and Closure Plan

All obligations closed. All theorems proved by `native_decide` or `omega`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Field.lean`: uses `F` type.
- `SuperNeo/Dimensions.lean`: uses `eta`, `d` for sanity cross-checks.

Downstream consumers:
- `SuperNeo/Thm3Core.lean`: uses `d_eq_54`, `eta_eq_81` for dimensional preconditions.
- `SuperNeo/InvertibilityAxioms.lean`: uses `kappa`, `B`, `b` for Theorem 8 preconditions.
- `SuperNeo/SamplingSet.lean`: uses `B`, `T` for expansion-factor checks.
- `SuperNeo/Decomp.lean`: uses `b`, `k` for decomposition base and depth.

## Implementation Plan

No further work required; module is proof-complete.

## Quality Expectations

Every constant must match the exact value from Appendix B.2 Table. Positivity and bound theorems must be directly usable without `native_decide` at call sites.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `concreteParameters` proves all sanity checks in one theorem.

## Out of Scope

- Mersenne-61 and Almost-Goldilocks parameter regimes.
- Derived bounds (e.g., `d · B < q`) — those belong in consuming modules.
