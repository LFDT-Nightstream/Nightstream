# Norm

## Purpose

- **What it is**: The infinity norm on field elements (`normInfF : F → Nat` via `centeredAbs`) and coefficient arrays (`normInfCoeffs : Coeffs → Nat` via element-wise max), plus norm-bound assumption bundles for vector-add, vector-scale, ring-mul, and subtraction.
- **Key property**: `normInfF(0) = 0` and `normInfCoeffs(#[]) = 0`; assumption bundles state that operation outputs satisfy target norm bounds given input norm bounds.
- **Protocol role**: Norm bounds are the core mechanism for SuperNeo's security reductions. Every folding step's witness growth is tracked via `normInfCoeffs`. The `AllChallengeCoeffs` predicate enforces `‖c_i‖_∞ ≤ 2` for challenge coefficients (Definition 14, Section 7.3). The `vecAddNormBoundFromOperands` and `mulRqNormBoundFromOperands` bundles feed Theorem 8 (MSIS hardness) through accumulated norm tracking.

## Target Formulas

- `‖a‖_∞ = |centeredRep(a)|` for `a : F`
- `‖v‖_∞ = max_i ‖v[i]‖_∞` for `v : Coeffs`
- `AllChallengeCoeffs(a) ↔ ∀ i, ‖a[i]‖_∞ ≤ 2`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 3 (Norm), Section 4, lines 290-291: `‖·‖_∞` on ring elements via coefficients.
- Theorem 8 (MSIS reduction), Section 6, lines 375-378: norm bounds enter security parameter.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Norm.lean` | Definition 3 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Norms | `normInfF` | def | Definitional | `centeredAbs` of field element |
| Norms | `normInfCoeffs` | def | Definitional | Max of element norms |
| Alias | `maxRhoNorm` | def | Definitional | `= normInfCoeffs` |
| Zeroes | `normInfF_zero` | theorem | Theorem-Target | `normInfF 0 = 0` |
| Zeroes | `normInfCoeffs_empty` | theorem | Theorem-Target | `normInfCoeffs #[] = 0` |
| Non-negativity | `normInfCoeffs_nonneg` | theorem | Theorem-Target | `0 ≤ normInfCoeffs a` |
| Bound bundles | `vecAddNormBoundFromOperands` | def | Definitional | Addition norm bound shape |
| Bound bundles | `vecScaleNormBoundFromOperands` | def | Definitional | Scaling norm bound shape |
| Bound bundles | `mulRqNormBoundFromOperands` | def | Definitional | Multiplication norm bound shape |
| Bound bundles | `coeffSubNormBoundFromOperands` | def | Definitional | Subtraction norm bound shape |
| Challenge | `AllChallengeCoeffs` | def | Definitional | `∀ i, normInfF a[i] ≤ 2` |
| Challenge | `allChallengeCoeffs_empty` | theorem | Theorem-Target | Vacuously true for `#[]` |
| Challenge | `allChallengeCoeffs_mono` | theorem | Theorem-Target | Monotonicity under larger challenge bounds |
| Challenge | `allChallengeCoeffs_of_bound`, `allChallengeCoeffs_weaken` | theorem | Theorem-Target | Introduction/elimination helpers for challenge bounds |
| Bundle constructors | `*_of_global` family | theorem | Theorem-Target | Lift global operation bounds into bundle shapes |

## Proof Obligations and Closure Plan

All module-scope obligations closed. Bundle definitions remain theorem contracts; constructor theorems now provide direct lifting from globally-proved operation bounds.

## Assumption Ledger

No open boundary assumptions in this module. The four `*NormBoundFromOperands` definitions are assumption *shapes* (parameterized propositions), not axioms — they define what it means for a bound to hold. Instantiating them with concrete values is a downstream obligation.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Ring.lean`: `Coeffs`, `vecAdd`, `vecScale`, `mulRq` for bound statements.

Downstream consumers:
- `SuperNeo/Decomp.lean`: uses norm bounds for digit-decomposition analysis.
- `SuperNeo/InvertibilityAxioms.lean`: uses `normInfCoeffs` bounds for Theorem 8 preconditions.
- `SuperNeo/ProtocolRelations.lean`: uses norm predicates for folding-step witness constraints.
- `SuperNeo/ArithmeticObligations.lean`: uses bound bundles for arithmetic obligation statements.

## Implementation Plan

Stable for module scope; downstream modules can instantiate bound bundles via the new `*_of_global` theorems.

## Quality Expectations

Norm definitions must match Definition 3 exactly (centered representative, not unsigned). Bound bundles must be universally quantified over all inputs satisfying the preconditions.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `normInfF_zero` and `normInfCoeffs_empty` are `@[simp]`-tagged.

## Out of Scope

- Euclidean / L2 norms.
- Concrete bound instantiation proofs.
- Submultiplicativity of `normInfCoeffs` under `mulRq`.
