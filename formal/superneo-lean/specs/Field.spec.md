# Field

## Purpose

- **What it is**: The prime field `F := Fin q` where `q` is the Goldilocks prime, equipped with modular arithmetic (`+`, `-`, `*`, `⁻¹`), canonical representatives `canonicalRep : F → Nat`, and centered representatives `centeredRep : F → Int` for norm computation.
- **Key property**: `canonicalRep (ofNat (canonicalRep a)) = canonicalRep a` (round-trip), and `∀ a : F, isCanonical a` (all elements are canonical).
- **Protocol role**: `F` is the base scalar type for all SuperNeo vectors, matrices, polynomials, and ring coefficients. `centeredRep` feeds Definition 3 norms via `Norm.lean`. Arithmetic rewrites (`val_add`, `val_mul`, etc.) are consumed by `Ring.lean`, `EqPoly.lean`, and `MLE.lean`.
- **Scope**: Base field `F` only; extension field `K` from Definition 1 is not instantiated.

## Target Formulas

- `F := Fin q` where `q = 2^64 - 2^32 + 1`
- `(a + b).val = (a.val + b.val) % q`
- `(a * b).val = (a.val * b.val) % q`
- `centeredRep a = if a.val ≤ halfQ then a.val else a.val - q` (as integers)
- `ofNat (canonicalRep a) = a` (round-trip)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 1 (Fields, Rings, and Dimensions), Section 4, lines 275-282: `F` is a finite field of prime order `q`.
- Definition 3 (Norm), Section 4, lines 290-291: uses centered representative for `‖a‖_∞`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Field.lean` | Definition 1 (field `F`) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Type | `F` | abbrev | Definitional | `F = Fin q` |
| Construction | `ofNat`, `zero`, `one` | def | Definitional | Field element constructors |
| Arithmetic | `Add F`, `Sub F`, `Mul F`, `Neg F` | instance | Definitional | Modular arithmetic on `Fin q` |
| Representatives | `canonicalRep` | def | Definitional | `canonicalRep a = a.val` |
| Representatives | `centeredRep` | def | Definitional | Signed representative in `[-(q-1)/2, (q-1)/2]` |
| Representatives | `centeredAbs` | def | Definitional | `|centeredRep a|` as `Nat` |
| Round-trip | `ofNat_val` | theorem | Theorem-Target | `ofNat a.val = a` |
| Round-trip | `ofNat_canonicalRep` | theorem | Theorem-Target | `ofNat (canonicalRep a) = a` |
| Canonicality | `canonical` | theorem | Theorem-Target | `∀ a : F, isCanonical a` |
| Arithmetic rewrites | `val_add`, `val_mul`, `val_sub`, `val_neg` | theorem | Theorem-Target | Operation semantics as `% q` |
| Centered rep | `centeredRep_eq_of_le_halfQ` | theorem | Theorem-Target | `a.val ≤ halfQ → centeredRep a = a.val` |
| Centered rep | `centeredRep_eq_sub_q_of_halfQ_lt` | theorem | Theorem-Target | `halfQ < a.val → centeredRep a = a.val - q` |

## Proof Obligations and Closure Plan

All obligations closed. All theorems are proved via `simp`, `omega`, or `Fin` arithmetic.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Goldilocks.lean`: uses `q`, `halfQ`, `q_pos`.

Downstream consumers:
- `SuperNeo/Ring.lean`: uses `F` as coefficient type for `Coeffs := Array F`.
- `SuperNeo/Norm.lean`: uses `centeredRep`/`centeredAbs` for `normInfF`.
- `SuperNeo/EqPoly.lean`: uses `F` arithmetic for `eqTerm`/`eqPoly`.
- `SuperNeo/MLE.lean`: uses `F` arithmetic for inner-product and folding evaluators.
- `SuperNeo/Decomp.lean`: uses `centeredRep` for balanced digit decomposition.

## Implementation Plan

No further work required; module is proof-complete for its scope.

## Quality Expectations

Arithmetic rewrite theorems must state exact modular semantics (not just type-level). Round-trip theorems must compose cleanly for downstream `ofNat`/`canonicalRep` reasoning.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry` in any theorem.
- `centeredRep` cases cover the full `[0, q)` range.

## Out of Scope

- Extension field `K` (Definition 1).
- Exponentiation / discrete-log properties.
- `Decidable` instances for field equality (Lean provides this for `Fin`).
