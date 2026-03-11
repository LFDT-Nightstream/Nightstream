# Goldilocks

## Purpose

- **What it is**: The Goldilocks prime `q = 2^64 - 2^32 + 1 = 18446744069414584321` and its half-point `halfQ = (q-1)/2`, used as the base field modulus throughout SuperNeo.
- **Key property**: `0 < q`, `1 < q`, `halfQ < q`, and `1 ≤ halfQ`.
- **Protocol role**: Every module that operates on `F = Fin q` depends on `q_pos` and `q_ne_zero` for well-formedness. The half-point `halfQ` partitions representatives into positive/negative halves for centered norms (Definition 3).
- **Scope**: Goldilocks field only (no Mersenne-61 or Almost-Goldilocks variants from Appendix B.1/B.3).

## Target Formulas

- `q = 2^64 - 2^32 + 1 = 18446744069414584321`
- `halfQ = q / 2 = 9223372034707292160`
- `0 < q ∧ 1 < q`
- `1 ≤ halfQ < q`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 1 (Fields, Rings, and Dimensions), Section 4, lines 275-282: `F` is a finite field of prime order `q`.
- Appendix B.2 (Goldilocks parameters), lines 709-727: concrete choice `q = 2^64 - 2^32 + 1`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Goldilocks.lean` | Definition 1 (field modulus), Appendix B.2 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Modulus | `q` | def | Definitional | `q = 18446744069414584321` |
| Modulus | `halfQ` | def | Definitional | `halfQ = q / 2` |
| Positivity | `q_pos` | theorem | Theorem-Target | `0 < q` |
| Positivity | `q_ne_zero` | theorem | Theorem-Target | `q ≠ 0` |
| Positivity | `q_gt_one` | theorem | Theorem-Target | `1 < q` |
| Half-point | `halfQ_lt_q` | theorem | Theorem-Target | `halfQ < q` |
| Half-point | `halfQ_le_q` | theorem | Theorem-Target | `halfQ ≤ q` |
| Half-point | `one_le_halfQ` | theorem | Theorem-Target | `1 ≤ halfQ` |

## Proof Obligations and Closure Plan

All obligations closed. Every theorem is proved by `native_decide` or `omega` over concrete values.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

Upstream dependencies: none.

Downstream consumers:
- `SuperNeo/Field.lean`: uses `q` to define `F := Fin q`; uses `q_pos` for `Fin q` inhabitedness.
- `SuperNeo/Decomp.lean`: uses `q` for field-element bounds in digit decomposition.
- `SuperNeo/Parameters.lean`: uses `q` via `modulus_def : modulus = q`.

## Implementation Plan

No further work required; module is proof-complete.

## Quality Expectations

All constants match the Goldilocks prime from Appendix B.2 exactly. Positivity and ordering theorems discharge downstream `0 < q` obligations without per-use `native_decide`.

## Acceptance Criteria

- `lake build` succeeds.
- All 8 declarations are proved (no `sorry`).
- Downstream `Field.lean` compiles using `q` and `q_pos` directly.

## Out of Scope

- Mersenne-61 (`2^61 - 1`) and Almost-Goldilocks (`(2^64 - 2^32 + 1) - 32`) variants from Appendix B.1/B.3.
- Field arithmetic (handled by `Field.lean`).
