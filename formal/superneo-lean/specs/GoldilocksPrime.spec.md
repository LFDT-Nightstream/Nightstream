# GoldilocksPrime

## Purpose

- **What it is**: Constructive primality closure for `Goldilocks.q`.
- **Key property**: `Goldilocks.q_prime : Nat.Prime Goldilocks.q`.
- **Protocol role**: Supplies `Fact (Nat.Prime Goldilocks.q)` needed by Mathlib finite-field root-count lemmas in SumCheck soundness bridges (`ZMod q` integral-domain path).

## Target Formulas

- `Goldilocks.q = 18446744069414584321`
- `Nat.Prime Goldilocks.q`
- `Fact (Nat.Prime Goldilocks.q)`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 1 (Fields, Rings, Dimensions), Section 4, lines 275-282: field modulus is prime.
- Appendix B.2, lines 709-727: Goldilocks modulus value.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/GoldilocksPrime.lean` | Definition 1 + Appendix B.2 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Prime closure | `Goldilocks.q_prime` | theorem | Theorem-Target | `Nat.Prime Goldilocks.q` |
| Prime closure | `Goldilocks.fact_q_prime` | instance | Theorem-Target | `Fact (Nat.Prime Goldilocks.q)` |

## Proof Obligations and Closure Plan

- Closed: constructive proof via Mathlib `lucas_primality` with witness `a = 7` and explicit factorization of `q - 1`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo/Goldilocks.lean`, `Mathlib/NumberTheory/LucasPrimality`.
- **Consumers**:
  - `SuperNeo/ProofSystem/SumCheck/General.lean`: discharges the prime-field prerequisite for `Polynomial.card_roots` over `ZMod q`.

## Acceptance Criteria

- `lake build` succeeds.
- `lake exe check` reports `all_checks=true`.
- No `axiom`/`sorry` in this module.
