# Decomp

## Purpose

- **What it is**: Base-2 and balanced (`split_b`) digit decomposition of field elements and coefficient vectors, with recomposition round-trip proofs, digit-bound proofs, and terminal-quotient-zero conditions.
- **Key property**: `recomposeBase2ScalarNat (splitBase2Scalar a k) = a.val % 2^k` (round-trip for base-2) and `splitBalancedRoundTripProp z b k` (round-trip for balanced split, including terminal-zero and digit-bound components).
- **Protocol role**: Decomposition is the core of Π_DEC (Section 7.5). The prover decomposes each witness coefficient vector into `k` digit-vectors satisfying `‖d_j‖_∞ ≤ b/2`, and the verifier checks recomposition and terminal-quotient conditions. The digit-bound predicates feed directly into norm-growth tracking for the MSIS security reduction (Theorem 8).

## Target Formulas

- `bitAt(n, i) = (n / 2^i) % 2 ∈ {0,1}`
- `splitBase2Scalar(a, k) = [F.ofNat(bitAt(a.val, 0)), ..., F.ofNat(bitAt(a.val, k-1))]`
- `a.val = splitBase2LowPartNat(a, k) + 2^k · splitBase2TerminalQuot(a, k)`
- `splitBalancedScalar(a, b, k)`: balanced-carry digit extraction
- `recomposeSplitBalancedScalar` inverts `splitBalancedScalar` when `splitBalancedTerminalZeroProp` holds

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 4, lines 294-296: `split_b` decomposition into base-`b` digits with `‖d_j‖_∞ ≤ ⌊b/2⌋`.
- Section 7.5 (Π_DEC), lines 490-520: decomposition check in the folding protocol.
- Appendix B.2, lines 709-727: concrete `b = 2`, `k = 14`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Decomp.lean` | `split_b` (Section 4), Π_DEC (Section 7.5) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Base-2 core | `bitAt` | def | Definitional | Bit at position `i` |
| Base-2 core | `splitBase2Scalar` | def | Definitional | `k` base-2 digits of `a` |
| Base-2 core | `recomposeBase2ScalarNat` | def | Definitional | Sum of `d_i · 2^i` |
| Base-2 core | `recomposeBase2Scalar` | def | Definitional | `F.ofNat` of above |
| Base-2 terminal | `splitBase2TerminalQuot` | def | Definitional | `a.val / 2^k` |
| Base-2 terminal | `splitBase2LowPartNat` | def | Definitional | `a.val % 2^k` |
| Base-2 terminal | `splitBase2TerminalZeroProp` | def | Definitional | Quotient = 0 |
| Base-2 bounds | `splitBase2DigitsWithinBoundProp` | def | Definitional | All digits ≤ 1 |
| Base-2 lift | `splitBase2Coeffs` | def | Definitional | Row-wise lift to vectors |
| Base-2 lift | `recomposeBase2Coeffs` | def | Definitional | Row-wise recomposition |
| Base-2 lift | `recomposeBase2Coeffs_splitBase2Coeffs_entry` | theorem | Theorem-Target | recomposed column equals canonical low part |
| Base-2 lift | `recomposeBase2Coeffs_splitBase2Coeffs_eq_of_terminal_zero` | theorem | Theorem-Target | exact row-wise round-trip under terminal-zero |
| Base-2 lift | `recomposeBase2Coeffs_splitBase2Coeffs_eq_of_val_lt_pow` | theorem | Theorem-Target | exact row-wise round-trip under range bound |
| Base-2 lift | `splitBase2RowsWithinBoundProp` | def | Definitional | Row-wise digit bounds |
| Balanced core | `splitBalancedScalar` | def | Definitional | Balanced-carry scalar split |
| Balanced core | `splitBalancedVec` | def | Definitional | Row-wise balanced split |
| Balanced recomp | `recomposeSplitBalancedScalar` | def | Definitional | Balanced recomposition |
| Balanced recomp | `recomposeSplitDigits` | def | Definitional | Vectorized recomposition |
| Balanced predicates | `digitsWithinBase` | def | Definitional | Executable bound check |
| Balanced predicates | `digitsWithinBaseProp` | def | Definitional | Propositional bound check |
| Balanced predicates | `splitBalancedRoundTripProp` | def | Definitional | Full round-trip predicate |
| Balanced predicates | `splitRoundTrip` | def | Definitional | Executable round-trip check |
| Base-2 theorems | `bitAt_lt_two` | theorem | Theorem-Target | `bitAt n i < 2` |
| Base-2 theorems | `bitAt_le_one` | theorem | Theorem-Target | `bitAt n i ≤ 1` |
| Base-2 theorems | `splitBase2DecompositionNat` | theorem | Theorem-Target | Euclidean decomposition |
| Base-2 theorems | `splitBase2DigitsWithinBound` | theorem | Theorem-Target | All digits in `{0,1}` |
| Base-2 theorems | `splitBase2RowsWithinBound` | theorem | Theorem-Target | Row-wise bounds |
| Balanced theorems | `splitBalancedDecompositionInt` | theorem | Theorem-Target | Integer decomposition identity |
| Balanced theorems | `splitBalancedDecompositionInt_of_terminal_zero` | theorem | Theorem-Target | Under terminal-zero |
| Balanced field lift | `splitBalancedScalarFieldLiftProp_holds_of_base_ge_two` | theorem | Theorem-Target | Field-lift for `b ≥ 2` |
| Balanced field lift | `splitBalancedVecFieldLiftProp_holds_of_base_ge_two` | theorem | Theorem-Target | Vectorized field-lift |
| Round-trip | `splitBalancedRoundTripProp_of_constructive_boundaries` | theorem | Theorem-Target | Constructive round-trip |
| Round-trip bridge | `splitRoundTrip_sound_prop` | theorem | Theorem-Target | `Bool → Prop` |
| Round-trip bridge | `splitRoundTrip_complete_prop` | theorem | Theorem-Target | `Prop → Bool` |

## Proof Obligations and Closure Plan

All core obligations closed. All base-2 and balanced decomposition theorems are proved.

## Assumption Ledger

No open boundary assumptions. All proofs are constructive.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Norm.lean`: `normInfF` for digit-bound predicates.

Downstream consumers:
- `SuperNeo/PiDEC.lean`: uses `splitBalancedVec`, `splitBalancedRoundTripProp` for Π_DEC protocol definition.
- `SuperNeo/ArithmeticObligations.lean`: uses decomposition predicates for arithmetic obligation statements.
- `SuperNeo/ProtocolRelations.lean`: uses decomposition conditions in folding-step relation checks.

## Implementation Plan

No further work required; module is proof-complete for its scope.

## Quality Expectations

Round-trip theorems must be constructive (no `sorry`, no axioms). Digit bounds must be universally quantified over all entries. The executable `splitRoundTrip` checker must be sound and complete w.r.t. `splitBalancedRoundTripProp`.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `splitRoundTrip_sound_prop` and `splitRoundTrip_complete_prop` both proved.

## Out of Scope

- Non-power-of-2 bases (only `b = 2` is concretely instantiated).
- Asymptotic complexity analysis of decomposition.
- Norm-growth formulas combining decomposition with folding (those belong in protocol modules).
