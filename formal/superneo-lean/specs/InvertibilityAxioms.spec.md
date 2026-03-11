# InvertibilityAxioms

## Purpose

- **What it is**: The invertibility predicate `invertibleRq` on ring elements, the weak norm-window bookkeeping predicate `invertibilityWindowProp`, the paper-faithful strict window `strictInvertibilityWindowProp`, the standalone boundary assumption `lowNormInvertibilityAssumption` for actual ring-shaped elements, and the narrower active protocol-path boundary `paperCarrierDiffInvertibilityAssumption`.
- **Key property**: `invertibleRq a ↔ ∃ aInv, mulRq a aInv = oneRq`; the actual Theorem-8 boundary uses `0 < ‖a‖∞ < B`, not the weak bookkeeping bound `‖a‖∞ ≤ B`.
- **Protocol role**: Provides the standalone low-norm invertibility boundary from Theorem 8 together with the narrower active protocol-path boundary on nonzero `paperCarrier` differences. Concrete Goldilocks instantiations may discharge either boundary in `InvertibilityGoldilocks.lean`, and downstream protocol routes may consume the narrower boundary directly or derive it from the stronger standalone route.

## Target Formulas

- `invertibleRq a ↔ ∃ aInv : Coeffs, mulRq a aInv = oneRq`
- `invertibilityWindowProp B a ↔ normInfCoeffs a ≤ B`
- `strictInvertibilityWindowProp B a ↔ 0 < normInfCoeffs a ∧ normInfCoeffs a < B`
- `lowNormInvertibilityAssumption B → ∀ a, (hasRingDegreeShape a → strictInvertibilityWindowProp B a → invertibleRq a)`
- `paperCarrierDiffInvertibilityAssumption → ∀ δ, samplingDiffSet paperCarrier δ → δ ≠ 0 → invertibleRq δ`
- `¬ invertibleRq zeroRq`
- `¬ (∀ a, invertibilityWindowProp B a → invertibleRq a)` (the old weak reading is false because `zeroRq` lies in the weak window)
- `goldilocksTheorem8Z = 3`, `goldilocksTheorem8Order = 27`, `goldilocksPaperBInv = 383`
- `Goldilocks.q % 3 = 1`
- `Goldilocks.q ^ 27 % 81 = 1`
- `∀ k ∣ 27, 0 < k → k < 27 → Goldilocks.q ^ k % 81 ≠ 1`
- `invertibilityPreconditionsProp = True` (trivial; reserved for protocol-level preconditions)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Theorem 8 (Low-norm invertibility), Section 5/6, lines 375-378.
- Definition 3 (Norm), Section 4, lines 290-291: norm bounds used for invertibility window preconditions.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/InvertibilityAxioms.lean` | Theorem 8 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Predicates | `invertibleRq` | def | Definitional | ∃ inverse in Rq |
| Predicates | `invertibilityWindowProp` | def | Definitional | weak bookkeeping bound `‖a‖∞ ≤ B` |
| Predicates | `strictInvertibilityWindowProp` | def | Definitional | paper-faithful strict window `0 < ‖a‖∞ < B` |
| Predicates | `invertibilityPreconditionsProp` | def | Definitional | Trivial (`True`); reserved for protocol-level preconditions |
| Constants | `goldilocksTheorem8Z` | def | Theorem-Target | concrete `z = 3` |
| Constants | `goldilocksTheorem8Order` | def | Theorem-Target | concrete `η / z = 27` |
| Constants | `goldilocksPaperBInv` | def | Theorem-Target | Appendix B.2 floor `383` |
| Boundary | `lowNormInvertibilityAssumption` | def | Boundary | ∀ a, ring-shaped strict window → invertible |
| Boundary | `paperCarrierDiffInvertibilityAssumption` | def | Boundary | ∀ δ, nonzero `paperCarrier` difference → invertible |
| Theorems | `invertibleRq_of_lowNormAssumption` | theorem | Boundary | Uses assumption to derive invertibility |
| Theorems | `paperCarrierDiffInvertibilityAssumption_of_lowNormFive` | theorem | Theorem-Target | Optional sufficient route from the standalone low-norm theorem target at `B = 5` into the narrower active protocol-path boundary |
| Theorems | `invertibilityWindowProp_of_strictWindow` | theorem | Theorem-Target | strict window implies weak bookkeeping bound |
| Theorems | `normInfCoeffs_zeroRq` | theorem | Theorem-Target | `‖zeroRq‖∞ = 0` |
| Theorems | `invertibilityWindowProp_zeroRq` | theorem | Theorem-Target | `zeroRq` lies in every weak window |
| Theorems | `not_invertibleRq_zeroRq` | theorem | Theorem-Target | `zeroRq` is not invertible |
| Theorems | `not_all_window_elements_invertible` | theorem | Theorem-Target | refutes the old weak-window reading |
| Theorems | `invertibilityPreconditions_from_constants` | theorem | Theorem-Target | Preconditions hold |
| Theorems | `goldilocksTheorem8Z_dvd_eta` | theorem | Theorem-Target | concrete `z = 3` divides `η = 81` |
| Theorems | `goldilocksModulus_mod_theorem8Z_eq_one` | theorem | Theorem-Target | `q ≡ 1 (mod 3)` |
| Theorems | `goldilocksModulus_pow_order_eq_one_mod_eta` | theorem | Theorem-Target | `q^27 ≡ 1 (mod 81)` |
| Theorems | `goldilocksModulus_order_mod_eta` | theorem | Theorem-Target | no proper positive divisor of `27` yields residue `1` mod `81` |
| Theorems | `goldilocksTheorem8Bound_gt_five` | theorem | Theorem-Target | concrete Theorem-8 modulus bound exceeds the active threshold `5` |
| Theorems | `goldilocksTheorem8Bound_gt_paperBInv` | theorem | Theorem-Target | concrete Theorem-8 modulus bound exceeds the Appendix B.2 floor `383` |
| Theorems | `eq_zeroRq_of_hasRingDegreeShape_of_normInfCoeffs_eq_zero` | theorem | Theorem-Target | Ring-shaped zero norm implies `zeroRq` |
| Theorems | `normInfCoeffs_pos_of_hasRingDegreeShape_of_ne_zeroRq` | theorem | Theorem-Target | Nonzero ring-shaped elements have positive norm |
| Theorems | `strictInvertibilityWindowProp_five_of_shape_norm_le_four_of_ne_zeroRq` | theorem | Theorem-Target | Ring shape + `‖a‖∞ ≤ 4` + nonzero implies strict window `< 5` |
| Theorems | `paperCarrierDiffInvertibilityAssumption_of_lowNormPaperBInv` | theorem | Theorem-Target | specialized bridge from the paper floor `B = 383` into the active protocol-path boundary |

## Proof Obligations and Closure Plan

The theorem-facing standalone boundary must quantify over actual ring elements, i.e. ring-shaped coefficient vectors satisfying the strict paper premise `0 < ‖a‖∞ < B`. The weaker bookkeeping window `‖a‖∞ ≤ B` is intentionally separate and is explicitly too weak to characterize invertibility because it contains `zeroRq`. Concrete Goldilocks arithmetic side conditions (`z = 3`, `ord_η(q) = 27`, `b_inv = 383`) live in this component, while specialized constructive Goldilocks proofs may live in `InvertibilityGoldilocks.lean`.

## Assumption Ledger

- `lowNormInvertibilityAssumption B`: standalone stronger boundary assumption that ring-shaped nonzero elements with `‖a‖∞ < B` are invertible in `Rq`.
- `paperCarrierDiffInvertibilityAssumption`: narrower active protocol-path boundary that every nonzero `paperCarrier` difference is invertible in `Rq`.
- Concrete Goldilocks proofs may instantiate either boundary in `InvertibilityGoldilocks.lean`.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Norm.lean`: imports `normInfCoeffs` for `invertibilityWindowProp`.
- `SuperNeo/Field.lean`: used indirectly through `normInfF_eq_zero_iff` in the zero-norm ring-shape theorem.

Downstream consumers:
- `SuperNeo/ProtocolTarget.lean`: consumes direct `invertibleRq` witnesses in the active protocol path and now also exposes the stricter `paperCarrier` challenge-difference bridge into those witnesses.
- `SuperNeo/ArithmeticBundle.lean`: depends on invertibility preconditions.

## Implementation Plan

Keep the weak bookkeeping window separate from the strict paper premise. Downstream protocol modules may consume direct `invertibleRq` witnesses, the narrower `paperCarrierDiffInvertibilityAssumption`, or the stronger standalone `lowNormInvertibilityAssumption` as appropriate.

## Quality Expectations

`invertibleRq` must match the paper's invertibility notion (existence of inverse in Rq). `strictInvertibilityWindowProp` must match the Theorem-8 premise `0 < ‖a‖∞ < B`. `invertibilityWindowProp` is only a weak bookkeeping predicate and must not be confused with the paper theorem premise.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.

## Out of Scope

- External proofs not formalized inside this repo.
- Number-theoretic invertibility analyses unrelated to the SuperNeo ring and its cited Goldilocks instantiation data.
