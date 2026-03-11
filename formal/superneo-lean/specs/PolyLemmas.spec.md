# PolyLemmas

## Purpose

- **What it is**: Boolean-cube polynomial helper layer for the Lemma 5 degree-vs-set-size bound surface and the Lemma 6 eq-lifting identity.
- **Protocol role**: supplies theorem-native Boolean-cube recovery/vanishing facts for truth-table MLEs and the arithmetic `d ≤ |S|` surface threaded into Schwartz-Zippel style soundness arguments.

## Mathematical Target

Let `qVals : Array F`, `ell : Nat`, and `n = 2^ell`.

- `eqLiftFromTable qVals z = Σ_{mask < n} eq(bits(mask), z) · qVals[mask]`, where `bits(mask)` is the little-endian Boolean embedding `bitsToFArray ell mask`.
- `eqLiftBooleanIndicatorProp qVals ell mask` packages:
  `qVals.size = n`, `mask < n`, and
  `eqLiftFromTable qVals (bits(mask)) = qVals[mask]`.
- `eqLiftAllBooleanProp qVals ell` packages:
  `qVals.size = n` and
  `∀ mask < n, eqLiftFromTable qVals (bits(mask)) = qVals[mask]`.
- `zeroOnBooleanCubeProp qVals ell` packages:
  `qVals.size = n` and
  `∀ mask < n, qVals[mask] = 0`.
- `eqLiftZeroOnBooleanCubeProp qVals ell` packages:
  `qVals.size = n` and
  `∀ mask < n, eqLiftFromTable qVals (bits(mask)) = 0`.
- `schwartzZippelBoundLeOneProp totalDegree setSize` means:
  `setSize ≠ 0 ∧ totalDegree ≤ setSize`.
- `eqLiftBooleanIndicator`, `eqLiftAllBoolean`, and `schwartzZippelBoundLeOne` are executable Boolean checkers equivalent to their proposition-level targets.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Lemma 5 (Schwartz-Zippel), Appendix C, lines 733-736: the `d/|S|` failure bound depends on the arithmetic side condition `d ≤ |S|` with `|S| ≠ 0`.
- Lemma 6 (eq-lifting), Appendix C, lines 737-740: `Σ_x eq(x,z) · Q(x) = Q(z)` for Boolean-cube points `z`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PolyLemmas.lean` | Lemma 5 (SZ), Lemma 6 (eq-lift) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Eq-lift | `eqLiftFromTable` | def | Definitional | `Σ eq(x,z) · q(x)` |
| Eq-lift | `eqLiftFromTable_bitsToFArray` | theorem | Theorem-Target | direct recovery at each Boolean point |
| Eq-lift | `eqLiftBooleanIndicator` | def | Definitional | Single-point check |
| Eq-lift | `eqLiftBooleanIndicatorProp` | def | Definitional | Single-point proposition |
| Eq-lift | `eqLiftAllBoolean` | def | Definitional | All-points check |
| Eq-lift | `eqLiftAllBooleanProp` | def | Theorem-Target | universal Boolean-cube recovery proposition |
| Eq-lift | `eqLiftAllBoolean_holds` | theorem | Theorem-Target | universal Boolean-cube recovery from the table size condition |
| Eq-lift | `eqLiftAllBoolean_sound` | theorem | Theorem-Target | Boolean all-points check implies proposition |
| Eq-lift | `eqLiftAllBoolean_complete` | theorem | Theorem-Target | proposition implies Boolean all-points check |
| Eq-lift | `eqLiftAllBoolean_eq_true_iff` | theorem | Theorem-Target | Boolean/proposition equivalence for all Boolean points |
| Vanishing | `zeroOnBooleanCubeProp` | def | Theorem-Target | truth-table vanishing on the Boolean cube |
| Vanishing | `eqLiftZeroOnBooleanCubeProp` | def | Theorem-Target | eq-lift vanishing on the Boolean cube |
| Vanishing | `eqLiftZeroOnBooleanCube_iff_zeroOnBooleanCube` | theorem | Theorem-Target | Boolean-cube vanishing is preserved by eq-lift |
| SZ bound | `schwartzZippelBoundLeOne` | def | Definitional | `d ≤ |S|` executable check |
| SZ bound | `schwartzZippelBoundLeOneProp` | def | Theorem-Target | proposition-level arithmetic precondition |
| SZ sound | `schwartzZippelBoundLeOne_sound` | theorem | Theorem-Target | `check = true → bound holds` |
| SZ complete | `schwartzZippelBoundLeOne_complete` | theorem | Theorem-Target | `bound holds → check = true` |
| Eq-lift bridge | `eqLiftBooleanIndicator_sound` | theorem | Theorem-Target | `Bool → Prop` |
| Eq-lift bridge | `eqLiftBooleanIndicator_complete` | theorem | Theorem-Target | `Prop → Bool` |
| Eq-lift bridge | `eqLiftBooleanIndicator_eq_true_iff` | theorem | Theorem-Target | Bool↔Prop closure |
| SZ bridge | `schwartzZippelBoundLeOne_eq_true_iff_prop` | theorem | Theorem-Target | Bool↔Prop closure |
| Sanity | `polyLemmaSanity` | def | Definitional | Cross-check harness |

## Assumption Ledger

None.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/EqPoly.lean`: `eqPoly`, `bitsToFArray`, and Boolean-cube selector facts.
- `SuperNeo/MLE.lean`: `bitsToFieldArray` and the MLE truth-table embedding.

Downstream consumers:
- `SuperNeo/Checks.lean`: uses `polyLemmaSanity` for cross-check validation.
- `SuperNeo/ArithmeticObligations.lean`: consumes theorem-native Boolean-cube recovery.
- `SuperNeo/ProofSystem/SumCheck/General.lean`: consumes the degree-vs-set-size arithmetic surface in the full Schwartz-Zippel/soundness layer.

## Regression Expectations

- `lake build` succeeds.
- `lake exe check` remains green.
- No `sorry`.
- The Boolean/proposition bridges remain biconditional theorem surfaces.
