# ExtensionMLE

## Purpose

Provide the extension-field analogue of the Boolean-cube equality polynomial
and multilinear-extension evaluators needed by Nightstream opening convergence.

This module owns the `SuperNeo.KExt` versions of:
- Boolean-cube bit embedding,
- `eq` polynomial evaluation,
- MLE inner-product evaluation,
- folding-style MLE evaluation.

It is the prerequisite layer for any paper-faithful extension-field SumCheck
component. The current base-field `SuperNeo/Primitives/MLE.lean` and `SuperNeo/Primitives/EqPoly.lean`
remain the canonical Section 4 closure for the base carrier `SuperNeo.F`;
this module closes the same evaluator surface for the quadratic extension
carrier used by opening convergence.

## Target Formulas

### Equality Polynomial

For `x, y ∈ K^ℓ`:

```text
eqPolyK(x, y) = ∏_{i=0}^{ℓ-1} (x_i * y_i + (1 - x_i) * (1 - y_i))
```

### Inner-Product MLE

For a truth table `f : {0,1}^ℓ -> K` and point `r ∈ K^ℓ`:

```text
mleEvalK(f, r) = Σ_{x ∈ {0,1}^ℓ} f(x) * eqPolyK(bits(x), r)
```

### Folding MLE

For the same table and point:

```text
mleByFoldingK(v, r)
```

is the recursive folding evaluator over `K`.

## Explicit Type Signatures

| Lean symbol | Type | Role |
|---|---|---|
| `bitsToKExtArray` | `Nat -> Nat -> Array KExt` | Boolean-cube embedding into `KExt` |
| `eqTermK` | `KExt -> KExt -> KExt` | Single-coordinate equality term |
| `eqPolyK` | `Array KExt -> Array KExt -> KExt` | Extension-field equality polynomial |
| `mleEvalK` | `Array KExt -> Array KExt -> KExt` | Extension-field MLE evaluator |
| `mleInnerProductFormK` | `Array KExt -> Array KExt -> KExt` | Inner-product target form |
| `foldLayerK` | `Array KExt -> KExt -> Array KExt` | One folding step |
| `mleByFoldingExecK` | `Array KExt -> Array KExt -> KExt` | Recursive executable folding evaluator |
| `mleByFoldingK` | `Array KExt -> Array KExt -> KExt` | Theorem-facing folding evaluator |
| `linCombK` | `KExt -> Array KExt -> Array KExt -> (f.size = g.size) -> Array KExt` | Pointwise linear combination `f + δ*g` |

## Paper Anchors

Source: `./formal/superneo-lean/SuperNeo.pdf.md`

- Section 4 (preliminaries): equality polynomial and multilinear extensions.
- Opening convergence design: Phase 1 point reduction is over the quadratic
  extension carrier `K`, not only the base field `F`.

## Contract Surface

| Lean symbol | Kind | Guarantee |
|---|---|---|
| `mleEvalK_eq_innerProductForm_of_size` | theorem | Valid-size extension-field MLE equals the inner-product form |
| `mleIdentityAssumptionK_holds` | theorem | Package-level closure of the extension-field MLE identity surface |
| `foldLayerK_size` | theorem | Folding halves the table width |
| `foldLayerK_get` | theorem | Explicit elementwise folding formula |
| `mleByFoldingK_step` | theorem | One-step unfolding for non-empty challenge vectors |
| `mleByFoldingK_empty` | theorem | Empty challenge vector returns the head element |
| `mleInnerProductFormK_eq_mleByFoldingK_of_size` | theorem | Valid-size extension-field inner-product MLE equals the folding evaluator |
| `linCombK_size` | theorem | Extension-field pointwise linear combination preserves table width |
| `mleInnerProductLinearityAssumptionK_holds` | theorem | Canonical closure of extension-field inner-product MLE linearity |
| `mleEvalK_linComb_of_assumption` | theorem | Uses a guarded extension-field MLE linearity hypothesis |
| `mleEvalK_linComb_of_assumptions` | theorem | Derives guarded extension-field MLE linearity from identity + inner-product linearity |
| `mleEvalLinearityAssumptionK_holds` | theorem | Canonical closure of guarded extension-field MLE linearity |
| `mleEvalK_eq_mleByFoldingK_of_size` | theorem | Valid-size extension-field guarded evaluator equals the folding evaluator |

## Out of Scope

- A full extension-field SumCheck theorem package.
- Probability-model / Schwartz-Zippel soundness over `KExt`.
- Any protocol-specific transcript construction.
