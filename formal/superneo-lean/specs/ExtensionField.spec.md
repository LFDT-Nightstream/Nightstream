# ExtensionField

## Purpose

- **What it is**: The quadratic extension-field carrier `K = F[u]/(u^2 - 7)` used by the opening-convergence design and by artifact-level SuperNeo checks that evaluate multilinear claims over an extension field.
- **Key property**: `K` is represented concretely as coefficient pairs `(a, b)` standing for `a + b*u`, with multiplication defined by `u^2 = 7`.
- **Protocol role**: This is the theorem-facing home for extension-field points, packed-column evaluations, and transcript challenges that live outside the base field `F`.
- **Scope**: Concrete carrier and arithmetic only. A full field-law package and theorem-facing CE generalization to extension-field points/evaluations are separate consumers.

## Target Formulas

- `K := F × F`
- `ofF(a) = (a, 0)`
- `(a₁ + b₁u) + (a₂ + b₂u) = (a₁ + a₂) + (b₁ + b₂)u`
- `-(a + bu) = (-a) + (-b)u`
- `(a₁ + b₁u) * (a₂ + b₂u) = (a₁a₂ + 7 b₁b₂) + (a₁b₂ + b₁a₂)u`
- `scaleBase(c, a + bu) = (ca) + (cb)u`

## Paper Anchors

Source: `./formal/superneo-lean/SuperNeo.pdf.md`

- Definition 1 (Fields, Rings, and Dimensions), Section 4, lines 275-282: extension field `K` of degree `t = 2`.
- Definition 2 / Remark 2 context: evaluations and ring coefficients lifted into the extension-field layer.

## Module Mapping

| Lean file | Role |
|---|---|
| `SuperNeo/Primitives/ExtensionField.lean` | Concrete carrier and arithmetic |
| `SuperNeo/Primitives/ExtensionFieldInterface.lean` | Theorem-facing surface |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Type | `KExt` | structure | Definitional | Concrete coefficient-pair carrier for `K` |
| Constants | `w` | def | Definitional | The binomial relation constant `7 ∈ F` |
| Constructors | `ofF`, `ofCoeffs` | def | Definitional | Embed base-field values / explicit coefficient pairs |
| Views | `coeffs` | def | Definitional | Expose `(re, im)` in canonical order |
| Arithmetic | `Zero`, `One`, `Add`, `Neg`, `Sub`, `Mul` | instance | Definitional | Concrete quadratic-extension arithmetic |
| Helpers | `scaleBase`, `pow` | def | Definitional | Base-field scaling and exponentiation |
| Rewrites | `ofF_re`, `ofF_im`, `coeffs_zero`, `coeffs_one`, `mul_re`, `mul_im`, `scaleBase_re`, `scaleBase_im` | theorem | Theorem-Target | Exact coordinate formulas |

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Primitives/Field.lean`: base field `F`
- `SuperNeo/Primitives/Parameters.lean`: extension degree `extDegreeK = 2`

Downstream consumers:
- opening-convergence formalization: concrete carrier for packed-column claims and extension-field points
- artifact validation: executable conversions and transcript checks over extension-field data
- future CE/opening generalization: theorem-facing extension-field points/evaluations

## Acceptance Criteria

- `lake build SuperNeo.ExtensionFieldInterface` succeeds.
- No `sorry` in the module.
- Coordinate theorems expose the exact `u^2 = 7` arithmetic used by downstream consumers.

## Out of Scope

- Full `Field KExt` instance and its laws.
- Irreducibility proof that `u^2 - 7` defines a field extension.
- CE / MLE / EqPoly generalization from `F` to `K`.
