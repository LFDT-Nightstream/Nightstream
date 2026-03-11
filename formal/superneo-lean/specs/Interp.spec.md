# Interp

## Purpose

- **What it is**: constructive univariate interpolation over the concrete Goldilocks field, presented through finite coefficient arrays.
- **Protocol role**: supplies the interpolation/evaluation obligation used by arithmetic side-condition packaging on the SuperNeo path.

## Mathematical Target

Let `xs, ys : Array F` with `xs.size = ys.size = n`.

- `coeffArrayPolynomial coeffs` is the polynomial over `ZMod q` whose first `n` coefficients are the entries of `coeffs`.
- `polyEval coeffs x` is evaluation of `coeffArrayPolynomial coeffs` at `x`.
- `interpolationNodesDistinct xs` means the sample nodes in `xs` are pairwise distinct.
- `interpolatesOn xs ys coeffs` means `coeffs.size = n` and `polyEval coeffs xs[i] = ys[i]` for every sample index `i : Fin n`.
- `interpolateCoeffs xs ys` is the constructive Lagrange interpolation coefficient array obtained from the sample pairs `(xs[i], ys[i])`.

Target properties:

1. If `interpolationNodesDistinct xs`, then `interpolateCoeffs xs ys` interpolates the full sample set:
   `interpolatesOn xs ys (interpolateCoeffs xs ys)`.
2. If `interpolationNodesDistinct xs` and `coeffs` interpolates the same sample set, then
   `coeffs = interpolateCoeffs xs ys`.
3. `interpolationProp xs ys coeffs evalPoint expectedEval` packages:
   distinct sample nodes, interpolation on the sample set, and evaluation of `coeffs` at `evalPoint`.
4. `interpolationCase xs ys coeffs evalPoint expectedEval` is a Boolean checker equivalent to
   `interpolationProp xs ys coeffs evalPoint expectedEval`.

## Paper Anchors

Source: `./formal/superneo-lean/SuperNeo.pdf.md`

- Definition 6 / Section 4: polynomial evaluation obligations appearing in sum-check style reasoning.
- Section 7.3 and Section 7.4: interpolation/evaluation side conditions threaded into protocol arithmetic targets.

## Module Mapping

| Lean file | Paper role |
|---|---|
| `SuperNeo/Interp.lean` | Constructive interpolation/evaluation infrastructure for protocol arithmetic obligations |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Relation | `interpolationNodesDistinct` | def | Theorem-Target | pairwise-distinct sample nodes |
| Relation | `interpolatesOn` | def | Theorem-Target | exact interpolation on the sample set |
| Executable | `polyEval` | def | Theorem-Target | coefficient-array polynomial evaluation |
| Executable | `interpolateCoeffs` | def | Theorem-Target | constructive Lagrange interpolation coefficients |
| Correctness | `interpolateCoeffs_interpolatesOn` | theorem | Theorem-Target | constructive interpolant matches all sample values |
| Uniqueness | `interpolateCoeffs_unique` | theorem | Theorem-Target | sample-wise interpolant is unique |
| Proposition | `interpolationProp` | def | Theorem-Target | distinctness + interpolation + extra evaluation |
| Executable | `interpolationCase` | def | Theorem-Target | Boolean interpolation/evaluation checker |
| Sound | `interpolationCase_sound` | theorem | Theorem-Target | `interpolationCase = true -> interpolationProp` |
| Complete | `interpolationCase_complete` | theorem | Theorem-Target | `interpolationProp -> interpolationCase = true` |
| Bridge | `interpolationCase_eq_true_iff` | theorem | Theorem-Target | Boolean/proposition equivalence |
| Structure | `interpolationProp_sizes` | theorem | Theorem-Target | extracts sample/coefficient size equalities |
| Structure | `interpolationProp_eval_eq` | theorem | Theorem-Target | extracts the extra evaluation equality |
| Legacy | `interpolationAssumption` | def | Legacy Boundary / Refuted | universal interpolation claim retained only as an explicit non-target surface |
| Refutation | `not_interpolationAssumption` | theorem | Theorem-Target | the legacy universal boundary is false as stated |

## Assumption Ledger

- `interpolationAssumption`: retained only as a legacy/refuted surface so downstream code can state explicitly that the old universal boundary is not a valid theorem target.

## Dependencies and Consumers

Upstream dependencies:
- `SuperNeo/Field.lean`
- `SuperNeo/PolynomialBridge.lean`

Downstream consumers:
- `SuperNeo/ArithmeticObligations.lean`
- `SuperNeo/ArithmeticBundle.lean`
- `SuperNeo/ProtocolTarget.lean`

## Regression Expectations

- `lake build` succeeds.
- `lake exe check` remains green.
- No `sorry` is introduced in the interpolation module.
