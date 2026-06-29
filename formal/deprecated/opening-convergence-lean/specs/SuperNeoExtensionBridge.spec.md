# SuperNeoExtensionBridge Specification

## Purpose

Freeze the canonical theorem-facing bridge from the opening-convergence
extension-field carrier `K = SuperNeo.KExt` to the current base-field
SuperNeo CE statement surface.

The opening-convergence protocol evaluates at extension-field points and
produces extension-field packed-column openings. The current proved
SuperNeo CE surface still carries:

- `point : Coeffs = Array F`
- `evaluations : Array Coeffs`

So this module owns the exact split encoding used to interpret one
extension-field claim inside that base-field CE boundary.

## Canonical Encoding

Let `x = a + b·u ∈ K`, with `a, b ∈ F` and `u^2 = 7`.

### Point Encoding

For one point `r : Fin ell → K`, encode:

```
pointReCoeffs(r) = [re(r_0), ..., re(r_{ell-1})]
pointImCoeffs(r) = [im(r_0), ..., im(r_{ell-1})]
pointToBaseCoeffs(r) = pointReCoeffs(r) ++ pointImCoeffs(r)
```

So the current base-field CE statement sees a point vector of length `2 * ell`.

### Packed Column Encoding

For one packed column evaluation `v : PackedColumnEval K`, encode:

```
packedColumnReCoeffs(v) = [re(v_0), ..., re(v_{D-1})]
packedColumnImCoeffs(v) = [im(v_0), ..., im(v_{D-1})]
```

Each packed extension-field column therefore becomes two base-field
coefficient vectors of length `D`.

### Payload Encoding

For one payload with `m = packedColumnCount(schema)` packed columns:

```
payloadReEvaluations(payload) = [packedColumnReCoeffs(v_0), ..., packedColumnReCoeffs(v_{m-1})]
payloadImEvaluations(payload) = [packedColumnImCoeffs(v_0), ..., packedColumnImCoeffs(v_{m-1})]
payloadToSplitEvaluations(payload) =
  payloadReEvaluations(payload) ++ payloadImEvaluations(payload)
```

So the current base-field CE statement sees `2 * m` evaluation vectors.

## Claim Translation

Given one current SuperNeo opened object `obj`, one extension-field point,
and one extension-field payload, the bridge-induced CE statement is:

```
claimStatementK(obj, point, payload).commitment = obj.commitment
claimStatementK(obj, point, payload).publicInput = obj.publicInput
claimStatementK(obj, point, payload).point = pointToBaseCoeffs(point)
claimStatementK(obj, point, payload).evaluations = payloadToSplitEvaluations(payload)
```

This module freezes the encoding boundary and the induced extension-field PCS
boundary used by the final concrete closure theorem in
`SuperNeoConvergenceClosure`.

## Target Properties

The bridge must prove:

1. size formulas for all split encodings
2. injectivity of the point split encoding
3. injectivity of the packed-column split encoding
4. exact statement-shape formulas for `claimStatementK`
5. one concrete `AjtaiPCSBoundary K` induced by the split CE statement
6. injectivity of the whole split point/payload/statement encoding once the
   opened-object schema is fixed

These facts prevent the final concrete closure theorem from hiding behind an
underspecified encoding choice.

## Contract Surface

| Lean symbol | Kind | Role |
|---|---|---|
| `pointReCoeffs` | Definition | Real-part block of one extension-field point |
| `pointImCoeffs` | Definition | Imag-part block of one extension-field point |
| `pointToBaseCoeffs` | Definition | Canonical flattened point encoding |
| `packedColumnReCoeffs` | Definition | Real-part coefficient vector of one packed column |
| `packedColumnImCoeffs` | Definition | Imag-part coefficient vector of one packed column |
| `payloadReEvaluations` | Definition | Real blocks for one payload |
| `payloadImEvaluations` | Definition | Imag blocks for one payload |
| `payloadToSplitEvaluations` | Definition | Canonical flattened payload encoding |
| `claimStatementK` | Definition | CE statement induced by one extension-field claim |
| `boundaryK` | Definition | Current extension-field PCS boundary induced by split CE statements |
| `point_eq_of_split_blocks_eq` | Theorem | Split point blocks determine the original point |
| `packedColumn_eq_of_split_blocks_eq` | Theorem | Split coefficient blocks determine the original packed column |
| `pointToBaseCoeffs_injective` | Theorem | Whole split point encoding determines the original extension-field point |
| `payloadToSplitEvaluations_injective` | Theorem | Whole split payload encoding determines the original payload when the schema is fixed |
| `claimStatementK_injective` | Theorem | For one fixed opened object, the split CE statement determines the original point and payload exactly |

## End State

This module closes the encoding choice and exposes the concrete boundary used
by Module 7's end-to-end closure theorem.
