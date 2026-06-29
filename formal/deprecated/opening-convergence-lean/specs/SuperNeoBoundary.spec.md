# SuperNeoBoundary Specification

## Purpose

Instantiate the generic `AjtaiPCSBoundary` used by the opening-convergence
package with the current concrete SuperNeo lattice / CE relation.

This gives theorem-facing meaning to the current imported PCS verifier
boundary. A claim is satisfied exactly when:

1. its `OpenedObjectId` resolves to one concrete SuperNeo opened object,
2. that object carries one concrete schema and one concrete CE relation,
3. the payload schema matches the opened-object schema,
4. the claim point / payload are translated into one CE statement, and
5. there exists a CE witness satisfying that statement.

## Core Translation

For the current imported CE surface, `K = SuperNeo.F`:

- `point : Fin ell → F`
  becomes
  `pointToCoeffs point : Coeffs`

- `PackedColumnEval F`
  becomes
  `packedColumnToCoeffs : Coeffs`

- `FamilyEvalPayload F`
  becomes
  `payloadToEvaluations : Array Coeffs`

- `(openedObject, point, payload)`
  becomes one CE statement:

```
stmt.commitment = openedObject.commitment
stmt.publicInput = openedObject.publicInput
stmt.point = pointToCoeffs point
stmt.evaluations = payloadToEvaluations payload
```

## Concrete Boundary

The concrete PCS boundary is:

```
verify(objectId, point, payload) :=
  exists obj, wit,
    registry.lookup objectId = some obj
    AND obj.schema = payload.schema
    AND obj.rowDomainLogSize = point.length
    AND CEHolds obj.ce (claimStatement obj point payload) wit
```

This is the paper-faithful local meaning of “the Ajtai opening verifies” for
the current base-field CE surface consumed by the opening-convergence package.

## Registry Contract

The registry is explicit theorem-facing data. It must satisfy:

- successful lookup returns the object keyed by that identity
- the object carries the concrete schema, commitment, public input, CE relation, and
  row-domain log size for the opened object

No hidden oracle predicate remains once this registry is supplied.

## Contract Surface

| Lean symbol | Kind | Role |
|---|---|---|
| `pointToCoeffs` | Definition | Point translation into CE statement coordinates |
| `packedColumnToCoeffs` | Definition | One packed column into one coefficient vector |
| `payloadToEvaluations` | Definition | Claim payload into CE statement evaluations |
| `OpenedObject` | Structure | Concrete theorem-facing opened-object semantics |
| `Registry` | Structure | Explicit object lookup boundary |
| `claimStatement` | Definition | CE statement induced by one convergence claim |
| `boundary` | Definition | Concrete `AjtaiPCSBoundary SuperNeo.F` |

## End State

Once a concrete Phase 0/2 artifact is packaged as a `Registry`, the local PCS
oracle is gone: the verifier predicate is interpreted directly by CE witness
existence in SuperNeo.

The full opening-convergence target is stronger than this current boundary:
the frozen protocol evaluates at the quadratic extension field `K`, while the
present CE surface here remains base-field-valued. Closing the final repo-wide
gap therefore requires a theorem-facing bridge from extension-field
points/evaluations into this base-field CE boundary, or a corresponding
generalization of the CE surface itself.
