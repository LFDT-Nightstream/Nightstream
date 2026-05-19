# Base-2 Reduced Transcript Authorization

This spec instantiates reduced transcript authorization with the concrete
canonical base-2 DEC surface used by the direct CCS `F'` model.

## Objects

- `source`: the reduced public challenge source, such as a parent `CE(B)` handle
  and child-table/proof commitment.
- `parent`: the base-2 parent claim.
- `children`: private base-2 children that will feed the next `Pi_CCS`.
- `proof`: proof object checked by the local verifier.
- `VerifyBase2DecProof`: verifier predicate for that proof object.

## Local Proof Soundness

For a fixed `source` and `parent`, the verifier predicate is acceptable only if
acceptance implies:

1. the children recompose to the parent using the base-2 DEC recomposition
   helper;
2. the children are the canonical base-2 split children of that parent.

This local condition is deliberately stronger than signed low-norm membership.
Signed low-norm decompositions are not unique for `b = 2`.

## Theorem Target

If two accepted proofs use the same reduced source and same parent, and both
proofs satisfy the local soundness condition above, then both authorizations
feed the same next `Pi_CCS` accumulator inputs.

Consequently, a deterministic Fiat-Shamir challenge over the reduced source is
not malleable by changing hidden children, because accepted hidden children are
unique.
