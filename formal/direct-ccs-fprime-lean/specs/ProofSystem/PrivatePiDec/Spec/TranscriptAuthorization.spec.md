# Reduced Transcript Authorization

This spec defines the exact condition under which a compact transcript input is
allowed to replace hashing full `CE(b)^k` children before deriving the next
`Pi_CCS` challenges.

## Objects

- `source`: the reduced public challenge source, such as a parent `CE(B)` handle
  plus a proof/table commitment.
- `parent`: the `CE(B)` claim authorized by that source.
- `children`: private `CE(b)^k` claims used as the next incoming accumulator.
- `proof`: proof data checked by `F'`.
- `Challenge`: a deterministic challenge function over `source`.

## Rule

A reduced challenge source is safe only if accepted authorizations with the same
source cannot feed different child accumulators to the next `Pi_CCS`.

This is stronger than saying "the digest is self-consistent." The reduced source
must be tied to the actual child wires by a verifier predicate whose soundness
implies DEC recomposition and child membership. The child membership predicate
must include the canonical-decomposition or equivalent uniqueness condition.

## Theorem Target

If:

1. the source binds the parent;
2. the DEC proof verifier is sound for recomposition and child membership;
3. the child decomposition is unique under that membership;
4. both accepted authorizations use the same source and parent;

then the next `Pi_CCS` inputs are equal and therefore the deterministic
challenge derived from the reduced source is not malleable by changing hidden
children.
