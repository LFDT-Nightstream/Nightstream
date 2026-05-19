# DEC Authorization

This spec defines the direct CCS `F'` authorization boundary for using a compact
parent `CE(B)` handle while keeping the reusable `CE(b)^k` children private.

## Objects

- `parent`: the bound `CE(B)` claim produced by `Pi_RLC`.
- `children`: the private `CE(b)^k` claims produced by `Pi_DEC`.
- `nextInputs`: the `CE(b)^k` claims consumed as the next `Pi_CCS` incoming
  accumulator.
- `HashBindsParent`: a proof-visible predicate saying the compact digest/handle
  binds the parent claim.
- `DecRecompose`: the `Pi_DEC` recomposition predicate from children to parent.
- `ChildCEMembership`: full child CE membership, including commitment opening,
  input projection, evaluation claims, and low-norm bounds.
- `WireIdentity`: equality between `nextInputs` and the proven `children`.

## Authorization Rule

The parent handle authorizes next-round accumulator inputs only when all of the
following hold:

1. the handle binds `parent`;
2. `children` recompose to `parent`;
3. `children` satisfy child `CE(b)` membership;
4. `nextInputs` are exactly `children`.

## Uniqueness Obligation

A concrete arithmetic instantiation must prove that the decomposition of a fixed
parent is unique under the chosen CE membership relation.

Signed low-norm bounds alone are not sufficient for `b = 2`; for example,
`1 = 1 + 2*0 = -1 + 2*1`. The instantiation must therefore enforce canonical
digit construction/bitness, or prove an equivalent production constraint that
rules out alternate signed decompositions.

## Theorem Target

If a parent-bound authorization is accepted and the low-norm decomposition is
unique, then a prover cannot use different next-round CE inputs without
violating DEC recomposition, child CE membership, or wire identity.
