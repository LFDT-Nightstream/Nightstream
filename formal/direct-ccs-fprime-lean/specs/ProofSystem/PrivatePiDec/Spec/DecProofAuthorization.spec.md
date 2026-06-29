# DEC Proof Authorization

This spec defines the proof-system boundary for replacing direct hashing of
`CE(b)^k` children with a compact parent `CE(B)` handle plus a proof about the
private children.

## Objects

- `parent`: the bound `CE(B)` claim produced by `Pi_RLC`.
- `children`: the private `CE(b)^k` claims that will be used as the next
  accumulator.
- `proof`: a proof object checked by a verifier such as a sumcheck-style
  verifier.
- `VerifyDecProof`: the verifier predicate for that proof object.

## Soundness Boundary

The verifier predicate is acceptable only if verification implies both:

1. `children` recompose to `parent` by the `Pi_DEC` equations;
2. `children` satisfy the required child membership predicate, including the
   canonical-decomposition or uniqueness condition needed to authorize the
   hidden children.

The proof object is not an input to Fiat-Shamir authority by itself. It is
authority only through the theorem that verification implies the exact
arithmetic predicates required by DEC authorization.

## Theorem Target

If a proof verifier is sound for DEC recomposition and child membership, and
the child decomposition is unique under that membership predicate, then two
accepted proof authorizations of the same parent must feed the same next
`Pi_CCS` accumulator inputs.
