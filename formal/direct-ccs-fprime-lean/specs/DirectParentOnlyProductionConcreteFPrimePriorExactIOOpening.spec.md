# DirectParentOnlyProductionConcreteFPrimePriorExactIOOpening

This module specifies the exact public-IO opening certificate for the
production prior `F'` verifier.

The verifier-visible surface contains the canonical prior statement, proof
statement, Construction-2 boundary, terminal public values, boundary public
values, terminal committed proof, terminal verifier public IO, replay
predicates, and fixed authority opener. It contains no theorem that accepted
proofs are sound.

Exact public-IO acceptance requires compact-image replay, Construction-2
boundary replay, transcript replay, public statement validity, proof-boundary
agreement, terminal verifier public IO, exact terminal values, and exact
boundary values for the canonical `(steps, image)` statement.

The raw public-vector bridge is value exact. If the raw vector equals
`terminal_public_values ++ boundary_public_values` and the terminal slice has
the canonical terminal length, then `raw = terminal ++ boundary` forces the
terminal and boundary slices to be exactly canonical. This is a structural list
equality argument, not an aggregate check.

The opening certificate has two cryptographic obligations:

- every exact bound statement accepted by the verifier opens through the fixed
  authority opener,
- every opened authority for an exact bound statement carries the same
  `(steps, image)` pair.

From these obligations Lean derives `FoldedFPrimeAuthority.Accepts`, packages a
`CertifiedPriorVerifier`, proves prior reachability and public-image
invariants, rejects unreachable prior images, and exposes the parent-only
terminal end-to-end, non-aggregate private DEC, and stage-audit consequences.
