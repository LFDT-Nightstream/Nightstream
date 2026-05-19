# Direct Parent Only Production SuperNeo Reuse Prior Opening

`DirectParentOnlyProductionSuperNeoReusePriorOpening` specifies the compressed
prior-authority boundary for the Section 7.1-backed parent-only production
context.

The context is `DirectParentOnlyProductionSuperNeoReuse.ProductionContext`.
Its direct stages are backed by upstream SuperNeo Section 7.1 contexts, and its
terminal public state remains parent-only.

An opaque prior proof is accepted by one of two equivalent authority-opening
surfaces:

```text
PriorAuthorityOpener.openAuthority(proof)
  = some proof_carrying_folded_F_prime_authority
```

or, for an externally defined verifier:

```text
VerifyPrior(steps, proof, image)
=>
proof opens to FoldedFPrimeAuthority for the same (steps, image)
```

Acceptance exposes the authority opening itself: the opened authority cannot be
absent, and if the fixed opener returns `some authority`, that exact authority
accepts the same `(steps, image)` pair. The Section 7.1-backed concrete
verifier-with-opening-certificate surface exposes the same direct opening and
acceptance facts. It also exposes prior-image reachability directly and rules
out accepting an unreachable prior image.

The fixed opener or fixed opening certificate also supplies generic same-proof
functionality:

```text
same opaque proof
two accepted prior (steps, image) pairs
=>
the two prior pairs are equal
```

This is the replay-stability condition that a bare per-acceptance
`SoundVerifier` does not provide.

The endpoint theorem composes that prior-opening requirement with the
Section 7.1-backed production context and derives:

```text
prior image reachability
terminal image reachability
Construction-2 public-image invariants
deterministic boundary update
pointwise private DEC child audit trail
contextual Pi_CCS -> Pi_RLC stage audit
```

The replay guard states that the same opaque prior proof cannot be reused under
the same opener or externally opened verifier to authorize a different prior
pair or terminal public image. Both the prior-pair and terminal-image guards are
exposed at this Section 7.1-backed boundary.

This module does not model Poseidon2 internals and does not treat digest
consistency as prior authority. Authority comes only from opening accepted
prior proofs to folded `F'` reachability for the exact public image.
