# Direct Parent Only Production Concrete FPrime Prior

`DirectParentOnlyProductionConcreteFPrimePrior` specifies the authority bridge
for a concrete compressed prior `F'` verifier in the Section 7.1-backed
parent-only production path.

The concrete verifier body contains the implementation checks that contribute
to verifier acceptance:

```text
compact image replay
Construction-2 public boundary replay
Fiat-Shamir / transcript replay
committed-step verifier acceptance
fixed authority opener
```

The checks-first surface separates the verifier body into:

```text
replay checks -> bound verifier statement
bound verifier statement + committed-step acceptance -> opened folded F' authority
```

This mirrors the implementation split: compact/public/transcript replay binds
what statement is being verified, and committed-step verifier soundness is the
only authority-producing obligation.

The canonical-statement binding surface refines the replay side:

```text
canonicalStatement(steps, image)
proofStatement(proof)
replay checks
=>
proofStatement(proof) = canonicalStatement(steps, image)
```

This is the typed shape for the verifier path that reconstructs the public
statement from the caller's `(steps, image)` pair and verifies the opaque proof
against that exact statement.

The statement-surface verifier refines the committed-verifier side to the
direct CCS verifier checks:

```text
statementPublicValid(canonicalStatement(steps, image))
proofBoundary(proof) = statementBoundary(canonicalStatement(steps, image))
terminalVerifierAccepted(
  terminalPublicValues(canonicalStatement(steps, image)),
  statementBoundary(canonicalStatement(steps, image)),
  terminalCommittedProof(proof))
```

The authority-producing theorem for this surface is:

```text
the direct CCS statement checks above
openAuthority(proof) = some authority
=>
FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

Thus the boundary matches the verifier shape that validates the final
Construction-2 public boundary, enforces exact public-boundary equality between
the proof and statement, and verifies the terminal committed proof against the
expected terminal public values and boundary.

The verifier acceptance predicate requires all replay checks and a successful
open through the fixed opener. Digest replay and public-image consistency are
treated as verifier-body checks, not as authority.

The verifier body must prove the authority theorem:

```text
all verifier-body checks pass
openAuthority(proof) = some authority
=>
FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

From that theorem the module derives the accepted-opens certificate consumed by
the production endpoint:

```text
VerifyPrior(steps, proof, image)
=>
exists authority:
  openAuthority(proof) = some authority
and FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

From the checks-first surface the module derives the same accepted-opens
certificate without exposing a loose accepted-opens premise to callers:

```text
compactImageReplay
construction2BoundaryReplay
transcriptReplay
=>
verifierStatement

verifierStatement
committedStepAccepted
openAuthority(proof) = some authority
=>
FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

From accepted prior verifier evidence the module derives the prior public-image
invariants of the folded `F'` chain:

```text
image.step = steps
initial.vkDigest = image.vkDigest
initial.initialBoundary = image.initialBoundary
WellFormed(image)
```

Verifier acceptance also has direct folded-authority consequences:

```text
VerifyPrior(body, steps, proof, image)
=>
Reachable(Transition(ctx.toProductionContext), ctx.initial, steps, image)

VerifyPrior(body, steps, proof, image)
and openAuthority(proof) = some authority
=>
FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

The checks, canonical-statement binding, and statement-surface verifier layers
expose the same reachability and exact-authority consequences. An accepted
concrete prior proof cannot authorize an unreachable prior image under the
production Construction-2 transition.

The module packages this certificate as a `CertifiedPriorVerifier` and exposes
the production latest-step end-to-end theorem through that packaged verifier.
The end-to-end theorem returns the existing terminal package: opened folded
`F'` authority, final public-image invariants, pointwise private `Pi_DEC`
audit, exact child-table no-swap, and contextual `Pi_CCS -> Pi_RLC` stage
audit.

The same-proof functionality theorem states that one opaque prior proof cannot
be accepted for two different `(steps, image)` pairs under the fixed opener.
