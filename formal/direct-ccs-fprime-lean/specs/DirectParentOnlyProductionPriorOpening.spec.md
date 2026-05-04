# DirectParentOnlyProductionPriorOpening

`DirectParentOnlyProductionPriorOpening` specifies the production opening
surface for opaque compressed prior proofs in the parent-only direct CCS `F'`
path.

The theorem-facing requirement is:

```text
VerifyPrior(steps, proof, image)
  => openAuthority(proof) = some authority
  => FoldedFPrimeAuthority.Accepts(steps, authority, image)
```

The opened authority is proof-carrying folded `F'` reachability under the
production context's exact transition and initial image. A digest or compact
proof representation may name that authority, but accepted verifier output is
authority only through this opened reachability object.

An opener-induced verifier accepts exactly when the opener returns authority for
the same `(steps, image)` pair. A separate concrete verifier can also use the
same shape by providing `PriorVerifierAuthorityOpening`: acceptance of that
verifier implies the fixed opener returns authority accepted for the same
terminal prior pair.

Accepted verifier output also exposes this opening directly: acceptance rules
out `openAuthority(proof) = none`, and if the opener returns `some authority`,
that exact authority accepts the same `(steps, image)` pair. The concrete
verifier-with-opening-certificate surface exposes the same opened authority and
the same exact acceptance fact.

Both forms construct the canonical
`CompressedFPrimeAuthority.SoundVerifier` object consumed by the strict
production theorem surfaces. The constructed sound verifier accepts exactly the
opener-induced verifier predicate, or exactly the supplied concrete verifier
predicate, respectively.

Both forms also satisfy the generic same-proof functionality property:

```text
same verifier
same opaque proof
two accepted prior (steps, image) pairs
=>
the two prior pairs are equal
```

This is the replay-stability requirement that is not implied by
`CompressedFPrimeAuthority.SoundVerifier` alone.

For a fixed opener, an opaque proof opens to at most one proof-carrying
authority. Consequently, if the same proof is accepted for two prior
`(steps, image)` pairs, those pairs are equal. This is the prior-side replay
and substitution guard: a compact proof cannot be reused as authority for a
different prior public image unless it opens to authority for that exact image.

At terminal compression level, the same guard applies to accepted terminal
objects: if two accepted terminal objects reuse one opaque prior proof under the
same opener or opening certificate, their prior `steps` and prior public image
are equal. The latest proof component may differ, but the prior authority pair
cannot be retargeted. Since the latest step relation is the production
Construction-2 transition, the second accepted latest step is also an alternate
latest transition for the first object once that prior pair is fixed. Therefore
the accepted terminal `nextImage` is functional for one opaque prior proof under
the same opener or opening certificate.

The terminal theorem composes this prior-authority opening with the parent-only
production theorem. Its conclusion includes:

```text
Reachable(Transition(ctx), ctx.initial, priorSteps, priorImage)
Reachable(Transition(ctx), ctx.initial, priorSteps + 1, nextImage)
nextImage = ComputedNextImage(ctx, priorSteps, priorImage, priorInputs)
altNext   = ComputedNextImage(ctx, priorSteps, priorImage, priorInputs)
TerminalChildAuditTrail(ctx, priorSteps, priorImage, nextImage, altNext)
FixedCEChildMembership(ctx.params, ctx.ce, priorInputs)
unique pointwise-valid priorInputs for the parent source
```

The audited public endpoint form additionally packages the flattened terminal
public-image facts:

```text
AuditedPublicEndpoint(ctx, priorSteps, priorImage, nextImage, altNext)
```

which includes prior reachability, terminal reachability at `priorSteps + 1`,
equality with any alternate latest transition from the same prior image,
deterministic boundary update, `step`, `vkDigest`, `initialBoundary`,
well-formedness, and the terminal child audit trail.

The stage-audited endpoint form additionally packages:

```text
AuditedPublicEndpointWithStageAudit(ctx, priorSteps, priorImage, nextImage, altNext)
```

which includes the contextual `Pi_CCS -> Pi_RLC` audit for the same private
child table and the same opened prior authority.

`TerminalChildAuditTrail` exposes the accepted private `Pi_DEC` decomposition,
binary fixed-length child columns, per-column Goldilocks recomposition to the
opened parent residues, witness-table identity, next-`Pi_CCS` wire identity,
and fixed CE child membership for the unique child table.

This rules out self-consistent prior digest acceptance as an authority source:
any accepted prior proof must open to folded `F'` reachability for the exact
prior image consumed by the terminal latest step, and terminal soundness is
derived from that reachable prior image plus the checked latest transition.
