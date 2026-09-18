# Direct Parent Only Production Endpoint

`DirectParentOnlyProductionEndpoint` specifies the packaged endpoint for the
optimized direct CCS `F'` terminal path whose public accumulator carries only
the parent `CE(B)` source.

The endpoint consumes the production context, a sound prior verifier or a raw
compressed prior verifier with an opening theorem, accepted terminal
compression, and an alternate latest transition checked against the same
context.

The unique-child conclusion is:

```text
PriorReachabilityAndUniquePointwiseChildren
```

which means:

```text
Reachable(Transition(ctx), ctx.initial, priorSteps, priorImage)
TerminalSoundness(ctx, priorSteps, priorImage, nextImage, altNext)
exists priorInputs:
  PointwisePrivateDecRequirements(ctx, priorImage.accumulator.parentSource, priorInputs)
  nextImage = ComputedNextImage(ctx, priorSteps, priorImage, priorInputs)
  altNext = ComputedNextImage(ctx, priorSteps, priorImage, priorInputs)
  every other pointwise-valid private DEC child table for the same parent
    source equals priorInputs
```

The audit-facing conclusion is:

```text
PriorReachabilityAndPointwiseChildAuditTrail
```

which means:

```text
Reachable(Transition(ctx), ctx.initial, priorSteps, priorImage)
TerminalSoundness(ctx, priorSteps, priorImage, nextImage, altNext)
TerminalChildAuditTrail(ctx, priorSteps, priorImage, nextImage, altNext)
```

The flattened public endpoint is:

```text
AuditedPublicEndpoint
```

which means:

```text
Reachable(Transition(ctx), ctx.initial, priorSteps, priorImage)
Reachable(Transition(ctx), ctx.initial, priorSteps + 1, nextImage)
nextImage = altNext
nextImage.accumulator.parentSource = altNext.accumulator.parentSource
nextImage.currentBoundary =
  ctx.computeBoundary(priorSteps, priorImage.currentBoundary)
altNext.currentBoundary =
  ctx.computeBoundary(priorSteps, priorImage.currentBoundary)
nextImage.step = priorSteps + 1
ctx.initial.vkDigest = nextImage.vkDigest
ctx.initial.initialBoundary = nextImage.initialBoundary
WellFormed(nextImage)
TerminalChildAuditTrail(ctx, priorSteps, priorImage, nextImage, altNext)
```

This endpoint treats the prior proof as authority only through folded `F'`
reachability. It treats private post-DEC children as valid only through the
pointwise requirements: accepted private `Pi_DEC`, fixed child CE relation and
Ajtai parameters, binary digits, exact child-column length, per-column
Goldilocks recomposition, witness-table identity, and wire identity into the
next `Pi_CCS` inputs.

No Poseidon2 permutation is modeled here. The parent hash assumption is the
implementation hash object's parent-encoding binding property supplied through
the production context.
