# Direct Parent Only Production SuperNeo Reuse Replay Endpoint

`DirectParentOnlyProductionSuperNeoReuseReplayEndpoint` specifies the replay
endpoint for the Section 7.1-backed parent-only production context.

The authority-opener replay scenario is:

```text
same production context
same prior-authority opener
same opaque prior proof
two accepted terminal compressions
```

The concrete-verifier replay scenario is:

```text
same production context
same concrete VerifyPrior predicate
same prior-authority opening certificate
same opaque prior proof
two accepted terminal compressions
```

The theorem proves that the two terminal acceptances have:

```text
same prior step count
same prior public image
same terminal public image
```

and also exposes the full audited endpoint:

```text
prior image reachability
terminal image reachability
Construction-2 public-image invariants
deterministic boundary update
pointwise private DEC child audit trail
contextual Pi_CCS -> Pi_RLC stage audit
```

The pointwise child replay binding projection exposes the private child table
from that audited endpoint:

```text
PointwisePrivateDecRequirements(parentSource, priorInputs)
PointwiseChildAuditTrail(parentSource, priorInputs)
nextImage = ComputedNextImage(priorImage, priorInputs)
altNext = ComputedNextImage(priorImage, priorInputs)
any other pointwise-valid table for parentSource equals priorInputs
ParentSourceStageAudit for both replayed terminal images
```

The computed-stage replay evidence projection additionally exposes the exact
deterministic computation carried by the replayed endpoint:

```text
nextImage = ComputedNextImage(priorImage, priorInputs)
altNext = ComputedNextImage(priorImage, priorInputs)
both replayed parent sources equal computePiRLC(priorSteps, out)
out.step = priorSteps
the imported Pi_CCS statement holds for out.ctx
the imported Pi_DEC statement holds for out.ctx
the imported Pi_RLC and Pi_DEC statements hold for the exact computed Pi_RLC context
```

The interface also exposes named projections from computed-stage replay
evidence to:

```text
the pointwise private DEC child table and the exact computed Pi_CCS context
the exact computed Pi_RLC context and both replayed parent-source equalities
```

The replay audit package combines the opening and replay facts:

```text
openAuthority(priorProof) is nonempty
the opened authority accepts the first prior (steps, image) pair
same prior step count
same prior public image
same terminal public image
computed-stage replay evidence
```

The child-table uniqueness theorem is stated against the full pointwise private
DEC requirement:

```text
PointwisePrivateDecRequirements(parentSource, otherInputs)
  =>
exists priorInputs,
  PointwisePrivateDecRequirements(parentSource, priorInputs)
  and PointwiseChildAuditTrail(parentSource, priorInputs)
  and both replayed terminal images are computed from priorInputs
  and otherInputs = priorInputs
  and both replayed parent sources carry ParentSourceStageAudit
```

This is an anti-retargeting surface for the optimized parent-only public state.
It does not hash the raw post-DEC children, and it does not accept digest
consistency as authority. The prior proof is authority only through an opening
to folded `F'` reachability for the exact `(steps, image)` pair.
