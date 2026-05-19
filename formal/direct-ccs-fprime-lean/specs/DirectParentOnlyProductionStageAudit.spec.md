# Direct Parent Only Production Stage Audit

`DirectParentOnlyProductionStageAudit` specifies the production endpoint that
exposes the SuperNeo stage facts behind the parent-only direct CCS `F'`
terminal path.

The public accumulator remains parent-only:

```text
parentSource
```

The audit relation reconstructs the child-carrying prior handle used internally
by the contextual stages:

```text
childCarryingPrior(parentSource, priorInputs)
```

For a parent source produced from a private pointwise child table, the stage
audit requires:

```text
out = computePiCCS(priorSteps, childCarryingPrior(parentSource, priorInputs))
out.step = priorSteps
source = computePiRLC(priorSteps, out)
Pi_CCS strong statement holds for out.ctx
Pi_DEC knowledge statement holds for out.ctx
Pi_RLC weak statement holds for piRLCContext(out, computePiRLC(priorSteps, out))
Pi_DEC knowledge statement holds for piRLCContext(out, computePiRLC(priorSteps, out))
```

The terminal audit trail combines:

```text
PointwisePrivateDecRequirements(parentSource, priorInputs)
PointwiseChildAuditTrail(parentSource, priorInputs)
nextImage = ComputedNextImage(priorImage, priorInputs)
altNext = ComputedNextImage(priorImage, priorInputs)
any other pointwise-valid table for parentSource equals priorInputs
ParentSourceStageAudit for nextImage.parentSource
ParentSourceStageAudit for altNext.parentSource
```

The interface exposes a direct projection from `ParentSourceStageAudit` to the
deterministic `Pi_RLC` source equality, the contextual `Pi_CCS` step fact, the
imported `Pi_CCS` strong statement, the imported `Pi_DEC` knowledge statement
for the exact computed `Pi_CCS` context, the imported `Pi_RLC` weak statement,
and the imported `Pi_DEC` knowledge statement for the exact computed `Pi_RLC`
context.

The terminal audit trail also exposes a flattened computed-stage evidence
projection:

```text
exists priorInputs,
  PointwisePrivateDecRequirements(parentSource, priorInputs)
  PointwiseChildAuditTrail(parentSource, priorInputs)
  nextImage = ComputedNextImage(priorImage, priorInputs)
  altNext = ComputedNextImage(priorImage, priorInputs)
  nextImage.parentSource = computePiRLC(priorSteps, out)
  altNext.parentSource = computePiRLC(priorSteps, out)
  any other pointwise-valid table for parentSource equals priorInputs
  out.step = priorSteps
  Pi_CCS strong statement holds for out.ctx
  Pi_DEC knowledge statement holds for out.ctx
  Pi_RLC weak statement holds for piRLCContext(out, computePiRLC(priorSteps, out))
  Pi_DEC knowledge statement holds for piRLCContext(out, computePiRLC(priorSteps, out))
```

The endpoint theorem combines the flattened public endpoint with that stage
audit trail. Prior authority may be supplied either as a sound prior verifier
object or as a concrete verifier plus an opening theorem to proof-carrying
folded `F'` authority for the same `(steps, image)`.

The theorem does not model Poseidon2 internals. It relies on the production
context's parent-hash binding object and on the contextual reused SuperNeo
stage package. The child table is authorized pointwise; aggregate child
summaries do not satisfy this audit relation.
