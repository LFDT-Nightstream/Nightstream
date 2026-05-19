# Direct Parent Only Production SuperNeo Reuse

`DirectParentOnlyProductionSuperNeoReuse` specifies the implementation-facing
production context for the parent-only direct CCS `F'` path when stage authority
is reused from theorem-native SuperNeo Section 7.1 contexts.

The production context fixes:

```text
Poseidon2 parent CE(B) hash binding object
concrete Ajtai-backed CE data
Section 7.1-backed contextual Pi_CCS/Pi_RLC stage computations
deterministic boundary update
deterministic parent commitment encoder
initial public image and base invariants
MSIS-to-Ajtai reduction
MSIS hardness assumption
```

The stage field is not an arbitrary direct-stage authority. It is
`DirectStageSuperNeoReuse.Section71ContextualStageComputations`, so the
computed `Pi_CCS` output context and the computed `Pi_RLC` parent-source context
must match upstream `ProtocolSection71Context` targets.

The context converts to `DirectParentOnlyProductionSoundness.Context` by
deriving `ContextualReusedStageComputations` from upstream `ceRelation`.

The context exposes the imported SuperNeo statements for its actual computed
stage contexts:

```text
Pi_CCS strong statement for computePiCCS(i, prior).ctx
Pi_RLC weak statement for piRLCContext(out, computePiRLC(i, out))
Pi_DEC knowledge statement for both computed contexts
```

It also exposes a Section 7.1 owner-target audit:

```text
source = computePiRLC(i, computePiCCS(i, prior_with_children))
piCCSSection71(i, prior_with_children).target =
  computePiCCS(i, prior_with_children).ctx
piRLCSection71(i, out).target =
  piRLCContext(out, computePiRLC(i, out))
```

The same audit carries `ceRelation` for both Section 7.1 owner targets and the
imported `Pi_CCS`, `Pi_RLC`, and `Pi_DEC` statements for those exact contexts.
The terminal audit form quantifies one pointwise-valid private child table,
proves both terminal images are the deterministic computed images for that
table, and proves any other pointwise-valid table for the same parent source is
equal to it.

The endpoint theorem is the existing stage-audited parent-only production
theorem applied to that induced context:

```text
accepted terminal compression
alternate latest transition from the same prior image
=>
public endpoint facts
+ pointwise private child audit trail
+ contextual Pi_CCS -> Pi_RLC stage audit
```

The prior compressed verifier remains a `SoundPriorVerifier`: accepted prior
proofs must open to folded `F'` authority for the same `(steps, image)`. This
module does not model Poseidon2 internals or replace the prior-verifier
soundness obligation with digest consistency.
