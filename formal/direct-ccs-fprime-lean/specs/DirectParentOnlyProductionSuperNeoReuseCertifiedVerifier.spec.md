# Direct Parent Only Production SuperNeo Reuse Certified Verifier

`DirectParentOnlyProductionSuperNeoReuseCertifiedVerifier` specifies the
implementation-facing object for a concrete compressed prior verifier in the
Section 7.1-backed parent-only production path.

The certified verifier object contains:

```text
verify : steps -> priorProof -> priorImage -> Prop
opening : PriorVerifierAuthorityOpening(ctx, verify)
```

The raw-verifier certification adapter builds this object from exactly:

```text
verify
fixed opener
accepted proof -> opened folded F' authority for the same (steps, image)
```

The adapter exposes the same one-terminal audit, one-terminal pointwise
uniqueness, same-proof replay audit package, and same-proof replay pointwise
uniqueness projection as the packaged certified verifier object.

The opening certificate is the authority boundary. For every accepted proof it
opens the same opaque proof to proof-carrying folded `F'` authority for the
same `(steps, image)` pair under the induced production transition.

The certified verifier exposes that fixed-opener boundary directly:

```text
verify(steps, proof, image)
=>
exists authority:
  opener.openAuthority(proof) = some authority
  and FoldedFPrimeAuthority.Accepts(steps, authority, image)

verify(steps, proof, image)
=>
opener.openAuthority(proof) != none

verify(steps, proof, image)
=>
Reachable(initial, steps, image)
```

It also rules out accepted proofs for unreachable prior public images.

The certified verifier induces the strict compressed-verifier object consumed
by the production theorem:

```text
CompressedFPrimeAuthority.SoundVerifier
```

and its accepted predicate is definitionally the certified `verify` predicate.

For one terminal acceptance and one alternate latest step under the same
context-fixed transition, the certified verifier exposes the production audited
endpoint directly:

```text
prior folded F' reachability
next folded F' reachability
same accepted and alternate terminal public image
step, vk digest, initial-boundary, and well-formedness invariants
pointwise private DEC child audit
contextual Pi_CCS -> Pi_RLC stage audit
```

It also exposes a flattened computed-stage projection for that endpoint:

```text
one pointwise-valid private DEC child table
both terminal images computed from that table
both terminal parent sources equal to the deterministic Pi_RLC(Pi_CCS(...)) result
pointwise uniqueness against every other valid child table for the same parent
imported Pi_CCS, Pi_RLC, and Pi_DEC statements for the exact computed contexts
```

The single-terminal audit package combines the opened prior authority, audited
endpoint, and flattened computed-stage endpoint evidence in one theorem. This
package is the terminal verifier surface for one accepted compressed proof;
replay stability is handled by the same-proof replay package.

The single-terminal pointwise uniqueness projection is quantified over an
arbitrary alternate child table. It concludes equality with the audited table
only after the alternate table satisfies the full private DEC requirements for
the same accepted parent source.

The certified verifier also satisfies same-proof functionality:

```text
verify(stepsA, proof, imageA)
verify(stepsB, proof, imageB)
=>
stepsA = stepsB and imageA = imageB
```

This replay-stability fact comes from the fixed opening certificate, not from a
bare `SoundVerifier`.

For two terminal acceptances under the same certified verifier and the same
opaque prior proof, the replay endpoint proves:

```text
same prior step count
same prior public image
same terminal public image
```

and exposes direct projections for:

```text
opened folded F' authority for the first prior pair
computed-stage replay evidence
pointwise private DEC child audit
contextual Pi_CCS -> Pi_RLC stage audit
pointwise no-swap uniqueness against any alternate child table
```

The child audit is pointwise: it requires the full private `Pi_DEC`
authorization, fixed CE/Ajtai membership, fixed child-column length,
per-column recomposition, witness-table identity, and next-`Pi_CCS` wire
identity. It is not an aggregate child-summary or norm-sum condition.

The no-swap projection is quantified over an arbitrary `otherInputs` table. If
that table satisfies the full `PointwisePrivateDecRequirements` for the same
replayed parent source, then it is equal to the audited private table that feeds
both terminal `Pi_CCS -> Pi_RLC` computations.

The raw-verifier no-swap projection has the same quantification. It may be
called with the concrete verifier predicate, fixed authority opener, and
accepted-opens theorem directly; no caller may replace the pointwise DEC
requirements with aggregate child summaries.

The raw-verifier replay audit package also exposes this no-swap projection as a
named conclusion. In addition to opened prior authority, same-proof replay, and
computed-stage evidence, it returns the audited child table together with the
universal statement that every other full pointwise DEC table for the same
parent source is equal to it.
