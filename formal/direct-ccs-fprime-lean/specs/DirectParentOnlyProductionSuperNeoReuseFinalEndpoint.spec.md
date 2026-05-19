# Direct Parent Only Production SuperNeo Reuse Final Endpoint

`DirectParentOnlyProductionSuperNeoReuseFinalEndpoint` specifies the final
implementation-facing replay endpoint for the Section 7.1-backed parent-only
production path.

The endpoint consumes a raw compressed prior verifier only through:

```text
verify
fixed prior-authority opener
accepted proof -> opened folded F' authority for the same (steps, image)
```

The endpoint also accepts the canonical `PriorVerifierAuthorityOpening`
certificate as the preferred production input. That certificate owns the fixed
opener and the accepted-opens theorem together, so the final call surface does
not treat verifier acceptance or digest consistency as authority by itself.
All endpoint facts exposed for the raw accepted-opens form are also exposed
directly from this certificate form.

Given two terminal acceptances under the same raw verifier and the same opaque
prior proof, the endpoint exposes:

```text
openAuthority(priorProof) is nonempty
the opened folded F' authority accepts the first prior (steps, image) pair
same prior step count
same prior public image
same terminal public image
computed Pi_CCS -> Pi_RLC -> Pi_DEC stage evidence
explicit pointwise no-swap evidence for the private DEC child table
```

The no-swap evidence is pointwise. It is quantified against every alternate
child table satisfying the full private DEC requirements for the same compact
parent source. Aggregate summaries, digest self-consistency, and norm-sum
equalities do not constitute authority at this boundary.

The endpoint exposes a direct no-swap theorem for an adversarial alternate
child table. Its hypothesis is the complete `PointwisePrivateDecRequirements`
predicate for the same parent source; its conclusion identifies the alternate
table with the audited table and returns the exact parent-source stage audits.

The endpoint also exposes named projections for same-proof replay stability,
opened prior folded authority, computed-stage replay evidence, and explicit
pointwise no-swap evidence.
The computed-stage projections also expose the exact computed `Pi_CCS` context
that consumes the pointwise private DEC child table and the exact computed
`Pi_RLC` context that produces the compact parent source.
The same projections are available from the canonical certificate surface, so
production callers can stay on the authority-carrying path after the verifier
opening has been proved.

The public-image projection exposes final reachability under the induced
Construction-2 `F'` transition, the final step value, verifier-key digest
preservation, initial-boundary preservation, and final well-formedness. The
stage-audit projection exposes the contextual `Pi_CCS -> Pi_RLC` audit for the
same pointwise private DEC child table.
