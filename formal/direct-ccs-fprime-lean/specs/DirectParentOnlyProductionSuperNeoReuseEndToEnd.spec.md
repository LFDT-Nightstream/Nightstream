# Direct Parent Only Production SuperNeo Reuse End To End

`DirectParentOnlyProductionSuperNeoReuseEndToEnd` specifies the final
theorem-facing soundness package for the Section 7.1-backed parent-only
production path.

The production entry point consumes:

```text
Section 7.1-backed production context
implementation-shaped exact-runtime F' prior verifier surface
public-IO layout binding for that verifier surface
exact-runtime prior verifier acceptance
latest Construction-2 step acceptance
```

Exact-runtime prior verifier acceptance must imply:

```text
fixed authority opener returns folded F' authority
opened authority accepts the same prior step count and public image
prior public-image invariants hold
the same opaque proof cannot be accepted for a different public pair
```

The latest-step evidence must prove:

```text
latest-step acceptance from priorProof.image to nextImage
private pointwise Pi_DEC requirements for the parent source
reuse of the same private child table in Pi_CCS
Pi_CCS -> Pi_RLC computation of the next parent source
```

The production projections expose the non-aggregate private DEC and stage facts
directly for the exact-runtime call path:

```text
accepted prior proof
latest-step acceptance
=>
accepted private Pi_DEC proof
fixed production CE relation and Ajtai parameters
binary length-14 child columns
per-column Goldilocks recomposition to the opened parent
CE-witness-derived child table identity
next-Pi_CCS wire identity
contextual Pi_CCS -> Pi_RLC parent-source computation
```

The same exact-runtime path also exposes the Section 7.1 owner-target audit
directly:

```text
one pointwise-valid private child table
both terminal images are computed from that table
computePiCCS consumes that child-carrying prior handle
computePiRLC produces both terminal parent sources from that Pi_CCS output
the computed Pi_CCS context is the target of piCCSSection71
the computed Pi_RLC context is the target of piRLCSection71
both Section 7.1 owner targets carry ceRelation
```

It exposes:

```text
opened folded F' authority for the same prior public image
same-proof replay stability
computed Pi_CCS -> Pi_RLC -> Pi_DEC stage evidence
exact computed Pi_CCS DEC knowledge over the pointwise private child table
exact computed Pi_RLC DEC knowledge for both replayed parent sources
flattened non-aggregate private DEC and stage facts
Section 7.1 owner-target audit
explicit pointwise no-swap evidence
all alternate private DEC child tables satisfy pointwise no-swap
flattened public endpoint
terminal stage audit
final Construction-2 public-image invariants
```

The certified end-to-end package itself contains the Section 7.1 owner-target
audit as a required field. A consumer of the final package therefore receives
the owner-target facts together with the public endpoint and terminal stage
audit, rather than relying on a separate theorem outside the package.

The flattened non-aggregate package exposes the accepted private `Pi_DEC`
proof, fixed production CE relation and Ajtai parameters, binary length-14
child columns, per-column Goldilocks recomposition to the opened parent,
CE-witness-derived child table identity, next-`Pi_CCS` wire identity, and the
same contextual `Pi_CCS -> Pi_RLC` computation that produces the parent source.

The alternate-child guarantee is quantified over every alternate table that
satisfies the full `PointwisePrivateDecRequirements` predicate for the same
parent source. Aggregate child summaries, norm-sum equalities, and
self-consistent digest recomputation do not satisfy this boundary.

The module does not specify the internals of Poseidon2. The parent hash enters
only through the parent-hash binding carried by the production context.
